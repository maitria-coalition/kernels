//! The NVRTC/driver-JIT host harness: device bring-up, module
//! compile/load with per-process caching, launch-geometry helpers, and
//! the zero-length-upload guard. Every lane in this crate rides it, and
//! sibling host libraries may import it instead of re-writing the same
//! plumbing.
//!
//! **Position among the host patterns** (repository `gpu/README.md`;
//! design: `crates/maitria-kernels-xla/DESIGN.md`): this repository
//! blesses two ways to put a hand-written kernel on device. The
//! *primary* pattern for new consumers is the XLA custom-call surface —
//! hand CUDA/PTX hosted behind `XLA_FFI` handlers, participating in
//! fused XLA programs (the `gpu/` lane family; per-instruction rounding
//! control rides it, which is what makes it the sound home for
//! enclosure arithmetic). The NVRTC/driver-JIT harness here is the
//! *transitional* pattern: a freestanding host with no XLA runtime in
//! sight, right for standalone lanes and incumbent consumers, and
//! deliberately shaped so a consumer can hold one struct and migrate
//! lanes off it one at a time.
//!
//! Never a trusted computing base: nothing here affects verdicts — the
//! lane law (repository `README.md`) is enforced by each lane's
//! conformance battery, not by this plumbing.
//!
//! Pipeline of record (ENGINEERING #6): NVRTC + driver JIT, arch
//! resolved from the live device. [`CompileSpec::dump_ptx_env`] is the
//! deployed-pipeline observability hook — the disassembly acceptance
//! item must read the PTX this pipeline actually emitted, not an
//! offline-nvcc proxy.

use cudarc::driver::{
    result, CudaContext, CudaEvent, CudaModule, CudaSlice, CudaStream, DevicePtrMut, DeviceRepr,
    LaunchConfig,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Typed host errors (ENGINEERING #7: refusals name their cause).
#[derive(Debug, thiserror::Error)]
pub enum HostError {
    /// CUDA driver error (device, memory, module load, launch).
    #[error("cuda: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    /// NVRTC compilation failure, naming the module that failed.
    #[error("nvrtc compile of '{module}' ({arch}): {log}")]
    Nvrtc {
        /// The cache key of the module that failed to compile.
        module: String,
        /// The architecture the compile targeted.
        arch: String,
        /// The NVRTC error/log text.
        log: String,
    },
    /// Geometry exceeds what a launch can express.
    #[error("launch geometry: {0}")]
    Geometry(&'static str),
}

/// Options for one NVRTC module build.
///
/// `Default` is the plain build: base architecture, no extra options,
/// no dump hook.
#[derive(Debug, Clone, Default)]
pub struct CompileSpec {
    /// Appended to the device's compute architecture — `""` for the
    /// plain target, `"a"` for the arch-specific target that gates
    /// arch-conditional instructions (e.g. block-scale tensor-core
    /// `mma`: `ptxas` refuses them on plain targets, so base modules
    /// never contain them and capable devices compile a suffixed twin).
    pub arch_suffix: &'static str,
    /// Extra NVRTC options, verbatim (e.g. `--device-int128`, `-DFOO`).
    pub options: Vec<String>,
    /// Deployed-pipeline observability (ENGINEERING #6): when set to
    /// `Some(var)` and the environment variable `var` names a path, the
    /// emitted PTX is written there after a successful compile.
    pub dump_ptx_env: Option<&'static str>,
}

/// One bound CUDA device: context, default stream, compute capability,
/// and a per-process module cache. Reusable across many kernel
/// invocations; cheap to share by reference.
///
/// Consumers own their kernel-source registries and pass `(key, src)`
/// pairs; the cache is keyed by `key` alone, so a source compiled under
/// several option sets needs one key per variant.
pub struct CudaHost {
    /// The bound device context.
    pub ctx: Arc<CudaContext>,
    /// The context's default stream.
    pub stream: Arc<CudaStream>,
    /// (major, minor) compute capability of the bound device.
    pub cc: (i32, i32),
    modules: Mutex<HashMap<String, Arc<CudaModule>>>,
}

impl CudaHost {
    /// Bind device 0.
    pub fn new() -> Result<Self, HostError> {
        Self::with_device(0)
    }

    /// Bind a device by ordinal.
    pub fn with_device(ordinal: usize) -> Result<Self, HostError> {
        let ctx = CudaContext::new(ordinal)?;
        let cc = ctx.compute_capability()?;
        let stream = ctx.default_stream();
        Ok(Self {
            ctx,
            stream,
            cc,
            modules: Mutex::new(HashMap::new()),
        })
    }

    /// One line of device identification for receipts.
    pub fn device_summary(&self) -> Result<String, HostError> {
        Ok(format!(
            "{} (compute_{}{}, {:.1} GiB)",
            self.ctx.name()?,
            self.cc.0,
            self.cc.1,
            self.ctx.total_mem()? as f64 / (1u64 << 30) as f64
        ))
    }

    /// The NVRTC target for this device: `compute_XY` + suffix.
    pub fn compute_arch(&self, suffix: &str) -> String {
        format!("compute_{}{}{}", self.cc.0, self.cc.1, suffix)
    }

    /// The SASS target for this device: `sm_XY` + suffix (the token
    /// hand-written PTX names in its `.target` directive).
    pub fn sm_target(&self, suffix: &str) -> String {
        format!("sm_{}{}{}", self.cc.0, self.cc.1, suffix)
    }

    /// NVRTC-compile CUDA source for the live device (once — cached by
    /// `key`) and JIT-load it.
    pub fn compile(
        &self,
        key: &str,
        src: &str,
        spec: &CompileSpec,
    ) -> Result<Arc<CudaModule>, HostError> {
        let mut cache = self.modules.lock().unwrap();
        if let Some(m) = cache.get(key) {
            return Ok(m.clone());
        }
        let arch_owned = self.compute_arch(spec.arch_suffix);
        // `CompileOptions.arch` wants `&'static str`; the leak is
        // deliberate and bounded — one short string per cache miss,
        // and each key compiles at most once per process.
        let arch: &'static str = Box::leak(arch_owned.clone().into_boxed_str());
        let opts = CompileOptions {
            arch: Some(arch),
            options: spec.options.clone(),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(src, opts).map_err(|e| HostError::Nvrtc {
            module: key.to_string(),
            arch: arch_owned,
            log: format!("{e:?}"),
        })?;
        if let Some(var) = spec.dump_ptx_env {
            if let Ok(path) = std::env::var(var) {
                let _ = std::fs::write(&path, ptx.to_src());
            }
        }
        let module = self.ctx.load_module(ptx)?;
        cache.insert(key.to_string(), module.clone());
        Ok(module)
    }

    /// JIT-load PTX text verbatim (once — cached by `key`): no NVRTC,
    /// the driver's JIT is the only compiler the text ever sees. The
    /// caller retargets any `.target` directive first ([`Self::sm_target`]
    /// is the live token).
    pub fn load_ptx(&self, key: &str, ptx_text: &str) -> Result<Arc<CudaModule>, HostError> {
        let mut cache = self.modules.lock().unwrap();
        if let Some(m) = cache.get(key) {
            return Ok(m.clone());
        }
        let module = self.ctx.load_module(Ptx::from_src(ptx_text))?;
        cache.insert(key.to_string(), module.clone());
        Ok(module)
    }
}

/// Thread-block width shared by the 1-D launch helpers.
pub const BLOCK_1D: u32 = 256;

/// Exact 1-D launch config: one thread per element, refusing geometry
/// the grid cannot express. For kernels written *without* a grid-stride
/// loop.
pub fn launch_1d(n: usize) -> Result<LaunchConfig, HostError> {
    let gx = (n as u64).div_ceil(BLOCK_1D as u64);
    if gx > i32::MAX as u64 {
        return Err(HostError::Geometry("1-d grid"));
    }
    Ok(LaunchConfig {
        grid_dim: (gx as u32, 1, 1),
        block_dim: (BLOCK_1D, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// Grid-stride 1-D launch config: the grid is clamped (65 535 blocks),
/// so the kernel must walk `blockDim.x * gridDim.x` strides. Total —
/// any `n`, including 0, yields a launchable config.
pub fn launch_1d_grid_stride(n: usize) -> LaunchConfig {
    let grid = ((n as u64).div_ceil(BLOCK_1D as u64)).min(65_535) as u32;
    LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK_1D, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Upload a host slice, padding the empty case to one zeroed element —
/// cudarc refuses zero-length allocations, and kernels index such
/// planes only under counts that are then also zero.
pub fn htod_nonempty<T: DeviceRepr + Default>(
    stream: &Arc<CudaStream>,
    v: &[T],
) -> Result<CudaSlice<T>, cudarc::driver::DriverError> {
    if v.is_empty() {
        stream.clone_htod(&[T::default()])
    } else {
        stream.clone_htod(v)
    }
}

// ---- pinned-host staging (the upload-wall lever) ----

/// Below this many bytes a "staged" upload delegates to the direct
/// pageable path: the staging turn's fixed costs (an event wait, a
/// separate DMA issue) exceed the pageable copy of a small plane, and
/// the delegation keeps zero-length semantics byte-for-byte identical
/// to `CudaStream::clone_htod`.
pub const STAGE_BYPASS_BYTES: usize = 64 << 10;

/// Double-buffered page-locked staging for host→device uploads.
///
/// The lane's phase receipts measured the rnsfold device lane
/// *upload-dominated*: pageable H2D ran at ~8–9 GB/s while the bus
/// could carry several times that from pinned memory, and the driver's
/// internal pageable staging serializes against the CPU. This stager
/// owns two page-locked buffers and rotates them: while the DMA out of
/// one buffer is in flight, the CPU fills the other (a parallel memcpy
/// — the buffers are write-combined, which is exactly right for
/// streaming stores that are never read back). Each buffer's CUDA
/// event guards its reuse; the rotation persists across planes, so the
/// last chunk of one plane overlaps the first fill of the next.
///
/// Cost, never verdicts (lane law, repository `README.md`): the staged
/// path moves exactly the caller's elements — a typed memcpy into the
/// pinned buffer, then an async DMA of those same elements — and
/// `tests/staging.rs` gates round-trip byte-identity plus the
/// engagement witness ([`Self::staged_chunks`]).
pub struct PinnedStager {
    bufs: [cudarc::driver::PinnedHostSlice<u8>; 2],
    events: [CudaEvent; 2],
    turn: usize,
    chunk: usize,
    staged_chunks: u64,
}

impl PinnedStager {
    /// Default per-buffer size: 32 MiB — ~20 turns at the X1-batch
    /// plane volume, small enough that two of them are noise against a
    /// GPU box's host RAM.
    pub const DEFAULT_CHUNK: usize = 32 << 20;

    /// Allocate the staging pair on `ctx`. `chunk` is the per-buffer
    /// byte size, rounded up to a 4 KiB multiple (so every lane
    /// element type divides it).
    pub fn new(ctx: &Arc<CudaContext>, chunk: usize) -> Result<Self, cudarc::driver::DriverError> {
        let chunk = chunk.next_multiple_of(4096).max(4096);
        // SAFETY: the buffers are uninitialized here; every read of
        // buffer bytes happens only after the staging loop has written
        // exactly those bytes (each turn fills `[..n]` then uploads
        // `[..n]`).
        let bufs = [unsafe { ctx.alloc_pinned::<u8>(chunk) }?, unsafe {
            ctx.alloc_pinned::<u8>(chunk)
        }?];
        let events = [ctx.new_event(None)?, ctx.new_event(None)?];
        Ok(Self {
            bufs,
            events,
            turn: 0,
            chunk,
            staged_chunks: 0,
        })
    }

    /// [`Self::new`] with the chunk size from
    /// `MAITRIA_KERNELS_STAGE_CHUNK` (bytes; unset or unparsable ⇒
    /// [`Self::DEFAULT_CHUNK`]) — the receipts' tuning knob.
    pub fn from_env(ctx: &Arc<CudaContext>) -> Result<Self, cudarc::driver::DriverError> {
        let chunk = std::env::var("MAITRIA_KERNELS_STAGE_CHUNK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(Self::DEFAULT_CHUNK);
        Self::new(ctx, chunk)
    }

    /// How many chunks have gone through the staged path — the
    /// engagement witness for the battery (a stager that silently
    /// delegated everything would report 0 and fail the test).
    pub fn staged_chunks(&self) -> u64 {
        self.staged_chunks
    }

    /// Upload a host slice to a fresh device buffer through the
    /// staging pair. Small and empty slices delegate to
    /// `CudaStream::clone_htod` ([`STAGE_BYPASS_BYTES`]).
    pub fn clone_htod<T: DeviceRepr + Copy + Send + Sync>(
        &mut self,
        stream: &Arc<CudaStream>,
        src: &[T],
    ) -> Result<CudaSlice<T>, cudarc::driver::DriverError> {
        if std::mem::size_of_val(src) <= STAGE_BYPASS_BYTES {
            return stream.clone_htod(src);
        }
        // SAFETY: uninitialized allocation; the staging loop below
        // writes every element before anything reads the buffer.
        let mut dst = unsafe { stream.alloc::<T>(src.len()) }?;
        self.stage_into(stream, src, &mut dst)?;
        Ok(dst)
    }

    /// [`htod_nonempty`], staged: the empty case pads to one zeroed
    /// element exactly as the direct helper does.
    pub fn htod_nonempty<T: DeviceRepr + Copy + Send + Sync + Default>(
        &mut self,
        stream: &Arc<CudaStream>,
        v: &[T],
    ) -> Result<CudaSlice<T>, cudarc::driver::DriverError> {
        if v.is_empty() {
            stream.clone_htod(&[T::default()])
        } else {
            self.clone_htod(stream, v)
        }
    }

    fn stage_into<T: DeviceRepr + Copy + Send + Sync>(
        &mut self,
        stream: &Arc<CudaStream>,
        src: &[T],
        dst: &mut CudaSlice<T>,
    ) -> Result<(), cudarc::driver::DriverError> {
        stream.context().bind_to_thread()?;
        let (dst_ptr, _record_dst) = dst.device_ptr_mut(stream);
        let chunk_elems = self.chunk / std::mem::size_of::<T>();
        debug_assert!(chunk_elems > 0);
        let mut off = 0usize;
        while off < src.len() {
            let n = chunk_elems.min(src.len() - off);
            let i = self.turn & 1;
            self.turn += 1;
            // CPU-side wait: the previous DMA out of buffer `i` must
            // have drained before its bytes are overwritten.
            self.events[i].synchronize()?;
            {
                // SAFETY: the pinned allocation is page-aligned (so
                // aligned for any lane element type) and `n * size_of
                // ::<T>() <= chunk` by construction; the embedded
                // event wait in `as_mut_ptr` is a no-op (that event is
                // never recorded on this path — `events[i]` above is
                // the guard).
                let stage: &mut [T] = unsafe {
                    std::slice::from_raw_parts_mut(self.bufs[i].as_mut_ptr()? as *mut T, n)
                };
                par_fill(stage, &src[off..off + n]);
                // SAFETY: `dst` was allocated with `src.len()` elements
                // and `off + n <= src.len()`, so the destination range
                // is in bounds; `stage` is page-locked, so the async
                // copy reads it at bus rate and `events[i]` (recorded
                // below) guards its reuse.
                unsafe {
                    result::memcpy_htod_async(
                        dst_ptr + (off * std::mem::size_of::<T>()) as u64,
                        &stage[..n],
                        stream.cu_stream(),
                    )
                }?;
            }
            self.events[i].record(stream)?;
            self.staged_chunks += 1;
            off += n;
        }
        Ok(())
    }
}

impl Drop for PinnedStager {
    fn drop(&mut self) {
        // A pinned buffer freed while a DMA still reads from it would
        // be undefined behavior; drain both guard events first (errors
        // ignored — there is nothing further to do at drop).
        for e in &self.events {
            let _ = e.synchronize();
        }
    }
}

/// Parallel memcpy for staging fills: sequential streaming stores per
/// rayon grain — the access pattern write-combined memory wants.
fn par_fill<T: Copy + Send + Sync>(dst: &mut [T], src: &[T]) {
    use rayon::prelude::*;
    const PAR_MIN_BYTES: usize = 4 << 20;
    const GRAIN_BYTES: usize = 2 << 20;
    debug_assert_eq!(dst.len(), src.len());
    if std::mem::size_of_val(src) < PAR_MIN_BYTES {
        dst.copy_from_slice(src);
    } else {
        let grain = (GRAIN_BYTES / std::mem::size_of::<T>()).max(1);
        dst.par_chunks_mut(grain)
            .zip(src.par_chunks(grain))
            .for_each(|(d, s)| d.copy_from_slice(s));
    }
}

/// Upload-path selector a lane threads through its plane uploads:
/// direct pageable (`CudaStream::clone_htod` — the pre-staging path,
/// kept selectable via `MAITRIA_KERNELS_PAGEABLE` for A/B receipts) or
/// pinned staging. Selection is cost-only; both paths move identical
/// bytes.
pub enum Uploader<'a> {
    /// Direct `clone_htod` from pageable memory.
    Pageable,
    /// Staged through a [`PinnedStager`].
    Pinned(&'a mut PinnedStager),
}

impl Uploader<'_> {
    /// Upload a (non-empty-semantics-preserving) host slice.
    pub fn clone_htod<T: DeviceRepr + Copy + Send + Sync>(
        &mut self,
        stream: &Arc<CudaStream>,
        v: &[T],
    ) -> Result<CudaSlice<T>, cudarc::driver::DriverError> {
        match self {
            Uploader::Pageable => stream.clone_htod(v),
            Uploader::Pinned(s) => s.clone_htod(stream, v),
        }
    }

    /// Upload with the zero-length pad ([`htod_nonempty`]).
    pub fn htod_nonempty<T: DeviceRepr + Copy + Send + Sync + Default>(
        &mut self,
        stream: &Arc<CudaStream>,
        v: &[T],
    ) -> Result<CudaSlice<T>, cudarc::driver::DriverError> {
        match self {
            Uploader::Pageable => htod_nonempty(stream, v),
            Uploader::Pinned(s) => s.htod_nonempty(stream, v),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_stride_config_clamps_and_floors() {
        let tiny = launch_1d_grid_stride(0);
        assert_eq!(tiny.grid_dim, (1, 1, 1));
        assert_eq!(tiny.block_dim, (BLOCK_1D, 1, 1));
        let mid = launch_1d_grid_stride(1000);
        assert_eq!(mid.grid_dim, (4, 1, 1));
        let huge = launch_1d_grid_stride(usize::MAX);
        assert_eq!(huge.grid_dim, (65_535, 1, 1));
    }

    #[test]
    fn exact_config_refuses_inexpressible_grids() {
        let ok = launch_1d(1000).unwrap();
        assert_eq!(ok.grid_dim, (4, 1, 1));
        assert_eq!(ok.block_dim, (BLOCK_1D, 1, 1));
        assert!(matches!(launch_1d(usize::MAX), Err(HostError::Geometry(_))));
    }
}
