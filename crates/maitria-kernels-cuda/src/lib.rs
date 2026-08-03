//! NVIDIA GPU lanes for the `maitria-kernels` families.
//!
//! Two lanes: `rnsfold` (residue-channel fold; see the core crate's
//! module documentation for semantics and the exactness argument) and
//! `roweq` (structural row membership; comparison-only, no arithmetic
//! anywhere on device). Lane law (repository `README.md`): a lane may
//! change cost, never verdicts — each lane's outcome must be
//! bit-identical to its core reference (`rnsfold::reference::verify`,
//! `roweq::reference::verify`) on every input, and the conformance
//! batteries in `tests/` gate exactly that. Never a trusted computing
//! base.
//!
//! Pipeline of record (ENGINEERING #6): NVRTC + driver JIT, arch
//! resolved from the live device. The disassembly branch-count
//! acceptance item rides the committed receipts.
//!
//! Host plumbing — device bring-up, NVRTC compile/load with caching,
//! launch helpers, the zero-length-upload guard — lives in [`host`],
//! shared by both lanes here and importable by sibling host libraries
//! (see that module's docs for where it sits among the repository's
//! blessed host patterns).

#![deny(missing_docs)]

pub mod host;

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use host::{htod_nonempty, CompileSpec, CudaHost, HostError, PinnedStager, Uploader};
use maitria_kernels::rnsfold::batch::{BatchError, RnsFoldBatch, RnsFoldOutcome};
use maitria_kernels::rnsfold::primes::{residue_of_limbs, residue_signed, PRIMES};
use maitria_kernels::roweq::batch::{RowEqBatch, RowEqError, RowEqOutcome};

/// The embedded rnsfold kernel source (compiled once per process, at
/// device arch, by NVRTC).
pub const RNSFOLD_CU: &str = include_str!("../kernels/rnsfold.cu");

/// The embedded roweq kernel source (compiled once per process, at
/// device arch, by NVRTC).
pub const ROWEQ_CU: &str = include_str!("../kernels/roweq.cu");

/// Typed lane errors (ENGINEERING #7: refusals name their cause).
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    /// The descriptor failed structural validation.
    #[error("descriptor invalid: {0}")]
    Batch(#[from] BatchError),
    /// The roweq descriptor failed structural validation.
    #[error("descriptor invalid: {0}")]
    RowEq(#[from] RowEqError),
    /// CUDA driver error (device, memory, launch).
    #[error("cuda: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    /// NVRTC compilation failure.
    #[error("nvrtc: {0}")]
    Nvrtc(String),
    /// Geometry exceeds what the launch can express.
    #[error("launch geometry: {0}")]
    Geometry(&'static str),
}

impl From<HostError> for GpuError {
    fn from(e: HostError) -> Self {
        match e {
            HostError::Driver(d) => GpuError::Driver(d),
            HostError::Nvrtc { module, arch, log } => {
                GpuError::Nvrtc(format!("'{module}' ({arch}): {log}"))
            }
            HostError::Geometry(g) => GpuError::Geometry(g),
        }
    }
}

/// `-p^{-1} mod 2^64` for odd `p` (Newton iteration; verified by the
/// caller's assert below and by the battery).
fn neg_inv_u64(p: u64) -> u64 {
    let mut x: u64 = p; // correct mod 2^3
    for _ in 0..5 {
        x = x.wrapping_mul(2u64.wrapping_sub(p.wrapping_mul(x)));
    }
    debug_assert_eq!(p.wrapping_mul(x), 1);
    x.wrapping_neg()
}

/// `(x << 64) % p` — enter Montgomery form.
fn to_mont(x: u64, p: u64) -> u64 {
    (((x as u128) << 64) % p as u128) as u64
}

/// The rnsfold CUDA lane: device context + compiled kernel, reusable
/// across many [`RnsFoldGpu::verify`] calls.
pub struct RnsFoldGpu {
    host: CudaHost,
    func: CudaFunction,
    /// Reusable pinned staging pair for plane uploads (the
    /// upload lever; see [`host::PinnedStager`]). Behind a mutex only
    /// because `verify` takes `&self`; contention is per-lane-object.
    stager: std::sync::Mutex<PinnedStager>,
    /// (major, minor) compute capability of the bound device.
    pub cc: (i32, i32),
}

impl RnsFoldGpu {
    /// Bind device 0 and NVRTC-compile the kernel for its arch.
    pub fn new() -> Result<Self, GpuError> {
        let host = CudaHost::new()?;
        // Deployed-pipeline observability (ENGINEERING #6): the SASS
        // branch-count acceptance item must disassemble the PTX the
        // NVRTC pipeline actually emitted, not an offline-nvcc proxy.
        let spec = CompileSpec {
            dump_ptx_env: Some("MAITRIA_KERNELS_DUMP_PTX"),
            ..Default::default()
        };
        let func = host
            .compile("rnsfold", RNSFOLD_CU, &spec)?
            .load_function("rnsfold_fold")?;
        let cc = host.cc;
        let stager = std::sync::Mutex::new(PinnedStager::from_env(&host.ctx)?);
        Ok(Self {
            host,
            func,
            stager,
            cc,
        })
    }

    /// One line of device identification for receipts.
    pub fn device_summary(&self) -> Result<String, GpuError> {
        Ok(self.host.device_summary()?)
    }

    /// Evaluate the batch on device. Outcome contract: bit-identical
    /// to the reference lane (same `fold_ok`, same `refused`, same
    /// `channels_used`) on every valid descriptor.
    ///
    /// Host-side phase walls print to stderr (one TSV line) when
    /// `MAITRIA_KERNELS_PHASE_LOG` is set — observability for the
    /// dispatch-threshold receipts, off by default, no output-path
    /// effect. The residue tables build in parallel (rayon, channel
    /// grain): per-entry results are independent and the arithmetic is
    /// unchanged, so the tables are bit-identical to the serial
    /// spelling's — cost, never verdicts.
    ///
    /// Plane uploads ride double-buffered pinned staging by default
    /// ([`host::PinnedStager`] — the upload-wall lever; knobs:
    /// `MAITRIA_KERNELS_PAGEABLE` restores the direct pageable path
    /// for A/B receipts, `MAITRIA_KERNELS_STAGE_CHUNK` sets the
    /// staging buffer size in bytes). Same bytes either way — cost,
    /// never verdicts.
    pub fn verify(&self, b: &RnsFoldBatch) -> Result<RnsFoldOutcome, GpuError> {
        use rayon::prelude::*;
        let phase_log = std::env::var_os("MAITRIA_KERNELS_PHASE_LOG").is_some();
        let t0 = std::time::Instant::now();
        b.validate()?;
        let t_validate = t0.elapsed();

        let t = std::time::Instant::now();
        let (channels, refused) = b.plan_channels();
        let t_plan = t.elapsed();
        let n_attempts = b.n_attempts();
        let n_acols = b.concl_slot.len();
        if n_attempts == 0 || n_acols == 0 {
            return Ok(RnsFoldOutcome {
                fold_ok: refused.iter().map(|r| !r).collect(),
                refused,
                channels_used: channels,
            });
        }
        if channels > u16::MAX as usize {
            return Err(GpuError::Geometry("channel count"));
        }

        // ---- host-side channel tables (Montgomery form) ----
        let t = std::time::Instant::now();
        let ps: Vec<u64> = PRIMES[..channels].to_vec();
        let pinvs: Vec<u64> = ps.iter().map(|&p| neg_inv_u64(p)).collect();
        // powr2[ch*k + l] = 2^{64(l+2)} mod p
        let k = b.k.max(1);
        let mut powr2 = vec![0u64; channels * k];
        powr2
            .par_chunks_mut(k)
            .zip(ps.par_iter())
            .for_each(|(row, &p)| {
                let base = ((1u128 << 64) % p as u128) as u64;
                // x = 2^{64*(l+2)} mod p, starting at l = 0 -> 2^128 mod p
                let mut x = to_mont(base, p); // base * 2^64 = 2^128 mod p
                for slot in row.iter_mut().take(b.k) {
                    *slot = x;
                    x = ((x as u128 * base as u128) % p as u128) as u64;
                }
            });
        let n_lams = b.lams.len().max(1);
        let mut lamres = vec![0u64; channels * n_lams];
        lamres
            .par_chunks_mut(n_lams)
            .zip(ps.par_iter())
            .for_each(|(row, &p)| {
                for (slot, (s, m)) in row.iter_mut().zip(b.lams.iter()) {
                    *slot = to_mont(residue_signed(*s, m, p), p);
                }
            });
        let n_mults = b.mults.len().max(1);
        let mut multres = vec![0u64; channels * n_mults];
        multres
            .par_chunks_mut(n_mults)
            .zip(ps.par_iter())
            .for_each(|(row, &p)| {
                for (slot, m) in row.iter_mut().zip(b.mults.iter()) {
                    *slot = to_mont(residue_of_limbs(m, p), p);
                }
            });
        // acol -> attempt
        let mut acol_attempt = vec![0u32; n_acols];
        for a in 0..n_attempts {
            for acol in b.acol_ptr[a]..b.acol_ptr[a + 1] {
                acol_attempt[acol as usize] = a as u32;
            }
        }
        let sign_i32: Vec<i32> = b.sign.par_iter().map(|&s| s as i32).collect();
        let t_tables = t.elapsed();

        // ---- upload ----
        // Pinned staging by default (the upload-wall lever);
        // `MAITRIA_KERNELS_PAGEABLE` selects the direct pageable path
        // for A/B receipts. Both paths move identical bytes — cost,
        // never verdicts. Phase-wall note: staged DMAs are truly
        // async, so `upload` measures fill+issue and the DMA tail of
        // the final chunks lands in `launch+sync` — compare the SUM of
        // the two walls across modes, not `upload` alone.
        let t = std::time::Instant::now();
        let st = &self.host.stream;
        let mut stager = self.stager.lock().unwrap();
        let mut up = if std::env::var_os("MAITRIA_KERNELS_PAGEABLE").is_some() {
            Uploader::Pageable
        } else {
            Uploader::Pinned(&mut stager)
        };
        let d_primes = up.clone_htod(st, &ps)?;
        let d_pinvs = up.clone_htod(st, &pinvs)?;
        let d_powr2 = up.clone_htod(st, &powr2)?;
        let d_lamres = up.clone_htod(st, &lamres)?;
        let d_multres = up.clone_htod(st, &multres)?;
        let d_sign = up.clone_htod(st, &sign_i32)?;
        let d_mag = up.clone_htod(st, &b.mag)?;
        let d_multid = up.clone_htod(st, &b.mult_id)?;
        let d_acolat = up.clone_htod(st, &acol_attempt)?;
        let d_cscptr = up.clone_htod(st, &b.csc_ptr)?;
        // nnz planes may be empty; keep cudarc away from zero-length allocs.
        let d_csclam = up.htod_nonempty(st, &b.csc_lam)?;
        let d_cscslot = up.htod_nonempty(st, &b.csc_slot)?;
        let d_concl = up.clone_htod(st, &b.concl_slot)?;
        let mut d_flags = st.alloc_zeros::<u32>(n_attempts)?;

        let t_upload = t.elapsed();

        // ---- launch: x over acols, y over channels ----
        let t = std::time::Instant::now();
        const BLOCK: u32 = 256;
        let gx = (n_acols as u64).div_ceil(BLOCK as u64);
        if gx > i32::MAX as u64 {
            return Err(GpuError::Geometry("acol grid"));
        }
        let cfg = LaunchConfig {
            grid_dim: (gx as u32, channels as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (n_acols_u, n_slots_u, k_u) = (n_acols as u32, b.n_slots as u32, b.k as u32);
        let (n_lams_u, n_mults_u) = (n_lams as u32, n_mults as u32);
        let mut la = st.launch_builder(&self.func);
        la.arg(&n_acols_u)
            .arg(&n_slots_u)
            .arg(&k_u)
            .arg(&n_lams_u)
            .arg(&n_mults_u)
            .arg(&d_primes)
            .arg(&d_pinvs)
            .arg(&d_powr2)
            .arg(&d_lamres)
            .arg(&d_multres)
            .arg(&d_sign)
            .arg(&d_mag)
            .arg(&d_multid)
            .arg(&d_acolat)
            .arg(&d_cscptr)
            .arg(&d_csclam)
            .arg(&d_cscslot)
            .arg(&d_concl)
            .arg(&mut d_flags);
        // SAFETY: kernel signature matches the argument list above;
        // all buffers sized per the validated descriptor.
        unsafe { la.launch(cfg) }?;
        st.synchronize()?;
        let t_launch = t.elapsed();
        let t = std::time::Instant::now();
        let flags: Vec<u32> = st.clone_dtoh(&d_flags)?;

        let fold_ok: Vec<bool> = (0..n_attempts)
            .map(|a| !refused[a] && flags[a] == 0)
            .collect();
        let t_download = t.elapsed();
        if phase_log {
            eprintln!(
                "rnsfold-gpu-phases\tvalidate\t{:.1}\tplan\t{:.1}\ttables\t{:.1}\tupload\t{:.1}\tlaunch+sync\t{:.1}\tdownload\t{:.1}\t(us; acols={} slots={} k={} channels={})",
                t_validate.as_secs_f64() * 1e6,
                t_plan.as_secs_f64() * 1e6,
                t_tables.as_secs_f64() * 1e6,
                t_upload.as_secs_f64() * 1e6,
                t_launch.as_secs_f64() * 1e6,
                t_download.as_secs_f64() * 1e6,
                n_acols,
                b.n_slots,
                b.k,
                channels,
            );
        }
        Ok(RnsFoldOutcome {
            fold_ok,
            refused,
            channels_used: channels,
        })
    }
}

/// The roweq CUDA lane: device context + compiled kernel, reusable
/// across many [`RowEqGpu::verify`] calls. Comparison-only on device;
/// outcome contract: bit-identical to
/// `maitria_kernels::roweq::reference::verify` on every valid
/// descriptor.
pub struct RowEqGpu {
    host: CudaHost,
    func: CudaFunction,
    /// (major, minor) compute capability of the bound device.
    pub cc: (i32, i32),
}

impl RowEqGpu {
    /// Bind device 0 and NVRTC-compile the kernel for its arch.
    pub fn new() -> Result<Self, GpuError> {
        let host = CudaHost::new()?;
        // Deployed-pipeline observability (ENGINEERING #6), same hook
        // shape as the rnsfold lane.
        let spec = CompileSpec {
            dump_ptx_env: Some("MAITRIA_KERNELS_DUMP_ROWEQ_PTX"),
            ..Default::default()
        };
        let func = host
            .compile("roweq", ROWEQ_CU, &spec)?
            .load_function("roweq_member")?;
        let cc = host.cc;
        Ok(Self { host, func, cc })
    }

    /// One line of device identification for receipts.
    pub fn device_summary(&self) -> Result<String, GpuError> {
        Ok(self.host.device_summary()?)
    }

    /// Evaluate the batch on device. Outcome contract: bit-identical
    /// to the reference lane on every valid descriptor.
    pub fn verify(&self, b: &RowEqBatch) -> Result<RowEqOutcome, GpuError> {
        b.validate()?;
        let n_attempts = b.n_attempts();

        // Per-query launch tables: row id + this attempt's pool range.
        let mut q_row: Vec<u32> = Vec::new();
        let mut p_lo: Vec<u32> = Vec::new();
        let mut p_hi: Vec<u32> = Vec::new();
        // query index range per attempt, for the host-side reduction.
        let mut aq: Vec<(usize, usize)> = Vec::with_capacity(n_attempts);
        for a in 0..n_attempts {
            let start = q_row.len();
            for q in b.arow_ptr[a]..b.split[a] {
                q_row.push(q);
                p_lo.push(b.split[a]);
                p_hi.push(b.arow_ptr[a + 1]);
            }
            aq.push((start, q_row.len()));
        }
        let n_queries = q_row.len();
        if n_queries == 0 {
            // Every attempt is vacuously true (no query rows anywhere).
            return Ok(RowEqOutcome {
                member_ok: vec![true; n_attempts],
            });
        }

        // ---- upload (zero-length planes guarded, cudarc convention) ----
        let st = &self.host.stream;
        let d_qrow = st.clone_htod(&q_row)?;
        let d_plo = st.clone_htod(&p_lo)?;
        let d_phi = st.clone_htod(&p_hi)?;
        let d_rowptr = st.clone_htod(&b.row_ptr)?;
        let d_col = htod_nonempty(st, &b.nnz_col)?;
        let d_slot = htod_nonempty(st, &b.nnz_slot)?;
        let sign_i32: Vec<i32> = b.sign.iter().map(|&s| s as i32).collect();
        let d_sign = htod_nonempty(st, &sign_i32)?;
        let d_mag = htod_nonempty(st, &b.mag)?;
        let d_den = htod_nonempty(st, &b.den_id)?;
        let mut d_matched = st.alloc_zeros::<u32>(n_queries)?;

        // One thread per query row, no grid-stride loop in the kernel:
        // the exact 1-D helper, whose refusal names the geometry.
        let cfg = host::launch_1d(n_queries)?;
        let (nq_u, ns_u, k_u) = (n_queries as u32, b.n_slots as u32, b.k as u32);
        let mut la = st.launch_builder(&self.func);
        la.arg(&nq_u)
            .arg(&ns_u)
            .arg(&k_u)
            .arg(&d_qrow)
            .arg(&d_plo)
            .arg(&d_phi)
            .arg(&d_rowptr)
            .arg(&d_col)
            .arg(&d_slot)
            .arg(&d_sign)
            .arg(&d_mag)
            .arg(&d_den)
            .arg(&mut d_matched);
        // SAFETY: kernel signature matches the argument list above;
        // all buffers sized per the validated descriptor.
        unsafe { la.launch(cfg) }?;
        st.synchronize()?;
        let matched: Vec<u32> = st.clone_dtoh(&d_matched)?;

        let member_ok = aq
            .iter()
            .map(|&(lo, hi)| matched[lo..hi].iter().all(|&m| m == 1))
            .collect();
        Ok(RowEqOutcome { member_ok })
    }
}
