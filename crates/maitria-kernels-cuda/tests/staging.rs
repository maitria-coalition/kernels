//! Pinned-staging battery: the staged upload path must move exactly
//! the caller's bytes (round-trip identity against a device readback),
//! across chunk boundaries, tails, the small-slice bypass, and the
//! zero-length pad — and it must actually ENGAGE (the
//! `staged_chunks` witness), so a stager that silently delegated
//! everything to the pageable path could not pass.
//!
//! GPU-gated like the conformance battery: no usable CUDA device ⇒
//! loud SKIP on stderr, not a failure — CPU CI boxes stay green; the
//! lane's receipts come from GPU boxes running exactly this file.

use maitria_kernels_cuda::host::{CudaHost, PinnedStager, Uploader, STAGE_BYPASS_BYTES};

fn cuda_host() -> Option<CudaHost> {
    match CudaHost::new() {
        Ok(h) => Some(h),
        Err(e) => {
            eprintln!("SKIP (no usable CUDA device): {e}");
            None
        }
    }
}

/// Deterministic non-trivial fill: distinct per index, both halves of
/// each word exercised.
fn pattern_u64(len: usize) -> Vec<u64> {
    (0..len as u64)
        .map(|x| x.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17) ^ 0xA5A5_5A5A_F00D_BEEF)
        .collect()
}

#[test]
fn staged_uploads_roundtrip_bit_identical() {
    let Some(host) = cuda_host() else { return };
    // Tiny chunk (one page) so modest slices cross many turns; the
    // bypass constant is independent of chunk size, so sizes just
    // above it exercise staged multi-chunk copies with tails.
    let mut stager = PinnedStager::new(&host.ctx, 4096).unwrap();

    let elems_at_bypass = STAGE_BYPASS_BYTES / std::mem::size_of::<u64>();
    let sizes = [
        1usize,              // bypass: single element
        elems_at_bypass - 1, // bypass: just under
        elems_at_bypass,     // bypass: exactly at (delegates, <=)
        elems_at_bypass + 1, // staged: just over, with a tail chunk
        3 * 4096 / 8,        // staged: exact multiple of the chunk
        3 * 4096 / 8 + 7,    // staged: multiple + ragged tail
        1_000_003,           // staged: ~8 MB, many turns, odd length
    ];
    for &len in &sizes {
        let src = pattern_u64(len);
        let d = stager.clone_htod(&host.stream, &src).unwrap();
        let back: Vec<u64> = host.stream.clone_dtoh(&d).unwrap();
        assert_eq!(back, src, "round-trip mismatch at len {len}");
    }
    assert!(
        stager.staged_chunks() > 0,
        "staged path never engaged — every size delegated to pageable"
    );

    // Narrower element types ride the same machinery.
    let src32: Vec<u32> = (0..300_000u32)
        .map(|x| x.wrapping_mul(2654435761) ^ 0xDEAD)
        .collect();
    let d32 = stager.clone_htod(&host.stream, &src32).unwrap();
    let back32: Vec<u32> = host.stream.clone_dtoh(&d32).unwrap();
    assert_eq!(back32, src32);

    let srci: Vec<i32> = (0..250_001i32)
        .map(|x| x.wrapping_mul(-40503) ^ 0x0BAD)
        .collect();
    let di = stager.clone_htod(&host.stream, &srci).unwrap();
    let backi: Vec<i32> = host.stream.clone_dtoh(&di).unwrap();
    assert_eq!(backi, srci);
}

#[test]
fn htod_nonempty_pads_zero_length_identically() {
    let Some(host) = cuda_host() else { return };
    let mut stager = PinnedStager::new(&host.ctx, 4096).unwrap();
    let empty: Vec<u32> = Vec::new();
    let d = stager.htod_nonempty(&host.stream, &empty).unwrap();
    let back: Vec<u32> = host.stream.clone_dtoh(&d).unwrap();
    assert_eq!(back, vec![0u32], "zero-length pad must match htod_nonempty");

    let nonempty = vec![7u32; 100_000];
    let d2 = stager.htod_nonempty(&host.stream, &nonempty).unwrap();
    let back2: Vec<u32> = host.stream.clone_dtoh(&d2).unwrap();
    assert_eq!(back2, nonempty);
}

#[test]
fn uploader_modes_move_identical_bytes() {
    let Some(host) = cuda_host() else { return };
    let mut stager = PinnedStager::new(&host.ctx, 8192).unwrap();
    let src = pattern_u64(200_000); // 1.6 MB: staged in one mode, pageable in the other
    let mut pinned = Uploader::Pinned(&mut stager);
    let d_pinned = pinned.clone_htod(&host.stream, &src).unwrap();
    let mut pageable = Uploader::Pageable;
    let d_pageable = pageable.clone_htod(&host.stream, &src).unwrap();
    let a: Vec<u64> = host.stream.clone_dtoh(&d_pinned).unwrap();
    let b: Vec<u64> = host.stream.clone_dtoh(&d_pageable).unwrap();
    assert_eq!(a, b, "the two upload modes disagreed on payload bytes");
    assert_eq!(a, src);
}
