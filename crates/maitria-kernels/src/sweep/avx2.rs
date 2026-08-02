//! The x86-64 AVX2 lane.
//!
//! AVX2 is not baseline on x86-64, so the public entry points do
//! runtime feature detection and fall back to the reference —
//! detection happens once per call, which is noise against any slab
//! worth vectorizing; callers on the hot path may pin the lane
//! explicitly. AVX-512 is noted as future work (its i64 min/max and
//! wider scans would simplify this file), not attempted while the
//! deployment fleet's smallest common denominator is AVX2.
//!
//! Shape mirrors the NEON lane: 8-wide unrolled block scans (two
//! 4-lane vectors per block), scalar re-scan of a hit block for exact
//! first-occurrence indices, two-pass `minmax` (vectorized value
//! fold, then first-index-of-value scans).

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256i, _mm256_blendv_epi8, _mm256_cmpeq_epi64, _mm256_cmpgt_epi64, _mm256_extract_epi64,
    _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_or_si256, _mm256_set1_epi64x,
    _mm256_setzero_si256,
};

use super::{reference, MinMax, SignPred};

/// Lane mask of entries violating `pred` (all-ones per violating
/// 64-bit lane).
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn violation_mask(pred: SignPred, v: __m256i) -> __m256i {
    // Register-only intrinsics are safe inside a `target_feature` fn
    // (target_feature 1.1); no unsafe block needed here.
    let zero = _mm256_setzero_si256();
    match pred {
        // violation: x < 0
        SignPred::NonNeg => _mm256_cmpgt_epi64(zero, v),
        // violation: x <= 0  ==  (0 > x) | (x == 0)
        SignPred::Pos => _mm256_or_si256(_mm256_cmpgt_epi64(zero, v), _mm256_cmpeq_epi64(v, zero)),
        // violation: x > 0
        SignPred::NonPos => _mm256_cmpgt_epi64(v, zero),
        // violation: x >= 0  ==  (x > 0) | (x == 0)
        SignPred::Neg => _mm256_or_si256(_mm256_cmpgt_epi64(v, zero), _mm256_cmpeq_epi64(v, zero)),
    }
}

/// # Safety
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
unsafe fn first_violation_avx2(nums: &[i64], pred: SignPred) -> Option<usize> {
    let mut chunks = nums.chunks_exact(8);
    let mut base = 0usize;
    for ch in &mut chunks {
        let p = ch.as_ptr();
        // SAFETY: `ch` is exactly 8 contiguous i64s; loadu tolerates
        // any alignment; AVX2 guaranteed by this fn's contract.
        let hit = unsafe {
            let m0 = violation_mask(pred, _mm256_loadu_si256(p.cast()));
            let m1 = violation_mask(pred, _mm256_loadu_si256(p.add(4).cast()));
            _mm256_movemask_epi8(_mm256_or_si256(m0, m1)) != 0
        };
        if hit {
            return ch.iter().position(|&x| !pred.holds(x)).map(|i| base + i);
        }
        base += 8;
    }
    chunks
        .remainder()
        .iter()
        .position(|&x| !pred.holds(x))
        .map(|i| base + i)
}

/// First index whose entry equals `target`; caller guarantees the
/// target occurs.
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
unsafe fn first_index_of_avx2(nums: &[i64], target: i64) -> usize {
    let mut chunks = nums.chunks_exact(8);
    let mut base = 0usize;
    // Register-only intrinsic: safe inside a `target_feature` fn.
    let t = _mm256_set1_epi64x(target);
    for ch in &mut chunks {
        let p = ch.as_ptr();
        // SAFETY: `ch` is exactly 8 contiguous i64s.
        let hit = unsafe {
            let m0 = _mm256_cmpeq_epi64(_mm256_loadu_si256(p.cast()), t);
            let m1 = _mm256_cmpeq_epi64(_mm256_loadu_si256(p.add(4).cast()), t);
            _mm256_movemask_epi8(_mm256_or_si256(m0, m1)) != 0
        };
        if hit {
            for (i, &x) in ch.iter().enumerate() {
                if x == target {
                    return base + i;
                }
            }
            unreachable!("hit block must contain target");
        }
        base += 8;
    }
    for (i, &x) in chunks.remainder().iter().enumerate() {
        if x == target {
            return base + i;
        }
    }
    unreachable!("target is an extremum of this very slab");
}

/// # Safety
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
unsafe fn minmax_avx2(nums: &[i64]) -> Option<MinMax> {
    if nums.len() < 16 {
        return reference::minmax(nums);
    }
    let first = nums[0];
    let mut chunks = nums.chunks_exact(8);
    // SAFETY: AVX2 guaranteed by this fn's contract; every load reads
    // exactly the 8 contiguous i64s of `ch`.
    let (mut min, mut max) = unsafe {
        let seed = _mm256_set1_epi64x(first);
        let mut vmin0 = seed;
        let mut vmin1 = seed;
        let mut vmax0 = seed;
        let mut vmax1 = seed;
        for ch in &mut chunks {
            let p = ch.as_ptr();
            let a = _mm256_loadu_si256(p.cast());
            let b = _mm256_loadu_si256(p.add(4).cast());
            vmin0 = _mm256_blendv_epi8(vmin0, a, _mm256_cmpgt_epi64(vmin0, a));
            vmax0 = _mm256_blendv_epi8(vmax0, a, _mm256_cmpgt_epi64(a, vmax0));
            vmin1 = _mm256_blendv_epi8(vmin1, b, _mm256_cmpgt_epi64(vmin1, b));
            vmax1 = _mm256_blendv_epi8(vmax1, b, _mm256_cmpgt_epi64(b, vmax1));
        }
        let vmin = _mm256_blendv_epi8(vmin0, vmin1, _mm256_cmpgt_epi64(vmin0, vmin1));
        let vmax = _mm256_blendv_epi8(vmax0, vmax1, _mm256_cmpgt_epi64(vmax1, vmax0));
        let mins = [
            _mm256_extract_epi64(vmin, 0),
            _mm256_extract_epi64(vmin, 1),
            _mm256_extract_epi64(vmin, 2),
            _mm256_extract_epi64(vmin, 3),
        ];
        let maxs = [
            _mm256_extract_epi64(vmax, 0),
            _mm256_extract_epi64(vmax, 1),
            _mm256_extract_epi64(vmax, 2),
            _mm256_extract_epi64(vmax, 3),
        ];
        (
            mins.into_iter().min().expect("nonempty"),
            maxs.into_iter().max().expect("nonempty"),
        )
    };
    for &x in chunks.remainder() {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
    }
    // SAFETY: AVX2 guaranteed by this fn's contract; both extrema
    // occur in `nums` by construction.
    let (min_idx, max_idx) = unsafe {
        (
            first_index_of_avx2(nums, min),
            first_index_of_avx2(nums, max),
        )
    };
    Some(MinMax {
        min,
        min_idx,
        max,
        max_idx,
    })
}

/// AVX2 [`super::first_violation`] behind runtime detection; falls
/// back to the reference when AVX2 is absent.
pub fn first_violation(nums: &[i64], pred: SignPred) -> Option<usize> {
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: feature just detected.
        unsafe { first_violation_avx2(nums, pred) }
    } else {
        reference::first_violation(nums, pred)
    }
}

/// AVX2 [`super::minmax`] behind runtime detection; falls back to the
/// reference when AVX2 is absent.
pub fn minmax(nums: &[i64]) -> Option<MinMax> {
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: feature just detected.
        unsafe { minmax_avx2(nums) }
    } else {
        reference::minmax(nums)
    }
}
