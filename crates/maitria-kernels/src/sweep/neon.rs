//! The aarch64 NEON lane.
//!
//! NEON is part of the aarch64 baseline target features, so this lane
//! needs no runtime detection; the intrinsics are stable Rust. SVE is
//! present on current server cores but its intrinsics are not yet
//! stable — noted as future work, not attempted here.
//!
//! Shape: 8-wide unrolled block scans (four 2-lane vectors per
//! block). A block that reports a hit is re-scanned scalar for the
//! exact index — the block is 8 entries, so the re-scan is constant
//! work and the first-occurrence semantics of the reference are
//! preserved exactly. `minmax` is two passes: a vectorized value
//! fold, then vectorized first-index-of-value scans; first index
//! attaining an extremum is the same whichever pass order finds it,
//! so the two-pass shape equals the reference's one-pass
//! first-occurrence semantics.

use core::arch::aarch64::{
    int64x2_t, uint64x2_t, vbslq_s64, vceqq_s64, vcgezq_s64, vcgtq_s64, vcgtzq_s64, vclezq_s64,
    vcltq_s64, vcltzq_s64, vdupq_n_s64, vgetq_lane_s64, vld1q_s64, vmaxvq_u32, vorrq_u64,
    vreinterpretq_u32_u64,
};

use super::{reference, MinMax, SignPred};

/// Lane mask of entries violating `pred` (all-ones per violating
/// 64-bit lane).
#[inline]
fn violation_mask(pred: SignPred, v: int64x2_t) -> uint64x2_t {
    // SAFETY: NEON is baseline on aarch64.
    unsafe {
        match pred {
            SignPred::NonNeg => vcltzq_s64(v), // violation: x < 0
            SignPred::Pos => vclezq_s64(v),    // violation: x <= 0
            SignPred::NonPos => vcgtzq_s64(v), // violation: x > 0
            SignPred::Neg => vcgezq_s64(v),    // violation: x >= 0
        }
    }
}

#[inline]
fn any_lane_set(m: uint64x2_t) -> bool {
    // SAFETY: NEON is baseline on aarch64.
    unsafe { vmaxvq_u32(vreinterpretq_u32_u64(m)) != 0 }
}

/// NEON [`super::first_violation`]; verdict-identical to the
/// reference by battery.
pub fn first_violation(nums: &[i64], pred: SignPred) -> Option<usize> {
    let mut chunks = nums.chunks_exact(8);
    let mut base = 0usize;
    for ch in &mut chunks {
        let p = ch.as_ptr();
        // SAFETY: `ch` is exactly 8 contiguous i64s; unaligned loads
        // are fine for vld1q; NEON is baseline on aarch64.
        let hit = unsafe {
            let m0 = violation_mask(pred, vld1q_s64(p));
            let m1 = violation_mask(pred, vld1q_s64(p.add(2)));
            let m2 = violation_mask(pred, vld1q_s64(p.add(4)));
            let m3 = violation_mask(pred, vld1q_s64(p.add(6)));
            any_lane_set(vorrq_u64(vorrq_u64(m0, m1), vorrq_u64(m2, m3)))
        };
        if hit {
            // Constant-size scalar re-scan preserves first-occurrence
            // order exactly.
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

/// First index whose entry equals `target`. Caller guarantees the
/// target occurs (it came out of the value fold over the same slab).
fn first_index_of(nums: &[i64], target: i64) -> usize {
    let mut chunks = nums.chunks_exact(8);
    let mut base = 0usize;
    // SAFETY: NEON is baseline on aarch64.
    let t = unsafe { vdupq_n_s64(target) };
    for ch in &mut chunks {
        let p = ch.as_ptr();
        // SAFETY: `ch` is exactly 8 contiguous i64s.
        let hit = unsafe {
            let m0 = vceqq_s64(vld1q_s64(p), t);
            let m1 = vceqq_s64(vld1q_s64(p.add(2)), t);
            let m2 = vceqq_s64(vld1q_s64(p.add(4)), t);
            let m3 = vceqq_s64(vld1q_s64(p.add(6)), t);
            any_lane_set(vorrq_u64(vorrq_u64(m0, m1), vorrq_u64(m2, m3)))
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

/// NEON [`super::minmax`]; verdict-identical to the reference by
/// battery.
pub fn minmax(nums: &[i64]) -> Option<MinMax> {
    // Below one unrolled block the vector setup outweighs the work.
    if nums.len() < 16 {
        return reference::minmax(nums);
    }
    let first = nums[0];
    let mut chunks = nums.chunks_exact(4);
    // SAFETY: NEON is baseline on aarch64; every load below reads
    // exactly the 4 contiguous i64s of `ch`.
    let (vmin, vmax) = unsafe {
        let seed = vdupq_n_s64(first);
        let mut vmin0 = seed;
        let mut vmin1 = seed;
        let mut vmax0 = seed;
        let mut vmax1 = seed;
        for ch in &mut chunks {
            let p = ch.as_ptr();
            let a = vld1q_s64(p);
            let b = vld1q_s64(p.add(2));
            vmin0 = vbslq_s64(vcltq_s64(a, vmin0), a, vmin0);
            vmax0 = vbslq_s64(vcgtq_s64(a, vmax0), a, vmax0);
            vmin1 = vbslq_s64(vcltq_s64(b, vmin1), b, vmin1);
            vmax1 = vbslq_s64(vcgtq_s64(b, vmax1), b, vmax1);
        }
        (
            vbslq_s64(vcltq_s64(vmin1, vmin0), vmin1, vmin0),
            vbslq_s64(vcgtq_s64(vmax1, vmax0), vmax1, vmax0),
        )
    };
    // SAFETY: lane extraction on values computed above.
    let (mut min, mut max) = unsafe {
        let m0 = vgetq_lane_s64(vmin, 0);
        let m1 = vgetq_lane_s64(vmin, 1);
        let x0 = vgetq_lane_s64(vmax, 0);
        let x1 = vgetq_lane_s64(vmax, 1);
        (m0.min(m1), x0.max(x1))
    };
    for &x in chunks.remainder() {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
    }
    Some(MinMax {
        min,
        min_idx: first_index_of(nums, min),
        max,
        max_idx: first_index_of(nums, max),
    })
}
