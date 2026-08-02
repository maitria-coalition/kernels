//! The conformance battery (ENGINEERING #2): every lane compiled on
//! this architecture is verdict-identical to the scalar reference,
//! property-tested over boundary-heavy slabs plus deterministic
//! block-boundary cases for every vector width in the tree.
//!
//! The battery also carries its own independently-derived formulation
//! of both kernels (iterator combinators, a different algorithm) as a
//! differential partner *for the reference itself* (ENGINEERING #5) —
//! the reference gates the lanes, and this formulation gates the
//! reference.

use maitria_kernels::sweep::{self, reference, MinMax, SignPred};
use proptest::prelude::*;

const PREDS: [SignPred; 4] = [
    SignPred::NonNeg,
    SignPred::Pos,
    SignPred::NonPos,
    SignPred::Neg,
];

/// Independently-derived first_violation: enumerate + find, rather
/// than position.
fn indep_first_violation(nums: &[i64], pred: SignPred) -> Option<usize> {
    nums.iter()
        .enumerate()
        .find(|&(_, &x)| !pred.holds(x))
        .map(|(i, _)| i)
}

/// Independently-derived minmax: std's min/max + first position of
/// each value.
fn indep_minmax(nums: &[i64]) -> Option<MinMax> {
    let min = nums.iter().copied().min()?;
    let max = nums.iter().copied().max()?;
    Some(MinMax {
        min,
        min_idx: nums.iter().position(|&x| x == min).expect("min occurs"),
        max,
        max_idx: nums.iter().position(|&x| x == max).expect("max occurs"),
    })
}

/// A named `first_violation` implementation under test.
type FvImpl = (&'static str, fn(&[i64], SignPred) -> Option<usize>);
/// A named `minmax` implementation under test.
type MmImpl = (&'static str, fn(&[i64]) -> Option<MinMax>);

/// Every implementation compiled on this arch, name + fn, for both
/// kernels. The dispatching entry point is always included, so the
/// battery covers whatever `active_lane()` selects too.
fn fv_impls() -> Vec<FvImpl> {
    let mut v: Vec<FvImpl> = vec![
        ("dispatch", sweep::first_violation),
        ("reference", reference::first_violation),
    ];
    #[cfg(target_arch = "aarch64")]
    v.push(("neon", sweep::neon::first_violation));
    #[cfg(target_arch = "x86_64")]
    v.push(("avx2", sweep::avx2::first_violation));
    v
}

fn mm_impls() -> Vec<MmImpl> {
    let mut v: Vec<MmImpl> = vec![
        ("dispatch", sweep::minmax),
        ("reference", reference::minmax),
    ];
    #[cfg(target_arch = "aarch64")]
    v.push(("neon", sweep::neon::minmax));
    #[cfg(target_arch = "x86_64")]
    v.push(("avx2", sweep::avx2::minmax));
    v
}

fn assert_all_agree(nums: &[i64]) {
    for pred in PREDS {
        let want = indep_first_violation(nums, pred);
        for (name, f) in fv_impls() {
            assert_eq!(
                f(nums, pred),
                want,
                "first_violation lane `{name}` diverged (pred {pred:?}, slab {nums:?})"
            );
        }
    }
    let want = indep_minmax(nums);
    for (name, f) in mm_impls() {
        assert_eq!(
            f(nums),
            want,
            "minmax lane `{name}` diverged (slab {nums:?})"
        );
    }
}

/// Boundary-heavy slab strategy: small magnitudes around zero (where
/// every predicate's edge lives), the i64 extremes, and arbitrary
/// values; lengths crossing several 8-wide blocks.
fn slab() -> impl Strategy<Value = Vec<i64>> {
    prop::collection::vec(
        prop_oneof![
            3 => any::<i64>(),
            4 => -3i64..=3,
            1 => Just(i64::MIN),
            1 => Just(i64::MAX),
        ],
        0..200,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn lanes_agree_on_boundary_heavy_slabs(nums in slab()) {
        assert_all_agree(&nums);
    }

    /// All-satisfying slabs (the accept verdict, the long-scan common
    /// case): strictly positive entries, checked under NonNeg/Pos.
    #[test]
    fn accept_verdict_scans(nums in prop::collection::vec(1i64..=i64::MAX, 0..300)) {
        assert_all_agree(&nums);
    }

    /// A single planted violation at every position of a positive
    /// slab: the first-occurrence index must be exact at every block
    /// offset.
    #[test]
    fn planted_violation_every_position(
        mut nums in prop::collection::vec(1i64..=1000, 1..64),
        pos_frac in 0.0f64..1.0,
    ) {
        let pos = ((nums.len() as f64) * pos_frac) as usize % nums.len();
        nums[pos] = -1;
        for (name, f) in fv_impls() {
            assert_eq!(
                f(&nums, SignPred::NonNeg),
                Some(pos),
                "lane `{name}` missed the planted violation at {pos}"
            );
        }
        assert_all_agree(&nums);
    }
}

#[test]
fn deterministic_edges() {
    // Empty slab: no violation to report, no range to fold.
    assert_all_agree(&[]);
    // Single entries of each sign.
    for x in [i64::MIN, -1, 0, 1, i64::MAX] {
        assert_all_agree(&[x]);
    }
    // Zeros exercise the strict/non-strict predicate split.
    assert_all_agree(&[0; 40]);
    // Duplicated extremes: first-occurrence must win.
    assert_all_agree(&[5, -7, 5, -7, 5, -7, 5, -7, 5, -7, 5, -7, 5, -7, 5, -7, 5]);
    // A violation at every index of a 3-block slab, one at a time.
    for k in 0..24 {
        let mut v = vec![2i64; 24];
        v[k] = -2;
        assert_all_agree(&v);
    }
    // Extremes at block boundaries.
    let mut v = vec![0i64; 33];
    v[7] = i64::MIN;
    v[8] = i64::MAX;
    v[31] = i64::MIN;
    v[32] = i64::MAX;
    assert_all_agree(&v);
}

#[test]
fn dispatch_lane_is_observable() {
    // ENGINEERING #7: the selected lane is a reportable fact. This
    // asserts only that the call succeeds and is arch-consistent.
    let lane = sweep::active_lane();
    #[cfg(target_arch = "aarch64")]
    assert_eq!(lane, sweep::Lane::Neon);
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    assert_eq!(lane, sweep::Lane::Reference);
    #[cfg(target_arch = "x86_64")]
    assert!(matches!(lane, sweep::Lane::Avx2 | sweep::Lane::Reference));
}
