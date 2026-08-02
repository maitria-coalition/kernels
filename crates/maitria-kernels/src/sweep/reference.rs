//! The scalar reference lane: the semantics of the `sweep` family.
//!
//! Written for obviousness over speed (ENGINEERING #1); always
//! compiled on every target; the conformance battery gates every
//! other lane against this one.

use super::{MinMax, SignPred};

/// First index whose entry violates `pred`, in slab order.
pub fn first_violation(nums: &[i64], pred: SignPred) -> Option<usize> {
    nums.iter().position(|&x| !pred.holds(x))
}

/// Exact range with first-occurrence extremum indices; `None` on an
/// empty slab. Strict comparisons keep the first occurrence.
pub fn minmax(nums: &[i64]) -> Option<MinMax> {
    let (&first, rest) = nums.split_first()?;
    let mut mm = MinMax {
        min: first,
        min_idx: 0,
        max: first,
        max_idx: 0,
    };
    for (i, &x) in rest.iter().enumerate() {
        if x < mm.min {
            mm.min = x;
            mm.min_idx = i + 1;
        }
        if x > mm.max {
            mm.max = x;
            mm.max_idx = i + 1;
        }
    }
    Some(mm)
}
