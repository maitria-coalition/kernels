//! First-violation sign sweeps and min/max enclosure folds over
//! exact-integer slabs.
//!
//! The coefficient-sweep verdict shape shared by the producer and
//! consumer sides of the system (see the crate README for the
//! shared-denominator reduction and the named differential partners):
//!
//! - [`first_violation`] — the index of the first slab entry
//!   violating a one-sided sign predicate, in slab order; `None` is
//!   the accept verdict. The consumer reads the verdict; the producer
//!   reads the index as its refinement witness.
//! - [`minmax`] — the exact range of the slab with first-occurrence
//!   extremum indices (the enclosure readout of a Bernstein
//!   coefficient table: min/max coefficients bound the polynomial).
//!
//! Semantics are defined by [`reference`]; the architecture lanes
//! ([`neon`] on aarch64, [`avx2`] on x86-64) are verdict-identical by
//! battery (`tests/conformance.rs`) and receipt-gated for dispatch
//! (`receipts/`, ENGINEERING #3).
//!
//! Slab entries are `i64`: the first rung of the callers' promotion
//! ladder. Callers pack exact values whose numerators fit 64 bits and
//! route wider values to their arbitrary-precision path — packing is
//! fit-checked upstream, so no rounding exists anywhere in or around
//! these kernels (ENGINEERING #4).

pub mod reference;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

/// One-sided sign predicates over exact integers.
///
/// The predicate states what every slab entry must satisfy; a sweep
/// reports the first entry that does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SignPred {
    /// Every entry must be `>= 0`.
    NonNeg,
    /// Every entry must be `> 0`.
    Pos,
    /// Every entry must be `<= 0`.
    NonPos,
    /// Every entry must be `< 0`.
    Neg,
}

impl SignPred {
    /// Does `x` satisfy the predicate?
    #[inline]
    pub fn holds(self, x: i64) -> bool {
        match self {
            SignPred::NonNeg => x >= 0,
            SignPred::Pos => x > 0,
            SignPred::NonPos => x <= 0,
            SignPred::Neg => x < 0,
        }
    }
}

/// Exact range of a slab with first-occurrence extremum indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MinMax {
    /// The minimum value.
    pub min: i64,
    /// The smallest index attaining `min`.
    pub min_idx: usize,
    /// The maximum value.
    pub max: i64,
    /// The smallest index attaining `max`.
    pub max_idx: usize,
}

/// The lane the dispatching entry points select on this machine
/// (ENGINEERING #7: dispatch is observable, never inferred).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    /// The scalar reference (the semantics).
    Reference,
    /// The aarch64 NEON lane.
    Neon,
    /// The x86-64 AVX2 lane.
    Avx2,
}

/// Which lane [`first_violation`] and [`minmax`] dispatch to here.
pub fn active_lane() -> Lane {
    #[cfg(target_arch = "aarch64")]
    {
        Lane::Neon
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            Lane::Avx2
        } else {
            Lane::Reference
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        Lane::Reference
    }
}

/// First index whose entry violates `pred`, in slab order; `None` if
/// every entry satisfies it (the accept verdict).
///
/// Dispatches per [`active_lane`]; semantics are
/// [`reference::first_violation`].
#[inline]
pub fn first_violation(nums: &[i64], pred: SignPred) -> Option<usize> {
    #[cfg(target_arch = "aarch64")]
    {
        neon::first_violation(nums, pred)
    }
    #[cfg(target_arch = "x86_64")]
    {
        avx2::first_violation(nums, pred)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        reference::first_violation(nums, pred)
    }
}

/// Exact range of the slab with first-occurrence extremum indices;
/// `None` on an empty slab.
///
/// Dispatches per [`active_lane`]; semantics are
/// [`reference::minmax`].
#[inline]
pub fn minmax(nums: &[i64]) -> Option<MinMax> {
    #[cfg(target_arch = "aarch64")]
    {
        neon::minmax(nums)
    }
    #[cfg(target_arch = "x86_64")]
    {
        avx2::minmax(nums)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        reference::minmax(nums)
    }
}
