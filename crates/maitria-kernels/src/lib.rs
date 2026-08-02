//! Shared multi-architecture fast paths for a certificate-checked
//! verification system.
//!
//! Every kernel family in this crate obeys the repository's lane law:
//! an acceleration lane may change *cost*, never *verdicts*. Each
//! family ships a scalar reference implementation (the semantics),
//! per-architecture lanes (the economics), and a property-based
//! conformance battery gating the lanes against the reference. See
//! the repository `README.md` for the admission test that decides
//! what belongs here, and `ENGINEERING.md` for the commitments this
//! code is reviewed against.
//!
//! This crate is **not a trusted computing base**: its outputs are
//! re-derived by a certified checking surface or screened by a cotest
//! regime downstream. It is also deliberately **zero-dependency**:
//! the entire verdict-relevant surface is auditable without a
//! dependency tree.

#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]

pub mod nullspace;
pub mod rnsfold;
pub mod roweq;
pub mod sweep;
