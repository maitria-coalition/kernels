//! `roweq` — batched exact sparse-row membership on limb planes (the
//! candidate-membership gate of batch certificate verification).
//!
//! ## What this family computes
//!
//! One *attempt* is the value-only core of a membership side-condition:
//! given a set of *query* rows (combination vehicles) and a set of
//! *pool* rows (premise-derived candidates), decide, per attempt:
//!
//! > does **every** query row equal **some** pool row?
//!
//! where a row is a sorted sparse vector of `(column, value)` pairs
//! and two rows are equal iff they have the same length and are
//! positionally equal — same column and *structurally* the same value
//! at every position. A value is a signed rational represented as
//! (sign, numerator magnitude limbs, denominator id); structural
//! equality is sign equality ∧ numerator limb equality ∧ denominator
//! id equality.
//!
//! This is the arithmetic-free sibling of the `rnsfold` family: the
//! two together cover the value-dependent checks of a batched
//! linear-combination side condition (membership here, the fold
//! there), shared by the producer side (re-checking a candidate
//! combination against its premise pool before emission) and the
//! consumer side (replaying a committed certificate's combination
//! rows) — the admission test's clause 1, by the same argument as
//! `rnsfold`'s. No machine-checked lemma names this code — clause 2.
//! (Both adjudications recorded here per the repository README; the
//! clause-1 reading is the maintainer-revisable one.)
//!
//! ## Why structural equality, and why it is exact
//!
//! The upstream CPU references decide membership by structural
//! equality of canonical rationals (numerator and denominator
//! compared as bignums). Equality of representations is therefore
//! *itself* the reference predicate — not a proxy for it — and a lane
//! that compares limb planes bit-for-bit computes the same verdict by
//! definition: no residue system, no soundness bound, no channel
//! plan, and consequently **no refusal surface anywhere in the
//! family** (DATERWI is satisfied vacuously: nothing is ever
//! inconclusive; VCARM is vacuous: no float exists).
//!
//! The one obligation this family places on its caller (recorded in
//! [`batch::RowEqBatch`]'s field docs): `den_id` must be an
//! attempt-scoped injective code of the actual denominators — equal
//! id ⟺ equal denominator, within each attempt. Callers obtain this
//! by construction (a byte-keyed interning dictionary per attempt);
//! the downstream system battery (provider conformance against its
//! own exact membership gate) is the differential partner positioned
//! to catch a packer that violates it.
//!
//! ## Lanes
//!
//! - [`reference::verify`] — the scalar reference: nested scan in the
//!   upstream evaluator's own access pattern, written for obviousness.
//!   The semantics.
//! - the CUDA lane (sibling crate `maitria-kernels-cuda`, kernel
//!   `roweq.cu`) — one thread per query row scanning its attempt's
//!   pool; conformance-gated against the reference on verdict
//!   bit-identity.
//!
//! Differential partners (ENGINEERING #5): the in-tree reference; the
//! battery's independent hash-set formulation (pool rows interned into
//! a set keyed on trimmed canonical row bytes, queries probed — a
//! different algorithm with a different access pattern,
//! `tests/roweq.rs`); and downstream, the consumer's own exact
//! membership gate (`qtsl-geolog-provider`'s batch-CSR lowerings).

pub mod batch;
pub mod reference;

pub use batch::{RowEqBatch, RowEqError, RowEqOutcome};
