//! maitria-hypernet — the hypernet certificate genus as data: exact
//! scalars, the graph representation, and the canonical `AXHN0001`
//! byte format with its content digest.
//!
//! This crate is the org's ONE canonical decoder. "Decodes
//! canonically" and "digest ties" are verdict-bearing gates of the
//! checking surface, so the codec lives at the bottom of the
//! dependency lattice — importable by producers (mtk) and checkers
//! (qtsl-geolog-provider) alike, owned by neither. It was factored
//! out of `mtk_catalogue::hypernet` during the walk-evaluator
//! provider port: a canonical decoder homed in the untrusted
//! producer's repo put bytes-semantics under the wrong roof, and the
//! checker could not reach it without an org-layer cycle.
//!
//! Not a fast path: a format. Cost may change; verdicts never. The
//! conformance battery (`tests/golden_codec.rs`) pins byte
//! compatibility against the frozen golden vectors; the
//! byte-compatibility authority is the X0 reference implementation
//! (`acasxu/hypernet.py`), and the Lean mirror is
//! `Core/Catalogue.lean` Layer 5 (qtsl repo).
//!
//! Contents:
//! - [`scalar`] — [`WitInt`] / [`WitRat`]: the exact arbitrary-
//!   precision scalar leaves (moved whole from
//!   `maitria-witness-types`, which re-exports them — one type
//!   identity across producer and checker; their CBOR rules ride
//!   along as serde impls, normative schema doc: mtk CONTRACTS.md
//!   §2.1).
//! - [`types`] — tensor types ([`TensorType`], [`Dtype`], [`Kind`],
//!   [`Sign`], [`Ext`]).
//! - [`app`] — the ten primitive boxes ([`App`], [`ConstPayload`],
//!   [`Wire`]).
//! - [`simplex`] — simplicial-complex payloads ([`SimpC`], [`VSet`]).
//! - [`graph`] — the container ([`Hypernet`]) + typed construction
//!   ([`HypernetBuilder`]).
//! - [`wf`] — well-formedness preflight ([`Violation`],
//!   [`TypeViolation`]).
//! - [`canon`] — canonical serialization, digest, and the validating
//!   decode ([`Hypernet::canonical_bytes`], [`Hypernet::digest`],
//!   [`Hypernet::from_canonical_bytes`], [`HypernetDigest`],
//!   [`CanonRefusal`], [`MAGIC`]).
//!
//! Producer-side operations (exact evaluation, substitution, TikZ
//! rendering) stay in `mtk_catalogue::hypernet`, operating on these
//! re-exported types.

pub mod app;
pub mod canon;
pub mod graph;
pub mod scalar;
pub mod simplex;
pub mod types;
pub mod wf;

pub use app::{App, ConstPayload, Wire};
pub use canon::{CanonRefusal, HypernetDigest, MAGIC};
pub use graph::{BuildRefusal, Hypernet, HypernetBuilder};
pub use scalar::{WitInt, WitRat};
pub use simplex::{SimpC, SimpCViolation, VSet};
pub use types::{Dtype, Ext, Kind, Sign, TensorType};
pub use wf::{TypeViolation, Violation};
