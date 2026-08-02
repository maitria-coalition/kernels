//! `rnsfold` — batched exact linear-combination equality over residue
//! channels (a residue-number-system lowering of the exact-integer
//! certificate fold).
//!
//! ## What this family computes
//!
//! One *attempt* is the value-only core of a linear-combination
//! side-condition check: given signed integer coefficients
//! $\hat\lambda_e$ (combination weights, pre-scaled by the caller to a
//! per-attempt common denominator $D$), signed integer vehicle values
//! $\hat v = v_{\text{num}} \cdot m$ (each raw numerator times a small
//! caller-chosen positive multiplier — the denominator cofactor
//! $D/\mathrm{den}$), and conclusion values
//! $\hat c = c_{\text{num}} \cdot m'$ (multiplier $D^2/\mathrm{den}$),
//! decide, per dense output column $\mathrm{col}$:
//!
//! $$\Delta(\mathrm{col}) \;=\; \sum_e \hat\lambda_e \cdot \hat
//! v_{e,\mathrm{col}} \;-\; \hat c_{\mathrm{col}} \;\stackrel{?}{=}\; 0$$
//!
//! and report, per attempt, whether **every** column's $\Delta$
//! vanishes. This is the arithmetic core shared verbatim by the
//! producer side (assembling and re-checking candidate Farkas/syzygy
//! combinations before emission) and the consumer side (replaying a
//! committed certificate's combination rows). No machine-checked
//! lemma names this code, and no trust story routes through it; its
//! assurance instruments are the in-tree reference, the independent
//! big-integer partner in the battery, and the system-level
//! differential its consumers operate.
//!
//! ## The RNS lowering, and why its verdicts are exact
//!
//! Every lane of this family evaluates $\Delta(\mathrm{col})$ modulo a
//! set of distinct 63-bit primes $p_0, \dots, p_{C-1}$ (the *channels*,
//! drawn in order from [`primes::PRIMES`]) and reports a mismatch iff
//! any channel of any column is nonzero. Two directions:
//!
//! - **A nonzero channel is a sound mismatch, unconditionally**:
//!   $\Delta \not\equiv 0 \pmod{p}$ implies $\Delta \neq 0$.
//! - **All-channels-zero is a sound equality** provided
//!   $|\Delta(\mathrm{col})| < \prod_{j<C} p_j$ for every column: an
//!   integer divisible by the product and smaller than it in magnitude
//!   is zero. The channel count is chosen — inside this module, from
//!   the batch's own planes, never by caller assertion — so that a
//!   conservative a-priori bound
//!   $|\Delta| \le T \cdot 2^{b_\lambda + b_v} + 2^{b_c}$ (with $T$ the
//!   per-column term count and $b_\cdot$ the measured operand bit
//!   widths, all read off the descriptor) clears that product with a
//!   two-bit margin. See [`RnsFoldBatch::required_bits`].
//!
//! No CRT reconstruction ever happens: only equality *verdicts* leave
//! the lanes, and equality is channel-local. Consequently there is no
//! fixed limb capacity anywhere in the family — wider operands cost
//! more channels, never a refusal — except the prime table itself
//! (64 channels ≈ 4032 bits of product): an attempt whose bound
//! exceeds the table is refused by typed outcome
//! ([`RnsFoldOutcome::refused`]), for the caller's own exact path.
//! That refusal is the family's entire deferral surface.
//!
//! ## Why RNS and not limb-plane schoolbook (provenance of the design)
//!
//! Chosen against the measured workload rather than from priors:
//! certificate numerators are
//! variable-length with high length entropy, which fights every
//! fixed-width-per-launch bignum layout (the CGBN critique), while an
//! equality-only verdict needs none of RNS's classically hard
//! operations (no sign detection, no magnitude comparison, no
//! division, no reconstruction). Residue channels give uniform u64
//! state, carry-free arithmetic, and per-batch width scaling — at the
//! price of a constant-factor arithmetic inflation the acceleration
//! lanes buy back in occupancy and regularity. The limb-plane
//! schoolbook fold (the sibling lowering, measured on CPU against the
//! same certificate corpora) remains the named alternative if a
//! consumer ever needs fold *values* rather than verdicts.
//!
//! ## Lanes
//!
//! - [`reference::verify`] — the scalar reference: plain `u128 %`
//!   residue arithmetic, written for obviousness. The semantics.
//! - the CUDA lane (sibling crate `maitria-kernels-cuda`, kernel
//!   `rnsfold.cu`) — Montgomery-form channel arithmetic on device;
//!   conformance-gated against the reference on verdict bit-identity.
//!
//! Differential partners (ENGINEERING #5): the in-tree reference; the
//! battery's independent big-integer formulation of $\Delta = 0$
//! (`tests/rnsfold.rs`, computed with `num-bigint`, no residues
//! anywhere); and downstream, the consumer's own exact evaluators.

pub mod batch;
pub mod primes;
pub mod reference;

pub use batch::{RnsFoldBatch, RnsFoldOutcome, ABSENT};
