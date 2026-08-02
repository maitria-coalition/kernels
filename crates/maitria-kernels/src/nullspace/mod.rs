//! `nullspace` — exact integer nullspace of a sparse integer matrix,
//! by mod-$p$ elimination and CRT lifting.
//!
//! ## What this family computes
//!
//! For a sparse integer matrix $A$ (given as accumulating triplets),
//! the family computes, per odd prime $p < 2^{63}$:
//!
//! - $\mathrm{rank}_p(A)$;
//! - the **leftmost column rank profile** mod $p$ (column $j$ is a
//!   pivot iff it strictly increases the rank of the prefix
//!   $A_{[:, 0..=j]}$) — an intrinsic invariant of the matrix, not of
//!   any pivoting strategy;
//! - the **canonical nullspace basis** mod $p$: one vector per free
//!   (non-pivot) column $f$, in ascending order of $f$, with
//!   $y_f = 1$, $y_c = 0$ for every other free column $c$, and the
//!   pivot-column coordinates read off the reduced row echelon form.
//!   This is the unique basis of $\ker_p(A)$ in
//!   identity-on-free-coordinates form.
//!
//! Canonicality is what makes multi-prime work meaningful: for every
//! prime whose rank profile agrees with the rational one (all but
//! finitely many), these residues are the reductions of **one**
//! well-defined rational object — the canonical rational nullspace
//! basis — so per-entry CRT combination and rational reconstruction
//! ([`lift`]) recover exact integer nullspace vectors, which callers
//! then confirm by exact arithmetic ([`verify`], or their own
//! big-integer path). An unlucky prime perturbs the profile or the
//! rank; profile disagreement between primes is a typed outcome
//! ([`lift::LiftError::ProfileMismatch`]), never a wrong answer.
//!
//! The motivating workload is Petri-net P-invariants at
//! model-checking-contest scale: the left nullspace
//! $\{y : y^\mathsf{T} C = 0\}$ of a $P \times T$ incidence matrix,
//! computed as the (right) nullspace of $A = C^\mathsf{T}$ — up to
//! $\sim 10^6$ rows and $\sim 25 \cdot 10^6$ nonzeros, far past dense
//! elimination. Both sides of the certificate system evaluate this
//! structure: the producer proposes invariant bases from it, and the
//! consumer's screening uses the same mod-$p$ elimination for the
//! rank bound $\mathrm{rank}_p \le \mathrm{rank}_\mathbb{Q}$ (so
//! $\mathrm{nullity}_p = 0$ proves the exact nullspace trivial and
//! dispatches the query with no exact work at all). No machine-checked
//! lemma names this code; its assurance instruments are the in-tree
//! reference, the independent big-rational partner in the battery,
//! and the exact re-verification its consumers run on every emitted
//! vector.
//!
//! ## The completeness pinch
//!
//! $\mathrm{rank}_p \le \mathrm{rank}_\mathbb{Q}$ unconditionally, so
//! $\mathrm{nullity}_p \ge \mathrm{nullity}_\mathbb{Q}$. When every
//! one of the $\mathrm{nullity}_p$ lifted vectors passes an exact
//! $A y = 0$ check, the two inequalities pinch: the vectors are
//! independent by their identity-on-free-coordinates pattern, so they
//! are a **complete** exact basis, and the only computation trusted
//! for that conclusion is the mod-$p$ rank at a single prime — which
//! is re-derivable from the emitted witness rows
//! ([`ModpNullspace::witness_rows`]) at a fraction of the original
//! cost. An unlucky prime cannot fake completeness: it can only
//! *inflate* $\mathrm{nullity}_p$, and the surplus vectors then fail
//! the exact check, which sends the caller to a fresh prime.
//!
//! ## Lanes
//!
//! - [`reference::nullspace_mod_p`] — dense Gauss–Jordan with leftmost
//!   column scanning, plain `u128 %` arithmetic, written for
//!   obviousness. The semantics.
//! - [`sparse::nullspace_mod_p`] — the economics: sparse
//!   row-merge elimination with fill-reducing local pivoting
//!   (sparsest remaining row, then least-populated column — the
//!   Dumas–Villard linear-pivoting heuristic), a switch-to-dense core
//!   once the active submatrix is small and full enough, Montgomery
//!   channel arithmetic, and a two-phase canonicalization (free-order
//!   row-space reduction, then leftmost reduced echelon on the
//!   surviving rows). Refuses by typed outcome when its fill budget
//!   is exceeded (a black-box/Wiedemann lane is the documented
//!   successor for inputs whose fill defeats elimination).
//!
//! Verdict fields — `rank`, `pivot_cols`, `basis` — are bit-identical
//! across lanes by battery (`tests/nullspace.rs`). `witness_rows` is
//! a *witness* field: lanes may select different row subsets, and the
//! battery checks validity (the selected rows alone reproduce the
//! rank) rather than identity.
//!
//! Differential partners (ENGINEERING #5): the in-tree reference; the
//! battery's independent big-rational Gauss–Jordan over `num-bigint`
//! (exact arithmetic, no residues anywhere); and downstream, the
//! exact re-verification every consumer runs on emitted vectors.

pub mod lift;
pub mod reference;
pub mod sparse;
pub mod verify;

/// A sparse integer matrix as accumulating triplets: entry
/// `(row, col, weight)`, duplicates summed. The natural form of a
/// Petri incidence matrix ($C = \mathrm{post} - \mathrm{pre}$, arcs as
/// triplets), and of most relational extractions.
#[derive(Clone, Copy, Debug)]
pub struct Triplets<'a> {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns (the nullspace lives in $\mathbb{Z}^{cols}$).
    pub cols: usize,
    /// The entries, in any order; duplicates accumulate.
    pub entries: &'a [(u32, u32, i64)],
}

impl<'a> Triplets<'a> {
    /// Every index in range?
    pub fn validate(&self) -> Result<(), NullspaceError> {
        for &(r, c, _) in self.entries {
            if r as usize >= self.rows || c as usize >= self.cols {
                return Err(NullspaceError::IndexOutOfRange {
                    row: r,
                    col: c,
                    rows: self.rows,
                    cols: self.cols,
                });
            }
        }
        Ok(())
    }
}

/// A sparse vector of residues mod the family prime: parallel arrays,
/// `idx` strictly ascending, values nonzero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseVec {
    /// Coordinate indices, strictly ascending.
    pub idx: Vec<u32>,
    /// Residues, nonzero, parallel to `idx`.
    pub val: Vec<u64>,
}

/// The mod-$p$ verdict-and-witness bundle for one prime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModpNullspace {
    /// The prime.
    pub prime: u64,
    /// $\mathrm{rank}_p(A)$.
    pub rank: usize,
    /// Leftmost column rank profile mod $p$, ascending. Verdict field.
    pub pivot_cols: Vec<u32>,
    /// A subset of original row indices, of size `rank`, whose rows
    /// alone have full rank mod $p$ — the re-derivable certificate
    /// that $\mathrm{rank}_\mathbb{Q} \ge \mathrm{rank}$ (hence the
    /// completeness pinch; see the module documentation). Witness
    /// field: valid-by-check, not bit-identical across lanes.
    pub witness_rows: Vec<u32>,
    /// The canonical basis of $\ker_p(A)$: one vector per free
    /// column, ascending by free column. Verdict field.
    pub basis: Vec<SparseVec>,
}

impl ModpNullspace {
    /// The free (non-pivot) column of basis vector `k` — its
    /// identity coordinate.
    pub fn free_col(&self, k: usize) -> u32 {
        // The identity coordinate is the unique index carrying
        // residue 1 at a non-pivot column; by construction it is
        // recoverable without a side table: basis vectors are emitted
        // in ascending free-column order, and each vector's free
        // column is the unique index in `idx` absent from
        // `pivot_cols`.
        let piv = &self.pivot_cols;
        *self.basis[k]
            .idx
            .iter()
            .find(|&&c| piv.binary_search(&c).is_err())
            .expect("canonical basis vector carries its free column")
    }
}

/// Typed refusals (ENGINEERING #7: an unsupported input is a typed
/// refusal naming the input; #10: inconclusive routes to the caller's
/// exact path, never into a guess).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NullspaceError {
    /// The prime must be odd and below $2^{63}$ (Montgomery-compatible
    /// with $R = 2^{64}$; the [`crate::rnsfold::primes::PRIMES`] table
    /// is the intended source).
    BadPrime {
        /// The offending value.
        prime: u64,
    },
    /// A triplet index exceeded the declared shape.
    IndexOutOfRange {
        /// Offending row index.
        row: u32,
        /// Offending column index.
        col: u32,
        /// Declared row count.
        rows: usize,
        /// Declared column count.
        cols: usize,
    },
    /// The elimination's live-entry count exceeded the caller's fill
    /// budget: elimination is the wrong algorithm for this input (the
    /// documented successor is a black-box/Wiedemann lane), or the
    /// budget was set below the answer's own size. Nothing partial is
    /// returned.
    FillBudget {
        /// Live sparse entries when the budget tripped.
        live_entries: usize,
        /// The budget that tripped.
        max_entries: usize,
        /// Pivots completed before refusal.
        pivots_done: usize,
        /// `1` = row-space reduction, `2` = leftmost canonicalization.
        phase: u8,
    },
    /// `rows * cols` exceeded the dense reference's guard (the
    /// reference is the semantics, not the economics; giant inputs
    /// belong to the sparse lane).
    DenseGuard {
        /// Declared row count.
        rows: usize,
        /// Declared column count.
        cols: usize,
    },
}

impl core::fmt::Display for NullspaceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NullspaceError::BadPrime { prime } => {
                write!(f, "prime {prime} is not an odd prime below 2^63")
            }
            NullspaceError::IndexOutOfRange {
                row,
                col,
                rows,
                cols,
            } => write!(
                f,
                "triplet ({row}, {col}) outside declared shape {rows}x{cols}"
            ),
            NullspaceError::FillBudget {
                live_entries,
                max_entries,
                pivots_done,
                phase,
            } => write!(
                f,
                "fill budget exceeded in phase {phase}: {live_entries} live entries > {max_entries} after {pivots_done} pivots"
            ),
            NullspaceError::DenseGuard { rows, cols } => {
                write!(f, "dense reference guard: {rows}x{cols} too large")
            }
        }
    }
}

impl std::error::Error for NullspaceError {}

/// Is `p` acceptable to every lane of this family? (Odd, below
/// $2^{63}$, and at least 3. Primality itself is the caller's promise
/// — the intended source is the audited table
/// [`crate::rnsfold::primes::PRIMES`], whose primality the rnsfold
/// battery re-verifies by deterministic Miller–Rabin.)
pub fn prime_ok(p: u64) -> bool {
    p >= 3 && p % 2 == 1 && p < (1u64 << 63)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triplets_validation() {
        let e = [(0u32, 0u32, 1i64)];
        assert!(Triplets {
            rows: 1,
            cols: 1,
            entries: &e
        }
        .validate()
        .is_ok());
        assert!(matches!(
            Triplets {
                rows: 1,
                cols: 1,
                entries: &[(1, 0, 1)]
            }
            .validate(),
            Err(NullspaceError::IndexOutOfRange { .. })
        ));
    }

    #[test]
    fn prime_gate() {
        assert!(prime_ok(3));
        assert!(prime_ok(9223372036854775783));
        assert!(!prime_ok(2));
        assert!(!prime_ok(1u64 << 63));
        assert!(!prime_ok(9223372036854775784));
    }
}
