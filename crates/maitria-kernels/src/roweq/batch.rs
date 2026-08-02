//! The `roweq` batch descriptor — the one shape every lane of the
//! family consumes. Purely structural: no arithmetic quantity is
//! derived from these planes anywhere in the family (contrast
//! `rnsfold`, whose descriptor is also the home of a soundness-
//! critical bound derivation — this family has none to derive).

/// One batched row-membership problem.
///
/// Index vocabulary: *attempt* → range of *rows* (`arow_ptr`), split
/// into query rows then pool rows by `split`; *row* → range of *nnz*
/// (`row_ptr`); *nnz* → (column, value slot). Value slots carry
/// signed numerator magnitudes as zero-padded limb planes plus a
/// denominator id.
///
/// Caller obligation (the family's single semantic contract, see the
/// module documentation): within one attempt, `den_id` equality must
/// coincide with denominator equality. Everything else is validated
/// structurally by [`RowEqBatch::validate`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RowEqBatch {
    /// Limb count of the magnitude planes (uniform; values narrower
    /// than `k` limbs are zero-padded above). May be 0 only when
    /// `n_slots == 0`.
    pub k: usize,
    /// Number of value slots.
    pub n_slots: usize,
    /// Per slot: −1 / 0 / +1 (sign of the numerator; 0 iff the
    /// magnitude is zero).
    pub sign: Vec<i8>,
    /// Numerator magnitude planes, `[k * n_slots]`: limb `l` of slot
    /// `s` at `l * n_slots + s` (limb-plane SoA, matching `rnsfold`).
    pub mag: Vec<u64>,
    /// Per slot: attempt-scoped denominator id (caller-interned).
    pub den_id: Vec<u32>,

    /// Attempt → row range; `len = n_attempts + 1`, starts at 0.
    /// Attempt `a`'s rows are `arow_ptr[a]..arow_ptr[a+1]`: first its
    /// query rows, then its pool rows.
    pub arow_ptr: Vec<u32>,
    /// Per attempt: the query/pool split point —
    /// `arow_ptr[a] <= split[a] <= arow_ptr[a+1]`; queries are
    /// `arow_ptr[a]..split[a]`, pool is `split[a]..arow_ptr[a+1]`.
    pub split: Vec<u32>,
    /// Row → nnz range; `len = total_rows + 1`, starts at 0.
    pub row_ptr: Vec<u32>,
    /// Per nnz: column.
    pub nnz_col: Vec<u32>,
    /// Per nnz: value slot.
    pub nnz_slot: Vec<u32>,
}

/// Per-attempt lane outcome. No refusal plane exists: the family has
/// no deferral surface (structural equality is always decidable).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowEqOutcome {
    /// Per attempt: every query row equals some pool row (vacuously
    /// true for an attempt with no query rows).
    pub member_ok: Vec<bool>,
}

/// A structural defect in a descriptor, named per ENGINEERING #7.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RowEqError {
    /// A pointer array is not monotone from zero to the data length.
    Structure(&'static str),
    /// An index plane references past its target's length.
    Index(&'static str),
    /// `split[a]` falls outside attempt `a`'s row range.
    Split(usize),
    /// A sign entry is outside {−1, 0, +1} or inconsistent with a
    /// zero magnitude.
    Sign(usize),
}

impl std::fmt::Display for RowEqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RowEqError::Structure(s) => write!(f, "malformed pointer plane: {s}"),
            RowEqError::Index(s) => write!(f, "index out of range: {s}"),
            RowEqError::Split(a) => write!(f, "attempt {a}: split outside its row range"),
            RowEqError::Sign(i) => write!(f, "slot {i}: sign not in -1/0/+1 or zero-inconsistent"),
        }
    }
}

impl std::error::Error for RowEqError {}

fn monotone_from_zero(ptr: &[u32], end: usize, what: &'static str) -> Result<(), RowEqError> {
    if ptr.is_empty() || ptr[0] != 0 {
        return Err(RowEqError::Structure(what));
    }
    if ptr.windows(2).any(|w| w[0] > w[1]) {
        return Err(RowEqError::Structure(what));
    }
    if *ptr.last().unwrap() as usize != end {
        return Err(RowEqError::Structure(what));
    }
    Ok(())
}

impl RowEqBatch {
    /// Attempts in the batch.
    pub fn n_attempts(&self) -> usize {
        self.arow_ptr.len().saturating_sub(1)
    }

    /// Total rows (query + pool, all attempts).
    pub fn n_rows(&self) -> usize {
        self.row_ptr.len().saturating_sub(1)
    }

    /// Bit length of slot `s`'s magnitude (0 for zero).
    pub fn slot_bits(&self, s: usize) -> u64 {
        for l in (0..self.k).rev() {
            let limb = self.mag[l * self.n_slots + s];
            if limb != 0 {
                return l as u64 * 64 + (64 - limb.leading_zeros() as u64);
            }
        }
        0
    }

    /// Full structural validation — every lane calls this before
    /// comparing anything; a lane never reads an unvalidated
    /// descriptor.
    pub fn validate(&self) -> Result<(), RowEqError> {
        if self.sign.len() != self.n_slots
            || self.mag.len() != self.k * self.n_slots
            || self.den_id.len() != self.n_slots
        {
            return Err(RowEqError::Structure("slot planes disagree with n_slots/k"));
        }
        monotone_from_zero(&self.arow_ptr, self.n_rows(), "arow_ptr")?;
        monotone_from_zero(&self.row_ptr, self.nnz_col.len(), "row_ptr")?;
        if self.nnz_col.len() != self.nnz_slot.len() {
            return Err(RowEqError::Structure("nnz_col/nnz_slot lengths differ"));
        }
        if self.split.len() != self.n_attempts() {
            return Err(RowEqError::Structure("split does not cover the attempts"));
        }
        for a in 0..self.n_attempts() {
            if self.split[a] < self.arow_ptr[a] || self.split[a] > self.arow_ptr[a + 1] {
                return Err(RowEqError::Split(a));
            }
        }
        for &s in &self.nnz_slot {
            if s as usize >= self.n_slots {
                return Err(RowEqError::Index("nnz_slot"));
            }
        }
        for (s, &sg) in self.sign.iter().enumerate() {
            let zero = self.slot_bits(s) == 0;
            if !matches!(sg, -1..=1) || (zero != (sg == 0)) {
                return Err(RowEqError::Sign(s));
            }
        }
        Ok(())
    }
}
