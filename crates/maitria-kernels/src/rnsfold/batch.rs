//! The `rnsfold` batch descriptor — the one shape every lane of the
//! family consumes, and the home of the soundness-critical channel
//! selection (the bound derivation lives HERE, from the planes, never
//! in a caller assertion).

use super::primes::{bits_of_limbs, channels_for_bits};

/// Sentinel for "no conclusion value at this column" (compare the
/// fold against zero there).
pub const ABSENT: u32 = u32::MAX;

/// One batched linear-combination equality problem. See the module
/// documentation for the semantics; field-level invariants here.
///
/// Index vocabulary: *attempt* → range of *acols* (`acol_ptr`); *acol*
/// (attempt-local dense output column) → range of *nnz* (`csc_ptr`);
/// *nnz* → (a λ id, a value slot). Value slots carry raw signed
/// numerators as limb planes plus a multiplier id; the effective
/// operand of the fold is `value[slot] * mult[mult_id[slot]]`, and the
/// effective conclusion operand at an acol is
/// `value[concl_slot] * mult[mult_id[concl_slot]]`.
///
/// `PartialEq` is derived deliberately (the upstream `CsrBatch`
/// precedent): packers that claim to emit this descriptor directly
/// are differential-tested against a reference packer by plane-for-
/// plane structural equality, not merely verdict agreement.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RnsFoldBatch {
    /// Limb count of the magnitude planes (uniform; values narrower
    /// than `k` limbs are zero-padded above).
    pub k: usize,
    /// Number of value slots.
    pub n_slots: usize,
    /// Per slot: −1 / 0 / +1.
    pub sign: Vec<i8>,
    /// Magnitude planes, `[k * n_slots]`: limb `l` of slot `s` at
    /// `l * n_slots + s` (limb-plane SoA — a warp's consecutive slots
    /// touch consecutive words).
    pub mag: Vec<u64>,
    /// Per slot: index into `mults`.
    pub mult_id: Vec<u32>,
    /// Positive multipliers, little-endian limbs, no fixed width.
    /// (`mults[0]` is conventionally `[1]` — callers with unscaled
    /// slots point them there.)
    pub mults: Vec<Vec<u64>>,

    /// Combination coefficients $\hat\lambda$: (sign, magnitude
    /// limbs), no fixed width.
    pub lams: Vec<(i8, Vec<u64>)>,

    /// Attempt → acol range; `len = n_attempts + 1`, starts at 0.
    pub acol_ptr: Vec<u32>,
    /// Acol → nnz range; `len = total_acols + 1`, starts at 0.
    pub csc_ptr: Vec<u32>,
    /// Per nnz: λ id.
    pub csc_lam: Vec<u32>,
    /// Per nnz: value slot.
    pub csc_slot: Vec<u32>,
    /// Per acol: conclusion value slot, or [`ABSENT`].
    pub concl_slot: Vec<u32>,
}

/// Per-attempt lane outcome. `fold_ok[a]` is meaningful only where
/// `refused[a]` is false; a refused attempt got **no** verdict (the
/// caller's exact path owns it — the family's entire deferral
/// surface).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RnsFoldOutcome {
    /// Per attempt: every column's fold equals the conclusion.
    pub fold_ok: Vec<bool>,
    /// Per attempt: the channel budget could not cover this attempt's
    /// bound (prime table exhausted) — no verdict was produced.
    pub refused: Vec<bool>,
    /// Channels actually evaluated (the max requirement over admitted
    /// attempts).
    pub channels_used: usize,
}

/// A structural defect in a descriptor, named per ENGINEERING #7.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchError {
    /// A pointer array is not monotone from zero to the data length.
    Structure(&'static str),
    /// An index plane references past its target's length.
    Index(&'static str),
    /// A sign entry is outside {−1, 0, +1} or inconsistent with a
    /// zero magnitude.
    Sign(usize),
    /// A multiplier is zero (multipliers are positive by contract).
    ZeroMult(usize),
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchError::Structure(s) => write!(f, "malformed pointer plane: {s}"),
            BatchError::Index(s) => write!(f, "index out of range: {s}"),
            BatchError::Sign(i) => write!(f, "slot {i}: sign not in -1/0/+1 or zero-inconsistent"),
            BatchError::ZeroMult(i) => write!(f, "mult {i} is zero (multipliers are positive)"),
        }
    }
}

impl std::error::Error for BatchError {}

fn monotone_from_zero(ptr: &[u32], end: usize, what: &'static str) -> Result<(), BatchError> {
    if ptr.is_empty() || ptr[0] != 0 {
        return Err(BatchError::Structure(what));
    }
    if ptr.windows(2).any(|w| w[0] > w[1]) {
        return Err(BatchError::Structure(what));
    }
    if *ptr.last().unwrap() as usize != end {
        return Err(BatchError::Structure(what));
    }
    Ok(())
}

impl RnsFoldBatch {
    /// Attempts in the batch.
    pub fn n_attempts(&self) -> usize {
        self.acol_ptr.len().saturating_sub(1)
    }

    /// Full structural validation — every lane calls this before
    /// arithmetic; a lane never reads an unvalidated descriptor.
    ///
    /// Under the `rayon` feature, descriptors past
    /// [`Self::VALIDATE_PAR_MIN`] run a parallel any-defect scan of
    /// the same predicates first; a clean scan returns `Ok` (identical
    /// to what the serial walk would return), and any hit falls back
    /// to the serial walk so the *canonical first error* is reported
    /// — behavior is exactly the serial spelling's on every input,
    /// only the clean path's cost changes (measured: 30–45 ms of
    /// serial validation per large batch, more than the device
    /// kernel itself).
    pub fn validate(&self) -> Result<(), BatchError> {
        #[cfg(feature = "rayon")]
        if self.n_slots.max(self.csc_lam.len()) >= Self::VALIDATE_PAR_MIN {
            if !self.any_defect_par() {
                return Ok(());
            }
            return self.validate_serial();
        }
        self.validate_serial()
    }

    /// Element count past which [`Self::validate`] uses the parallel
    /// scan (cost-only; see there).
    #[cfg(feature = "rayon")]
    pub const VALIDATE_PAR_MIN: usize = 1 << 16;

    /// Parallel any-defect scan: true iff [`Self::validate_serial`]
    /// would return an error — same predicates, order-free.
    #[cfg(feature = "rayon")]
    fn any_defect_par(&self) -> bool {
        use rayon::prelude::*;
        // small planes + structure, serially (cheap):
        if self.sign.len() != self.n_slots
            || self.mag.len() != self.k * self.n_slots
            || self.mult_id.len() != self.n_slots
            || self.acol_ptr.is_empty()
            || self.acol_ptr[0] != 0
            || self.acol_ptr.windows(2).any(|w| w[0] > w[1])
            || *self.acol_ptr.last().unwrap() as usize != self.concl_slot.len()
            || self.csc_ptr.is_empty()
            || self.csc_ptr[0] != 0
            || self.csc_ptr.windows(2).any(|w| w[0] > w[1])
            || *self.csc_ptr.last().unwrap() as usize != self.csc_lam.len()
            || self.csc_lam.len() != self.csc_slot.len()
            || self.csc_ptr.len() != self.concl_slot.len() + 1
            || self
                .lams
                .iter()
                .any(|&(s, ref m)| !matches!(s, -1..=1) || ((bits_of_limbs(m) == 0) != (s == 0)))
            || self.mults.iter().any(|m| bits_of_limbs(m) == 0)
        {
            return true;
        }
        // big planes, in parallel:
        (0..self.n_slots).into_par_iter().any(|s| {
            let sg = self.sign[s];
            !matches!(sg, -1..=1) || ((self.slot_bits(s) == 0) != (sg == 0))
        }) || self
            .csc_lam
            .par_iter()
            .any(|&l| l as usize >= self.lams.len())
            || self
                .csc_slot
                .par_iter()
                .any(|&s| s as usize >= self.n_slots)
            || self
                .concl_slot
                .par_iter()
                .any(|&c| c != ABSENT && c as usize >= self.n_slots)
            || self
                .mult_id
                .par_iter()
                .any(|&m| m as usize >= self.mults.len())
    }

    /// The serial walk (the original spelling; canonical error order).
    fn validate_serial(&self) -> Result<(), BatchError> {
        if self.sign.len() != self.n_slots
            || self.mag.len() != self.k * self.n_slots
            || self.mult_id.len() != self.n_slots
        {
            return Err(BatchError::Structure("slot planes disagree with n_slots/k"));
        }
        monotone_from_zero(&self.acol_ptr, self.concl_slot.len(), "acol_ptr")?;
        monotone_from_zero(&self.csc_ptr, self.csc_lam.len(), "csc_ptr")?;
        if self.csc_lam.len() != self.csc_slot.len() {
            return Err(BatchError::Structure("csc_lam/csc_slot lengths differ"));
        }
        if self.csc_ptr.len() != self.concl_slot.len() + 1 {
            return Err(BatchError::Structure(
                "csc_ptr does not cover the acol space",
            ));
        }
        for (i, &(s, ref m)) in self.lams.iter().enumerate() {
            let zero = bits_of_limbs(m) == 0;
            if !matches!(s, -1..=1) || (zero != (s == 0)) {
                return Err(BatchError::Sign(i));
            }
        }
        for (s, &sg) in self.sign.iter().enumerate() {
            let zero = self.slot_bits(s) == 0;
            if !matches!(sg, -1..=1) || (zero != (sg == 0)) {
                return Err(BatchError::Sign(s));
            }
        }
        for (i, m) in self.mults.iter().enumerate() {
            if bits_of_limbs(m) == 0 {
                return Err(BatchError::ZeroMult(i));
            }
        }
        for &l in &self.csc_lam {
            if l as usize >= self.lams.len() {
                return Err(BatchError::Index("csc_lam"));
            }
        }
        for &s in &self.csc_slot {
            if s as usize >= self.n_slots {
                return Err(BatchError::Index("csc_slot"));
            }
        }
        for &c in &self.concl_slot {
            if c != ABSENT && c as usize >= self.n_slots {
                return Err(BatchError::Index("concl_slot"));
            }
        }
        for &m in &self.mult_id {
            if m as usize >= self.mults.len() {
                return Err(BatchError::Index("mult_id"));
            }
        }
        Ok(())
    }

    /// Bit length of slot `s`'s raw magnitude.
    pub fn slot_bits(&self, s: usize) -> u64 {
        let mut bits = 0;
        for l in (0..self.k).rev() {
            let limb = self.mag[l * self.n_slots + s];
            if limb != 0 {
                bits = l as u64 * 64 + (64 - limb.leading_zeros() as u64);
                break;
            }
        }
        bits
    }

    /// The soundness-critical bound, derived from the planes: a
    /// conservative bit length `B(a)` such that every column of
    /// attempt `a` satisfies $|\Delta| < 2^{B(a)}$.
    ///
    /// Derivation (all quantities measured, per attempt): with
    /// $b_\lambda$ = max λ bit length, $b_v$ = max over the attempt's
    /// fold slots of raw-value bits + its multiplier's bits, $b_c$ =
    /// the same maximum over its conclusion slots, and $T$ = max
    /// per-column term count,
    /// $|\Delta| \le T \cdot 2^{b_\lambda + b_v} + 2^{b_c}
    ///   < 2^{\max(b_\lambda + b_v + \lceil \log_2 T \rceil,\, b_c) + 2}.$
    pub fn required_bits(&self, attempt: usize) -> u64 {
        let (a0, a1) = (
            self.acol_ptr[attempt] as usize,
            self.acol_ptr[attempt + 1] as usize,
        );
        let mut b_lam: u64 = 0;
        let mut b_v: u64 = 0;
        let mut b_c: u64 = 0;
        let mut t_max: u64 = 0;
        let mult_bits =
            |s: usize| self.slot_bits(s) + bits_of_limbs(&self.mults[self.mult_id[s] as usize]);
        for acol in a0..a1 {
            let (n0, n1) = (self.csc_ptr[acol] as usize, self.csc_ptr[acol + 1] as usize);
            t_max = t_max.max((n1 - n0) as u64);
            for i in n0..n1 {
                b_lam = b_lam.max(bits_of_limbs(&self.lams[self.csc_lam[i] as usize].1));
                b_v = b_v.max(mult_bits(self.csc_slot[i] as usize));
            }
            let c = self.concl_slot[acol];
            if c != ABSENT {
                b_c = b_c.max(mult_bits(c as usize));
            }
        }
        let log_t = 64 - t_max.leading_zeros() as u64; // ceil(log2(T)) + slack
        (b_lam + b_v + log_t).max(b_c) + 2
    }

    /// [`Self::required_bits`], evaluated against precomputed per-slot
    /// and per-λ bit-length arrays (same maxima, same formula — the
    /// arrays only replace repeated `slot_bits`/`bits_of_limbs` walks
    /// with u64 reads; `tests/rnsfold.rs` pins the equality per
    /// attempt).
    fn required_bits_from(&self, attempt: usize, slot_eff: &[u64], lam_bits: &[u64]) -> u64 {
        let (a0, a1) = (
            self.acol_ptr[attempt] as usize,
            self.acol_ptr[attempt + 1] as usize,
        );
        let mut b_lam: u64 = 0;
        let mut b_v: u64 = 0;
        let mut b_c: u64 = 0;
        let mut t_max: u64 = 0;
        for acol in a0..a1 {
            let (n0, n1) = (self.csc_ptr[acol] as usize, self.csc_ptr[acol + 1] as usize);
            t_max = t_max.max((n1 - n0) as u64);
            for i in n0..n1 {
                b_lam = b_lam.max(lam_bits[self.csc_lam[i] as usize]);
                b_v = b_v.max(slot_eff[self.csc_slot[i] as usize]);
            }
            let c = self.concl_slot[acol];
            if c != ABSENT {
                b_c = b_c.max(slot_eff[c as usize]);
            }
        }
        let log_t = 64 - t_max.leading_zeros() as u64; // ceil(log2(T)) + slack
        (b_lam + b_v + log_t).max(b_c) + 2
    }

    /// Channel count covering every attempt's bound, plus the refusal
    /// set (attempts beyond the prime table's capacity).
    ///
    /// Identical outputs to mapping [`Self::required_bits`] over the
    /// attempts (the battery pins this); the batched spelling
    /// precomputes per-slot effective bit lengths once instead of
    /// re-walking the limb planes per nnz, and the `rayon` feature
    /// parallelizes the (associative, commutative) max-reductions —
    /// cost may change, outputs may not.
    pub fn plan_channels(&self) -> (usize, Vec<bool>) {
        let mult_bits: Vec<u64> = self.mults.iter().map(|m| bits_of_limbs(m)).collect();
        let slot_eff_of = |s: usize| self.slot_bits(s) + mult_bits[self.mult_id[s] as usize];
        let lam_bits_of = |l: &(i8, Vec<u64>)| bits_of_limbs(&l.1);
        let n = self.n_attempts();

        #[cfg(feature = "rayon")]
        let (slot_eff, lam_bits, req): (Vec<u64>, Vec<u64>, Vec<u64>) = {
            use rayon::prelude::*;
            let slot_eff: Vec<u64> = (0..self.n_slots).into_par_iter().map(slot_eff_of).collect();
            let lam_bits: Vec<u64> = self.lams.par_iter().map(lam_bits_of).collect();
            let req = (0..n)
                .into_par_iter()
                .map(|a| self.required_bits_from(a, &slot_eff, &lam_bits))
                .collect();
            (slot_eff, lam_bits, req)
        };
        #[cfg(not(feature = "rayon"))]
        let (slot_eff, lam_bits, req): (Vec<u64>, Vec<u64>, Vec<u64>) = {
            let slot_eff: Vec<u64> = (0..self.n_slots).map(slot_eff_of).collect();
            let lam_bits: Vec<u64> = self.lams.iter().map(lam_bits_of).collect();
            let req = (0..n)
                .map(|a| self.required_bits_from(a, &slot_eff, &lam_bits))
                .collect();
            (slot_eff, lam_bits, req)
        };
        let _ = (&slot_eff, &lam_bits);

        let mut refused = vec![false; n];
        let mut channels = 1usize;
        for (r, &bits) in refused.iter_mut().zip(req.iter()) {
            match channels_for_bits(bits) {
                Some(c) => channels = channels.max(c),
                None => *r = true,
            }
        }
        (channels, refused)
    }
}
