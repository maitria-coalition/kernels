//! The scalar reference lane — the semantics of the family, written
//! for obviousness: for every query row, scan the attempt's pool in
//! order and stop at the first structurally equal row (the upstream
//! evaluator's own access pattern). Every acceleration lane is
//! conformance-gated against this function's outcome, bit-for-bit.

use super::batch::{RowEqBatch, RowEqError, RowEqOutcome};

/// Two slots structurally equal: sign, denominator id, and every
/// magnitude limb (planes are uniformly zero-padded, so plane
/// equality is canonical-limb equality).
fn slots_eq(b: &RowEqBatch, s1: usize, s2: usize) -> bool {
    if b.sign[s1] != b.sign[s2] || b.den_id[s1] != b.den_id[s2] {
        return false;
    }
    (0..b.k).all(|l| b.mag[l * b.n_slots + s1] == b.mag[l * b.n_slots + s2])
}

/// Two rows equal: same length, positionally same column and
/// structurally same value.
fn rows_eq(b: &RowEqBatch, r1: u32, r2: u32) -> bool {
    let (a0, a1) = (
        b.row_ptr[r1 as usize] as usize,
        b.row_ptr[r1 as usize + 1] as usize,
    );
    let (c0, c1) = (
        b.row_ptr[r2 as usize] as usize,
        b.row_ptr[r2 as usize + 1] as usize,
    );
    a1 - a0 == c1 - c0
        && (0..a1 - a0).all(|k| {
            let (i, j) = (a0 + k, c0 + k);
            b.nnz_col[i] == b.nnz_col[j]
                && slots_eq(b, b.nnz_slot[i] as usize, b.nnz_slot[j] as usize)
        })
}

/// Evaluate the batch on the reference lane: per attempt, every query
/// row must equal some pool row (vacuously true with no queries).
pub fn verify(b: &RowEqBatch) -> Result<RowEqOutcome, RowEqError> {
    b.validate()?;
    let member_ok = (0..b.n_attempts())
        .map(|a| {
            let queries = b.arow_ptr[a]..b.split[a];
            let pool = b.split[a]..b.arow_ptr[a + 1];
            queries
                .clone()
                .all(|q| pool.clone().any(|p| rows_eq(b, q, p)))
        })
        .collect();
    Ok(RowEqOutcome { member_ok })
}
