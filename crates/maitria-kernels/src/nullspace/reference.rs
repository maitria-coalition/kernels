//! The scalar reference: dense Gauss–Jordan to reduced row echelon
//! form with leftmost column scanning, plain `u128 %` residue
//! arithmetic, written for obviousness over speed. The semantics of
//! the family (the sparse lane is conformance-gated against this).

use super::{prime_ok, ModpNullspace, NullspaceError, SparseVec, Triplets};

/// Guard on `rows * cols` for the dense build: the reference exists
/// to define semantics on battery-sized inputs, not to run giants.
const DENSE_GUARD: usize = 1 << 26;

#[inline]
fn addmod(a: u64, b: u64, p: u64) -> u64 {
    let s = a + b; // both < p < 2^63: no overflow
    if s >= p {
        s - p
    } else {
        s
    }
}

#[inline]
fn submod(a: u64, b: u64, p: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a + p - b
    }
}

#[inline]
fn mulmod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

/// Modular inverse by Fermat (`p` prime, `a` nonzero mod `p`).
fn invmod(a: u64, p: u64) -> u64 {
    let mut e = p - 2;
    let mut base = a % p;
    let mut acc = 1u64;
    while e > 0 {
        if e & 1 == 1 {
            acc = mulmod(acc, base, p);
        }
        base = mulmod(base, base, p);
        e >>= 1;
    }
    acc
}

#[inline]
fn residue(w: i64, p: u64) -> u64 {
    if w >= 0 {
        w as u64 % p
    } else {
        (p - w.unsigned_abs() % p) % p
    }
}

/// Canonical mod-$p$ nullspace of `a` (see the module documentation
/// for the contract): dense reduced row echelon form, pivots chosen
/// by leftmost column, then lowest original row index.
pub fn nullspace_mod_p(a: Triplets<'_>, p: u64) -> Result<ModpNullspace, NullspaceError> {
    if !prime_ok(p) {
        return Err(NullspaceError::BadPrime { prime: p });
    }
    a.validate()?;
    let (m, n) = (a.rows, a.cols);
    if m.checked_mul(n).is_none_or(|mn| mn > DENSE_GUARD) {
        return Err(NullspaceError::DenseGuard { rows: m, cols: n });
    }

    // Dense accumulation (duplicates sum).
    let mut mat = vec![0u64; m * n];
    for &(r, c, w) in a.entries {
        let cell = &mut mat[r as usize * n + c as usize];
        *cell = addmod(*cell, residue(w, p), p);
    }
    // Original row index per current row position (swaps tracked so
    // witness rows are original indices).
    let mut orig: Vec<u32> = (0..m as u32).collect();

    // Gauss–Jordan, leftmost column scanning: full reduction, so the
    // surviving rows are the reduced row echelon form.
    let mut pivot_cols: Vec<u32> = Vec::new();
    let mut witness_rows: Vec<u32> = Vec::new();
    let mut row = 0usize;
    for col in 0..n {
        let Some(pr) = (row..m).find(|&r| mat[r * n + col] != 0) else {
            continue; // free column
        };
        if pr != row {
            for c in 0..n {
                mat.swap(row * n + c, pr * n + c);
            }
            orig.swap(row, pr);
        }
        let inv = invmod(mat[row * n + col], p);
        for c in col..n {
            mat[row * n + c] = mulmod(mat[row * n + c], inv, p);
        }
        for r in 0..m {
            if r != row && mat[r * n + col] != 0 {
                let f = mat[r * n + col];
                for c in col..n {
                    let d = mulmod(f, mat[row * n + c], p);
                    mat[r * n + c] = submod(mat[r * n + c], d, p);
                }
            }
        }
        pivot_cols.push(col as u32);
        witness_rows.push(orig[row]);
        row += 1;
        if row == m {
            break;
        }
    }
    let rank = pivot_cols.len();

    // Canonical basis: one vector per free column f, ascending; the
    // pivot-column coordinates are the negated RREF entries in f's
    // column.
    let is_pivot = {
        let mut flags = vec![false; n];
        for &c in &pivot_cols {
            flags[c as usize] = true;
        }
        flags
    };
    let mut basis = Vec::with_capacity(n - rank);
    for f in 0..n {
        if is_pivot[f] {
            continue;
        }
        let mut idx = Vec::new();
        let mut val = Vec::new();
        // Coordinates in ascending index order: pivot columns before
        // f that carry a nonzero RREF entry, then f itself (residue
        // 1), then pivot columns after f. Pivot column of RREF row j
        // is pivot_cols[j], and rows are sorted by pivot column, so a
        // single merge in column order suffices.
        for (j, &pc) in pivot_cols.iter().enumerate() {
            if pc as usize > f {
                break;
            }
            let v = mat[j * n + f];
            if v != 0 {
                idx.push(pc);
                val.push(p - v);
            }
        }
        idx.push(f as u32);
        val.push(1);
        for (j, &pc) in pivot_cols.iter().enumerate() {
            if (pc as usize) < f {
                continue;
            }
            let v = mat[j * n + f];
            if v != 0 {
                idx.push(pc);
                val.push(p - v);
            }
        }
        basis.push(SparseVec { idx, val });
    }

    Ok(ModpNullspace {
        prime: p,
        rank,
        pivot_cols,
        witness_rows,
        basis,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_has_trivial_nullspace() {
        let e = [(0u32, 0u32, 1i64), (1, 1, 1)];
        let ns = nullspace_mod_p(
            Triplets {
                rows: 2,
                cols: 2,
                entries: &e,
            },
            9223372036854775783,
        )
        .unwrap();
        assert_eq!(ns.rank, 2);
        assert!(ns.basis.is_empty());
    }

    #[test]
    fn zero_matrix_yields_identity_basis() {
        let ns = nullspace_mod_p(
            Triplets {
                rows: 3,
                cols: 4,
                entries: &[],
            },
            9223372036854775783,
        )
        .unwrap();
        assert_eq!(ns.rank, 0);
        assert_eq!(ns.basis.len(), 4);
        for (k, v) in ns.basis.iter().enumerate() {
            assert_eq!(v.idx, vec![k as u32]);
            assert_eq!(v.val, vec![1]);
        }
    }

    #[test]
    fn kernel_vector_of_sum_row() {
        // A = [1 1]: nullspace basis {(-1, 1)} => canonical (p-1, 1).
        let p = 9223372036854775783u64;
        let e = [(0u32, 0u32, 1i64), (0, 1, 1)];
        let ns = nullspace_mod_p(
            Triplets {
                rows: 1,
                cols: 2,
                entries: &e,
            },
            p,
        )
        .unwrap();
        assert_eq!(ns.rank, 1);
        assert_eq!(ns.pivot_cols, vec![0]);
        assert_eq!(ns.basis.len(), 1);
        assert_eq!(ns.basis[0].idx, vec![0, 1]);
        assert_eq!(ns.basis[0].val, vec![p - 1, 1]);
        assert_eq!(ns.free_col(0), 1);
    }
}
