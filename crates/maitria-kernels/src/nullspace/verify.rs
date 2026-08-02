//! Exact re-verification of a lifted vector against the original
//! triplets — the family's own first differential partner, in the
//! house verdict shape: `None` is the accept verdict, `Some(row)` the
//! first violated row (the refinement witness), and overflow is a
//! typed deferral to the caller's arbitrary-precision path.

use super::Triplets;

/// 128-bit accumulation overflowed; the exact answer needs the
/// caller's arbitrary-precision ladder (ENGINEERING #10).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifyOverflow {
    /// The row whose term overflowed, where attributable.
    pub row: u32,
    /// The column being accumulated.
    pub col: u32,
}

impl core::fmt::Display for VerifyOverflow {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "i128 overflow accumulating row {} into column {}",
            self.row, self.col
        )
    }
}

impl std::error::Error for VerifyOverflow {}

/// Is `y` (sparse over the *columns* of `a`, ascending) in the
/// nullspace of `a` — does every row's weighted sum
/// $\sum_c A_{r,c} \, y_c$ vanish? Exact `i128` arithmetic;
/// `Ok(None)` accepts, `Ok(Some(row))` names the lowest-index
/// violated row.
///
/// For the Petri workload, `a` is $C^\mathsf{T}$ (rows = transitions,
/// columns = places) — so this checks $A y = 0$, i.e.
/// $y^\mathsf{T} C = 0$: `y` is a P-invariant iff accept, and a
/// violation names the offending transition.
pub fn check_nullvector(a: Triplets<'_>, y: &[(u32, i128)]) -> Result<Option<u32>, VerifyOverflow> {
    // y as a dense-by-need lookup: y is sparse over columns of a.
    // (Coordinates of y index a's *columns*.)
    let mut acc: Vec<i128> = Vec::new();
    let mut touched: Vec<u32> = Vec::new();
    let mut acc_of: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    let lookup: std::collections::HashMap<u32, i128> = y.iter().copied().collect();
    for &(r, c, w) in a.entries {
        let Some(&yc) = lookup.get(&c) else {
            continue;
        };
        let term = yc
            .checked_mul(w as i128)
            .ok_or(VerifyOverflow { row: r, col: c })?;
        let slot = *acc_of.entry(r).or_insert_with(|| {
            acc.push(0);
            touched.push(r);
            acc.len() - 1
        });
        acc[slot] = acc[slot]
            .checked_add(term)
            .ok_or(VerifyOverflow { row: r, col: c })?;
    }
    let mut violated: Option<u32> = None;
    for (k, &r) in touched.iter().enumerate() {
        if acc[k] != 0 && violated.is_none_or(|v| r < v) {
            violated = Some(r);
        }
    }
    Ok(violated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_true_nullvector_and_rejects_mutant() {
        // A = [[1, 1], [2, 2]] (2 rows, 2 cols):
        // A y = (y0 + y1, 2 y0 + 2 y1); y = (1, -1) vanishes, and
        // y = (1, 1) violates row 0 first.
        let e = [(0u32, 0u32, 1i64), (0, 1, 1), (1, 0, 2), (1, 1, 2)];
        let a = Triplets {
            rows: 2,
            cols: 2,
            entries: &e,
        };
        assert_eq!(check_nullvector(a, &[(0, 1), (1, -1)]), Ok(None));
        assert_eq!(check_nullvector(a, &[(0, 1), (1, 1)]), Ok(Some(0)));
    }
}
