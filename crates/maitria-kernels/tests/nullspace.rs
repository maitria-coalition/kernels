//! Conformance battery for the `nullspace` family.
//!
//! Partners (ENGINEERING #5):
//! - the in-tree dense reference (`nullspace::reference`) — the
//!   semantics every lane must match bit-identically on verdict
//!   fields (`rank`, `pivot_cols`, `basis`);
//! - an independent exact partner in this file: big-rational
//!   Gauss–Jordan over `num-bigint` — no residues, no Montgomery, no
//!   shared code with the lanes — whose primitive integer basis the
//!   full pipeline (per-prime kernels → CRT → rational
//!   reconstruction → primitive clearing) must reproduce;
//! - the family's own exact re-verification (`nullspace::verify`),
//!   itself cross-checked here against designed mutants.
//!
//! Witness fields (`witness_rows`) are checked for *validity* (the
//! selected rows alone reproduce the rank), not identity — lanes may
//! legitimately select different row subsets.

use maitria_kernels::nullspace::{lift, reference, sparse, verify, ModpNullspace, Triplets};
use maitria_kernels::rnsfold::primes::PRIMES;
use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};
use proptest::prelude::*;

// ----------------------------------------------------------------
// Independent exact partner: big-rational Gauss–Jordan.
// ----------------------------------------------------------------

fn gcd_big(a: &BigInt, b: &BigInt) -> BigInt {
    let mut a = a.abs();
    let mut b = b.abs();
    while !b.is_zero() {
        let r = &a % &b;
        a = b;
        b = r;
    }
    a
}

#[derive(Clone, PartialEq, Eq)]
struct Q {
    n: BigInt,
    d: BigInt, // > 0
}

impl Q {
    fn int(n: i64) -> Q {
        Q {
            n: BigInt::from(n),
            d: BigInt::one(),
        }
    }
    fn norm(mut n: BigInt, mut d: BigInt) -> Q {
        assert!(!d.is_zero());
        if d.is_negative() {
            n = -n;
            d = -d;
        }
        let g = gcd_big(&n, &d);
        if !g.is_zero() && !g.is_one() {
            n /= &g;
            d /= g;
        }
        if n.is_zero() {
            d = BigInt::one();
        }
        Q { n, d }
    }
    fn is_zero(&self) -> bool {
        self.n.is_zero()
    }
    fn sub_mul(&self, f: &Q, o: &Q) -> Q {
        // self - f * o
        Q::norm(
            &self.n * &f.d * &o.d - &f.n * &o.n * &self.d,
            &self.d * &f.d * &o.d,
        )
    }
    fn div(&self, o: &Q) -> Q {
        Q::norm(&self.n * &o.d, &self.d * &o.n)
    }
}

struct ExactNullspace {
    rank: usize,
    pivot_cols: Vec<u32>,
    /// Primitive integer basis vectors, sparse, ascending index,
    /// first nonzero positive; one per free column, ascending.
    basis: Vec<Vec<(u32, BigInt)>>,
}

fn exact_nullspace(rows: usize, cols: usize, entries: &[(u32, u32, i64)]) -> ExactNullspace {
    // Dense accumulation over Q.
    let mut a: Vec<Vec<Q>> = vec![vec![Q::int(0); cols]; rows];
    for &(r, c, w) in entries {
        let cell = &a[r as usize][c as usize];
        a[r as usize][c as usize] = Q::norm(
            &cell.n * BigInt::one() + BigInt::from(w) * &cell.d,
            cell.d.clone(),
        );
    }
    // Gauss–Jordan, leftmost scanning.
    let mut pivot_cols: Vec<u32> = Vec::new();
    let mut row = 0usize;
    for col in 0..cols {
        let Some(pr) = (row..rows).find(|&r| !a[r][col].is_zero()) else {
            continue;
        };
        a.swap(row, pr);
        let inv = a[row][col].clone();
        for x in &mut a[row] {
            *x = x.div(&inv);
        }
        for r in 0..rows {
            if r != row && !a[r][col].is_zero() {
                let f = a[r][col].clone();
                let pivot_row = a[row].clone();
                for (c, cell) in a[r].iter_mut().enumerate() {
                    *cell = cell.sub_mul(&f, &pivot_row[c]);
                }
            }
        }
        pivot_cols.push(col as u32);
        row += 1;
        if row == rows {
            break;
        }
    }
    let rank = pivot_cols.len();
    let is_pivot: Vec<bool> = {
        let mut v = vec![false; cols];
        for &c in &pivot_cols {
            v[c as usize] = true;
        }
        v
    };
    let mut basis = Vec::new();
    for f in 0..cols {
        if is_pivot[f] {
            continue;
        }
        // Rational vector: y_f = 1, y_{pc_j} = -a[j][f].
        let mut ent: Vec<(u32, Q)> = Vec::new();
        for (j, &pc) in pivot_cols.iter().enumerate() {
            if !a[j][f].is_zero() {
                ent.push((pc, Q::norm(-a[j][f].n.clone(), a[j][f].d.clone())));
            }
        }
        ent.push((f as u32, Q::int(1)));
        ent.sort_by_key(|&(c, _)| c);
        // Clear to primitive integers.
        let mut l = BigInt::one();
        for (_, q) in &ent {
            let g = gcd_big(&l, &q.d);
            l = &l / g * &q.d;
        }
        let mut iv: Vec<(u32, BigInt)> =
            ent.iter().map(|(c, q)| (*c, &q.n * (&l / &q.d))).collect();
        let mut g = BigInt::zero();
        for (_, v) in &iv {
            g = gcd_big(&g, v);
        }
        if !g.is_zero() && !g.is_one() {
            for (_, v) in &mut iv {
                *v = &*v / &g;
            }
        }
        if let Some((_, first)) = iv.iter().find(|(_, v)| !v.is_zero()) {
            if first.is_negative() {
                for (_, v) in &mut iv {
                    *v = -&*v;
                }
            }
        }
        basis.push(iv);
    }
    ExactNullspace {
        rank,
        pivot_cols,
        basis,
    }
}

// ----------------------------------------------------------------
// Generation: rank-deficient sparse matrices with bounded minors
// (dims <= 6, small entries: every Hadamard bound sits far below the
// 63-bit table primes, so those primes are deterministically lucky
// and mod-p profiles MUST equal the exact profile).
// ----------------------------------------------------------------

#[derive(Clone, Debug)]
struct Mat {
    rows: usize,
    cols: usize,
    entries: Vec<(u32, u32, i64)>,
}

fn arb_mat() -> impl Strategy<Value = Mat> {
    (1usize..7, 1usize..7)
        .prop_flat_map(|(m, n)| {
            let outer = proptest::collection::vec(
                (
                    proptest::collection::vec(-3i64..4, m),
                    proptest::collection::vec(-3i64..4, n),
                ),
                0..3,
            );
            let noise = proptest::collection::vec((0..m as u32, 0..n as u32, -5i64..6), 0..10);
            (Just(m), Just(n), outer, noise)
        })
        .prop_map(|(m, n, outer, noise)| {
            let mut entries = noise;
            for (u, v) in outer {
                for (r, &ur) in u.iter().enumerate() {
                    for (c, &vc) in v.iter().enumerate() {
                        let w = ur * vc;
                        if w != 0 {
                            entries.push((r as u32, c as u32, w));
                        }
                    }
                }
            }
            Mat {
                rows: m,
                cols: n,
                entries,
            }
        })
}

fn tp(m: &Mat) -> Triplets<'_> {
    Triplets {
        rows: m.rows,
        cols: m.cols,
        entries: &m.entries,
    }
}

/// Witness validity: the selected rows alone reproduce the rank.
fn witness_valid(m: &Mat, ns: &ModpNullspace) {
    let keep: std::collections::HashSet<u32> = ns.witness_rows.iter().copied().collect();
    assert_eq!(keep.len(), ns.rank, "witness rows distinct");
    let sub: Vec<(u32, u32, i64)> = m
        .entries
        .iter()
        .filter(|(r, _, _)| keep.contains(r))
        .map(|&(r, c, w)| (r, c, w))
        .collect();
    let sub_ns = reference::nullspace_mod_p(
        Triplets {
            rows: m.rows,
            cols: m.cols,
            entries: &sub,
        },
        ns.prime,
    )
    .unwrap();
    assert_eq!(sub_ns.rank, ns.rank, "witness rows reproduce the rank");
}

proptest! {
    /// The lane law: sparse lane verdict fields bit-identical to the
    /// reference, lucky primes and unlucky primes alike.
    #[test]
    fn sparse_conforms_to_reference(m in arb_mat()) {
        for p in [PRIMES[0], PRIMES[7], 3u64, 5u64] {
            let r = reference::nullspace_mod_p(tp(&m), p).unwrap();
            let s = sparse::nullspace_mod_p(tp(&m), p, sparse::Params::default()).unwrap();
            prop_assert_eq!(r.rank, s.rank);
            prop_assert_eq!(&r.pivot_cols, &s.pivot_cols);
            prop_assert_eq!(&r.basis, &s.basis);
            witness_valid(&m, &r);
            witness_valid(&m, &s);
        }
    }

    /// Cost knobs change cost, never verdicts: a degenerate parameter
    /// set that forces the dense core immediately must agree with the
    /// default (which, at battery sizes, stays sparse until the
    /// density trigger).
    #[test]
    fn dense_core_conforms(m in arb_mat()) {
        let p = PRIMES[3];
        let forced = sparse::Params {
            max_entries: 1 << 22,
            dense_cap: 1 << 20,
            dense_inv_density: usize::MAX,
        };
        let a = sparse::nullspace_mod_p(tp(&m), p, sparse::Params::default()).unwrap();
        let b = sparse::nullspace_mod_p(tp(&m), p, forced).unwrap();
        prop_assert_eq!(a.rank, b.rank);
        prop_assert_eq!(&a.pivot_cols, &b.pivot_cols);
        prop_assert_eq!(&a.basis, &b.basis);
        witness_valid(&m, &b);
    }

    /// End-to-end against the independent exact partner: at
    /// deterministically lucky primes, profiles match the exact
    /// profile, and the lifted primitive vectors are the exact
    /// primitive vectors.
    #[test]
    fn lift_matches_exact_partner(m in arb_mat()) {
        let exact = exact_nullspace(m.rows, m.cols, &m.entries);
        let n1 = sparse::nullspace_mod_p(tp(&m), PRIMES[0], sparse::Params::default()).unwrap();
        let n2 = sparse::nullspace_mod_p(tp(&m), PRIMES[1], sparse::Params::default()).unwrap();
        prop_assert_eq!(n1.rank, exact.rank);
        prop_assert_eq!(&n1.pivot_cols, &exact.pivot_cols);
        let lifted = lift::crt2_ratrec(&n1, &n2).unwrap();
        prop_assert_eq!(lifted.len(), exact.basis.len());
        for (lv, ev) in lifted.iter().zip(&exact.basis) {
            let lv_big: Vec<(u32, BigInt)> =
                lv.iter().map(|&(c, v)| (c, BigInt::from(v))).collect();
            prop_assert_eq!(&lv_big, ev);
        }
    }

    /// Every lifted vector passes exact re-verification; a designed
    /// single-coordinate mutant fails it with a named row.
    #[test]
    fn lifted_vectors_verify_and_mutants_fail(m in arb_mat()) {
        let n1 = sparse::nullspace_mod_p(tp(&m), PRIMES[0], sparse::Params::default()).unwrap();
        let n2 = sparse::nullspace_mod_p(tp(&m), PRIMES[1], sparse::Params::default()).unwrap();
        let lifted = lift::crt2_ratrec(&n1, &n2).unwrap();
        // Columns of the matrix that are nonzero (a +1 perturbation
        // there must break the nullspace property).
        let mut col_nonzero = vec![false; m.cols];
        {
            // Accumulate: a column is "nonzero" if some accumulated
            // entry survives.
            let mut acc = std::collections::HashMap::<(u32, u32), i64>::new();
            for &(r, c, w) in &m.entries {
                *acc.entry((r, c)).or_insert(0) += w;
            }
            for ((_, c), w) in acc {
                if w != 0 {
                    col_nonzero[c as usize] = true;
                }
            }
        }
        for y in &lifted {
            prop_assert_eq!(verify::check_nullvector(tp(&m), y), Ok(None));
            if let Some(k) = y.iter().position(|&(c, _)| col_nonzero[c as usize]) {
                let mut mutant = y.clone();
                mutant[k].1 += 1;
                let mutant: Vec<(u32, i128)> =
                    mutant.into_iter().filter(|&(_, v)| v != 0).collect();
                let verdict = verify::check_nullvector(tp(&m), &mutant).unwrap();
                prop_assert!(verdict.is_some(), "mutant must violate some row");
            }
        }
    }
}

// ----------------------------------------------------------------
// Deterministic refusals and edges.
// ----------------------------------------------------------------

#[test]
fn fill_budget_refusal_is_typed() {
    let e = [(0u32, 0u32, 1i64), (0, 1, 1), (1, 0, 1), (1, 1, 2)];
    let out = sparse::nullspace_mod_p(
        Triplets {
            rows: 2,
            cols: 2,
            entries: &e,
        },
        PRIMES[0],
        sparse::Params {
            max_entries: 1,
            dense_cap: 0,
            dense_inv_density: 8,
        },
    );
    assert!(matches!(
        out,
        Err(maitria_kernels::nullspace::NullspaceError::FillBudget { .. })
    ));
}

#[test]
fn profile_mismatch_between_unlucky_primes_is_typed() {
    // [[3]] has rank 1 exactly, rank 0 mod 3.
    let e = [(0u32, 0u32, 3i64)];
    let t = Triplets {
        rows: 1,
        cols: 1,
        entries: &e,
    };
    let n3 = reference::nullspace_mod_p(t, 3).unwrap();
    let n5 = reference::nullspace_mod_p(t, 5).unwrap();
    assert_eq!(n3.rank, 0);
    assert_eq!(n5.rank, 1);
    assert!(matches!(
        lift::crt2_ratrec(&n3, &n5),
        Err(lift::LiftError::ProfileMismatch { .. })
    ));
}

#[test]
fn lift_capacity_refusal_is_typed() {
    // Three pairwise-coprime ~2^62 denominators: each coordinate
    // reconstructs, but the vector's denominator lcm (~2^186)
    // overflows the two-channel clearing path — the caller's
    // arbitrary-precision ladder owns it.
    let a1 = (1i64 << 62) + 1;
    let a2 = (1i64 << 62) - 1;
    let a3 = (1i64 << 62) + 3;
    let e = [
        (0u32, 0u32, a1),
        (0, 3, 1),
        (1, 1, a2),
        (1, 3, 1),
        (2, 2, a3),
        (2, 3, 1),
    ];
    let t = Triplets {
        rows: 3,
        cols: 4,
        entries: &e,
    };
    let n1 = sparse::nullspace_mod_p(t, PRIMES[0], sparse::Params::default()).unwrap();
    let n2 = sparse::nullspace_mod_p(t, PRIMES[1], sparse::Params::default()).unwrap();
    assert!(matches!(
        lift::crt2_ratrec(&n1, &n2),
        Err(lift::LiftError::Capacity { .. })
    ));
}

#[test]
fn empty_and_zero_shapes() {
    // No rows at all: everything is free.
    let t = Triplets {
        rows: 0,
        cols: 3,
        entries: &[],
    };
    for ns in [
        reference::nullspace_mod_p(t, PRIMES[0]).unwrap(),
        sparse::nullspace_mod_p(t, PRIMES[0], sparse::Params::default()).unwrap(),
    ] {
        assert_eq!(ns.rank, 0);
        assert_eq!(ns.basis.len(), 3);
    }
    // Duplicate triplets cancelling to a zero matrix.
    let e = [(0u32, 0u32, 5i64), (0, 0, -5)];
    let t = Triplets {
        rows: 1,
        cols: 1,
        entries: &e,
    };
    let ns = sparse::nullspace_mod_p(t, PRIMES[0], sparse::Params::default()).unwrap();
    assert_eq!(ns.rank, 0);
    assert_eq!(ns.basis.len(), 1);
}

#[test]
fn determinism() {
    let e = [
        (0u32, 0u32, 1i64),
        (0, 2, -1),
        (1, 1, 2),
        (1, 2, 2),
        (2, 0, 1),
        (2, 1, 1),
    ];
    let t = Triplets {
        rows: 3,
        cols: 3,
        entries: &e,
    };
    let a = sparse::nullspace_mod_p(t, PRIMES[0], sparse::Params::default()).unwrap();
    let b = sparse::nullspace_mod_p(t, PRIMES[0], sparse::Params::default()).unwrap();
    assert_eq!(a, b);
}
