//! roweq battery (ENGINEERING #2/#5): the reference lane against an
//! independent hash-set formulation of membership (pool rows interned
//! by trimmed canonical bytes, queries probed — a different algorithm
//! with a different access pattern, no positional scanning anywhere),
//! plus accept-shaped constructions whose mutants must flip, the
//! padding-invariance property, structural-validation refusals, and
//! edge shapes.

use maitria_kernels::roweq::batch::{RowEqBatch, RowEqError};
use maitria_kernels::roweq::reference;
use proptest::collection::vec as pvec;
use proptest::prelude::*;
use std::collections::HashSet;

// ---- logical model ---------------------------------------------------

/// A value in the logical model: sign, trimmed magnitude limbs,
/// denominator id.
type Val = (i8, Vec<u64>, u32);
/// A row: sorted (column, value) pairs.
type Row = Vec<(u32, Val)>;
/// An attempt: query rows and pool rows.
#[derive(Debug, Clone)]
struct Attempt {
    queries: Vec<Row>,
    pool: Vec<Row>,
}

fn trim(mut limbs: Vec<u64>) -> Vec<u64> {
    while limbs.last() == Some(&0) {
        limbs.pop();
    }
    limbs
}

/// Descriptor construction from the logical model. `extra_pad` grows
/// the uniform limb count past the minimum (the padding-invariance
/// property runs the same model at two paddings).
fn build(attempts: &[Attempt], extra_pad: usize) -> RowEqBatch {
    let mut b = RowEqBatch::default();
    let mut k = 0usize;
    for a in attempts {
        for r in a.queries.iter().chain(&a.pool) {
            for (_, (_, m, _)) in r {
                k = k.max(trim(m.clone()).len());
            }
        }
    }
    let k = k + extra_pad;
    b.k = k;
    b.arow_ptr.push(0);
    b.row_ptr.push(0);
    let mut slots: Vec<Val> = Vec::new();
    for a in attempts {
        for (qi, r) in a.queries.iter().chain(&a.pool).enumerate() {
            let _ = qi;
            for (col, v) in r {
                b.nnz_col.push(*col);
                b.nnz_slot.push(slots.len() as u32);
                slots.push(v.clone());
            }
            b.row_ptr.push(b.nnz_col.len() as u32);
        }
        b.split
            .push(b.arow_ptr.last().unwrap() + a.queries.len() as u32);
        b.arow_ptr
            .push(b.arow_ptr.last().unwrap() + (a.queries.len() + a.pool.len()) as u32);
    }
    b.n_slots = slots.len();
    b.mag = vec![0u64; k * slots.len()];
    for (s, (sg, m, d)) in slots.iter().enumerate() {
        let t = trim(m.clone());
        // Canonical sign for the model: zero magnitude forces sign 0.
        b.sign.push(if t.is_empty() { 0 } else { *sg });
        b.den_id.push(*d);
        for (l, limb) in t.iter().enumerate() {
            b.mag[l * slots.len() + s] = *limb;
        }
    }
    b
}

// ---- the independent partner -----------------------------------------

/// Canonical bytes of a row: trimmed limbs, canonical sign, den id,
/// column — order-preserving, hash-set interned. No positional scan,
/// no plane arithmetic: a genuinely different algorithm.
fn row_key(r: &Row) -> Vec<u8> {
    let mut out = Vec::new();
    for (col, (sg, m, d)) in r {
        let t = trim(m.clone());
        out.extend(col.to_le_bytes());
        out.push((if t.is_empty() { 0i8 } else { *sg }) as u8);
        out.extend(d.to_le_bytes());
        out.extend((t.len() as u32).to_le_bytes());
        for l in &t {
            out.extend(l.to_le_bytes());
        }
    }
    out
}

fn membership_hashset(attempts: &[Attempt]) -> Vec<bool> {
    attempts
        .iter()
        .map(|a| {
            let pool: HashSet<Vec<u8>> = a.pool.iter().map(row_key).collect();
            a.queries.iter().all(|q| pool.contains(&row_key(q)))
        })
        .collect()
}

fn check(attempts: &[Attempt]) {
    let want = membership_hashset(attempts);
    for pad in [0usize, 2] {
        let b = build(attempts, pad);
        let out = reference::verify(&b).expect("valid descriptor");
        assert_eq!(out.member_ok, want, "reference vs hash partner (pad={pad})");
    }
}

// ---- generators --------------------------------------------------------

fn arb_val() -> BoxedStrategy<Val> {
    (
        prop_oneof![Just(-1i8), Just(1i8)],
        pvec(any::<u64>(), 0..4),
        0u32..5,
    )
        .prop_map(|(s, m, d)| (s, m, d))
        .boxed()
}

fn arb_row() -> BoxedStrategy<Row> {
    pvec((0u32..12, arb_val()), 0..6)
        .prop_map(|mut v| {
            v.sort_by_key(|(c, _)| *c);
            v.dedup_by_key(|(c, _)| *c);
            v
        })
        .boxed()
}

fn arb_attempt() -> BoxedStrategy<Attempt> {
    (pvec(arb_row(), 0..5), pvec(arb_row(), 0..6), any::<u64>())
        .prop_map(|(queries, mut pool, seed)| {
            // Half the time, make membership plausible: seed the pool
            // with copies of some queries (otherwise random rows almost
            // never collide and the accept arm is untested).
            if seed & 1 == 0 {
                for (i, q) in queries.iter().enumerate() {
                    if (seed >> (i % 60)) & 2 == 0 {
                        pool.push(q.clone());
                    }
                }
            }
            Attempt { queries, pool }
        })
        .boxed()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Reference ≡ hash-set partner on arbitrary batches, at two
    /// paddings (padding invariance rides along).
    #[test]
    fn reference_matches_hash_partner(atts in pvec(arb_attempt(), 0..5)) {
        check(&atts);
    }

    /// Accept-shaped attempts (pool ⊇ queries) accept; single-field
    /// mutants (sign, limb, den id, column, dropped nnz, dropped pool
    /// row) flip the mutated attempt exactly when the partner says so.
    #[test]
    fn accepts_and_mutants(
        mut queries in pvec(arb_row(), 1..4),
        noise in pvec(arb_row(), 0..3),
        which in any::<prop::sample::Index>(),
        mode in 0u8..6,
    ) {
        // Force at least one nonempty query so mutants have a target.
        if queries.iter().all(|q| q.is_empty()) {
            queries[0].push((3, (1, vec![7], 0)));
        }
        let mut pool = queries.clone();
        pool.extend(noise);
        let a = Attempt { queries: queries.clone(), pool: pool.clone() };
        let accept = build(std::slice::from_ref(&a), 0);
        let out = reference::verify(&accept).expect("valid");
        assert!(out.member_ok == vec![true], "pool ⊇ queries must accept");

        // Mutate one query value/structure; the pool keeps the
        // original, so equality of the mutant is (usually) broken —
        // the hash partner adjudicates, the reference must agree.
        let nonempty: Vec<usize> =
            (0..queries.len()).filter(|i| !queries[*i].is_empty()).collect();
        let qi = nonempty[which.index(nonempty.len())];
        let pos = which.index(queries[qi].len());
        let mut m = a.clone();
        match mode {
            0 => m.queries[qi][pos].1.0 = -m.queries[qi][pos].1.0,
            1 => {
                let limbs = &mut m.queries[qi][pos].1.1;
                if limbs.is_empty() { limbs.push(1); } else { limbs[0] ^= 1; }
            }
            2 => m.queries[qi][pos].1.2 ^= 1,
            3 => m.queries[qi][pos].0 = m.queries[qi][pos].0.wrapping_add(100),
            4 => { m.queries[qi].remove(pos); }
            _ => {
                // Drop the pool copy of the query instead.
                let key = row_key(&m.queries[qi]);
                if let Some(pi) = m.pool.iter().position(|r| row_key(r) == key) {
                    m.pool.remove(pi);
                } else {
                    return Ok(());
                }
            }
        }
        check(&[m]);
    }
}

// ---- fixed edges and validation ---------------------------------------

#[test]
fn edges() {
    // No attempts.
    check(&[]);
    // No queries: vacuously true, even with an empty pool.
    check(&[Attempt {
        queries: vec![],
        pool: vec![],
    }]);
    // Queries but empty pool: false.
    check(&[Attempt {
        queries: vec![vec![(0, (1, vec![3], 0))]],
        pool: vec![],
    }]);
    // Empty row equals empty row.
    check(&[Attempt {
        queries: vec![vec![]],
        pool: vec![vec![]],
    }]);
    // Empty query row, no empty pool row: false.
    check(&[Attempt {
        queries: vec![vec![]],
        pool: vec![vec![(0, (1, vec![1], 0))]],
    }]);
    // Duplicate pool rows are harmless.
    let r: Row = vec![(1, (-1, vec![9, 9], 2)), (4, (1, vec![], 0))];
    check(&[Attempt {
        queries: vec![r.clone()],
        pool: vec![r.clone(), r.clone(), r],
    }]);
    // Same limbs, different den id: not equal (the den contract is
    // part of the predicate, not decoration).
    check(&[Attempt {
        queries: vec![vec![(0, (1, vec![5], 0))]],
        pool: vec![vec![(0, (1, vec![5], 1))]],
    }]);
    // Adversarial deep-compare: rows equal except the LAST limb of the
    // LAST position (exercises full-depth comparison, no early out).
    let long: Row = (0..6u32).map(|c| (c, (1i8, vec![0xAB; 3], 1u32))).collect();
    let mut long2 = long.clone();
    long2[5].1 .1[2] ^= 1;
    check(&[Attempt {
        queries: vec![long.clone()],
        pool: vec![long2, long],
    }]);
}

#[test]
fn validation_refusals() {
    // Bad split.
    let mut b = build(
        &[Attempt {
            queries: vec![vec![]],
            pool: vec![vec![]],
        }],
        0,
    );
    b.split[0] = 7;
    assert_eq!(reference::verify(&b), Err(RowEqError::Split(0)));

    // Sign inconsistent with zero magnitude.
    let mut b = build(
        &[Attempt {
            queries: vec![vec![(0, (1, vec![1], 0))]],
            pool: vec![vec![(0, (1, vec![1], 0))]],
        }],
        0,
    );
    b.mag.fill(0);
    assert!(matches!(reference::verify(&b), Err(RowEqError::Sign(_))));

    // nnz_slot out of range.
    let mut b = build(
        &[Attempt {
            queries: vec![vec![(0, (1, vec![1], 0))]],
            pool: vec![vec![(0, (1, vec![1], 0))]],
        }],
        0,
    );
    b.nnz_slot[0] = 99;
    assert_eq!(reference::verify(&b), Err(RowEqError::Index("nnz_slot")));

    // Non-monotone row_ptr.
    let mut b = build(
        &[Attempt {
            queries: vec![vec![(0, (1, vec![1], 0))], vec![]],
            pool: vec![vec![(0, (1, vec![1], 0))]],
        }],
        0,
    );
    b.row_ptr[1] = 2;
    assert!(matches!(
        reference::verify(&b),
        Err(RowEqError::Structure(_))
    ));
}
