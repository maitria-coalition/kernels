//! CUDA-lane conformance battery: the device outcome must equal the
//! core reference lane's outcome — same `fold_ok`, same `refused`,
//! same `channels_used` — on generated batches spanning magnitude
//! tiers, accept-shaped constructions, mutants, and edge shapes.
//!
//! GPU-gated: when no CUDA device is reachable the battery SKIPS
//! (stderr note), it does not fail — CPU CI boxes stay green; the
//! lane's receipts come from GPU boxes running exactly this file.

use maitria_kernels::rnsfold::batch::{RnsFoldBatch, ABSENT};
use maitria_kernels::rnsfold::reference;
use maitria_kernels_cuda::RnsFoldGpu;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::Zero;
use proptest::collection::vec as pvec;
use proptest::prelude::*;

fn gpu() -> Option<RnsFoldGpu> {
    match RnsFoldGpu::new() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("SKIP (no usable CUDA device): {e}");
            None
        }
    }
}

// ---- a small independent builder (test-side; mirrors the intended
// consumer packing without sharing its code) ----

fn limbs_of(v: &BigUint) -> Vec<u64> {
    v.to_u64_digits()
}

fn big_from_limbs(limbs: &[u64]) -> BigUint {
    let mut digits = Vec::with_capacity(limbs.len() * 2);
    for &l in limbs {
        digits.push((l & 0xFFFF_FFFF) as u32);
        digits.push((l >> 32) as u32);
    }
    BigUint::new(digits)
}

#[derive(Debug, Clone)]
struct At {
    n_cols: u32,
    lams: Vec<BigInt>,
    rows: Vec<Vec<(u32, BigInt)>>,
    concl: Vec<(u32, BigInt)>,
}

fn build(attempts: &[At], mults: &[BigUint]) -> RnsFoldBatch {
    let mut b = RnsFoldBatch {
        mults: mults.iter().map(limbs_of).collect(),
        ..Default::default()
    };
    let mut slots: Vec<(i8, BigUint, u32)> = Vec::new();
    let push = |v: &BigInt, mult: u32, slots: &mut Vec<(i8, BigUint, u32)>| -> u32 {
        let s = match v.sign() {
            Sign::NoSign => 0i8,
            Sign::Plus => 1,
            Sign::Minus => -1,
        };
        slots.push((s, v.magnitude().clone(), mult));
        (slots.len() - 1) as u32
    };
    b.acol_ptr.push(0);
    b.csc_ptr.push(0);
    for at in attempts {
        let lam_base = b.lams.len() as u32;
        for l in &at.lams {
            let s = match l.sign() {
                Sign::NoSign => 0i8,
                Sign::Plus => 1,
                Sign::Minus => -1,
            };
            b.lams.push((s, limbs_of(l.magnitude())));
        }
        let mut cols: Vec<Vec<(u32, u32)>> = vec![Vec::new(); at.n_cols as usize];
        for (e, row) in at.rows.iter().enumerate() {
            for (col, v) in row {
                let mult = (e % b.mults.len()) as u32;
                let slot = push(v, mult, &mut slots);
                cols[*col as usize].push((lam_base + e as u32, slot));
            }
        }
        let mut concl = vec![ABSENT; at.n_cols as usize];
        for (col, v) in &at.concl {
            concl[*col as usize] = push(v, 0, &mut slots);
        }
        for col in 0..at.n_cols as usize {
            for &(l, s) in &cols[col] {
                b.csc_lam.push(l);
                b.csc_slot.push(s);
            }
            b.csc_ptr.push(b.csc_lam.len() as u32);
            b.concl_slot.push(concl[col]);
        }
        b.acol_ptr.push(b.concl_slot.len() as u32);
    }
    b.n_slots = slots.len();
    b.k = slots
        .iter()
        .map(|(_, m, _)| limbs_of(m).len())
        .max()
        .unwrap_or(1)
        .max(1);
    b.sign = slots.iter().map(|(s, _, _)| *s).collect();
    b.mult_id = slots.iter().map(|(_, _, m)| *m).collect();
    b.mag = vec![0; b.k * b.n_slots];
    for (i, (_, m, _)) in slots.iter().enumerate() {
        for (l, limb) in limbs_of(m).into_iter().enumerate() {
            b.mag[l * b.n_slots + i] = limb;
        }
    }
    b
}

fn mults() -> Vec<BigUint> {
    vec![
        BigUint::from(1u32),
        BigUint::from(7u32),
        BigUint::from(360360u32),
        BigUint::from(u64::MAX) * BigUint::from(u64::MAX),
    ]
}

fn assert_lanes_agree(b: &RnsFoldBatch, g: &RnsFoldGpu) {
    let want = reference::verify(b).expect("reference verify");
    let got = g.verify(b).expect("gpu verify");
    assert_eq!(want, got, "CUDA lane diverged from reference");
}

fn arb_bigint(bits: u32) -> BoxedStrategy<BigInt> {
    (
        pvec(any::<u64>(), 0..=(bits as usize).div_ceil(64)),
        any::<bool>(),
    )
        .prop_map(move |(limbs, neg)| {
            let mut v = BigInt::from_biguint(Sign::Plus, big_from_limbs(&limbs));
            v %= BigInt::from(1) << bits;
            if neg {
                -v
            } else {
                v
            }
        })
        .boxed()
}

fn arb_at(bits: u32) -> BoxedStrategy<At> {
    (1u32..8, 1usize..6)
        .prop_flat_map(move |(n_cols, n_entries)| {
            let row = pvec((0..n_cols, arb_bigint(bits)), 0..=n_cols as usize);
            (
                pvec(arb_bigint(bits), n_entries),
                pvec(row, n_entries),
                pvec((0..n_cols, arb_bigint(bits)), 0..=n_cols as usize),
                Just(n_cols),
            )
        })
        .prop_map(|(lams, rows, concl, n_cols)| {
            let dedupe = |v: Vec<(u32, BigInt)>| {
                let mut m = std::collections::BTreeMap::new();
                for (c, x) in v {
                    m.insert(c, x);
                }
                m.into_iter().collect::<Vec<_>>()
            };
            At {
                n_cols,
                lams,
                rows: rows.into_iter().map(dedupe).collect(),
                concl: dedupe(concl),
            }
        })
        .boxed()
}

#[test]
fn conformance_proptest_tiers() {
    let Some(g) = gpu() else { return };
    let mut runner = proptest::test_runner::TestRunner::new(ProptestConfig {
        cases: 64,
        ..ProptestConfig::default()
    });
    for &bits in &[40u32, 120, 600] {
        runner
            .run(&pvec(arb_at(bits), 1..5), |ats| {
                let b = build(&ats, &mults());
                prop_assert!(b.validate().is_ok());
                assert_lanes_agree(&b, &g);
                Ok(())
            })
            .unwrap();
    }
}

#[test]
fn conformance_accept_shaped_and_mutants() {
    let Some(g) = gpu() else { return };
    let ms = mults();
    // deterministic accept-shaped attempt at the ~600-bit tier
    let lam = |x: i64, shift: u32| BigInt::from(x) << shift;
    for shift in [0u32, 60, 500] {
        let lams = vec![lam(3, shift), lam(-2, shift), lam(7, 0)];
        let rows: Vec<Vec<(u32, BigInt)>> = vec![
            vec![(0, lam(11, shift)), (2, lam(-5, 0))],
            vec![(0, lam(1, 0)), (1, lam(9, shift))],
            vec![(2, lam(4, shift)), (3, lam(1, 1))],
        ];
        // conclusion = exact fold with the builder's mult convention
        let mut fold = vec![BigInt::zero(); 4];
        for (e, row) in rows.iter().enumerate() {
            let m = BigInt::from_biguint(Sign::Plus, ms[e % ms.len()].clone());
            for (c, v) in row {
                fold[*c as usize] += &lams[e] * v * &m;
            }
        }
        let concl: Vec<(u32, BigInt)> = fold
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_zero())
            .map(|(c, v)| (c as u32, v.clone()))
            .collect();
        let at = At {
            n_cols: 4,
            lams,
            rows,
            concl,
        };
        let mut b = build(std::slice::from_ref(&at), &ms);
        let out = reference::verify(&b).unwrap();
        assert!(
            out.fold_ok[0],
            "constructed accept must accept (shift {shift})"
        );
        assert_lanes_agree(&b, &g);

        // mutants: flip one low limb of every slot in turn
        for s in 0..b.n_slots {
            b.mag[s] ^= 1;
            let bits = b.slot_bits(s);
            let old_sign = b.sign[s];
            if bits == 0 {
                b.sign[s] = 0;
            } else if b.sign[s] == 0 {
                b.sign[s] = 1;
            }
            assert_lanes_agree(&b, &g);
            b.mag[s] ^= 1;
            b.sign[s] = old_sign;
        }
    }
}

#[test]
fn conformance_edges() {
    let Some(g) = gpu() else { return };
    let ms = mults();
    // empty batch
    let b = build(&[], &ms);
    assert_lanes_agree(&b, &g);
    // attempt with zero columns
    let b = build(
        &[At {
            n_cols: 1,
            lams: vec![],
            rows: vec![],
            concl: vec![],
        }],
        &ms,
    );
    assert_lanes_agree(&b, &g);
    // absent conclusions with nonzero fold (must reject), zero fold (accept)
    let b = build(
        &[
            At {
                n_cols: 2,
                lams: vec![BigInt::from(1)],
                rows: vec![vec![(0, BigInt::from(5))]],
                concl: vec![],
            },
            At {
                n_cols: 2,
                lams: vec![BigInt::from(0)],
                rows: vec![vec![(0, BigInt::from(5))]],
                concl: vec![],
            },
        ],
        &ms,
    );
    let out = reference::verify(&b).unwrap();
    assert_eq!(out.fold_ok, vec![false, true]);
    assert_lanes_agree(&b, &g);
}
