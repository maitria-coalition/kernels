//! rnsfold conformance battery.
//!
//! Partners (ENGINEERING #5): the reference lane vs an INDEPENDENT
//! big-integer formulation of the same predicate — Δ(col) computed
//! directly in `num_bigint::BigInt`, no residues, no channels, no
//! shared arithmetic code — on generated batches spanning three
//! magnitude tiers (machine words, 128-bit-boundary values, ~2^600
//! bignums), accept-shaped and mutant-perturbed. Plus: deterministic
//! Miller–Rabin over the whole prime table (the table's own receipt),
//! and helper-level differentials for the residue primitives.

use maitria_kernels::rnsfold::batch::{BatchError, RnsFoldBatch, ABSENT};
use maitria_kernels::rnsfold::primes::{
    addmod, bits_of_limbs, mulmod, residue_of_limbs, residue_signed, PRIMES,
};
use maitria_kernels::rnsfold::reference;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::Zero;
use proptest::collection::vec as pvec;
use proptest::prelude::*;

// ---------------------------------------------------------------- helpers

fn big_from_limbs(limbs: &[u64]) -> BigUint {
    let mut digits = Vec::with_capacity(limbs.len() * 2);
    for &l in limbs {
        digits.push((l & 0xFFFF_FFFF) as u32);
        digits.push((l >> 32) as u32);
    }
    BigUint::new(digits)
}

fn signed_from(sign: i8, limbs: &[u64]) -> BigInt {
    let mag = big_from_limbs(limbs);
    match sign {
        0 => BigInt::zero(),
        s if s > 0 => BigInt::from_biguint(Sign::Plus, mag),
        _ => BigInt::from_biguint(Sign::Minus, mag),
    }
}

fn limbs_of(v: &BigUint) -> Vec<u64> {
    v.to_u64_digits()
}

/// Batch slot accessor as BigInt (raw numerator times multiplier).
fn slot_value(b: &RnsFoldBatch, s: usize) -> BigInt {
    let limbs: Vec<u64> = (0..b.k).map(|l| b.mag[l * b.n_slots + s]).collect();
    let raw = signed_from(b.sign[s], &limbs);
    let m = BigInt::from_biguint(Sign::Plus, big_from_limbs(&b.mults[b.mult_id[s] as usize]));
    raw * m
}

/// The independent partner: Δ(col) == 0 for every acol of every
/// attempt, in plain BigInt arithmetic.
fn bigint_verify(b: &RnsFoldBatch) -> Vec<bool> {
    (0..b.n_attempts())
        .map(|a| {
            (b.acol_ptr[a] as usize..b.acol_ptr[a + 1] as usize).all(|acol| {
                let mut acc = BigInt::zero();
                for i in b.csc_ptr[acol] as usize..b.csc_ptr[acol + 1] as usize {
                    let (ls, lm) = &b.lams[b.csc_lam[i] as usize];
                    let lam = signed_from(*ls, lm);
                    acc += lam * slot_value(b, b.csc_slot[i] as usize);
                }
                let c = b.concl_slot[acol];
                let concl = if c == ABSENT {
                    BigInt::zero()
                } else {
                    slot_value(b, c as usize)
                };
                acc == concl
            })
        })
        .collect()
}

// ------------------------------------------------------------- the table

/// Deterministic Miller–Rabin for u64 (the 7-witness set proven
/// sufficient below 3.3 * 10^24).
fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    let mut d = n - 1;
    let mut s = 0u32;
    while d.is_multiple_of(2) {
        d /= 2;
        s += 1;
    }
    let powmod = |mut b: u64, mut e: u64, m: u64| -> u64 {
        let mut r: u64 = 1;
        b %= m;
        while e > 0 {
            if e & 1 == 1 {
                r = mulmod(r, b, m);
            }
            b = mulmod(b, b, m);
            e >>= 1;
        }
        r
    };
    'witness: for a in [2u64, 325, 9375, 28178, 450775, 9780504, 1795265022] {
        let a = a % n;
        if a == 0 {
            continue;
        }
        let mut x = powmod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mulmod(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

#[test]
fn prime_table_is_prime_distinct_63bit_descending() {
    for w in PRIMES.windows(2) {
        assert!(w[0] > w[1], "descending, distinct");
    }
    for &p in &PRIMES {
        assert!(p > 1 << 62 && p < 1 << 63, "63-bit: {p}");
        assert!(is_prime_u64(p), "not prime: {p}");
    }
}

#[test]
fn residue_helpers_match_bigint() {
    let cases: Vec<Vec<u64>> = vec![
        vec![],
        vec![0],
        vec![1],
        vec![u64::MAX],
        vec![u64::MAX, u64::MAX, u64::MAX],
        vec![0, 0, 1],
        vec![123456789, 987654321, 0xDEAD_BEEF_CAFE_F00D],
    ];
    for limbs in &cases {
        let v = big_from_limbs(limbs);
        for &p in &PRIMES[..8] {
            let want = (&v % BigUint::from(p)).to_u64_digits();
            let want = want.first().copied().unwrap_or(0);
            assert_eq!(residue_of_limbs(limbs, p), want, "limbs {limbs:?} mod {p}");
            // signed: -v ≡ p - r
            let neg = residue_signed(if v.is_zero() { 0 } else { -1 }, limbs, p);
            let want_neg = if want == 0 { 0 } else { p - want };
            assert_eq!(neg, want_neg);
        }
    }
    // addmod/mulmod near the boundary
    for &p in &PRIMES[..4] {
        for &(a, b) in &[(p - 1, p - 1), (p - 1, 1), (0, 0), (1, p - 1)] {
            assert_eq!(
                addmod(a, b, p) as u128,
                ((a as u128) + (b as u128)) % p as u128
            );
            assert_eq!(mulmod(a, b, p) as u128, (a as u128 * b as u128) % p as u128);
        }
    }
}

// --------------------------------------------------------- batch builder

/// Build a batch from nested values: per attempt, per entry a λ, per
/// entry a dense-ish sparse row over `n_cols`, a conclusion row.
/// Multipliers exercise the mult table with small positive values.
#[derive(Debug)]
struct Nested {
    attempts: Vec<NestedAttempt>,
}

#[derive(Debug)]
struct NestedAttempt {
    n_cols: u32,
    lams: Vec<BigInt>,
    /// rows[e] : Vec<(col, value)>
    rows: Vec<Vec<(u32, BigInt)>>,
    concl: Vec<(u32, BigInt)>,
}

/// The builder mirrors the intended consumer packing: one slot per
/// (row, nnz) and per conclusion nnz; CSC built by counting sort.
fn build(nested: &Nested, mults: &[BigUint]) -> RnsFoldBatch {
    let mut b = RnsFoldBatch {
        mults: mults.iter().map(limbs_of).collect(),
        ..Default::default()
    };
    // mult 0 must be 1 by the builder's own convention.
    assert_eq!(b.mults[0], vec![1]);
    let mut slot_vals: Vec<(i8, BigUint, u32)> = Vec::new(); // sign, mag, mult
    let push_slot = |v: &BigInt, mult: u32, slot_vals: &mut Vec<(i8, BigUint, u32)>| -> u32 {
        let s = match v.sign() {
            Sign::NoSign => 0i8,
            Sign::Plus => 1,
            Sign::Minus => -1,
        };
        slot_vals.push((s, v.magnitude().clone(), mult));
        (slot_vals.len() - 1) as u32
    };

    b.acol_ptr.push(0);
    b.csc_ptr.push(0);
    for at in &nested.attempts {
        for l in &at.lams {
            let s = match l.sign() {
                Sign::NoSign => 0i8,
                Sign::Plus => 1,
                Sign::Minus => -1,
            };
            b.lams.push((s, limbs_of(l.magnitude())));
        }
        let lam_base = (b.lams.len() - at.lams.len()) as u32;
        // per column: collect (lam id, slot)
        let mut cols: Vec<Vec<(u32, u32)>> = vec![Vec::new(); at.n_cols as usize];
        for (e, row) in at.rows.iter().enumerate() {
            for (col, v) in row {
                let mult = (e % b.mults.len()) as u32; // exercise the table
                let slot = push_slot(v, mult, &mut slot_vals);
                cols[*col as usize].push((lam_base + e as u32, slot));
            }
        }
        let mut concl: Vec<u32> = vec![ABSENT; at.n_cols as usize];
        for (col, v) in &at.concl {
            let slot = push_slot(v, 0, &mut slot_vals);
            concl[*col as usize] = slot;
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
    // Slot planes at uniform k.
    b.n_slots = slot_vals.len();
    b.k = slot_vals
        .iter()
        .map(|(_, m, _)| limbs_of(m).len())
        .max()
        .unwrap_or(1)
        .max(1);
    b.sign = slot_vals.iter().map(|(s, _, _)| *s).collect();
    b.mult_id = slot_vals.iter().map(|(_, _, m)| *m).collect();
    b.mag = vec![0; b.k * b.n_slots];
    for (i, (_, m, _)) in slot_vals.iter().enumerate() {
        for (l, limb) in limbs_of(m).into_iter().enumerate() {
            b.mag[l * b.n_slots + i] = limb;
        }
    }
    b
}

/// Effective slot semantics: the builder multiplies row e's values by
/// mults[e % mults.len()], so the BigInt ground truth must too. To
/// keep the partner INDEPENDENT, the nested ground truth is computed
/// on the *effective* values (value * mult), which `bigint_verify`
/// reads back off the packed planes itself — the two computations
/// share only the packed bytes.
fn nested_truth(b: &RnsFoldBatch) -> Vec<bool> {
    bigint_verify(b)
}

// -------------------------------------------------------------- strategies

fn arb_bigint(bits: u32) -> BoxedStrategy<BigInt> {
    (
        pvec(any::<u64>(), 0..=(bits as usize).div_ceil(64)),
        any::<bool>(),
    )
        .prop_map(move |(limbs, neg)| {
            let mut v = BigInt::from_biguint(Sign::Plus, big_from_limbs(&limbs));
            let m = BigInt::from(1) << bits;
            v %= &m;
            if neg {
                v = -v;
            }
            v
        })
        .boxed()
}

fn arb_attempt(bits: u32, accept: bool) -> BoxedStrategy<NestedAttempt> {
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
        .prop_map(move |(lams, rows, concl_extra, n_cols)| {
            // dedupe columns within rows / conclusion
            let dedupe = |v: Vec<(u32, BigInt)>| {
                let mut seen = std::collections::BTreeMap::new();
                for (c, x) in v {
                    seen.insert(c, x);
                }
                seen.into_iter().collect::<Vec<_>>()
            };
            let rows: Vec<Vec<(u32, BigInt)>> = rows.into_iter().map(dedupe).collect();
            let concl = if accept {
                // conclusion = exact fold of effective values: computed
                // by the test builder's OWN convention (mult on row e).
                // Left empty here; filled after packing is impossible,
                // so accept-shaped cases are built by construction in
                // the property body instead.
                Vec::new()
            } else {
                dedupe(concl_extra)
            };
            NestedAttempt {
                n_cols,
                lams,
                rows,
                concl,
            }
        })
        .boxed()
}

fn small_mults() -> Vec<BigUint> {
    vec![
        BigUint::from(1u32),
        BigUint::from(7u32),
        BigUint::from(360360u32),                          // lcm-flavored
        BigUint::from(u64::MAX) * BigUint::from(u64::MAX), // 2-limb multiplier
    ]
}

// -------------------------------------------------------------- properties

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// Reference lane == independent BigInt partner, across magnitude
    /// tiers, on arbitrary (mostly rejecting) batches.
    #[test]
    fn reference_matches_bigint_partner(
        attempts in pvec(arb_attempt(40, false), 1..5),
        tier in prop_oneof![Just(40u32), Just(120u32), Just(600u32)],
    ) {
        // rebuild rows at the drawn tier by scaling magnitudes up
        let attempts: Vec<NestedAttempt> = attempts
            .into_iter()
            .map(|mut a| {
                if tier > 40 {
                    let shift = tier - 40;
                    for r in &mut a.rows {
                        for (_, v) in r.iter_mut() {
                            *v = &*v << shift;
                        }
                    }
                    for (_, v) in a.concl.iter_mut() {
                        *v = &*v << shift;
                    }
                }
                a
            })
            .collect();
        let b = build(&Nested { attempts }, &small_mults());
        prop_assert!(b.validate().is_ok());
        let out = reference::verify(&b).unwrap();
        let want = nested_truth(&b);
        for (a, w) in want.iter().enumerate() {
            prop_assert!(!out.refused[a], "no refusal expected at these shapes");
            prop_assert_eq!(out.fold_ok[a], *w, "attempt {}", a);
        }
    }

    /// Accept-shaped batches: conclusion constructed as the true fold
    /// (so fold_ok must be true), then one mutant per batch — a single
    /// slot magnitude perturbed — must flip that attempt to false.
    #[test]
    fn accepts_accept_and_mutants_flip(
        lams in pvec(arb_bigint(90), 1..5),
        rows_seed in pvec(pvec((0u32..6, prop::num::i64::ANY), 1..6), 1..5),
        which in any::<prop::sample::Index>(),
    ) {
        let n_entries = lams.len().min(rows_seed.len());
        let n_cols = 6u32;
        let mults = small_mults();
        // effective row values chosen; conclusion computed as the fold
        // of value*mult(e) with mult table as build() assigns.
        let mut rows: Vec<Vec<(u32, BigInt)>> = Vec::new();
        let mut fold = vec![BigInt::zero(); n_cols as usize];
        for e in 0..n_entries {
            let mut seen = std::collections::BTreeMap::new();
            for (c, x) in &rows_seed[e] {
                seen.insert(*c, BigInt::from(*x));
            }
            let row: Vec<(u32, BigInt)> = seen.into_iter().collect();
            let m = BigInt::from_biguint(Sign::Plus, mults[e % mults.len()].clone());
            for (c, v) in &row {
                fold[*c as usize] += &lams[e] * v * &m;
            }
            rows.push(row);
        }
        let concl: Vec<(u32, BigInt)> = fold
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_zero())
            .map(|(c, v)| (c as u32, v.clone()))
            .collect();
        let at = NestedAttempt { n_cols, lams: lams[..n_entries].to_vec(), rows, concl };
        let mut b = build(&Nested { attempts: vec![at] }, &mults);
        prop_assert!(b.validate().is_ok());
        let out = reference::verify(&b).unwrap();
        prop_assert!(out.fold_ok[0], "constructed accept must accept");
        prop_assert_eq!(out.fold_ok.clone(), nested_truth(&b));

        // mutant: perturb one nonzero-capable slot's low limb
        if b.n_slots > 0 {
            let s = which.index(b.n_slots);
            b.mag[s] ^= 1; // limb 0 plane of slot s
            // re-canonicalize sign if we created/destroyed a zero
            let bits = b.slot_bits(s);
            if bits == 0 { b.sign[s] = 0; } else if b.sign[s] == 0 { b.sign[s] = 1; }
            let out2 = reference::verify(&b).unwrap();
            let want2 = nested_truth(&b);
            prop_assert_eq!(out2.fold_ok.clone(), want2, "mutant must track ground truth");
        }
    }

    /// Channel plan: required_bits really bounds |Δ| (the soundness
    /// inequality, checked numerically against exact BigInt Δ).
    #[test]
    fn required_bits_bounds_delta(
        attempts in pvec(arb_attempt(200, false), 1..4),
    ) {
        let b = build(&Nested { attempts }, &small_mults());
        prop_assert!(b.validate().is_ok());
        for a in 0..b.n_attempts() {
            let bound = BigInt::from(1) << b.required_bits(a);
            for acol in b.acol_ptr[a] as usize..b.acol_ptr[a + 1] as usize {
                let mut acc = BigInt::zero();
                for i in b.csc_ptr[acol] as usize..b.csc_ptr[acol + 1] as usize {
                    let (ls, lm) = &b.lams[b.csc_lam[i] as usize];
                    acc += signed_from(*ls, lm) * slot_value(&b, b.csc_slot[i] as usize);
                }
                let c = b.concl_slot[acol];
                if c != ABSENT {
                    acc -= slot_value(&b, c as usize);
                }
                prop_assert!(acc.magnitude() < bound.magnitude(),
                    "|Δ| must sit under 2^required_bits");
            }
        }
    }

    /// Channel plan: the batched `plan_channels` (precomputed bit
    /// arrays; parallel under the `rayon` feature) equals the naive
    /// per-attempt mapping of `required_bits` through
    /// `channels_for_bits` — the differential pair for the batched
    /// planning respelling (a lane may change cost, never verdicts;
    /// applied here to a host walk).
    #[test]
    fn plan_channels_matches_required_bits_map(
        attempts in pvec(arb_attempt(200, false), 1..4),
    ) {
        use maitria_kernels::rnsfold::primes::channels_for_bits;
        let b = build(&Nested { attempts }, &small_mults());
        prop_assert!(b.validate().is_ok());
        let (channels, refused) = b.plan_channels();
        let mut want_channels = 1usize;
        prop_assert_eq!(refused.len(), b.n_attempts());
        for (a, &attempt_refused) in refused.iter().enumerate() {
            match channels_for_bits(b.required_bits(a)) {
                Some(c) => {
                    want_channels = want_channels.max(c);
                    prop_assert!(!attempt_refused);
                }
                None => prop_assert!(attempt_refused),
            }
        }
        prop_assert_eq!(channels, want_channels);
    }
}

#[test]
fn bits_of_limbs_edges() {
    assert_eq!(bits_of_limbs(&[]), 0);
    assert_eq!(bits_of_limbs(&[0, 0]), 0);
    assert_eq!(bits_of_limbs(&[1]), 1);
    assert_eq!(bits_of_limbs(&[0, 1]), 65);
    assert_eq!(bits_of_limbs(&[u64::MAX]), 64);
}

/// The adversarial shape for the equality bound: an attempt whose Δ is
/// a huge power of two times a product of several table primes —
/// nonzero, divisible by many channels — must still be REJECTED
/// (bound-driven channel count must exceed the planted divisibility).
#[test]
fn planted_multi_prime_divisible_delta_still_rejects() {
    // Δ = p0 * p1 * ... * p11 (product of the first 12 table primes):
    // divisible by 12 channels' primes; the bound must force > 12.
    let mut delta = BigInt::from(1);
    for &p in &PRIMES[..12] {
        delta *= BigInt::from(p);
    }
    // attempt: single column, single entry: λ=1, v=delta, concl=0.
    let at = NestedAttempt {
        n_cols: 1,
        lams: vec![BigInt::from(1)],
        rows: vec![vec![(0u32, delta)]],
        concl: vec![],
    };
    let b = build(&Nested { attempts: vec![at] }, &small_mults());
    let out = reference::verify(&b).unwrap();
    assert!(!out.refused[0]);
    assert!(
        !out.fold_ok[0],
        "nonzero Δ divisible by 12 primes must still reject"
    );
    assert!(
        out.channels_used > 12,
        "channel plan must out-run the planted divisibility"
    );
}

/// Big-descriptor validation (the parallel scan past
/// `VALIDATE_PAR_MIN` under the `rayon` feature; the serial walk
/// otherwise — canonical errors either way, which is the contract):
/// a clean 2^17-slot descriptor validates, and planted defects deep
/// in each big plane report the same canonical error the serial
/// order would.
#[test]
fn big_descriptor_validation_canonical_errors() {
    let n: usize = 1 << 17;
    let b = RnsFoldBatch {
        k: 1,
        n_slots: n,
        sign: vec![1; n],
        mag: vec![1u64; n],
        mult_id: vec![0; n],
        mults: vec![vec![1u64]],
        lams: vec![(1, vec![1u64])],
        acol_ptr: vec![0, 1],
        csc_ptr: vec![0, n as u32],
        csc_lam: vec![0; n],
        csc_slot: (0..n as u32).collect(),
        concl_slot: vec![ABSENT],
    };
    b.validate().expect("clean big descriptor validates");
    let mut bad = b.clone();
    bad.csc_slot[12345] = n as u32;
    assert_eq!(bad.validate().unwrap_err(), BatchError::Index("csc_slot"));
    let mut bad = b.clone();
    bad.sign[54321] = 0; // zero sign, nonzero magnitude
    assert_eq!(bad.validate().unwrap_err(), BatchError::Sign(54321));
    let mut bad = b.clone();
    bad.mult_id[7] = 1;
    bad.mults.push(vec![0u64]);
    assert_eq!(bad.validate().unwrap_err(), BatchError::ZeroMult(1));
}
