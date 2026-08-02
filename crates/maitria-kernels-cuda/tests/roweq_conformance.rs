//! roweq CUDA-lane conformance battery: the device outcome must equal
//! the core reference lane's outcome — same `member_ok`, attempt for
//! attempt — on generated batches, accept-shaped constructions,
//! mutants, and edge shapes.
//!
//! GPU-gated: when no CUDA device is reachable the battery SKIPS
//! (stderr note), it does not fail — CPU CI boxes stay green; the
//! lane's receipts come from GPU boxes running exactly this file.

use maitria_kernels::roweq::batch::RowEqBatch;
use maitria_kernels::roweq::reference;
use maitria_kernels_cuda::RowEqGpu;
use proptest::collection::vec as pvec;
use proptest::prelude::*;

fn gpu() -> Option<RowEqGpu> {
    match RowEqGpu::new() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("SKIP (no usable CUDA device): {e}");
            None
        }
    }
}

// ---- test-side builder (mirrors the battery's logical model) ----

type Val = (i8, Vec<u64>, u32);
type Row = Vec<(u32, Val)>;

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
        for r in a.queries.iter().chain(&a.pool) {
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
        b.sign.push(if t.is_empty() { 0 } else { *sg });
        b.den_id.push(*d);
        for (l, limb) in t.iter().enumerate() {
            b.mag[l * slots.len() + s] = *limb;
        }
    }
    b
}

fn check_conformance(g: &RowEqGpu, attempts: &[Attempt]) {
    for pad in [0usize, 2] {
        let b = build(attempts, pad);
        let want = reference::verify(&b).expect("valid descriptor");
        let got = g.verify(&b).expect("device verify");
        assert_eq!(got, want, "device diverged from reference (pad={pad})");
    }
}

// ---- generators (the core battery's family) ----

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
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Device ≡ reference on arbitrary batches (two paddings).
    #[test]
    fn device_matches_reference(atts in pvec(arb_attempt(), 0..5)) {
        // One context per process would be ideal; per-case creation is
        // tolerable at 64 cases and keeps the test self-contained.
        use std::sync::OnceLock;
        static GPU: OnceLock<Option<RowEqGpu>> = OnceLock::new();
        let Some(g) = GPU.get_or_init(gpu).as_ref() else { return Ok(()); };
        check_conformance(g, &atts);
    }
}

#[test]
fn edges_and_mutants() {
    let Some(g) = gpu() else {
        return;
    };
    // Empty batch.
    check_conformance(&g, &[]);
    // Vacuous attempts, empty rows, missing pools.
    check_conformance(
        &g,
        &[
            Attempt {
                queries: vec![],
                pool: vec![],
            },
            Attempt {
                queries: vec![vec![]],
                pool: vec![vec![]],
            },
            Attempt {
                queries: vec![vec![(0, (1, vec![3], 0))]],
                pool: vec![],
            },
            Attempt {
                queries: vec![vec![]],
                pool: vec![vec![(0, (1, vec![1], 0))]],
            },
        ],
    );
    // Accept + slot-by-slot mutants (sign, limb, den, col, length).
    let base: Row = vec![
        (0, (1, vec![0xDEAD, 0xBEEF], 1)),
        (3, (-1, vec![7], 0)),
        (9, (1, vec![], 2)),
    ];
    let accept = Attempt {
        queries: vec![base.clone()],
        pool: vec![vec![(1, (1, vec![2], 0))], base.clone()],
    };
    check_conformance(&g, &[accept]);
    let mutate = |f: &dyn Fn(&mut Row)| {
        let mut q = base.clone();
        f(&mut q);
        Attempt {
            queries: vec![q],
            pool: vec![base.clone()],
        }
    };
    check_conformance(
        &g,
        &[
            mutate(&|r| r[0].1 .0 = -1),
            mutate(&|r| r[0].1 .1[1] ^= 1),
            mutate(&|r| r[1].1 .2 = 9),
            mutate(&|r| r[1].0 = 4),
            mutate(&|r| {
                r.pop();
            }),
        ],
    );
    // Adversarial deep compare: equal except the last limb of the
    // last position — plus a shared-slot fast path check (query and
    // pool rows referencing the SAME slot indices are equal by
    // construction; the builder never shares slots, so this rides the
    // long-row case instead).
    let long: Row = (0..8u32).map(|c| (c, (1i8, vec![0x55; 3], 1u32))).collect();
    let mut long2 = long.clone();
    long2[7].1 .1[2] ^= 1;
    check_conformance(
        &g,
        &[Attempt {
            queries: vec![long.clone()],
            pool: vec![long2, long],
        }],
    );
}
