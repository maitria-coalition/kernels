//! Receipt generator (ENGINEERING #3): reference vs the compiled
//! lane, medians over repeated runs, on the shapes that matter —
//! all-pass scans (the accept verdict walks the whole slab), planted
//! mid-slab violations (early exit), and minmax folds. Deliberately
//! dependency-free; run with `--release` and commit the table under
//! `receipts/` with machine + toolchain noted.

use std::hint::black_box;
use std::time::Instant;

use maitria_kernels::sweep::{self, reference, SignPred};

fn median_ns(mut f: impl FnMut(), runs: usize) -> u128 {
    let mut samples: Vec<u128> = (0..runs)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos()
        })
        .collect();
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn fmt_ns(ns: u128) -> String {
    if ns >= 1_000_000 {
        format!("{:.2} ms", ns as f64 / 1e6)
    } else if ns >= 1_000 {
        format!("{:.2} µs", ns as f64 / 1e3)
    } else {
        format!("{ns} ns")
    }
}

fn main() {
    let runs = 201;
    println!(
        "# sweep lane receipt — lane under test: {:?}",
        sweep::active_lane()
    );
    println!();
    println!("| shape | n | reference | lane | ratio |");
    println!("|---|---:|---:|---:|---:|");
    for &n in &[1_024usize, 65_536, 1_048_576] {
        // all-pass: every entry positive, full scan.
        let pass: Vec<i64> = (0..n).map(|i| 1 + (i as i64 % 997)).collect();
        // planted violation at the midpoint.
        let mut mid = pass.clone();
        mid[n / 2] = -1;

        for (shape, slab) in [("all-pass", &pass), ("mid-violation", &mid)] {
            let r = median_ns(
                || {
                    black_box(reference::first_violation(
                        black_box(slab),
                        SignPred::NonNeg,
                    ));
                },
                runs,
            );
            let l = median_ns(
                || {
                    black_box(sweep::first_violation(black_box(slab), SignPred::NonNeg));
                },
                runs,
            );
            println!(
                "| first_violation {shape} | {n} | {} | {} | {:.2}x |",
                fmt_ns(r),
                fmt_ns(l),
                r as f64 / l as f64
            );
        }

        let r = median_ns(
            || {
                black_box(reference::minmax(black_box(&pass)));
            },
            runs,
        );
        let l = median_ns(
            || {
                black_box(sweep::minmax(black_box(&pass)));
            },
            runs,
        );
        println!(
            "| minmax | {n} | {} | {} | {:.2}x |",
            fmt_ns(r),
            fmt_ns(l),
            r as f64 / l as f64
        );
    }
}
