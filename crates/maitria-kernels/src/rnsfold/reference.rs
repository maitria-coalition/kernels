//! The scalar reference lane — the semantics of the family, written
//! for obviousness: plain `u128 %` residue arithmetic, one channel at
//! a time, no Montgomery form, no batching tricks. Every acceleration
//! lane is conformance-gated against this function's outcome,
//! bit-for-bit.

use super::batch::{BatchError, RnsFoldBatch, RnsFoldOutcome, ABSENT};
use super::primes::{addmod, mulmod, residue_signed, PRIMES};

/// Evaluate the batch on the reference lane.
///
/// For each admitted attempt, for each of its acols, for each channel
/// `j < channels`: fold the products `λ̂ · (value · mult)` modulo
/// `PRIMES[j]` and compare against the conclusion operand's residue.
/// `fold_ok[a]` is the conjunction over the attempt's acols and
/// channels. Refused attempts (bound past the prime table) receive no
/// verdict.
pub fn verify(b: &RnsFoldBatch) -> Result<RnsFoldOutcome, BatchError> {
    b.validate()?;
    let (channels, refused) = b.plan_channels();
    let n = b.n_attempts();
    let mut fold_ok = vec![false; n];

    // Slot residue: raw signed numerator times its positive
    // multiplier, mod p.
    let slot_limbs =
        |s: usize| -> Vec<u64> { (0..b.k).map(|l| b.mag[l * b.n_slots + s]).collect() };
    let slot_res = |s: usize, p: u64| -> u64 {
        let raw = residue_signed(b.sign[s], &slot_limbs(s), p);
        let m = super::primes::residue_of_limbs(&b.mults[b.mult_id[s] as usize], p);
        mulmod(raw, m, p)
    };

    for a in 0..n {
        if refused[a] {
            continue;
        }
        let mut ok = true;
        'attempt: for acol in b.acol_ptr[a] as usize..b.acol_ptr[a + 1] as usize {
            for &p in &PRIMES[..channels] {
                let mut acc: u64 = 0;
                for i in b.csc_ptr[acol] as usize..b.csc_ptr[acol + 1] as usize {
                    let (ls, lm) = &b.lams[b.csc_lam[i] as usize];
                    let lam = residue_signed(*ls, lm, p);
                    let v = slot_res(b.csc_slot[i] as usize, p);
                    acc = addmod(acc, mulmod(lam, v, p), p);
                }
                let c = b.concl_slot[acol];
                let cres = if c == ABSENT {
                    0
                } else {
                    slot_res(c as usize, p)
                };
                if acc != cres {
                    ok = false;
                    break 'attempt;
                }
            }
        }
        fold_ok[a] = ok;
    }
    Ok(RnsFoldOutcome {
        fold_ok,
        refused,
        channels_used: channels,
    })
}
