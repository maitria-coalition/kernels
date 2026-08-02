//! The channel table: 64 distinct 63-bit primes, descending from
//! $2^{63}$, plus the plain residue helpers every lane's host side
//! shares. Primality and size are re-verified by the battery
//! (deterministic Miller–Rabin over the whole table), not taken from
//! this comment.

/// The 64 largest primes below $2^{63}$, descending. Channel $j$ uses
/// `PRIMES[j]`. All odd (Montgomery-compatible with $R = 2^{64}$ for
/// lanes that want it); product exceeds $2^{4032}$.
pub const PRIMES: [u64; 64] = [
    9223372036854775783,
    9223372036854775643,
    9223372036854775549,
    9223372036854775507,
    9223372036854775433,
    9223372036854775421,
    9223372036854775417,
    9223372036854775399,
    9223372036854775351,
    9223372036854775337,
    9223372036854775291,
    9223372036854775279,
    9223372036854775259,
    9223372036854775181,
    9223372036854775159,
    9223372036854775139,
    9223372036854775097,
    9223372036854775073,
    9223372036854775057,
    9223372036854774959,
    9223372036854774937,
    9223372036854774917,
    9223372036854774893,
    9223372036854774797,
    9223372036854774739,
    9223372036854774713,
    9223372036854774679,
    9223372036854774629,
    9223372036854774587,
    9223372036854774571,
    9223372036854774559,
    9223372036854774511,
    9223372036854774509,
    9223372036854774499,
    9223372036854774451,
    9223372036854774413,
    9223372036854774341,
    9223372036854774319,
    9223372036854774307,
    9223372036854774277,
    9223372036854774257,
    9223372036854774247,
    9223372036854774233,
    9223372036854774199,
    9223372036854774179,
    9223372036854774173,
    9223372036854774053,
    9223372036854773999,
    9223372036854773977,
    9223372036854773953,
    9223372036854773899,
    9223372036854773867,
    9223372036854773783,
    9223372036854773639,
    9223372036854773561,
    9223372036854773557,
    9223372036854773519,
    9223372036854773507,
    9223372036854773489,
    9223372036854773477,
    9223372036854773443,
    9223372036854773429,
    9223372036854773407,
    9223372036854773353,
];

/// Number of channels whose prime product provably exceeds
/// $2^{\text{bits}}$, or `None` if the whole table cannot. Uses the
/// conservative fact that every table prime exceeds $2^{62}$, so $C$
/// channels give strictly more than $62 C$ bits of product.
pub fn channels_for_bits(bits: u64) -> Option<usize> {
    let c = (bits / 62 + 1) as usize;
    if c <= PRIMES.len() {
        Some(c)
    } else {
        None
    }
}

/// `(a * b) % p` in plain `u128` arithmetic. The reference lane's one
/// multiplication primitive — deliberately the obvious spelling.
#[inline]
pub fn mulmod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

/// `(a + b) % p` for `a, b < p`.
#[inline]
pub fn addmod(a: u64, b: u64, p: u64) -> u64 {
    let (s, ovf) = a.overflowing_add(b);
    if ovf || s >= p {
        s.wrapping_sub(p)
    } else {
        s
    }
}

/// Residue of a little-endian u64-limb magnitude, by Horner over
/// $2^{64} \bmod p$.
pub fn residue_of_limbs(limbs: &[u64], p: u64) -> u64 {
    let base = ((1u128 << 64) % p as u128) as u64;
    let mut r: u64 = 0;
    for &l in limbs.iter().rev() {
        r = mulmod(r, base, p);
        r = addmod(r, (l as u128 % p as u128) as u64, p);
    }
    r
}

/// Residue of a signed value given as (sign, magnitude limbs):
/// `p - r` for negatives (with the zero-residue case left at zero).
pub fn residue_signed(sign: i8, limbs: &[u64], p: u64) -> u64 {
    let r = residue_of_limbs(limbs, p);
    if sign < 0 && r != 0 {
        p - r
    } else {
        r
    }
}

/// Bit length of a little-endian limb magnitude (0 for zero).
pub fn bits_of_limbs(limbs: &[u64]) -> u64 {
    for (i, &l) in limbs.iter().enumerate().rev() {
        if l != 0 {
            return i as u64 * 64 + (64 - l.leading_zeros() as u64);
        }
    }
    0
}
