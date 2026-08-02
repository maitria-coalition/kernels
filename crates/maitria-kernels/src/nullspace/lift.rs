//! Two-channel CRT combination and rational reconstruction: the
//! fixed-width fast path from per-prime canonical residues to exact
//! primitive integer nullspace vectors.
//!
//! Two 63-bit channels give a modulus $M = p_1 p_2 < 2^{126}$, hence
//! a reconstruction range of $|n|, d \le \lfloor\sqrt{M/2}\rfloor
//! \approx 2^{62}$ — far beyond every invariant vector observed in
//! the motivating corpus (entries are small integers, denominators
//! almost always 1), yet all in `u128`/`i128` arithmetic. Anything
//! wider is a typed refusal ([`LiftError::Capacity`]) for the
//! caller's arbitrary-precision ladder (ENGINEERING #4/#10: the
//! fixed-width path refuses what it cannot represent exactly; nothing
//! rounds).
//!
//! Reconstruction is the classic extended-Euclidean rational
//! recovery (Wang's bound; Monagan's maximal-quotient refinement is
//! the named upgrade if channel economy ever matters here): descend
//! the remainder sequence of $(M, x)$ until the remainder falls below
//! $B = \lfloor\sqrt{M/2}\rfloor$; the stopping convergent is the
//! unique fraction $n/d$ with $|n|, d \le B$ congruent to $x$ — when
//! one exists, which the caller confirms anyway by exact
//! re-verification of every lifted vector (the pinch; see the family
//! documentation).

use super::{ModpNullspace, SparseVec};

/// Typed refusals of the fixed-width lift.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LiftError {
    /// The two bundles must come from distinct primes.
    SamePrime {
        /// The repeated prime.
        prime: u64,
    },
    /// Rank or leftmost profile disagree between the primes: at least
    /// one prime is unlucky for this matrix. Drop the bundle with the
    /// *smaller* rank (it is certainly unlucky —
    /// $\mathrm{rank}_p \le \mathrm{rank}_\mathbb{Q}$) or, at equal
    /// ranks, either one, and retry with a fresh prime.
    ProfileMismatch {
        /// Rank under the first prime.
        rank_a: usize,
        /// Rank under the second prime.
        rank_b: usize,
    },
    /// A coordinate did not fit the two-channel reconstruction range,
    /// or clearing denominators overflowed 128-bit arithmetic. The
    /// caller's arbitrary-precision ladder owns this input.
    Capacity {
        /// Basis vector index (ascending free-column order).
        vector: usize,
        /// Coordinate (column) index, where attributable.
        coord: u32,
    },
}

impl core::fmt::Display for LiftError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LiftError::SamePrime { prime } => write!(f, "both bundles use prime {prime}"),
            LiftError::ProfileMismatch { rank_a, rank_b } => {
                write!(
                    f,
                    "rank profiles disagree ({rank_a} vs {rank_b}): unlucky prime"
                )
            }
            LiftError::Capacity { vector, coord } => write!(
                f,
                "vector {vector}, coordinate {coord}: outside two-channel capacity"
            ),
        }
    }
}

impl std::error::Error for LiftError {}

/// Lift two agreeing per-prime bundles to exact primitive integer
/// vectors: per coordinate CRT + rational reconstruction, then per
/// vector denominator clearing, content division, and sign
/// normalization (first nonzero coordinate positive) — the canonical
/// integer form of the canonical rational basis.
pub fn crt2_ratrec(
    a: &ModpNullspace,
    b: &ModpNullspace,
) -> Result<Vec<Vec<(u32, i128)>>, LiftError> {
    if a.prime == b.prime {
        return Err(LiftError::SamePrime { prime: a.prime });
    }
    if a.rank != b.rank || a.pivot_cols != b.pivot_cols {
        return Err(LiftError::ProfileMismatch {
            rank_a: a.rank,
            rank_b: b.rank,
        });
    }
    let (pa, pb) = (a.prime, b.prime);
    let m = pa as u128 * pb as u128;
    let bound = isqrt_u128(m / 2);
    // Garner: x = ra + pa * ((rb - ra) * pa^{-1} mod pb).
    let pa_inv_mod_pb = invmod64(pa % pb, pb);

    let mut out = Vec::with_capacity(a.basis.len());
    for (k, (va, vb)) in a.basis.iter().zip(&b.basis).enumerate() {
        let mut rat: Vec<(u32, i128, u128)> = Vec::with_capacity(va.idx.len().max(vb.idx.len()));
        for (c, ra, rb) in union_residues(va, vb) {
            let diff = submod64(rb % pb, ra % pb, pb);
            let x = ra as u128 + pa as u128 * mulmod64(diff, pa_inv_mod_pb, pb) as u128;
            let (num, den) = ratrec(x, m, bound).ok_or(LiftError::Capacity {
                vector: k,
                coord: c,
            })?;
            rat.push((c, num, den));
        }
        out.push(clear_to_primitive(k, rat)?);
    }
    Ok(out)
}

/// Merged iteration over two sparse residue vectors: coordinates in
/// either support, missing side read as residue 0.
fn union_residues<'v>(
    a: &'v SparseVec,
    b: &'v SparseVec,
) -> impl Iterator<Item = (u32, u64, u64)> + 'v {
    let mut i = 0usize;
    let mut j = 0usize;
    std::iter::from_fn(move || {
        let ai = a.idx.get(i).copied();
        let bj = b.idx.get(j).copied();
        match (ai, bj) {
            (None, None) => None,
            (Some(c), None) => {
                i += 1;
                Some((c, a.val[i - 1], 0))
            }
            (None, Some(c)) => {
                j += 1;
                Some((c, 0, b.val[j - 1]))
            }
            (Some(ca), Some(cb)) => {
                if ca < cb {
                    i += 1;
                    Some((ca, a.val[i - 1], 0))
                } else if cb < ca {
                    j += 1;
                    Some((cb, 0, b.val[j - 1]))
                } else {
                    i += 1;
                    j += 1;
                    Some((ca, a.val[i - 1], b.val[j - 1]))
                }
            }
        }
    })
}

/// Clear a vector of reconstructed rationals to the primitive integer
/// form: multiply by the denominator lcm, divide by the content,
/// make the first nonzero coordinate positive.
fn clear_to_primitive(
    k: usize,
    rat: Vec<(u32, i128, u128)>,
) -> Result<Vec<(u32, i128)>, LiftError> {
    let cap = |coord: u32| LiftError::Capacity { vector: k, coord };
    let mut l: u128 = 1;
    for &(c, _, d) in &rat {
        let g = gcd_u128(l, d);
        l = (l / g).checked_mul(d).ok_or(cap(c))?;
    }
    let mut out: Vec<(u32, i128)> = Vec::with_capacity(rat.len());
    for &(c, n, d) in &rat {
        let scale = l / d;
        let scale: i128 = scale.try_into().map_err(|_| cap(c))?;
        let v = n.checked_mul(scale).ok_or(cap(c))?;
        debug_assert_ne!(v, 0, "union support carries no zero rationals");
        out.push((c, v));
    }
    let mut g: u128 = 0;
    for &(_, v) in &out {
        g = gcd_u128(g, v.unsigned_abs());
    }
    if g > 1 {
        for (_, v) in &mut out {
            *v /= g as i128;
        }
    }
    if let Some(&(_, first)) = out.first() {
        if first < 0 {
            for (_, v) in &mut out {
                *v = -*v;
            }
        }
    }
    Ok(out)
}

/// Extended-Euclidean rational reconstruction: the unique `n/d` with
/// `|n|, d <= bound`, `gcd(n, d) = 1`, and `n ≡ x·d (mod m)` — or
/// `None` when no such fraction exists in range.
fn ratrec(x: u128, m: u128, bound: u128) -> Option<(i128, u128)> {
    let (mut r0, mut r1) = (m, x);
    let (mut t0, mut t1): (i128, i128) = (0, 1);
    while r1 > bound {
        if r1 == 0 {
            return None;
        }
        let q = r0 / r1;
        let q_i: i128 = q.try_into().ok()?;
        (r0, r1) = (r1, r0 - q * r1);
        (t0, t1) = (t1, t0.checked_sub(q_i.checked_mul(t1)?)?);
    }
    // r1 <= bound now (possibly 0 => x was a multiple of every
    // convergent — the fraction is 0/1 only if x == 0).
    if t1 == 0 {
        return None;
    }
    let d = t1.unsigned_abs();
    if d > bound {
        return None;
    }
    let n = if t1 < 0 {
        -(TryInto::<i128>::try_into(r1).ok()?)
    } else {
        r1.try_into().ok()?
    };
    let g = gcd_u128(n.unsigned_abs(), d);
    if g != 1 {
        return None;
    }
    Some((n, d))
}

fn isqrt_u128(v: u128) -> u128 {
    if v == 0 {
        return 0;
    }
    let mut x = 1u128 << (v.ilog2() / 2 + 1);
    loop {
        let y = (x + v / x) / 2;
        if y >= x {
            return x;
        }
        x = y;
    }
}

fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[inline]
fn submod64(a: u64, b: u64, p: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a + p - b
    }
}

#[inline]
fn mulmod64(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

fn invmod64(a: u64, p: u64) -> u64 {
    let mut e = p - 2;
    let mut base = a % p;
    let mut acc = 1u64;
    while e > 0 {
        if e & 1 == 1 {
            acc = mulmod64(acc, base, p);
        }
        base = mulmod64(base, base, p);
        e >>= 1;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isqrt_matches_squares() {
        for v in [0u128, 1, 2, 3, 4, 8, 9, 15, 16, 1 << 63, (1 << 63) + 1] {
            let s = isqrt_u128(v);
            assert!(s * s <= v);
            assert!((s + 1).checked_mul(s + 1).is_none_or(|sq| sq > v));
        }
    }

    #[test]
    fn ratrec_recovers_small_fractions() {
        let m: u128 = 1000003u128 * 999983;
        let bound = isqrt_u128(m / 2);
        // x ≡ -3/7 (mod m): x = (m - 3) * inv(7) mod m — build by hand.
        // 7^{-1} mod m via extended Euclid on small numbers: use
        // Fermat on each prime and CRT would be circular; instead
        // check the forward direction: pick n/d, embed, recover.
        for (n, d) in [(1i128, 1u128), (-3, 7), (22, 5), (-1000, 999)] {
            let dinv = modinv_u128(d % m, m).unwrap();
            let nm = if n >= 0 {
                n as u128 % m
            } else {
                (m - (n.unsigned_abs() % m)) % m
            };
            let x = mulmod_u128(nm, dinv, m);
            let (rn, rd) = ratrec(x, m, bound).unwrap();
            assert_eq!((rn, rd), (n, d));
        }
    }

    fn mulmod_u128(a: u128, b: u128, m: u128) -> u128 {
        // Schoolbook via double-and-add: test-only helper, m < 2^126.
        let mut acc = 0u128;
        let mut a = a % m;
        let mut b = b;
        while b > 0 {
            if b & 1 == 1 {
                acc = (acc + a) % m;
            }
            a = (a + a) % m;
            b >>= 1;
        }
        acc
    }

    fn modinv_u128(a: u128, m: u128) -> Option<u128> {
        let (mut r0, mut r1) = (m as i128, a as i128);
        let (mut t0, mut t1) = (0i128, 1i128);
        while r1 != 0 {
            let q = r0 / r1;
            (r0, r1) = (r1, r0 - q * r1);
            (t0, t1) = (t1, t0 - q * t1);
        }
        if r0 != 1 {
            return None;
        }
        Some(t0.rem_euclid(m as i128) as u128)
    }
}
