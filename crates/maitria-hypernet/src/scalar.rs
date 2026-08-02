//! Exact scalar leaves of the witness contract: arbitrary-precision
//! integers ([`WitInt`]) and rationals ([`WitRat`]) with the CBOR
//! encoding the landed producer corpus already uses.
//!
//! Encoding (CONTRACTS.md §2.1, exactness class E):
//! - integer: CBOR major-0/1 int when `-2^64 <= x < 2^64`, else RFC 8949
//!   tag 2 (positive bignum) / tag 3 (negative bignum, bytes encode
//!   `-1 - x`) with big-endian magnitude bytes. This is exactly what
//!   python `cbor2` emits for `int`, so every landed fixture conforms
//!   by construction (witness: `tests/golden.rs` decodes
//!   `cegis2d_selftest_WR.cbor`, whose `tau_exact` numerator is a
//!   55-digit bignum).
//! - bignum payload padding (encode-side rule, CONTRACTS.md §2.1):
//!   this crate pads bignum payloads with leading zero bytes to a
//!   minimum of 17 bytes. Rationale: ciborium 0.2.2's deserializer
//!   eagerly converts tag-2/3 payloads of <= 16 bytes to native
//!   integers and hard-errors on tag-3 magnitudes >= 2^127 before any
//!   visitor runs; >= 17-byte payloads take the faithful tag path.
//!   Leading zeroes are explicitly legal (RFC 8949 §3.4.3: decoders
//!   "MUST be able to decode bignums that do have leading zeroes").
//!   Known residue: a MINIMALLY-encoded tag-3 from a third-party
//!   canonical encoder (python cbor2) with value in
//!   `[-(2^128), -(2^127)-1]` cannot pass ciborium 0.2.2 at all; the
//!   decode side here recovers every other case (including positive
//!   16-byte bignums, via `visit_u128`).
//! - rational: 2-array `[num, den]`, `den >= 1`, `gcd(num, den) = 1`
//!   (the `[numerator, denominator]` incumbent of `leaf_doc` /
//!   `cegis2d._efrac`). Producers MUST emit canonical form; consumers
//!   MAY verify (`WitRat::is_canonical`).

use num_bigint::{BigInt, Sign};
use num_integer::Integer as _;
use num_traits::{Signed, Zero};
use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Arbitrary-precision integer (exactness class E).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WitInt(pub BigInt);

impl From<i64> for WitInt {
    fn from(x: i64) -> Self {
        WitInt(BigInt::from(x))
    }
}

impl From<BigInt> for WitInt {
    fn from(x: BigInt) -> Self {
        WitInt(x)
    }
}

impl WitInt {
    /// The CBOR value this integer encodes to (int within the native
    /// range, tag 2/3 bignum outside it; payload padded to >= 17 bytes
    /// — see the module doc's padding rule).
    pub fn to_cbor_value(&self) -> ciborium::value::Value {
        use ciborium::value::{Integer, Value};
        // Native CBOR int range: -2^64 <= x < 2^64. i128 covers it.
        if let Ok(small) = i128::try_from(&self.0) {
            if let Ok(int) = Integer::try_from(small) {
                return Value::Integer(int);
            }
        }
        fn padded(bytes: Vec<u8>) -> Vec<u8> {
            const MIN: usize = 17;
            if bytes.len() >= MIN {
                bytes
            } else {
                let mut out = vec![0u8; MIN - bytes.len()];
                out.extend_from_slice(&bytes);
                out
            }
        }
        if self.0.sign() != Sign::Minus {
            let (_, bytes) = self.0.to_bytes_be();
            Value::Tag(2, Box::new(Value::Bytes(padded(bytes))))
        } else {
            // tag 3 encodes -1 - n; n = -x - 1 >= 0.
            let n = -&self.0 - BigInt::from(1u8);
            let (_, bytes) = n.to_bytes_be();
            Value::Tag(3, Box::new(Value::Bytes(padded(bytes))))
        }
    }

    /// Parse from a decoded CBOR value (int or tag 2/3 bignum).
    pub fn from_cbor_value(v: &ciborium::value::Value) -> Result<Self, String> {
        use ciborium::value::Value;
        match v {
            Value::Integer(i) => Ok(WitInt(BigInt::from(i128::from(*i)))),
            Value::Tag(2, inner) => match inner.as_ref() {
                Value::Bytes(b) => Ok(WitInt(BigInt::from_bytes_be(Sign::Plus, b))),
                _ => Err("tag 2 payload is not a byte string".into()),
            },
            Value::Tag(3, inner) => match inner.as_ref() {
                Value::Bytes(b) => {
                    let n = BigInt::from_bytes_be(Sign::Plus, b);
                    Ok(WitInt(-n - BigInt::from(1u8)))
                }
                _ => Err("tag 3 payload is not a byte string".into()),
            },
            other => Err(format!("expected int or bignum tag, got {other:?}")),
        }
    }
}

impl Serialize for WitInt {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_cbor_value().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WitInt {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct IntVisitor;

        impl<'de> serde::de::Visitor<'de> for IntVisitor {
            type Value = WitInt;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "an integer or a CBOR bignum (tag 2/3)")
            }

            fn visit_u64<E: DeError>(self, x: u64) -> Result<WitInt, E> {
                Ok(WitInt(BigInt::from(x)))
            }

            fn visit_i64<E: DeError>(self, x: i64) -> Result<WitInt, E> {
                Ok(WitInt(BigInt::from(x)))
            }

            // ciborium converts tag-2/3 payloads <= 16 bytes to native
            // integers before visiting: positives arrive here as u128
            // (full range — this is what recovers (i128::MAX, 2^128)),
            // negatives as i128.
            fn visit_u128<E: DeError>(self, x: u128) -> Result<WitInt, E> {
                Ok(WitInt(BigInt::from(x)))
            }

            fn visit_i128<E: DeError>(self, x: i128) -> Result<WitInt, E> {
                Ok(WitInt(BigInt::from(x)))
            }

            // Payloads >= 17 bytes arrive via ciborium's tag protocol:
            // enum variant "@@TAGGED@@", then (tag: u64, bytes).
            fn visit_enum<A: serde::de::EnumAccess<'de>>(self, acc: A) -> Result<WitInt, A::Error> {
                use serde::de::VariantAccess;

                struct TagBody;

                impl<'de> serde::de::Visitor<'de> for TagBody {
                    type Value = WitInt;

                    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(f, "a (tag, bytes) bignum pair")
                    }

                    fn visit_seq<A: serde::de::SeqAccess<'de>>(
                        self,
                        mut acc: A,
                    ) -> Result<WitInt, A::Error> {
                        let tag: u64 = acc
                            .next_element()?
                            .ok_or_else(|| DeError::custom("bignum: missing tag"))?;
                        let bytes: serde_bytes::ByteBuf = acc
                            .next_element()?
                            .ok_or_else(|| DeError::custom("bignum: missing bytes"))?;
                        let n = BigInt::from_bytes_be(Sign::Plus, &bytes);
                        match tag {
                            2 => Ok(WitInt(n)),
                            3 => Ok(WitInt(-n - BigInt::from(1u8))),
                            t => Err(DeError::custom(format!(
                                "expected bignum tag 2/3, got tag {t}"
                            ))),
                        }
                    }
                }

                let (name, data): (String, _) = acc.variant()?;
                if name != "@@TAGGED@@" {
                    return Err(DeError::custom(format!(
                        "expected a tagged value, got variant {name:?}"
                    )));
                }
                data.tuple_variant(2, TagBody)
            }
        }

        deserializer.deserialize_any(IntVisitor)
    }
}

/// Exact rational as `[num, den]` (exactness class E). Canonical form:
/// `den >= 1`, `gcd(|num|, den) = 1`, and `0` is `[0, 1]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WitRat(pub WitInt, pub WitInt);

impl WitRat {
    /// Construct in canonical form (normalizes sign and gcd).
    /// Panics on a zero denominator — a witness with `den = 0` is
    /// malformed at the producer, not a checkable state.
    pub fn new(num: BigInt, den: BigInt) -> Self {
        assert!(!den.is_zero(), "WitRat: zero denominator");
        let (mut num, mut den) = if den.is_negative() {
            (-num, -den)
        } else {
            (num, den)
        };
        let g = num.gcd(&den);
        if !g.is_zero() && g != BigInt::from(1u8) {
            num /= &g;
            den /= &g;
        }
        if num.is_zero() {
            den = BigInt::from(1u8);
        }
        WitRat(WitInt(num), WitInt(den))
    }

    pub fn from_i64(num: i64, den: i64) -> Self {
        Self::new(BigInt::from(num), BigInt::from(den))
    }

    pub fn num(&self) -> &BigInt {
        &self.0 .0
    }

    pub fn den(&self) -> &BigInt {
        &self.1 .0
    }

    /// Canonicity check (consumers MAY enforce; producers MUST emit
    /// canonical — CONTRACTS.md §2.1).
    pub fn is_canonical(&self) -> bool {
        let num = self.num();
        let den = self.den();
        if !den.is_positive() {
            return false;
        }
        if num.is_zero() {
            return den == &BigInt::from(1u8);
        }
        num.gcd(den) == BigInt::from(1u8)
    }

    /// log2 of the denominator, rounded up — the quantity the
    /// `den_bound_log2` fast-path hint bounds (CONTRACTS.md §2.4).
    pub fn den_log2(&self) -> u64 {
        self.den().bits().saturating_sub(1) + u64::from(!is_pow2(self.den()))
    }
}

/// Canonical rational text: `"n"` when `den = 1`, else `"n/d"` — the
/// same rendering the node boundary's structured refusal payloads use
/// (`mtk-node`'s `sr`), so message strings and payloads agree.
impl core::fmt::Display for WitRat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.den() == &BigInt::from(1u8) {
            write!(f, "{}", self.num())
        } else {
            write!(f, "{}/{}", self.num(), self.den())
        }
    }
}

fn is_pow2(x: &BigInt) -> bool {
    let (sign, bytes) = x.to_bytes_be();
    sign == Sign::Plus && {
        let ones: u32 = bytes.iter().map(|b| b.count_ones()).sum();
        ones == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalization() {
        let r = WitRat::new(BigInt::from(4), BigInt::from(-6));
        assert_eq!(r, WitRat::from_i64(-2, 3));
        assert!(r.is_canonical());
        let z = WitRat::new(BigInt::from(0), BigInt::from(-7));
        assert_eq!(z, WitRat::from_i64(0, 1));
    }

    #[test]
    fn display_is_canonical_text() {
        assert_eq!(WitRat::from_i64(3, 2).to_string(), "3/2");
        assert_eq!(WitRat::from_i64(-2, 3).to_string(), "-2/3");
        assert_eq!(WitRat::from_i64(4, 2).to_string(), "2");
        assert_eq!(WitRat::from_i64(0, -7).to_string(), "0");
    }

    #[test]
    fn den_log2() {
        assert_eq!(WitRat::from_i64(1, 1).den_log2(), 0);
        assert_eq!(WitRat::from_i64(1, 2).den_log2(), 1);
        assert_eq!(WitRat::from_i64(1, 3).den_log2(), 2);
        assert_eq!(WitRat::from_i64(1, 1024).den_log2(), 10);
        assert_eq!(WitRat::from_i64(1, 100000).den_log2(), 17);
    }
}
