//! Catalogue: canonical serialization + SHA-256 content digest (H-GRAPH's
//! byte layer).
//!
//! Byte-compatible with the reference serializer it was minted
//! against (`acasxu/hypernet.py::canonical_bytes`, format magic
//! `AXHN0001`) — golden vectors under
//! `tests/golden/` are the cross-language proof. All integers little-
//! endian. Wire ids u32, shape extents u64. Rational elements: u32
//! num_len, u32 den_len, numerator as minimal two's-complement LE,
//! denominator as minimal unsigned LE, canonical form (den ≥ 1, gcd 1).
//! Vertex-index sets are LSB-first bitsets over the vertex count.
//! Excluded from bytes and digest: wire debug names, complex hints.
//!
//! Equal digests certify equal computation graphs, not merely equal
//! functions; the decoder ([`Hypernet::from_canonical_bytes`]) refuses
//! non-canonical encodings, so decode∘encode is the identity on bytes
//! by construction.

use std::fmt;

use num_bigint::{BigInt, Sign as BigSign};
use num_integer::Integer as _;
use num_traits::{One, Signed, Zero};
use sha2::{Digest as _, Sha256};

use crate::scalar::{WitInt, WitRat};

use crate::app::{App, ConstPayload, Wire};
use crate::graph::Hypernet;
use crate::simplex::{SimpC, VSet};
use crate::types::{Dtype, Ext, Kind, Sign, TensorType};

/// Wire-format magic (version 1 of the X0 canonicalization).
pub const MAGIC: &[u8; 8] = b"AXHN0001";

/// The rounding-mode byte every X0 application carries (exact).
const ROUND_EXACT: u8 = 4;

/// A hypernet's SHA-256 content digest. Content-addresses the
/// computation graph: certificates cite computations by this value, and
/// downstream catalogue families (System manifests' mechanism content)
/// reference hypernets through it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HypernetDigest(pub [u8; 32]);

impl HypernetDigest {
    /// Lowercase hex, the interchange spelling (matches the python
    /// reference's `digest()` and the committed digest goldens).
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.0 {
            use fmt::Write as _;
            write!(s, "{b:02x}").expect("infallible");
        }
        s
    }

    /// Parse the 64-char lowercase/uppercase hex spelling.
    pub fn from_hex(s: &str) -> Option<HypernetDigest> {
        if s.len() != 64 {
            return None;
        }
        let mut out = [0u8; 32];
        for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
            let hi = (chunk[0] as char).to_digit(16)?;
            let lo = (chunk[1] as char).to_digit(16)?;
            out[i] = (hi * 16 + lo) as u8;
        }
        Some(HypernetDigest(out))
    }
}

impl fmt::Display for HypernetDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// CBOR wire form: a 32-byte byte string — identical bytes to the
/// pre-unification `#[serde(with = "serde_bytes")] Vec<u8>` spelling
/// (System manifests' `HypernetRef.digest`), so the union of the Rust
/// types changed no wire. Decode enforces the 32-byte length (a
/// preflight courtesy; the checker re-derives digests regardless).
impl serde::Serialize for HypernetDigest {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(&self.0)
    }
}

impl<'de> serde::Deserialize<'de> for HypernetDigest {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let b = serde_bytes::ByteBuf::deserialize(d)?;
        let arr: [u8; 32] = b
            .as_slice()
            .try_into()
            .map_err(|_| serde::de::Error::invalid_length(b.len(), &"a 32-byte hypernet digest"))?;
        Ok(HypernetDigest(arr))
    }
}

/// Refusals of the canonical codec — encode-side (a graph whose ids the
/// format cannot express) and decode-side (bytes that are truncated,
/// ill-tagged, or non-canonical), each carrying its geometry.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CanonRefusal {
    #[error("encode: wire {wire} out of range (table has {table_len}) at application {app}")]
    EncodeWireOutOfRange {
        app: u32,
        wire: Wire,
        table_len: u32,
    },
    #[error("encode: rational is not canonical (den >= 1, gcd 1 required)")]
    EncodeNonCanonicalRational,
    #[error("decode: need {needed} bytes for {what} at offset {offset}, only {available} remain")]
    Truncated {
        offset: usize,
        needed: usize,
        available: usize,
        what: &'static str,
    },
    #[error("decode: bad magic {got:02x?} (expected AXHN0001)")]
    BadMagic { got: Vec<u8> },
    #[error("decode: unknown {what} byte {got} at offset {offset}")]
    UnknownByte {
        offset: usize,
        what: &'static str,
        got: u8,
    },
    #[error("decode: unknown application tag {tag} at offset {offset}")]
    UnknownTag { offset: usize, tag: u8 },
    #[error("decode: application header at offset {offset} is not canonical (round byte {round}, pad {pad})")]
    BadAppHeader { offset: usize, round: u8, pad: u16 },
    #[error("decode: non-canonical rational encoding at offset {offset}: {why}")]
    NonCanonicalRational { offset: usize, why: &'static str },
    #[error("decode: bitset at offset {offset} has bits set beyond member count {n}")]
    BitsetOverflow { offset: usize, n: u32 },
    #[error("decode: simplex list at offset {offset} is not in canonical (size, members) order")]
    NonCanonicalSimplexOrder { offset: usize },
    #[error("decode: wire {wire} out of range (table has {table_len}) at offset {offset}")]
    DecodeWireOutOfRange {
        offset: usize,
        wire: Wire,
        table_len: u32,
    },
    #[error("decode: application at offset {offset} carries dtype/shape bytes disagreeing with the wire-type table (field {what})")]
    RedundantFieldMismatch { offset: usize, what: &'static str },
    #[error("decode: {got} trailing bytes after a complete hypernet")]
    TrailingBytes { got: usize },
    #[error("decode: length {got} at offset {offset} exceeds the format bound for {what}")]
    LengthOverflow {
        offset: usize,
        got: u64,
        what: &'static str,
    },
}

// ========================================================================
// primitive writers (all little-endian, mirroring the reference)
// ========================================================================

fn w_u16(out: &mut Vec<u8>, x: u16) {
    out.extend_from_slice(&x.to_le_bytes());
}
fn w_u32(out: &mut Vec<u8>, x: u32) {
    out.extend_from_slice(&x.to_le_bytes());
}
fn w_u64(out: &mut Vec<u8>, x: u64) {
    out.extend_from_slice(&x.to_le_bytes());
}

/// Minimal two's-complement little-endian bytes of a signed integer
/// (python `int.to_bytes(..., signed=True)` + the reference's shrink
/// loop; 0 encodes as one zero byte).
fn twos_complement_le_minimal(x: &BigInt) -> Vec<u8> {
    if x.is_zero() {
        return vec![0];
    }
    let mut b = x.to_signed_bytes_le();
    while b.len() > 1 {
        let last = b[b.len() - 1];
        let prev = b[b.len() - 2];
        if (last == 0 && prev < 0x80) || (last == 0xFF && prev >= 0x80) {
            b.pop();
        } else {
            break;
        }
    }
    b
}

/// Minimal unsigned little-endian magnitude bytes (denominators; ≥ 1).
fn unsigned_le_minimal(x: &BigInt) -> Vec<u8> {
    debug_assert!(x.is_positive());
    x.to_bytes_le().1
}

/// One rational element: u32 num_len | u32 den_len | num bytes | den
/// bytes (reference `_q_element`). Refuses non-canonical values — the
/// producer must construct via [`WitRat::new`].
fn w_q_element(out: &mut Vec<u8>, q: &WitRat) -> Result<(), CanonRefusal> {
    if !q.is_canonical() {
        return Err(CanonRefusal::EncodeNonCanonicalRational);
    }
    let nb = twos_complement_le_minimal(q.num());
    let db = unsigned_le_minimal(q.den());
    w_u32(out, nb.len() as u32);
    w_u32(out, db.len() as u32);
    out.extend_from_slice(&nb);
    out.extend_from_slice(&db);
    Ok(())
}

/// LSB-first bitset over `n` members (reference `_bitset`).
fn w_bitset(out: &mut Vec<u8>, members: &VSet, n: u32) {
    let nbytes = n.div_ceil(8) as usize;
    let start = out.len();
    out.resize(start + nbytes, 0);
    for &i in members {
        out[start + (i / 8) as usize] |= 1 << (i % 8);
    }
}

fn w_dtype(out: &mut Vec<u8>, d: Dtype) {
    out.push(d.kind as u8);
    out.push(d.sign as u8);
    out.push(d.ext as u8);
    out.push(0);
}

fn w_shape(out: &mut Vec<u8>, shape: &[u64]) {
    w_u32(out, shape.len() as u32);
    for &e in shape {
        w_u64(out, e);
    }
}

impl Hypernet {
    /// The canonical byte serialization (reference
    /// `canonical_bytes`). Wire ids serialize as-is — emit normal form
    /// (`renumber`) or normalize before digesting, per the digest-
    /// coherence discipline.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, CanonRefusal> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);

        // --- wire-type table
        w_u32(&mut out, self.wire_types.len() as u32);
        for t in &self.wire_types {
            w_dtype(&mut out, t.dtype);
            w_shape(&mut out, &t.shape);
        }

        // --- complex pool
        w_u32(&mut out, self.complexes.len() as u32);
        for c in &self.complexes {
            w_u32(&mut out, c.vertices.len() as u32);
            w_u32(&mut out, c.amb_dim);
            for v in &c.vertices {
                for coord in v {
                    w_q_element(&mut out, coord)?;
                }
            }
            // canonical simplex order: by (size, ascending members)
            let mut simps: Vec<&VSet> = c.simplices.iter().collect();
            simps.sort_by_key(|s| (s.len(), s.iter().copied().collect::<Vec<u32>>()));
            w_u32(&mut out, simps.len() as u32);
            let n = c.vertices.len() as u32;
            for s in simps {
                w_bitset(&mut out, s, n);
            }
            w_u32(&mut out, c.cells.len() as u32);
            for (mask, ext) in &c.cells {
                w_bitset(&mut out, mask, n);
                w_bitset(&mut out, ext, n);
            }
        }

        // --- applications (SSA order)
        w_u32(&mut out, self.apps.len() as u32);
        for (idx, a) in self.apps.iter().enumerate() {
            out.push(a.tag());
            out.push(ROUND_EXACT);
            w_u16(&mut out, 0);
            let wt = |w: Wire| -> Result<&TensorType, CanonRefusal> {
                self.wtype(w).ok_or(CanonRefusal::EncodeWireOutOfRange {
                    app: idx as u32,
                    wire: w,
                    table_len: self.wire_types.len() as u32,
                })
            };
            match a {
                App::Const { out: o, value } => {
                    w_u32(&mut out, *o);
                    match value {
                        ConstPayload::Rat(qs) => {
                            for q in qs {
                                w_q_element(&mut out, q)?;
                            }
                        }
                        ConstPayload::Bits(bits) => {
                            let members: VSet = bits
                                .iter()
                                .enumerate()
                                .filter(|(_, &b)| b)
                                .map(|(i, _)| i as u32)
                                .collect();
                            w_bitset(&mut out, &members, bits.len() as u32);
                        }
                    }
                }
                App::Contract {
                    inputs,
                    output,
                    extents,
                    in_bindings,
                    out_binding,
                } => {
                    let tt = wt(*output)?;
                    w_dtype(&mut out, tt.dtype);
                    w_u32(&mut out, inputs.len() as u32);
                    for &w in inputs {
                        w_u32(&mut out, w);
                    }
                    w_u32(&mut out, *output);
                    w_u32(&mut out, extents.len() as u32);
                    for &e in extents {
                        w_u64(&mut out, e);
                    }
                    for bnd in in_bindings {
                        w_u32(&mut out, bnd.len() as u32);
                        for &b in bnd {
                            w_u32(&mut out, b);
                        }
                    }
                    w_u32(&mut out, out_binding.len() as u32);
                    for &b in out_binding {
                        w_u32(&mut out, b);
                    }
                }
                App::Cast { inp, out: o } => {
                    let ti = wt(*inp)?.clone();
                    let to = wt(*o)?;
                    w_dtype(&mut out, ti.dtype);
                    w_dtype(&mut out, to.dtype);
                    w_shape(&mut out, &ti.shape);
                    w_u32(&mut out, *inp);
                    w_u32(&mut out, *o);
                }
                App::Cell {
                    complex_idx,
                    inp,
                    outs,
                } => {
                    w_u32(&mut out, *complex_idx);
                    w_u32(&mut out, *inp);
                    w_u32(&mut out, outs.len() as u32);
                    for &w in outs {
                        w_u32(&mut out, w);
                    }
                }
                App::Reshape {
                    new_shape,
                    inp,
                    out: o,
                } => {
                    w_shape(&mut out, new_shape);
                    w_u32(&mut out, *inp);
                    w_u32(&mut out, *o);
                }
                App::Broadcast {
                    to_shape,
                    inp,
                    out: o,
                } => {
                    w_shape(&mut out, to_shape);
                    w_u32(&mut out, *inp);
                    w_u32(&mut out, *o);
                }
                App::Copy { inp, outs } => {
                    w_u32(&mut out, outs.len() as u32);
                    w_u32(&mut out, *inp);
                    for &w in outs {
                        w_u32(&mut out, w);
                    }
                }
                App::Delete { inp } => {
                    w_u32(&mut out, *inp);
                }
                App::Swap {
                    in1,
                    in2,
                    out1,
                    out2,
                } => {
                    w_u32(&mut out, *in1);
                    w_u32(&mut out, *in2);
                    w_u32(&mut out, *out1);
                    w_u32(&mut out, *out2);
                }
                App::Stack { inputs, output } => {
                    w_u32(&mut out, inputs.len() as u32);
                    for &w in inputs {
                        w_u32(&mut out, w);
                    }
                    w_u32(&mut out, *output);
                }
            }
        }

        // --- interface
        w_u32(&mut out, self.graph_inputs.len() as u32);
        for &w in &self.graph_inputs {
            w_u32(&mut out, w);
        }
        w_u32(&mut out, self.graph_outputs.len() as u32);
        for &w in &self.graph_outputs {
            w_u32(&mut out, w);
        }
        Ok(out)
    }

    /// SHA-256 of the canonical bytes — the content address every
    /// certificate row cites.
    pub fn digest(&self) -> Result<HypernetDigest, CanonRefusal> {
        let bytes = self.canonical_bytes()?;
        let d = Sha256::digest(&bytes);
        Ok(HypernetDigest(d.into()))
    }

    /// Parse canonical bytes back into a hypernet. Strict: refuses
    /// truncation, unknown tags, non-canonical rationals, stray bitset
    /// bits, redundant-field disagreement with the wire-type table, and
    /// trailing bytes — so `from_canonical_bytes(b)?.canonical_bytes()`
    /// reproduces `b` exactly. Structural only; run
    /// [`Hypernet::preflight`] for well-formedness.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Hypernet, CanonRefusal> {
        let mut r = Reader { b: bytes, off: 0 };
        let magic = r.take(8, "magic")?;
        if magic != MAGIC {
            return Err(CanonRefusal::BadMagic {
                got: magic.to_vec(),
            });
        }

        // --- wire-type table
        let n_wires = r.u32("wire-type count")?;
        let mut wire_types = Vec::with_capacity(r.cap(n_wires, 8));
        for _ in 0..n_wires {
            let dtype = r.dtype()?;
            let shape = r.shape()?;
            wire_types.push(TensorType { dtype, shape });
        }
        let table_len = wire_types.len() as u32;
        let wt = |w: Wire, off: usize| -> Result<&TensorType, CanonRefusal> {
            wire_types
                .get(w as usize)
                .ok_or(CanonRefusal::DecodeWireOutOfRange {
                    offset: off,
                    wire: w,
                    table_len,
                })
        };

        // --- complex pool
        let n_complexes = r.u32("complex count")?;
        let mut complexes = Vec::with_capacity(r.cap(n_complexes, 16));
        for _ in 0..n_complexes {
            let n_vertices = r.u32("vertex count")?;
            let amb_dim = r.u32("ambient dimension")?;
            let vcap = r.check_count(n_vertices, amb_dim as usize * 9, "vertices")?;
            let mut vertices = Vec::with_capacity(vcap);
            for _ in 0..n_vertices {
                let mut v = Vec::with_capacity(r.cap(amb_dim, 9));
                for _ in 0..amb_dim {
                    v.push(r.q_element()?);
                }
                vertices.push(v);
            }
            let n_simplices = r.u32("simplex count")?;
            let bitset_bytes = n_vertices.div_ceil(8) as usize;
            let scap = r.check_count(n_simplices, bitset_bytes, "simplices")?;
            let mut simplices = Vec::with_capacity(scap);
            for _ in 0..n_simplices {
                let off = r.off;
                let s = r.bitset(n_vertices)?;
                // Enforce the encoder's canonical (size, ascending members)
                // sort — non-decreasing, mirroring exactly what encode can
                // emit — so decode∘encode stays the identity on bytes.
                // (Decode-side twin of the `simps.sort_by_key` above; hole
                // found by the single-byte-corruption proptest during
                // merge hardening, seed banked in tests/.)
                if let Some(prev) = simplices.last() {
                    let prev: &VSet = prev;
                    let cmp = prev
                        .len()
                        .cmp(&s.len())
                        .then_with(|| prev.iter().cmp(s.iter()));
                    if cmp == std::cmp::Ordering::Greater {
                        return Err(CanonRefusal::NonCanonicalSimplexOrder { offset: off });
                    }
                }
                simplices.push(s);
            }
            let n_cells = r.u32("cell count")?;
            let ccap = r.check_count(n_cells, 2 * bitset_bytes, "cells")?;
            let mut cells = Vec::with_capacity(ccap);
            for _ in 0..n_cells {
                let mask = r.bitset(n_vertices)?;
                let ext = r.bitset(n_vertices)?;
                cells.push((mask, ext));
            }
            complexes.push(SimpC::new(amb_dim, vertices, simplices, cells));
        }

        // --- applications
        let n_apps = r.u32("application count")?;
        let mut apps = Vec::with_capacity(r.cap(n_apps, 8));
        for _ in 0..n_apps {
            let hdr_off = r.off;
            let tag = r.u8("application tag")?;
            let round = r.u8("rounding mode")?;
            let pad = r.u16("application pad")?;
            if round != ROUND_EXACT || pad != 0 {
                return Err(CanonRefusal::BadAppHeader {
                    offset: hdr_off,
                    round,
                    pad,
                });
            }
            let app = match tag {
                0 => {
                    let out_off = r.off;
                    let out = r.u32("const output wire")?;
                    let tt = wt(out, out_off)?;
                    let value = if tt.dtype.kind == Kind::Rat {
                        let mut qs =
                            Vec::with_capacity(r.cap(tt.size().min(u32::MAX as u64) as u32, 9));
                        for _ in 0..tt.size() {
                            qs.push(r.q_element()?);
                        }
                        ConstPayload::Rat(qs)
                    } else {
                        let members = r.bitset(tt.size() as u32)?;
                        let bits = (0..tt.size() as u32)
                            .map(|i| members.contains(&i))
                            .collect();
                        ConstPayload::Bits(bits)
                    };
                    App::Const { out, value }
                }
                1 => {
                    let dt_off = r.off;
                    let dtype = r.dtype()?;
                    let n_inputs = r.u32("contract input count")?;
                    let mut inputs = Vec::with_capacity(r.cap(n_inputs, 4));
                    for _ in 0..n_inputs {
                        inputs.push(r.u32("contract input wire")?);
                    }
                    let out_off = r.off;
                    let output = r.u32("contract output wire")?;
                    if wt(output, out_off)?.dtype != dtype {
                        return Err(CanonRefusal::RedundantFieldMismatch {
                            offset: dt_off,
                            what: "contract output dtype",
                        });
                    }
                    let n_extents = r.u32("contract extent count")?;
                    let mut extents = Vec::with_capacity(r.cap(n_extents, 8));
                    for _ in 0..n_extents {
                        extents.push(r.u64("contract extent")?);
                    }
                    let mut in_bindings = Vec::with_capacity(r.cap(n_inputs, 4));
                    for _ in 0..n_inputs {
                        let n = r.u32("in-binding length")?;
                        let mut bnd = Vec::with_capacity(r.cap(n, 4));
                        for _ in 0..n {
                            bnd.push(r.u32("in-binding entry")?);
                        }
                        in_bindings.push(bnd);
                    }
                    let n_out = r.u32("out-binding length")?;
                    let mut out_binding = Vec::with_capacity(r.cap(n_out, 4));
                    for _ in 0..n_out {
                        out_binding.push(r.u32("out-binding entry")?);
                    }
                    App::Contract {
                        inputs,
                        output,
                        extents,
                        in_bindings,
                        out_binding,
                    }
                }
                2 => {
                    let field_off = r.off;
                    let from = r.dtype()?;
                    let to = r.dtype()?;
                    let shape = r.shape()?;
                    let inp_off = r.off;
                    let inp = r.u32("cast input wire")?;
                    let out_off = r.off;
                    let out = r.u32("cast output wire")?;
                    let ti = wt(inp, inp_off)?;
                    if ti.dtype != from || ti.shape != shape {
                        return Err(CanonRefusal::RedundantFieldMismatch {
                            offset: field_off,
                            what: "cast input dtype/shape",
                        });
                    }
                    if wt(out, out_off)?.dtype != to {
                        return Err(CanonRefusal::RedundantFieldMismatch {
                            offset: field_off,
                            what: "cast output dtype",
                        });
                    }
                    App::Cast { inp, out }
                }
                3 => {
                    let complex_idx = r.u32("cell complex index")?;
                    let inp = r.u32("cell input wire")?;
                    let n_outs = r.u32("cell output count")?;
                    let mut outs = Vec::with_capacity(r.cap(n_outs, 4));
                    for _ in 0..n_outs {
                        outs.push(r.u32("cell output wire")?);
                    }
                    App::Cell {
                        complex_idx,
                        inp,
                        outs,
                    }
                }
                4 => {
                    let new_shape = r.shape()?;
                    let inp = r.u32("reshape input wire")?;
                    let out = r.u32("reshape output wire")?;
                    App::Reshape {
                        new_shape,
                        inp,
                        out,
                    }
                }
                5 => {
                    let to_shape = r.shape()?;
                    let inp = r.u32("broadcast input wire")?;
                    let out = r.u32("broadcast output wire")?;
                    App::Broadcast { to_shape, inp, out }
                }
                6 => {
                    let n_outs = r.u32("copy output count")?;
                    let inp = r.u32("copy input wire")?;
                    let mut outs = Vec::with_capacity(r.cap(n_outs, 4));
                    for _ in 0..n_outs {
                        outs.push(r.u32("copy output wire")?);
                    }
                    App::Copy { inp, outs }
                }
                7 => App::Delete {
                    inp: r.u32("delete input wire")?,
                },
                8 => App::Swap {
                    in1: r.u32("swap in1")?,
                    in2: r.u32("swap in2")?,
                    out1: r.u32("swap out1")?,
                    out2: r.u32("swap out2")?,
                },
                9 => {
                    let n_inputs = r.u32("stack input count")?;
                    let mut inputs = Vec::with_capacity(r.cap(n_inputs, 4));
                    for _ in 0..n_inputs {
                        inputs.push(r.u32("stack input wire")?);
                    }
                    let output = r.u32("stack output wire")?;
                    App::Stack { inputs, output }
                }
                _ => {
                    return Err(CanonRefusal::UnknownTag {
                        offset: hdr_off,
                        tag,
                    })
                }
            };
            apps.push(app);
        }

        // --- interface
        let n_in = r.u32("graph input count")?;
        let mut graph_inputs = Vec::with_capacity(r.cap(n_in, 4));
        for _ in 0..n_in {
            graph_inputs.push(r.u32("graph input wire")?);
        }
        let n_out = r.u32("graph output count")?;
        let mut graph_outputs = Vec::with_capacity(r.cap(n_out, 4));
        for _ in 0..n_out {
            graph_outputs.push(r.u32("graph output wire")?);
        }

        if r.off != bytes.len() {
            return Err(CanonRefusal::TrailingBytes {
                got: bytes.len() - r.off,
            });
        }
        Ok(Hypernet {
            wire_types,
            complexes,
            apps,
            graph_inputs,
            graph_outputs,
            wire_names: Default::default(),
        })
    }
}

// ========================================================================
// strict reader
// ========================================================================

struct Reader<'a> {
    b: &'a [u8],
    off: usize,
}

impl<'a> Reader<'a> {
    /// Safe pre-allocation capacity for `n` declared elements of at
    /// least `min_size` bytes each: never trusts a length field beyond
    /// what the remaining input could actually supply (a corrupted
    /// count must fail by truncation, not by allocation).
    fn cap(&self, n: u32, min_size: usize) -> usize {
        let remaining = self.b.len().saturating_sub(self.off);
        (n as usize).min(remaining / min_size.max(1) + 1)
    }

    /// Like [`Reader::cap`], but for element kinds that may legally
    /// occupy ZERO bytes (vertices at ambient dimension 0, bitsets over
    /// zero vertices): a count of such elements cannot be bounded by
    /// truncation, so counts far beyond the input length are refused
    /// outright — they are unrepresentable as honest data and would
    /// otherwise be a memory bomb. The slack keeps every graph the
    /// reference serializer can actually emit from a well-formed (or
    /// even validate-passing) structure decodable.
    fn check_count(
        &self,
        n: u32,
        min_size: usize,
        what: &'static str,
    ) -> Result<usize, CanonRefusal> {
        if min_size == 0 {
            let bound = self.b.len().saturating_sub(self.off) + 1024;
            if n as usize > bound {
                return Err(CanonRefusal::LengthOverflow {
                    offset: self.off,
                    got: n as u64,
                    what,
                });
            }
            return Ok(n as usize);
        }
        Ok(self.cap(n, min_size))
    }

    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], CanonRefusal> {
        if self.off + n > self.b.len() {
            return Err(CanonRefusal::Truncated {
                offset: self.off,
                needed: n,
                available: self.b.len() - self.off,
                what,
            });
        }
        let s = &self.b[self.off..self.off + n];
        self.off += n;
        Ok(s)
    }

    fn u8(&mut self, what: &'static str) -> Result<u8, CanonRefusal> {
        Ok(self.take(1, what)?[0])
    }

    fn u16(&mut self, what: &'static str) -> Result<u16, CanonRefusal> {
        let s = self.take(2, what)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }

    fn u32(&mut self, what: &'static str) -> Result<u32, CanonRefusal> {
        let s = self.take(4, what)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn u64(&mut self, what: &'static str) -> Result<u64, CanonRefusal> {
        let s = self.take(8, what)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        Ok(u64::from_le_bytes(a))
    }

    fn dtype(&mut self) -> Result<Dtype, CanonRefusal> {
        let off = self.off;
        let s = self.take(4, "dtype triple")?;
        let kind = Kind::from_u8(s[0]).ok_or(CanonRefusal::UnknownByte {
            offset: off,
            what: "dtype kind",
            got: s[0],
        })?;
        let sign = Sign::from_u8(s[1]).ok_or(CanonRefusal::UnknownByte {
            offset: off + 1,
            what: "dtype sign",
            got: s[1],
        })?;
        let ext = Ext::from_u8(s[2]).ok_or(CanonRefusal::UnknownByte {
            offset: off + 2,
            what: "dtype ext",
            got: s[2],
        })?;
        if s[3] != 0 {
            return Err(CanonRefusal::UnknownByte {
                offset: off + 3,
                what: "dtype pad",
                got: s[3],
            });
        }
        Ok(Dtype { kind, sign, ext })
    }

    fn shape(&mut self) -> Result<Vec<u64>, CanonRefusal> {
        let n = self.u32("shape rank")?;
        let mut shape = Vec::with_capacity(self.cap(n, 8));
        for _ in 0..n {
            shape.push(self.u64("shape extent")?);
        }
        Ok(shape)
    }

    fn q_element(&mut self) -> Result<WitRat, CanonRefusal> {
        let off = self.off;
        let num_len = self.u32("rational numerator length")? as usize;
        let den_len = self.u32("rational denominator length")? as usize;
        if num_len == 0 || den_len == 0 {
            return Err(CanonRefusal::NonCanonicalRational {
                offset: off,
                why: "zero-length component",
            });
        }
        let nb = self.take(num_len, "rational numerator bytes")?;
        let db = self.take(den_len, "rational denominator bytes")?;
        let num = BigInt::from_signed_bytes_le(nb);
        let den = BigInt::from_bytes_le(BigSign::Plus, db);
        // canonical-form refusals (minimality via re-encode identity)
        if twos_complement_le_minimal(&num) != nb {
            return Err(CanonRefusal::NonCanonicalRational {
                offset: off,
                why: "non-minimal numerator encoding",
            });
        }
        if den < BigInt::one() {
            return Err(CanonRefusal::NonCanonicalRational {
                offset: off,
                why: "denominator < 1",
            });
        }
        if unsigned_le_minimal(&den) != db {
            return Err(CanonRefusal::NonCanonicalRational {
                offset: off,
                why: "non-minimal denominator encoding",
            });
        }
        if num.gcd(&den) != BigInt::one() {
            return Err(CanonRefusal::NonCanonicalRational {
                offset: off,
                why: "gcd(num, den) != 1",
            });
        }
        if num.is_zero() && den != BigInt::one() {
            return Err(CanonRefusal::NonCanonicalRational {
                offset: off,
                why: "zero with denominator != 1",
            });
        }
        Ok(WitRat(WitInt(num), WitInt(den)))
    }

    fn bitset(&mut self, n: u32) -> Result<VSet, CanonRefusal> {
        let off = self.off;
        let nbytes = n.div_ceil(8) as usize;
        let s = self.take(nbytes, "bitset")?;
        let mut set = VSet::new();
        for (byte_idx, &byte) in s.iter().enumerate() {
            for bit in 0..8u32 {
                if byte & (1 << bit) != 0 {
                    let i = byte_idx as u32 * 8 + bit;
                    if i >= n {
                        return Err(CanonRefusal::BitsetOverflow { offset: off, n });
                    }
                    set.insert(i);
                }
            }
        }
        Ok(set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(num: i64, den: i64) -> WitRat {
        WitRat::from_i64(num, den)
    }

    /// Byte vectors pinned live from the reference implementation
    /// (`python3 -c "... _q_element ..."`, 2026-07-21).
    #[test]
    fn q_element_matches_reference_vectors() {
        let cases: Vec<(WitRat, &str)> = vec![
            (q(0, 1), "01000000010000000001"),
            (q(1, 1), "01000000010000000101"),
            (q(-1, 1), "0100000001000000ff01"),
            (q(127, 1), "01000000010000007f01"),
            (q(128, 1), "0200000001000000800001"),
            (q(-128, 1), "01000000010000008001"),
            (q(-129, 1), "02000000010000007fff01"),
            (q(255, 1), "0200000001000000ff0001"),
            (q(256, 1), "0200000001000000000101"),
            (q(3, 2), "01000000010000000302"),
            (q(-7, 3), "0100000001000000f903"),
        ];
        for (val, hex) in &cases {
            let mut out = Vec::new();
            w_q_element(&mut out, val).unwrap();
            let got: String = out.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(&got, hex, "encoding of {val:?}");
        }
        // bignum cases: ±2^80 over small denominators
        let big = BigInt::from(2u8).pow(80);
        let pos = WitRat::new(big.clone(), BigInt::from(3u8));
        let mut out = Vec::new();
        w_q_element(&mut out, &pos).unwrap();
        assert_eq!(
            out.iter().map(|b| format!("{b:02x}")).collect::<String>(),
            "0b00000001000000000000000000000000000103"
        );
        let neg = WitRat::new(-big, BigInt::from(5u8));
        let mut out = Vec::new();
        w_q_element(&mut out, &neg).unwrap();
        assert_eq!(
            out.iter().map(|b| format!("{b:02x}")).collect::<String>(),
            "0b0000000100000000000000000000000000ff05"
        );
    }

    #[test]
    fn q_element_roundtrip() {
        for (n, d) in [(0i64, 1i64), (1, 1), (-1, 1), (128, 1), (-129, 1), (22, 7)] {
            let val = q(n, d);
            let mut out = Vec::new();
            w_q_element(&mut out, &val).unwrap();
            let mut r = Reader { b: &out, off: 0 };
            let back = r.q_element().unwrap();
            assert_eq!(back, val);
            assert_eq!(r.off, out.len());
        }
    }
}
