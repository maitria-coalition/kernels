//! Catalogue: tensor types ⟨shape, dtype⟩ (`qtsl-hypernets.tex` §"The
//! intermediate representation"; wire-format dtype triple = (kind, sign,
//! ext), matching the reference serializer's constants).

/// Dtype kind byte of the wire format. `Bool`/`Int`/`Rat` are the
/// configurations the book uses; the float kinds exist in the format
/// (class-F carriers) and are decodable, but nothing in this producer
/// surface constructs them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Kind {
    Bool = 0,
    Int = 1,
    Rat = 2,
    F16 = 3,
    Bf16 = 4,
    F32 = 5,
    F64 = 6,
}

impl Kind {
    pub fn from_u8(x: u8) -> Option<Kind> {
        Some(match x {
            0 => Kind::Bool,
            1 => Kind::Int,
            2 => Kind::Rat,
            3 => Kind::F16,
            4 => Kind::Bf16,
            5 => Kind::F32,
            6 => Kind::F64,
            _ => return None,
        })
    }
}

/// Signedness byte of the wire format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Sign {
    NonNeg = 0,
    Signed = 1,
}

impl Sign {
    pub fn from_u8(x: u8) -> Option<Sign> {
        Some(match x {
            0 => Sign::NonNeg,
            1 => Sign::Signed,
            _ => return None,
        })
    }
}

/// Extension byte of the wire format (finite vs extended carrier).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Ext {
    Fin = 0,
    Ext = 1,
}

impl Ext {
    pub fn from_u8(x: u8) -> Option<Ext> {
        Some(match x {
            0 => Ext::Fin,
            1 => Ext::Ext,
            _ => return None,
        })
    }
}

/// The dtype triple as it rides the wire: kind, sign, extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dtype {
    pub kind: Kind,
    pub sign: Sign,
    pub ext: Ext,
}

impl Dtype {
    /// Exact rational ℚ — the dtype of every numeric wire in the book's
    /// configurations.
    pub const RAT: Dtype = Dtype {
        kind: Kind::Rat,
        sign: Sign::Signed,
        ext: Ext::Fin,
    };

    /// Boolean.
    pub const BOOL: Dtype = Dtype {
        kind: Kind::Bool,
        sign: Sign::NonNeg,
        ext: Ext::Fin,
    };

    /// Signed finite integer.
    pub const INT: Dtype = Dtype {
        kind: Kind::Int,
        sign: Sign::Signed,
        ext: Ext::Fin,
    };
}

/// Renders the kind name bare when sign/ext are the kind's
/// conventional configuration (`Bool` is nonneg, everything else
/// signed; finite everywhere), with explicit qualifiers otherwise:
/// `Rat`, `Bool`, `Int(nonneg)`, `Rat(ext)`, `Bool(signed,ext)`.
impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self.kind {
            Kind::Bool => "Bool",
            Kind::Int => "Int",
            Kind::Rat => "Rat",
            Kind::F16 => "F16",
            Kind::Bf16 => "Bf16",
            Kind::F32 => "F32",
            Kind::F64 => "F64",
        };
        write!(f, "{kind}")?;
        let conventional_sign = match self.kind {
            Kind::Bool => Sign::NonNeg,
            _ => Sign::Signed,
        };
        let sign = (self.sign != conventional_sign).then_some(match self.sign {
            Sign::NonNeg => "nonneg",
            Sign::Signed => "signed",
        });
        let ext = (self.ext == Ext::Ext).then_some("ext");
        match (sign, ext) {
            (None, None) => Ok(()),
            (Some(q), None) | (None, Some(q)) => write!(f, "({q})"),
            (Some(s), Some(e)) => write!(f, "({s},{e})"),
        }
    }
}

/// Catalogue: tensor type ⟨shape, dtype⟩. Scalars have the empty shape.
///
/// ```
/// use maitria_hypernet::TensorType;
/// let t = TensorType::rat([3, 2]);
/// assert_eq!(t.size(), 6);
/// assert_eq!(TensorType::boolean([]).size(), 1);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorType {
    pub dtype: Dtype,
    /// Extent list; row-major entry order everywhere in the format.
    pub shape: Vec<u64>,
}

impl TensorType {
    pub fn new(dtype: Dtype, shape: impl Into<Vec<u64>>) -> TensorType {
        TensorType {
            dtype,
            shape: shape.into(),
        }
    }

    /// Exact-rational tensor type (the python reference's `RatT`).
    pub fn rat(shape: impl Into<Vec<u64>>) -> TensorType {
        TensorType::new(Dtype::RAT, shape)
    }

    /// Boolean tensor type (the python reference's `BoolT`).
    pub fn boolean(shape: impl Into<Vec<u64>>) -> TensorType {
        TensorType::new(Dtype::BOOL, shape)
    }

    /// Total entry count (product of extents; 1 for scalars).
    pub fn size(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// Compact mathematical rendering: dtype, then bracketed extents for
/// non-scalars — matching the `Rat[{amb_dim}]` convention the
/// well-formedness refusals already use.
///
/// ```
/// use maitria_hypernet::TensorType;
/// assert_eq!(TensorType::rat([3, 2]).to_string(), "Rat[3,2]");
/// assert_eq!(TensorType::boolean([]).to_string(), "Bool");
/// ```
impl std::fmt::Display for TensorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.dtype)?;
        if !self.shape.is_empty() {
            write!(f, "[")?;
            for (i, e) in self.shape.iter().enumerate() {
                if i > 0 {
                    write!(f, ",")?;
                }
                write!(f, "{e}")?;
            }
            write!(f, "]")?;
        }
        Ok(())
    }
}

/// numpy trailing-axis broadcast compatibility (`acasxu/hypernet.py
/// numpy_broadcasts_to`; Lean `Shape.broadcastsTo`).
pub(crate) fn broadcasts_to(src: &[u64], dst: &[u64]) -> bool {
    if src.len() > dst.len() {
        return false;
    }
    for i in 0..src.len() {
        let s = src[src.len() - 1 - i];
        let d = dst[dst.len() - 1 - i];
        if s != d && s != 1 {
            return false;
        }
    }
    true
}
