//! Catalogue: the ten primitive boxes (H-CONST…H-STACK; wire tags 0–9).
//!
//! Constructor order matches the mechanized catalogue and the wire
//! format's tag order (`qtsl-hypernets.tex` §"The ten primitive boxes";
//! Lean `PrimApp` constructors 0–9; `acasxu/hypernet.py::APP_TAG`).

use crate::scalar::WitRat;

/// Wire identifier — an index into the hypernet's wire-type table.
pub type Wire = u32;

/// A `const` box's literal payload. Exactness by type: rational payloads
/// are exact [`WitRat`]s (class E — an IEEE float read from a model file
/// denotes a dyadic rational and is stored as one); Boolean/other kinds
/// carry bits. Which arm is *legal* is decided by the output wire's
/// dtype kind (the well-formedness preflight refuses mismatches).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstPayload {
    /// Flat row-major exact-rational entries (dtype kind `Rat`).
    Rat(Vec<WitRat>),
    /// Flat row-major bits (dtype kinds other than `Rat`; serialized
    /// LSB-first bit-packed).
    Bits(Vec<bool>),
}

impl ConstPayload {
    pub fn len(&self) -> usize {
        match self {
            ConstPayload::Rat(v) => v.len(),
            ConstPayload::Bits(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Catalogue: a primitive-box application (one row of the SSA list).
///
/// The ten variants are the ten boxes, tag order 0–9. Fan-out is
/// explicit `Copy`, fan-in is explicit `Stack`, discard is explicit
/// `Delete`, exchange is explicit `Swap` — linearity makes structure
/// syntax. `Cell` is the calculus's only source of disjunction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum App {
    /// Tag 0 — emit a literal tensor. Parameters are part of the graph,
    /// hence part of the digest.
    Const { out: Wire, value: ConstPayload },
    /// Tag 1 — n-ary generalized Einstein summation with explicit index
    /// bindings (the entire multilinear engine). `extents[i]` is index
    /// variable `i`'s extent; input `k` binds its axes in order via
    /// `in_bindings[k]`; the output likewise via `out_binding`.
    Contract {
        inputs: Vec<Wire>,
        output: Wire,
        extents: Vec<u64>,
        in_bindings: Vec<Vec<u32>>,
        out_binding: Vec<u32>,
    },
    /// Tag 2 — change dtype at fixed shape; the only box where numeric
    /// representation changes (X0 carries the exact rounding mode only).
    Cast { inp: Wire, out: Wire },
    /// Tag 3 — one-hot region membership over a pooled simplicial
    /// complex; one Boolean scalar output per cell. The only branching
    /// box.
    Cell {
        complex_idx: u32,
        inp: Wire,
        outs: Vec<Wire>,
    },
    /// Tag 4 — same entries, new shape (row-major re-indexing).
    Reshape {
        new_shape: Vec<u64>,
        inp: Wire,
        out: Wire,
    },
    /// Tag 5 — replicate along new/singleton axes (numpy trailing-axis
    /// rule).
    Broadcast {
        to_shape: Vec<u64>,
        inp: Wire,
        out: Wire,
    },
    /// Tag 6 — explicit fan-out: duplicate one wire into `outs.len()`
    /// wires of the same type.
    Copy { inp: Wire, outs: Vec<Wire> },
    /// Tag 7 — explicit discard: the discard is a row, never an absence.
    Delete { inp: Wire },
    /// Tag 8 — exchange two wires (the symmetry, as an explicit row).
    Swap {
        in1: Wire,
        in2: Wire,
        out1: Wire,
        out2: Wire,
    },
    /// Tag 9 — explicit fan-in: k wires of one common type ⟨s, δ⟩ to one
    /// wire ⟨k::s, δ⟩ (the tenth box; the packing-gap repair — sums are
    /// `stack` + ones-`contract`).
    Stack { inputs: Vec<Wire>, output: Wire },
}

impl App {
    /// Wire-format tag (0–9), matching the mechanized catalogue's
    /// constructor order.
    pub fn tag(&self) -> u8 {
        match self {
            App::Const { .. } => 0,
            App::Contract { .. } => 1,
            App::Cast { .. } => 2,
            App::Cell { .. } => 3,
            App::Reshape { .. } => 4,
            App::Broadcast { .. } => 5,
            App::Copy { .. } => 6,
            App::Delete { .. } => 7,
            App::Swap { .. } => 8,
            App::Stack { .. } => 9,
        }
    }

    /// The box's name in catalogue vocabulary.
    pub fn name(&self) -> &'static str {
        match self {
            App::Const { .. } => "const",
            App::Contract { .. } => "contract",
            App::Cast { .. } => "cast",
            App::Cell { .. } => "cell",
            App::Reshape { .. } => "reshape",
            App::Broadcast { .. } => "broadcast",
            App::Copy { .. } => "copy",
            App::Delete { .. } => "delete",
            App::Swap { .. } => "swap",
            App::Stack { .. } => "stack",
        }
    }

    /// Consumed wires, in application order (mirrors
    /// `acasxu/hypernet.py::app_inputs` / Lean `PrimApp.inputWires`).
    pub fn input_wires(&self) -> Vec<Wire> {
        match self {
            App::Const { .. } => vec![],
            App::Contract { inputs, .. } => inputs.clone(),
            App::Cast { inp, .. }
            | App::Cell { inp, .. }
            | App::Reshape { inp, .. }
            | App::Broadcast { inp, .. }
            | App::Copy { inp, .. }
            | App::Delete { inp } => vec![*inp],
            App::Swap { in1, in2, .. } => vec![*in1, *in2],
            App::Stack { inputs, .. } => inputs.clone(),
        }
    }

    /// Produced wires, in application order (mirrors
    /// `acasxu/hypernet.py::app_outputs` / Lean `PrimApp.outputWires`).
    pub fn output_wires(&self) -> Vec<Wire> {
        match self {
            App::Const { out, .. } => vec![*out],
            App::Contract { output, .. } => vec![*output],
            App::Cast { out, .. } | App::Reshape { out, .. } | App::Broadcast { out, .. } => {
                vec![*out]
            }
            App::Cell { outs, .. } => outs.clone(),
            App::Copy { outs, .. } => outs.clone(),
            App::Delete { .. } => vec![],
            App::Swap { out1, out2, .. } => vec![*out1, *out2],
            App::Stack { output, .. } => vec![*output],
        }
    }

    /// Rebuild this application with every wire id passed through `wmap`
    /// and (for `Cell`) the complex index through `cmap` — a pure
    /// renaming (mirrors `acasxu/substitution.py::_remap_app`).
    pub fn remap_wires(&self, wmap: impl Fn(Wire) -> Wire, cmap: impl Fn(u32) -> u32) -> App {
        self.remap(&mut |w| wmap(w), &mut |c| cmap(c))
    }

    /// Internal `FnMut` flavor of [`App::remap_wires`] — the
    /// substitution/renumbering workhorse.
    /// Public since the shared-home factoring: substitution
    /// (producer-side) remaps wire and complex indices across the
    /// crate boundary; `remap_wires` above stays the by-value face.
    pub fn remap(
        &self,
        wmap: &mut impl FnMut(Wire) -> Wire,
        cmap: &mut impl FnMut(u32) -> u32,
    ) -> App {
        match self {
            App::Const { out, value } => App::Const {
                out: wmap(*out),
                value: value.clone(),
            },
            App::Contract {
                inputs,
                output,
                extents,
                in_bindings,
                out_binding,
            } => App::Contract {
                inputs: inputs.iter().map(|&w| wmap(w)).collect(),
                output: wmap(*output),
                extents: extents.clone(),
                in_bindings: in_bindings.clone(),
                out_binding: out_binding.clone(),
            },
            App::Cast { inp, out } => App::Cast {
                inp: wmap(*inp),
                out: wmap(*out),
            },
            App::Cell {
                complex_idx,
                inp,
                outs,
            } => App::Cell {
                complex_idx: cmap(*complex_idx),
                inp: wmap(*inp),
                outs: outs.iter().map(|&w| wmap(w)).collect(),
            },
            App::Reshape {
                new_shape,
                inp,
                out,
            } => App::Reshape {
                new_shape: new_shape.clone(),
                inp: wmap(*inp),
                out: wmap(*out),
            },
            App::Broadcast { to_shape, inp, out } => App::Broadcast {
                to_shape: to_shape.clone(),
                inp: wmap(*inp),
                out: wmap(*out),
            },
            App::Copy { inp, outs } => App::Copy {
                inp: wmap(*inp),
                outs: outs.iter().map(|&w| wmap(w)).collect(),
            },
            App::Delete { inp } => App::Delete { inp: wmap(*inp) },
            App::Swap {
                in1,
                in2,
                out1,
                out2,
            } => App::Swap {
                in1: wmap(*in1),
                in2: wmap(*in2),
                out1: wmap(*out1),
                out2: wmap(*out2),
            },
            App::Stack { inputs, output } => App::Stack {
                inputs: inputs.iter().map(|&w| wmap(w)).collect(),
                output: wmap(*output),
            },
        }
    }
}
