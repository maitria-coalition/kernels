//! Catalogue: the hypernet container (H-GRAPH) and the typed builder for
//! the ten primitive boxes (H-CONST…H-STACK).

use std::collections::{BTreeMap, BTreeSet};

use crate::app::{App, ConstPayload, Wire};
use crate::simplex::SimpC;
use crate::types::{broadcasts_to, Dtype, Kind, TensorType};
use crate::wf::{TypeViolation, Violation};

/// Catalogue: hypernet — wire-type table, complex pool, SSA application
/// list, interface (`qtsl-hypernets.tex` Definition "hypernet";
/// `acasxu/hypernet.py::Hypernet`; Lean `Hypernet`, whose structure
/// carries well-formedness as proof fields — here the same three
/// conditions are checked by [`Hypernet::preflight`]).
///
/// Plain owned data. `wire_names` is debug metadata: excluded from
/// equality, from the canonical serialization, and from the digest.
#[derive(Debug, Clone, Default)]
pub struct Hypernet {
    /// Wire id = index into this table.
    pub wire_types: Vec<TensorType>,
    /// Payload pool for `cell` applications.
    pub complexes: Vec<SimpC>,
    /// Applications in SSA order.
    pub apps: Vec<App>,
    /// Ordered graph inputs (wire ids).
    pub graph_inputs: Vec<Wire>,
    /// Ordered graph outputs (wire ids).
    pub graph_outputs: Vec<Wire>,
    /// Debug names — not compared, not serialized, not digested.
    pub wire_names: BTreeMap<Wire, String>,
}

impl PartialEq for Hypernet {
    /// Content equality: everything the canonical serialization carries
    /// (`wire_names` and complex `hint`s excluded).
    fn eq(&self, other: &Hypernet) -> bool {
        self.wire_types == other.wire_types
            && self.complexes == other.complexes
            && self.apps == other.apps
            && self.graph_inputs == other.graph_inputs
            && self.graph_outputs == other.graph_outputs
    }
}
impl Eq for Hypernet {}

impl Hypernet {
    /// The tensor type of wire `w`, if `w` is in range.
    pub fn wtype(&self, w: Wire) -> Option<&TensorType> {
        self.wire_types.get(w as usize)
    }
}

/// Refusals from the typed builder — construction-time typing failures,
/// with the geometry attached (which wire, which axis, which extent).
/// Witnesses always, never bare verdicts.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BuildRefusal {
    #[error("unknown wire {wire}: not allocated by this builder")]
    UnknownWire { wire: Wire },
    #[error(
        "wire {wire} already consumed by application {by_app} — fan-out is the explicit `copy` box"
    )]
    WireAlreadyConsumed { wire: Wire, by_app: u32 },
    #[error("contract with no inputs: the builder derives the dtype from its inputs (construct a raw `App::Contract` if a nullary contract is genuinely meant)")]
    ContractNoInputs,
    #[error("cell: complex index {complex_idx} out of range (pool has {pool_len})")]
    ComplexIndexOutOfRange { complex_idx: u32, pool_len: u32 },
    #[error("cell: complex {complex_idx} is invalid: {violations:?}")]
    InvalidComplex {
        complex_idx: u32,
        violations: Vec<crate::simplex::SimpCViolation>,
    },
    #[error("copy with zero branches: a value nobody reads is discarded by `delete`, never by a 0-ary copy")]
    CopyNoBranches,
    #[error("stack with no inputs: the output type ⟨k::s, δ⟩ needs a common input type to exist")]
    StackNoInputs,
    #[error("graph output {wire} was consumed by application {by_app}: an interface listing needs an unconsumed wire")]
    OutputConsumed { wire: Wire, by_app: u32 },
    #[error("typing: {0}")]
    Typing(TypeViolation),
    #[error("finish: preflight found violations (builder invariant breach): {0:?}")]
    Preflight(Vec<Violation>),
}

/// Typed constructor surface for the ten primitive boxes: each method
/// checks its box's typing rule at construction and returns the produced
/// wire(s), so a builder-made graph satisfies SSA and per-primitive
/// typing by construction (linearity is enforced eagerly via
/// consumed-wire tracking; [`HypernetBuilder::finish`] runs the full
/// preflight as a belt).
///
/// ```
/// use maitria_hypernet::{HypernetBuilder, TensorType, ConstPayload};
/// use maitria_hypernet::WitRat;
///
/// // u + v as stack-and-ones (the sum idiom of the tenth box)
/// let mut b = HypernetBuilder::new();
/// let u = b.input(TensorType::rat([]));
/// let v = b.input(TensorType::rat([]));
/// let t = b.stack(&[u, v]).unwrap();
/// let ones = b
///     .constant(
///         TensorType::rat([2]),
///         ConstPayload::Rat(vec![WitRat::from_i64(1, 1), WitRat::from_i64(1, 1)]),
///     )
///     .unwrap();
/// let sum = b
///     .contract(&[t, ones], &[2], &[vec![0], vec![0]], &[])
///     .unwrap();
/// let h = b.finish(vec![sum]).unwrap();
/// assert_eq!(h.apps.len(), 3);
/// assert!(h.preflight().is_empty());
/// ```
#[derive(Debug, Clone, Default)]
pub struct HypernetBuilder {
    net: Hypernet,
    /// wire -> index of the app that consumed it (linearity, eagerly).
    consumed: BTreeMap<Wire, u32>,
}

impl HypernetBuilder {
    pub fn new() -> HypernetBuilder {
        HypernetBuilder::default()
    }

    /// Declare a graph input of type `ty`; returns its wire.
    pub fn input(&mut self, ty: TensorType) -> Wire {
        let w = self.alloc(ty, "");
        self.net.graph_inputs.push(w);
        w
    }

    /// Declare a named graph input (name is debug metadata only).
    pub fn input_named(&mut self, ty: TensorType, name: &str) -> Wire {
        let w = self.alloc(ty, name);
        self.net.graph_inputs.push(w);
        w
    }

    /// Intern a complex into the pool (validated); returns its index for
    /// [`HypernetBuilder::cell`]. Pool entries are appended as given —
    /// content dedup is `renumber`'s job, so builder output digests match
    /// the reference implementation's for the same construction sequence.
    pub fn add_complex(&mut self, c: SimpC) -> Result<u32, BuildRefusal> {
        let violations = c.validate();
        if !violations.is_empty() {
            return Err(BuildRefusal::InvalidComplex {
                complex_idx: self.net.complexes.len() as u32,
                violations,
            });
        }
        self.net.complexes.push(c);
        Ok(self.net.complexes.len() as u32 - 1)
    }

    /// Rule catalogue: hypernet primitive `const` (H-CONST) — emit a
    /// literal tensor of type `ty`. Payload arity must equal the type's
    /// size; rational kinds take `Rat` payloads, all others `Bits`.
    pub fn constant(&mut self, ty: TensorType, value: ConstPayload) -> Result<Wire, BuildRefusal> {
        if value.len() as u64 != ty.size() {
            return Err(BuildRefusal::Typing(TypeViolation::ConstPayloadSize {
                expected: ty.size(),
                got: value.len() as u64,
            }));
        }
        let want_rat = ty.dtype.kind == Kind::Rat;
        let is_rat = matches!(value, ConstPayload::Rat(_));
        if want_rat != is_rat {
            return Err(BuildRefusal::Typing(TypeViolation::ConstPayloadKind {
                kind: ty.dtype.kind,
                rational_payload: is_rat,
            }));
        }
        let out = self.alloc(ty, "");
        self.net.apps.push(App::Const { out, value });
        Ok(out)
    }

    /// Rule catalogue: hypernet primitive `contract` (H-CONTRACT) — the
    /// n-ary generalized Einstein summation. The output type is derived:
    /// dtype from the (necessarily uniform) inputs, shape from
    /// `out_binding`'s index extents.
    pub fn contract(
        &mut self,
        inputs: &[Wire],
        extents: &[u64],
        in_bindings: &[Vec<u32>],
        out_binding: &[u32],
    ) -> Result<Wire, BuildRefusal> {
        if inputs.is_empty() {
            return Err(BuildRefusal::ContractNoInputs);
        }
        if inputs.len() != in_bindings.len() {
            return Err(BuildRefusal::Typing(TypeViolation::ContractArity {
                inputs: inputs.len() as u32,
                bindings: in_bindings.len() as u32,
            }));
        }
        let dtype = self.peek(inputs[0])?.dtype;
        for (k, (&w, bnd)) in inputs.iter().zip(in_bindings).enumerate() {
            let ti = self.peek(w)?.clone();
            if ti.dtype != dtype {
                return Err(BuildRefusal::Typing(TypeViolation::ContractInputDtype {
                    input: k as u32,
                }));
            }
            if ti.shape.len() != bnd.len() {
                return Err(BuildRefusal::Typing(TypeViolation::ContractInputRank {
                    input: k as u32,
                    rank: ti.shape.len() as u32,
                    bindings: bnd.len() as u32,
                }));
            }
            for (ax, &b) in bnd.iter().enumerate() {
                if b as usize >= extents.len() || ti.shape[ax] != extents[b as usize] {
                    return Err(BuildRefusal::Typing(TypeViolation::ContractInputExtent {
                        input: k as u32,
                        axis: ax as u32,
                        extent: ti.shape[ax],
                        index: b,
                    }));
                }
            }
        }
        let mut out_shape = Vec::with_capacity(out_binding.len());
        for (ax, &b) in out_binding.iter().enumerate() {
            if b as usize >= extents.len() {
                return Err(BuildRefusal::Typing(TypeViolation::ContractOutputExtent {
                    axis: ax as u32,
                    index: b,
                }));
            }
            out_shape.push(extents[b as usize]);
        }
        for &w in inputs {
            self.consume(w)?;
        }
        let output = self.alloc(TensorType::new(dtype, out_shape), "");
        self.net.apps.push(App::Contract {
            inputs: inputs.to_vec(),
            output,
            extents: extents.to_vec(),
            in_bindings: in_bindings.to_vec(),
            out_binding: out_binding.to_vec(),
        });
        Ok(output)
    }

    /// Rule catalogue: hypernet primitive `cast` (H-CAST) — change dtype
    /// at fixed shape (X0 carries the exact rounding mode only).
    pub fn cast(&mut self, inp: Wire, to: Dtype) -> Result<Wire, BuildRefusal> {
        let shape = self.peek(inp)?.shape.clone();
        self.consume(inp)?;
        let out = self.alloc(TensorType::new(to, shape), "");
        self.net.apps.push(App::Cast { inp, out });
        Ok(out)
    }

    /// Rule catalogue: hypernet primitive `cell` (H-CELL) — one-hot
    /// region membership over pooled complex `complex_idx`; input must be
    /// a rank-1 rational wire of extent `amb_dim`; returns one Boolean
    /// scalar wire per cell, in cell order. The calculus's only source of
    /// disjunction.
    pub fn cell(&mut self, complex_idx: u32, inp: Wire) -> Result<Vec<Wire>, BuildRefusal> {
        let Some(c) = self.net.complexes.get(complex_idx as usize) else {
            return Err(BuildRefusal::ComplexIndexOutOfRange {
                complex_idx,
                pool_len: self.net.complexes.len() as u32,
            });
        };
        let (amb_dim, n_cells) = (c.amb_dim, c.cells.len());
        let ti = self.peek(inp)?;
        if ti.shape != [amb_dim as u64] || ti.dtype.kind != Kind::Rat {
            return Err(BuildRefusal::Typing(TypeViolation::CellInputType {
                amb_dim,
                got: ti.clone(),
            }));
        }
        self.consume(inp)?;
        let outs: Vec<Wire> = (0..n_cells)
            .map(|_| self.alloc(TensorType::boolean([]), ""))
            .collect();
        self.net.apps.push(App::Cell {
            complex_idx,
            inp,
            outs: outs.clone(),
        });
        Ok(outs)
    }

    /// Rule catalogue: hypernet primitive `reshape` (H-RESHAPE) — same
    /// entries, new shape of the same total size.
    pub fn reshape(&mut self, inp: Wire, new_shape: &[u64]) -> Result<Wire, BuildRefusal> {
        let ti = self.peek(inp)?.clone();
        let new_size: u64 = new_shape.iter().product();
        if ti.size() != new_size {
            return Err(BuildRefusal::Typing(TypeViolation::ReshapeSize {
                from_size: ti.size(),
                to_size: new_size,
            }));
        }
        self.consume(inp)?;
        let out = self.alloc(TensorType::new(ti.dtype, new_shape.to_vec()), "");
        self.net.apps.push(App::Reshape {
            new_shape: new_shape.to_vec(),
            inp,
            out,
        });
        Ok(out)
    }

    /// Rule catalogue: hypernet primitive `broadcast` (H-BROADCAST) —
    /// replicate along new/singleton axes (numpy trailing-axis rule).
    pub fn broadcast(&mut self, inp: Wire, to_shape: &[u64]) -> Result<Wire, BuildRefusal> {
        let ti = self.peek(inp)?.clone();
        if !broadcasts_to(&ti.shape, to_shape) {
            return Err(BuildRefusal::Typing(TypeViolation::BroadcastShape {
                from: ti.shape.clone(),
                to: to_shape.to_vec(),
            }));
        }
        self.consume(inp)?;
        let out = self.alloc(TensorType::new(ti.dtype, to_shape.to_vec()), "");
        self.net.apps.push(App::Broadcast {
            to_shape: to_shape.to_vec(),
            inp,
            out,
        });
        Ok(out)
    }

    /// Rule catalogue: hypernet primitive `copy` (H-COPY) — explicit
    /// fan-out into `k` wires of the input's type. Under linearity this
    /// is the only way a value acquires multiple consumers.
    pub fn copy(&mut self, inp: Wire, k: u32) -> Result<Vec<Wire>, BuildRefusal> {
        if k == 0 {
            return Err(BuildRefusal::CopyNoBranches);
        }
        let ti = self.peek(inp)?.clone();
        self.consume(inp)?;
        let outs: Vec<Wire> = (0..k).map(|_| self.alloc(ti.clone(), "")).collect();
        self.net.apps.push(App::Copy {
            inp,
            outs: outs.clone(),
        });
        Ok(outs)
    }

    /// Rule catalogue: hypernet primitive `delete` (H-DELETE) — explicit
    /// discard: the discard is a row, never an absence.
    pub fn delete(&mut self, inp: Wire) -> Result<(), BuildRefusal> {
        self.peek(inp)?;
        self.consume(inp)?;
        self.net.apps.push(App::Delete { inp });
        Ok(())
    }

    /// Rule catalogue: hypernet primitive `swap` (H-SWAP) — exchange two
    /// wires of possibly different types; returns `(out1, out2)` with
    /// `out1 : type(in2)` and `out2 : type(in1)`.
    pub fn swap(&mut self, in1: Wire, in2: Wire) -> Result<(Wire, Wire), BuildRefusal> {
        let t1 = self.peek(in1)?.clone();
        let t2 = self.peek(in2)?.clone();
        self.consume(in1)?;
        self.consume(in2)?;
        let out1 = self.alloc(t2, "");
        let out2 = self.alloc(t1, "");
        self.net.apps.push(App::Swap {
            in1,
            in2,
            out1,
            out2,
        });
        Ok((out1, out2))
    }

    /// Rule catalogue: hypernet primitive `stack` (H-STACK) — explicit
    /// fan-in: k wires of one common type ⟨s, δ⟩ to one wire ⟨k::s, δ⟩.
    /// The sum idiom: stack, then contract against a ones vector.
    pub fn stack(&mut self, inputs: &[Wire]) -> Result<Wire, BuildRefusal> {
        let Some((&first, rest)) = inputs.split_first() else {
            return Err(BuildRefusal::StackNoInputs);
        };
        let t0 = self.peek(first)?.clone();
        for &w in rest {
            if self.peek(w)? != &t0 {
                return Err(BuildRefusal::Typing(TypeViolation::StackInputType {
                    wire: w,
                }));
            }
        }
        for &w in inputs {
            self.consume(w)?;
        }
        let mut out_shape = Vec::with_capacity(t0.shape.len() + 1);
        out_shape.push(inputs.len() as u64);
        out_shape.extend_from_slice(&t0.shape);
        let output = self.alloc(TensorType::new(t0.dtype, out_shape), "");
        self.net.apps.push(App::Stack {
            inputs: inputs.to_vec(),
            output,
        });
        Ok(output)
    }

    /// The type of a wire this builder has allocated (inputs and every
    /// constructor output), if it exists.
    pub fn wire_type(&self, w: Wire) -> Option<&TensorType> {
        self.net.wire_types.get(w as usize)
    }

    /// Attach a debug name to a wire (metadata only).
    pub fn name_wire(&mut self, w: Wire, name: &str) {
        if !name.is_empty() {
            self.net.wire_names.insert(w, name.to_string());
        }
    }

    /// Close the graph with its ordered outputs; runs the full
    /// well-formedness preflight as a belt (a builder-made graph passes
    /// by construction).
    pub fn finish(mut self, graph_outputs: Vec<Wire>) -> Result<Hypernet, BuildRefusal> {
        for &w in &graph_outputs {
            if w as usize >= self.net.wire_types.len() {
                return Err(BuildRefusal::UnknownWire { wire: w });
            }
            if let Some(&by_app) = self.consumed.get(&w) {
                return Err(BuildRefusal::OutputConsumed { wire: w, by_app });
            }
        }
        self.net.graph_outputs = graph_outputs;
        let violations = self.net.preflight();
        if !violations.is_empty() {
            return Err(BuildRefusal::Preflight(violations));
        }
        Ok(self.net)
    }

    // ---- internals ------------------------------------------------------

    fn alloc(&mut self, ty: TensorType, name: &str) -> Wire {
        self.net.wire_types.push(ty);
        let w = (self.net.wire_types.len() - 1) as Wire;
        if !name.is_empty() {
            self.net.wire_names.insert(w, name.to_string());
        }
        w
    }

    fn peek(&self, w: Wire) -> Result<&TensorType, BuildRefusal> {
        self.net
            .wire_types
            .get(w as usize)
            .ok_or(BuildRefusal::UnknownWire { wire: w })
    }

    fn consume(&mut self, w: Wire) -> Result<(), BuildRefusal> {
        if let Some(&by_app) = self.consumed.get(&w) {
            return Err(BuildRefusal::WireAlreadyConsumed { wire: w, by_app });
        }
        self.consumed.insert(w, self.net.apps.len() as u32);
        Ok(())
    }
}

/// Internal: consumed-wire set of an application list (used by the seam
/// logic and preflight).
/// Public since the shared-home factoring: the substitution operation
/// (producer-side, `mtk_catalogue::hypernet::subst`) reads the
/// consumed-wire set across the crate boundary.
pub fn consumed_wires(apps: &[App]) -> BTreeSet<Wire> {
    apps.iter().flat_map(|a| a.input_wires()).collect()
}
