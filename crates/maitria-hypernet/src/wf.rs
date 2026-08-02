//! Catalogue: hypernet validity preflight (H-WF) — linearity, SSA
//! acyclicity, per-primitive type consistency, complex validity.
//!
//! This is a *courtesy check*: it saves checker round-trips, and the
//! checker's replay is the only authority. Clause-for-clause mirror of
//! `acasxu/hypernet.py::check_wellformed` (Lean: `Linearity`, `Acyclic`,
//! `PrimApp.TypeConsistent`), with string violations replaced by
//! witness-shaped ones. Two deliberate deltas from the python reference,
//! both on inputs where it would raise instead of report: out-of-range
//! wire ids and complex indices come back as violations here.

use std::collections::BTreeMap;

use crate::app::{App, ConstPayload, Wire};
use crate::graph::Hypernet;
use crate::simplex::SimpCViolation;
use crate::types::{broadcasts_to, Kind, TensorType};

/// A well-formedness violation, with its geometry (which wire, which
/// application, which axis). `app` indices are positions in the SSA list.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Violation {
    #[error("linearity: wire {wire} consumed by applications {first_app} and {second_app}")]
    WireConsumedTwice {
        wire: Wire,
        first_app: u32,
        second_app: u32,
    },
    #[error("linearity: wire {wire} produced by applications {first_app} and {second_app}")]
    WireProducedTwice {
        wire: Wire,
        first_app: u32,
        second_app: u32,
    },
    #[error("linearity: graph input {wire} also produced by application {app}")]
    GraphInputProduced { wire: Wire, app: u32 },
    #[error("acyclicity/SSA: application {app} ({box_name}) reads wire {wire} before production")]
    ReadBeforeProduction {
        app: u32,
        box_name: &'static str,
        wire: Wire,
    },
    #[error("graph output {wire} never produced")]
    GraphOutputNeverProduced { wire: Wire },
    #[error("wire {wire} referenced by application {app} is out of range (table has {table_len})")]
    WireOutOfRange {
        app: u32,
        wire: Wire,
        table_len: u32,
    },
    #[error("application {app}: complex index {complex_idx} out of range (pool has {pool_len})")]
    ComplexIndexOutOfRange {
        app: u32,
        complex_idx: u32,
        pool_len: u32,
    },
    #[error("type: application {app} ({box_name}): {violation}")]
    Typing {
        app: u32,
        box_name: &'static str,
        violation: TypeViolation,
    },
    #[error("complex {complex_idx}: {violation}")]
    Complex {
        complex_idx: u32,
        violation: SimpCViolation,
    },
}

/// A per-primitive typing violation (the geometry of a typing failure).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TypeViolation {
    #[error("const payload has {got} entries, type size is {expected}")]
    ConstPayloadSize { expected: u64, got: u64 },
    #[error(
        "const payload arm mismatch: dtype kind {kind:?} with rational_payload={rational_payload}"
    )]
    ConstPayloadKind { kind: Kind, rational_payload: bool },
    #[error("contract: {inputs} inputs vs {bindings} bindings")]
    ContractArity { inputs: u32, bindings: u32 },
    #[error("contract: input {input} dtype differs from output dtype")]
    ContractInputDtype { input: u32 },
    #[error("contract: input {input} rank {rank} != {bindings} bound axes")]
    ContractInputRank {
        input: u32,
        rank: u32,
        bindings: u32,
    },
    #[error("contract: input {input} axis {axis} extent {extent} vs index {index}")]
    ContractInputExtent {
        input: u32,
        axis: u32,
        extent: u64,
        index: u32,
    },
    #[error("contract: output rank {rank} != {bindings} bound axes")]
    ContractOutputRank { rank: u32, bindings: u32 },
    #[error("contract: output axis {axis} extent mismatch (index {index})")]
    ContractOutputExtent { axis: u32, index: u32 },
    #[error("cast: input and output shapes differ")]
    CastShape,
    #[error("cell: input must be Rat[{amb_dim}], got {got}")]
    CellInputType { amb_dim: u32, got: TensorType },
    #[error("cell: {outs} outputs vs {cells} cells")]
    CellArity { outs: u32, cells: u32 },
    #[error("cell: output wire {wire} is not a Bool scalar")]
    CellOutputNotBoolScalar { wire: Wire },
    #[error(
        "reshape: size {from_size} -> {to_size} mismatch, or output shape differs from declared"
    )]
    ReshapeSize { from_size: u64, to_size: u64 },
    #[error("reshape: input and output dtypes differ")]
    ReshapeDtype,
    #[error(
        "broadcast: {from:?} does not broadcast to {to:?}, or output shape differs from declared"
    )]
    BroadcastShape { from: Vec<u64>, to: Vec<u64> },
    #[error("copy: output wire {wire} type differs from input")]
    CopyOutputType { wire: Wire },
    #[error("swap: output types are not the exchanged input types")]
    SwapType,
    #[error("stack: input wire {wire} type differs from the first input")]
    StackInputType { wire: Wire },
    #[error("stack: output shape {got:?} != {expected:?}")]
    StackOutputShape { got: Vec<u64>, expected: Vec<u64> },
    #[error("stack: output dtype differs from the inputs'")]
    StackDtype,
}

impl Hypernet {
    /// Well-formedness preflight: returns every violation found (empty =
    /// the three conditions hold and every pooled complex validates).
    pub fn preflight(&self) -> Vec<Violation> {
        let mut fails = Vec::new();
        let table_len = self.wire_types.len() as u32;

        // LINEARITY
        let mut produced: BTreeMap<Wire, u32> = BTreeMap::new();
        let mut consumed: BTreeMap<Wire, u32> = BTreeMap::new();
        for (idx, a) in self.apps.iter().enumerate() {
            let idx = idx as u32;
            for w in a.input_wires() {
                if let Some(&first) = consumed.get(&w) {
                    fails.push(Violation::WireConsumedTwice {
                        wire: w,
                        first_app: first,
                        second_app: idx,
                    });
                }
                consumed.insert(w, idx);
            }
            for w in a.output_wires() {
                if let Some(&first) = produced.get(&w) {
                    fails.push(Violation::WireProducedTwice {
                        wire: w,
                        first_app: first,
                        second_app: idx,
                    });
                }
                produced.insert(w, idx);
            }
        }
        for &w in &self.graph_inputs {
            if let Some(&app) = produced.get(&w) {
                fails.push(Violation::GraphInputProduced { wire: w, app });
            }
        }

        // ACYCLICITY (SSA order)
        let mut seen: std::collections::BTreeSet<Wire> =
            self.graph_inputs.iter().copied().collect();
        for (idx, a) in self.apps.iter().enumerate() {
            for w in a.input_wires() {
                if !seen.contains(&w) {
                    fails.push(Violation::ReadBeforeProduction {
                        app: idx as u32,
                        box_name: a.name(),
                        wire: w,
                    });
                }
            }
            for w in a.output_wires() {
                seen.insert(w);
            }
        }
        for &w in &self.graph_outputs {
            if !seen.contains(&w) {
                fails.push(Violation::GraphOutputNeverProduced { wire: w });
            }
        }

        // TYPE CONSISTENCY
        for (idx, a) in self.apps.iter().enumerate() {
            self.check_app_types(idx as u32, a, table_len, &mut fails);
        }

        // COMPLEX VALIDITY
        for (ci, c) in self.complexes.iter().enumerate() {
            for violation in c.validate() {
                fails.push(Violation::Complex {
                    complex_idx: ci as u32,
                    violation,
                });
            }
        }
        fails
    }

    fn check_app_types(&self, idx: u32, a: &App, table_len: u32, fails: &mut Vec<Violation>) {
        // every referenced wire must be in range before its type is read
        let mut in_range = true;
        for w in a.input_wires().into_iter().chain(a.output_wires()) {
            if self.wtype(w).is_none() {
                fails.push(Violation::WireOutOfRange {
                    app: idx,
                    wire: w,
                    table_len,
                });
                in_range = false;
            }
        }
        if !in_range {
            return;
        }
        let t = |w: Wire| self.wtype(w).expect("range-checked above");
        let err = |violation: TypeViolation, fails: &mut Vec<Violation>| {
            fails.push(Violation::Typing {
                app: idx,
                box_name: a.name(),
                violation,
            });
        };
        match a {
            App::Const { out, value } => {
                let tt = t(*out);
                if value.len() as u64 != tt.size() {
                    err(
                        TypeViolation::ConstPayloadSize {
                            expected: tt.size(),
                            got: value.len() as u64,
                        },
                        fails,
                    );
                }
                let want_rat = tt.dtype.kind == Kind::Rat;
                let is_rat = matches!(value, ConstPayload::Rat(_));
                if want_rat != is_rat {
                    err(
                        TypeViolation::ConstPayloadKind {
                            kind: tt.dtype.kind,
                            rational_payload: is_rat,
                        },
                        fails,
                    );
                }
            }
            App::Contract {
                inputs,
                output,
                extents,
                in_bindings,
                out_binding,
            } => {
                if inputs.len() != in_bindings.len() {
                    err(
                        TypeViolation::ContractArity {
                            inputs: inputs.len() as u32,
                            bindings: in_bindings.len() as u32,
                        },
                        fails,
                    );
                    return; // mirror: reference skips the rest on arity mismatch
                }
                let dt = t(*output);
                for (k, (&w, bnd)) in inputs.iter().zip(in_bindings).enumerate() {
                    let ti = t(w);
                    if ti.dtype != dt.dtype {
                        err(TypeViolation::ContractInputDtype { input: k as u32 }, fails);
                    }
                    if ti.shape.len() != bnd.len() {
                        err(
                            TypeViolation::ContractInputRank {
                                input: k as u32,
                                rank: ti.shape.len() as u32,
                                bindings: bnd.len() as u32,
                            },
                            fails,
                        );
                        continue;
                    }
                    for (ax, &b) in bnd.iter().enumerate() {
                        if b as usize >= extents.len() || ti.shape[ax] != extents[b as usize] {
                            err(
                                TypeViolation::ContractInputExtent {
                                    input: k as u32,
                                    axis: ax as u32,
                                    extent: ti.shape[ax],
                                    index: b,
                                },
                                fails,
                            );
                        }
                    }
                }
                if dt.shape.len() != out_binding.len() {
                    err(
                        TypeViolation::ContractOutputRank {
                            rank: dt.shape.len() as u32,
                            bindings: out_binding.len() as u32,
                        },
                        fails,
                    );
                } else {
                    for (ax, &b) in out_binding.iter().enumerate() {
                        if b as usize >= extents.len() || dt.shape[ax] != extents[b as usize] {
                            err(
                                TypeViolation::ContractOutputExtent {
                                    axis: ax as u32,
                                    index: b,
                                },
                                fails,
                            );
                        }
                    }
                }
            }
            App::Cast { inp, out } => {
                if t(*inp).shape != t(*out).shape {
                    err(TypeViolation::CastShape, fails);
                }
            }
            App::Cell {
                complex_idx,
                inp,
                outs,
            } => {
                let Some(c) = self.complexes.get(*complex_idx as usize) else {
                    fails.push(Violation::ComplexIndexOutOfRange {
                        app: idx,
                        complex_idx: *complex_idx,
                        pool_len: self.complexes.len() as u32,
                    });
                    return;
                };
                let ti = t(*inp);
                if ti.shape != [c.amb_dim as u64] || ti.dtype.kind != Kind::Rat {
                    err(
                        TypeViolation::CellInputType {
                            amb_dim: c.amb_dim,
                            got: ti.clone(),
                        },
                        fails,
                    );
                }
                if outs.len() != c.cells.len() {
                    err(
                        TypeViolation::CellArity {
                            outs: outs.len() as u32,
                            cells: c.cells.len() as u32,
                        },
                        fails,
                    );
                }
                for &w in outs {
                    if t(w) != &TensorType::boolean([]) {
                        err(TypeViolation::CellOutputNotBoolScalar { wire: w }, fails);
                    }
                }
            }
            App::Reshape {
                new_shape,
                inp,
                out,
            } => {
                let (ti, to) = (t(*inp), t(*out));
                if ti.size() != to.size() || &to.shape != new_shape {
                    err(
                        TypeViolation::ReshapeSize {
                            from_size: ti.size(),
                            to_size: to.size(),
                        },
                        fails,
                    );
                }
                if ti.dtype != to.dtype {
                    err(TypeViolation::ReshapeDtype, fails);
                }
            }
            App::Broadcast { to_shape, inp, out } => {
                let (ti, to) = (t(*inp), t(*out));
                if !broadcasts_to(&ti.shape, to_shape) || &to.shape != to_shape {
                    err(
                        TypeViolation::BroadcastShape {
                            from: ti.shape.clone(),
                            to: to_shape.clone(),
                        },
                        fails,
                    );
                }
                // NOTE deliberate mirror: the reference implementation
                // does not constrain broadcast's dtype; neither do we.
            }
            App::Copy { inp, outs } => {
                for &w in outs {
                    if t(w) != t(*inp) {
                        err(TypeViolation::CopyOutputType { wire: w }, fails);
                    }
                }
            }
            App::Delete { .. } => {}
            App::Swap {
                in1,
                in2,
                out1,
                out2,
            } => {
                if t(*out1) != t(*in2) || t(*out2) != t(*in1) {
                    err(TypeViolation::SwapType, fails);
                }
            }
            App::Stack { inputs, output } => {
                if inputs.is_empty() {
                    // The reference implementation indexes inputs[0] and
                    // would raise; report the shape clause instead: an
                    // empty stack's output cannot have a leading axis of
                    // extent 0 matching "no common input type".
                    err(
                        TypeViolation::StackOutputShape {
                            got: t(*output).shape.clone(),
                            expected: vec![0],
                        },
                        fails,
                    );
                    return;
                }
                let t0 = t(inputs[0]);
                for &w in &inputs[1..] {
                    if t(w) != t0 {
                        err(TypeViolation::StackInputType { wire: w }, fails);
                    }
                }
                let to = t(*output);
                let mut expected = Vec::with_capacity(t0.shape.len() + 1);
                expected.push(inputs.len() as u64);
                expected.extend_from_slice(&t0.shape);
                if to.shape != expected {
                    err(
                        TypeViolation::StackOutputShape {
                            got: to.shape.clone(),
                            expected,
                        },
                        fails,
                    );
                }
                if to.dtype != t0.dtype {
                    err(TypeViolation::StackDtype, fails);
                }
            }
        }
    }
}
