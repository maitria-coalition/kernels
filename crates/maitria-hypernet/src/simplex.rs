//! Catalogue: simplicial-complex payloads of the `cell` box (H-CELL's
//! branching schemas; `qtsl-hypernets.tex` §"Branching: cell";
//! `acasxu/hypernet.py::SimpC`, Lean `SimpC`).

use std::collections::BTreeSet;

use crate::scalar::WitRat;

/// A vertex-index set (simplex or extension mask). Ordered so iteration
/// is deterministic; serialized as an LSB-first bitset.
pub type VSet = BTreeSet<u32>;

/// Content identity of a complex — everything except the hint.
pub(crate) type SimpCKey<'a> = (
    u32,
    &'a Vec<Vec<WitRat>>,
    &'a Vec<VSet>,
    &'a Vec<(VSet, VSet)>,
);

/// Catalogue: simplicial complex — the payload of a `cell` application.
///
/// Vertices carry exact-ℚ coordinates; a *cell* is a pair
/// (simplex mask, extension mask): the region of points reachable as
/// affine combinations of the simplex's vertices with nonnegative
/// weights on the core vertices and nonpositive weights on the
/// extension vertices (each extension vertex opens the region outward
/// through the facet opposite it).
///
/// `hint` is evaluator metadata: excluded from equality, from the
/// canonical serialization, and hence from the digest (mirrors the
/// python reference's `compare=False` field).
#[derive(Debug, Clone)]
pub struct SimpC {
    pub amb_dim: u32,
    /// Vertex coordinates, `amb_dim` exact rationals each.
    pub vertices: Vec<Vec<WitRat>>,
    /// Downward-closed, conformal simplex list (validated by
    /// [`SimpC::validate`], enforced by the well-formedness preflight).
    pub simplices: Vec<VSet>,
    /// Cells as (simplex mask, extension mask) pairs; `cell` output
    /// order = this list's order.
    pub cells: Vec<(VSet, VSet)>,
    /// Evaluator hint — not compared, not serialized, not digested.
    pub hint: String,
}

impl PartialEq for SimpC {
    fn eq(&self, other: &SimpC) -> bool {
        self.content_key() == other.content_key()
    }
}
impl Eq for SimpC {}

impl SimpC {
    pub fn new(
        amb_dim: u32,
        vertices: Vec<Vec<WitRat>>,
        simplices: Vec<VSet>,
        cells: Vec<(VSet, VSet)>,
    ) -> SimpC {
        SimpC {
            amb_dim,
            vertices,
            simplices,
            cells,
            hint: String::new(),
        }
    }

    /// Content identity (everything except `hint`) — the dedup key the
    /// canonical renumbering uses for the complex pool.
    pub(crate) fn content_key(&self) -> SimpCKey<'_> {
        (self.amb_dim, &self.vertices, &self.simplices, &self.cells)
    }

    /// Structural validation, witness-shaped (mirrors
    /// `acasxu/hypernet.py::SimpC.validate` clause for clause; an empty
    /// return is a *preflight* pass, never a verdict).
    pub fn validate(&self) -> Vec<SimpCViolation> {
        let mut v = Vec::new();
        let n = self.vertices.len();
        for (i, vert) in self.vertices.iter().enumerate() {
            if vert.len() != self.amb_dim as usize {
                v.push(SimpCViolation::VertexDimMismatch {
                    vertex: i as u32,
                    len: vert.len() as u32,
                    amb_dim: self.amb_dim,
                });
            }
        }
        let simplex_set: BTreeSet<&VSet> = self.simplices.iter().collect();
        for (si, s) in self.simplices.iter().enumerate() {
            if s.is_empty() {
                v.push(SimpCViolation::EmptySimplex { simplex: si as u32 });
                continue;
            }
            if let Some(&bad) = s.iter().find(|&&i| i as usize >= n) {
                v.push(SimpCViolation::VertexOutOfRange {
                    simplex: si as u32,
                    vertex: bad,
                });
                continue;
            }
            // closure: every one-vertex-removed face is present
            if s.len() > 1 {
                for &v0 in s {
                    let mut face = s.clone();
                    face.remove(&v0);
                    if !simplex_set.contains(&face) {
                        v.push(SimpCViolation::MissingFace {
                            simplex: si as u32,
                            removed_vertex: v0,
                        });
                    }
                }
            }
        }
        // conformity: nonempty pairwise intersections are simplices
        for (i, s1) in self.simplices.iter().enumerate() {
            for (j, s2) in self.simplices.iter().enumerate() {
                let inter: VSet = s1.intersection(s2).copied().collect();
                if !inter.is_empty() && !simplex_set.contains(&inter) {
                    v.push(SimpCViolation::MissingIntersection {
                        simplex_a: i as u32,
                        simplex_b: j as u32,
                    });
                }
            }
        }
        for (ci, (mask, ext)) in self.cells.iter().enumerate() {
            if !simplex_set.contains(mask) {
                v.push(SimpCViolation::CellMaskNotSimplex { cell: ci as u32 });
            }
            if !ext.is_subset(mask) {
                v.push(SimpCViolation::CellExtNotSubset { cell: ci as u32 });
            }
            if mask.difference(ext).next().is_none() {
                v.push(SimpCViolation::CellNoCoreVertex { cell: ci as u32 });
            }
        }
        v
    }
}

/// Witness-shaped violations of the complex discipline (which vertex,
/// which simplex, which cell — failure geometry is routing advice).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SimpCViolation {
    #[error("vertex {vertex} has {len} coordinates, ambient dimension is {amb_dim}")]
    VertexDimMismatch { vertex: u32, len: u32, amb_dim: u32 },
    #[error("simplex {simplex} is empty")]
    EmptySimplex { simplex: u32 },
    #[error("simplex {simplex} names vertex {vertex}, out of range")]
    VertexOutOfRange { simplex: u32, vertex: u32 },
    #[error("closure: simplex {simplex} minus vertex {removed_vertex} is not in the complex")]
    MissingFace { simplex: u32, removed_vertex: u32 },
    #[error("conformity: simplices {simplex_a} and {simplex_b} intersect outside the complex")]
    MissingIntersection { simplex_a: u32, simplex_b: u32 },
    #[error("cell {cell}: mask is not a simplex of the complex")]
    CellMaskNotSimplex { cell: u32 },
    #[error("cell {cell}: extension set is not a subset of the mask")]
    CellExtNotSubset { cell: u32 },
    #[error("cell {cell}: no core vertex (extension set equals the mask)")]
    CellNoCoreVertex { cell: u32 },
}
