# kernels — shared multi-architecture fast paths

**kernels** is the performance substrate shared by the producer toolkit
(`mtk`) and the database's ground decision procedures (`geolog-alpha`):
the computational structures that *both* sides of a certificate-checked
verification system build and evaluate, implemented once, per
architecture, behind one conformance discipline.

The system's trust architecture (see the
`geolog-alpha`
README and the `qtsl`
repository) splits the world into
untrusted producers and a small verified checking surface. Between
them sits a large body of computation that is neither: Bernstein-basis
table arithmetic, exact-rational batch verification, semiring
contraction of sparse relational tensors. The producer runs these
structures forward to *propose* certificates; the consumer re-derives
the same structures to *screen and dispatch* them ahead of certified
checking. Implementing that shared middle twice — once per side, per
architecture — is how systems drift. This repository is the single
home.

## The admission test

A kernel belongs here iff **both** clauses hold:

1. **Shared by producer and consumer.** The same computational
   structure is built or evaluated on the synthesis side (candidate
   generation, LP/tensor assembly, refinement search) *and* on the
   checking side (side-condition decision procedures, batch
   screening). Producer-only conveniences — float prefilters whose
   only role is candidate pruning, search heuristics, training
   machinery — stay in `mtk`. Consumer-only certified cores stay out
   (clause 2).
2. **Not proof-coupled.** If a machine-checked soundness lemma names
   an implementation, that implementation lives beside its lemma in
   `qtsl` — mechanization-side coupling outranks performance-side
   grouping. What lives here is *performance with external checking*:
   every kernel's assurance instrument is a differential battery
   against an in-tree reference (and, transitively, the system-level
   cotest regime), never a proof about the kernel itself.

Both clauses are decidable per kernel, and future kernels
self-adjudicate against them; a kernel that fails either clause is
misfiled, not grandfathered.

## The lane law

Inherited from the ground kernel's feature-flag doctrine
(`geolog-alpha` README) and binding on every kernel here:

> An acceleration lane may change *cost*, never *verdicts*. Every lane
> is conformance-gated against the always-on reference by differential
> batteries; a lane that cannot pass its battery does not ship.

Three consequences, made structural (see `ENGINEERING.md`):

- **Reference first.** Every kernel ships a scalar reference
  implementation — obviously correct, dependency-free, always
  compiled. A kernel without a reference does not exist here.
- **Receipts before dispatch.** A lane becomes the default on its
  architecture only with a measured receipt on real hardware,
  committed under `receipts/`. Vector code that does not win against
  the reference on its target stays available and un-dispatched.
- **Fit-detection, never approximation.** Fixed-width fast paths
  refuse inputs they cannot represent exactly (the caller's promotion
  ladder handles them); no rounding exists on any verdict-relevant
  path.

## Architecture coverage

[shipped] The implemented lanes and their evidence are:

| target | shipped lanes | evidence |
|---|---|---|
| portable CPU | scalar Rust references for `sweep`, `nullspace`, `rnsfold`, and `roweq` | always compiled by [`ci.sh`](ci.sh) |
| aarch64 CPU | NEON `sweep`; host directed-rounding interval lane | differential tests plus the [Grace/GB10 sweep receipt](receipts/sweep-2026-07-19.md#aarch64-neon-lane-dispatched) and [interval parity receipt](receipts/gpu-ivl-2026-07-20.md#battery-semantics) |
| x86-64 CPU | runtime-detected AVX2 `sweep` with scalar fallback; host directed-rounding interval lane | differential tests plus the [Skylake-W sweep receipt](receipts/sweep-2026-07-19.md#x86-64-avx2-lane-dispatched-where-detected) and [interval parity receipt](receipts/gpu-ivl-2026-07-20.md#battery-semantics) |
| NVIDIA GPU, Rust host | CUDA lanes for `rnsfold` and `roweq` | Rust [device/reference batteries](crates/maitria-kernels-cuda/tests/) |
| NVIDIA GPU, Python/JAX host | emitted-XLA and CUDA-FFI contraction rows; directed-rounding CUDA-FFI and Mosaic-GPU interval lanes | [`gpu/`](gpu/) batteries and the [contraction-row](receipts/gpu-row-2026-07-20.md) and [interval](receipts/gpu-ivl-2026-07-20.md) receipts |

Notable missing acceleration targets include SVE/SVE2, AVX-512,
RISC-V vectors, WebAssembly SIMD, AMD GPU, Intel GPU, Apple GPU/Metal,
Vulkan, and WebGPU. NVIDIA is the only GPU architecture with committed
on-device conformance and timing receipts. The Rust XLA custom-call
host is [argued] in
[`crates/maitria-kernels-xla/DESIGN.md`](crates/maitria-kernels-xla/DESIGN.md),
but no implementation ships yet.

Per-device-class *strategy* (what to fuse, what to keep resident, what
to recompute) belongs to the consumers; this repository provides the
lanes and their conformance instruments.

## Trust and auditability

This repository is not a mechanized trust root: no soundness theorem
names these implementations. It nevertheless has a soundness duty.
Consumer-side kernels must be sound in practice, including species for
which no separate certified checker re-derives the result. The defenses
are scalar or exact references, property-based differential batteries,
typed deferral to exact arithmetic, exact confirmation where the API
provides it, deterministic pipelines, and committed hardware receipts.
The `qtsl` checkers add a
further boundary only for the fragments they actually re-check.

## Shipped components

[shipped] The repository currently contains:

- **`crates/maitria-kernels`** contains four exact families under one
  reference-first API: `sweep` (first-violation and min/max folds),
  `nullspace` (sparse modular elimination with CRT lifting and exact
  re-verification), `rnsfold` (batched exact linear-combination equality
  over residue channels), and `roweq` (batched structural equality of
  sparse limb-plane rows). Fixed-width capacity and fill limits produce
  typed refusals rather than approximate answers.
- **`crates/maitria-kernels-cuda`** provides the shared CUDA host and
  verdict-identical device lanes for `rnsfold` and `roweq`. Its
  property-based batteries cover magnitude tiers, accepting examples,
  deliberate mutants, empty shapes, and descriptor padding.
- **`gpu/`** contains the contraction-row reference, emitted-XLA,
  CUDA-FFI, and Mosaic-GPU lanes, plus directed-rounding interval
  evaluation and a three-state screen that defers inconclusive cases to
  exact rational arithmetic. The interval-contraction receipt reports
  33/33 tests green, each property run with 100 generated examples, and
  bitwise agreement among device, directed reference, and host
  rounding-mode implementations.
- **`maitria-hypernet`** owns the canonical `AXHN0001` hypernet data
  model and codec shared by producers and checkers. Golden vectors pin
  canonical bytes, digest behavior, validating decode, and
  well-formedness refusals.

Committed [`receipts/`](receipts/) record the hardware, toolchains,
conformance runs, and timing measurements behind dispatch choices.

## The wider system

This repository is one of six:

- **[geolog-alpha](https://github.com/maitria-coalition/geolog-alpha)** —
  a local-first database whose tables are logical consequences; consumes
  these lanes behind its acceleration feature flags.
- **[qtsl](https://github.com/maitria-coalition/qtsl)** — the
  Quantitative Temporal Specification Logic: theories, rule catalogue,
  mechanized soundness developments, certified checkers, fixtures, and
  the book.
- **[mtk](https://github.com/maitria-coalition/mtk)** — the modelling
  toolkit: the untrusted producer side; consumes these lanes for
  candidate generation and batch assembly.
- **[qtsl-experiments](https://github.com/maitria-coalition/qtsl-experiments)** —
  the reproduction spine under experiment tables: producers,
  certificate fixtures, receipts, and table generators.
- **kernels** (here) — the shared fast paths.
- **[maitria.org](https://maitria.org)** — the front door: narrative
  site, guided reading paths, and book.

## Verification status

[shipped] The CPU workspace gate runs formatting, warning-free Clippy,
and the complete Rust test suite in both debug and release modes. The
CPU architecture receipts record green differential batteries on
aarch64 and x86-64; the GPU receipts record on-device batteries on
NVIDIA `sm_120`. GPU tests skip explicitly when no CUDA device is
available, so a CPU-only run is not evidence about a GPU lane.

[shipped] Optimized lanes are checked against independent references;
performance determines dispatch only after verdict agreement. The
published measurements show NEON and AVX2 wins for `sweep`, XLA winning
large integer contractions, CUDA custom calls competitive at small
contractions, and directed-rounding CUDA as the available sound device
path for interval contractions. These are scoped measurements from the
committed machines and shapes, not universal performance claims.

[argued] Any additional architecture lane is subject to the same
reference-first API, typed-refusal behavior, property-based differential
batteries, and hardware-receipt requirement before dispatch.

[aspirational] The missing acceleration targets listed above may gain
lanes as demand and hardware access warrant; no schedule is claimed.
