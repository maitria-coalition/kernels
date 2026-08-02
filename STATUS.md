# STATUS — assembly coverage

This repository is populated by deliberate re-authoring from a private
development lineage (see README, *Status*). This table is the current
coverage.

| component | status |
|---|---|
| `README.md`, `ENGINEERING.md`, `LICENSING.md`, `ci.sh` | here |
| `crates/maitria-kernels` — `sweep` family: `first_violation`, `minmax`; reference + NEON + AVX2 lanes; conformance battery | here |
| `receipts/` — aarch64 (Grace/GB10, NEON) and x86-64 (Skylake-W, AVX2) lane receipts; batteries green on both boxes | here |
| Bernstein-basis table transforms (box/simplex forward-difference towers, subdivision, degree elevation) | re-authoring queued — arrives with its exact-arithmetic substrate |
| exact-rational limb-plane batch substrate (integer/rational batch add · mul · cmp · dot; the promotion ladder) | re-authoring queued |
| `gpu/` — contraction-row GPU/XLA lanes: PROVISIONAL row IR (`rowir.py`, with the DATERWI verdict type + `fits_i64`/`fits_f64` gates), definitional Python reference, XLA-emission lane, FFI CUDA microkernel (rows as data; aliased in-place accumulate), Pallas/Mosaic-GPU inline-PTX lane (`mad.lo.s64`; directed-rounding interval adds), f64 interval family with exact-rational oracle; hypothesis battery + on-hardware receipts | here (first arrival; the canonical row IR still follows the compiler wave — `gpu/rowir.py` is marked provisional and everything reads rows through its surface) |
| `gpu/` — **interval-enclosure contraction family** (the VCARM consumer fast-path): the same rows-as-data kernel over the enclosure semiring — lo/hi f64 pairs, per-instruction outward rounding, branch-free four-corner product (case table + monotonicity equivalence worked in `gpu/README.md`), directed accumulation, on-device three-state verdict reduction (pass / fail-with-witness / INCONCLUSIVE-defer); reference strata (exact-rational oracle, ideal enclosure, op-for-op directed mirror), fesetround host lane, fused-Pallas interval hadamard-accumulate; battery: soundness in exact arithmetic, ideal sandwich, three-corner bitwise parity, verdict truth | here |
| `crates/maitria-kernels` — **`nullspace` family**: exact sparse integer nullspace via mod-p elimination + CRT lift — canonical per-prime basis contract (leftmost rank profile, identity-on-free form), dense reference + sparse lane (linear pivoting, hybrid dense core, Montgomery axpy, typed fill-budget refusal), two-channel ratrec lift with typed capacity refusal, exact i128 re-verification; battery: reference/sparse bit-identity across lucky and unlucky primes, independent big-rational partner end-to-end, designed mutants, witness validity | here |
| `crates/maitria-kernels` — **`rnsfold` family**: batched exact linear-combination equality over residue channels (the RNS lowering of the certificate L-int fold) — descriptor with in-module bound-driven channel selection, 63-bit prime table, scalar reference lane; battery with an independent big-integer partner, bound-inequality property, planted multi-prime-divisible adversary, Miller–Rabin over the table | here |
| `crates/maitria-kernels-cuda` — first GPU-lane sibling crate (cudarc + NVRTC, pipeline-of-record per ENGINEERING #6 with a deployed-PTX dump hook for the disassembly acceptance item): the `rnsfold` CUDA lane, Montgomery channels, conformance-gated bit-identical to the core reference | here |
| `crates/maitria-kernels-cuda::host` — the shared NVRTC/driver-JIT host harness (device bring-up, cached compile/load, arch-suffix + define variants, verbatim-PTX load, launch helpers, zero-length-upload guard, typed errors): both lanes above ride it; sibling host libraries import it instead of re-writing the plumbing. The *transitional* host pattern — see the module docs and `crates/maitria-kernels-xla/DESIGN.md` for how it sits relative to the XLA custom-call pattern | here |
| `crates/maitria-kernels-xla` — the Rust host for the XLA custom-call lanes (PJRT plugin loading, FFI handler registration, StableHLO-text emission around the existing `gpu/ffi` handler library) | design committed (`DESIGN.md`); probe-first order stated there; no code yet |
| `gpu/` — **checker embed** (`ivl_screen.py`): per-row jit-cached `IvlRowCheck` programs (descriptor static, verdict scalars read late — retrace-free candidate screening), the DATERWI screening ladder (float screen → exact-rational authority; fail witnesses exactly re-derived via single-entry exact evaluation), and the contract-shaped row seam (`row_from_bindings`, diagonal-embedded outputs handled with exact off-diagonal classification); battery: ladder truth against the exact stratum, confirm-mode semantics, contract-seam ground truth | here |
| contraction-row data model, canonical form (the multi-target lowering spine as the compiler wave settles it) | re-authoring queued; `gpu/rowir.py` is the deliberately-thin stand-in |
| CUDA lane for the Bernstein/exact-rational families (pipeline-of-record + disassembly acceptance per ENGINEERING #6) | re-authoring queued; `gpu/ffi/` is the embedding pattern it will ride |
| XLA lane, full emission (JAX-side programs consuming the settled row IR) | re-authoring queued; mechanics settled by measured recon, `gpu/xla_lane.py` is the first emitted fragment |

Deliberately absent, by the admission test (README): certified
decision procedures and anything a soundness lemma names (→ `qtsl`);
producer-only search/prefilter machinery (→ `mtk`); consumer engine
strategy (→ `geolog-alpha`).
