# gpu — contraction-row evaluation lanes (GPU / XLA)

The first arrival of the NVIDIA-GPU and XLA rows of the repository's
targets table: evaluate a **contraction row** — index map + output
set as data, never an eagerly expanded sum-product — on device, under
the same lane law as every CPU kernel here. The scalar carrier is s64
(two's-complement; the DATERWI gate below is what makes that sound),
plus an f64 directed-rounding interval family that exists to
demonstrate the rounding law.

## The provisional row IR

`rowir.py` defines the row form every module here reads:
`Row(extents, factors, out)` — extents of the index universe, one
axis-map per operand (repeated axes select diagonals; operand shapes
are *induced*, so maps are well-typed by construction), ordered
output axes. Axes hit by no factor must be outputs (broadcast axes);
an axis nobody reads is refused at construction. `pack_descriptor()`
is the wire form the CUDA kernel consumes.

**PROVISIONAL**, stated loudly: the compiler wave's architecture work
will settle the canonical row IR. Everything downstream reads rows
through this one module's surface, so re-targeting is one localized
change; nothing may grow dependencies on this exact encoding.

## The lanes

| lane | file | mechanism | scope |
|---|---|---|---|
| reference | `reference.py` | pure Python, zero dependencies, direct transcription of the contraction definition; exact integers wrapped once per entry | the semantics; every row |
| XLA emission | `xla_lane.py` | `jnp.einsum` + broadcast insertion — an independently-derived formulation, fusible into the surrounding XLA program | every row (m <= 26); s64 ONLY — no sound emission exists for enclosures (below) |
| FFI custom call | `ffi_lane.py` + `ffi/rowkernel.cu` | one compiled CUDA kernel; the row rides as data (packed descriptor), so new rows never touch device code; `RowAccum` is registered with `input_output_aliases` for in-place accumulate | rows with m <= 8, factors <= 4 (typed refusal to the XLA lane above the bounds) |
| Pallas / Mosaic-GPU | `pallas_lane.py` | literal-PTX islands (`inline_mgpu`) inside a fused kernel: `mad.lo.s64` for hadamard-accumulate; `add.rm.f64`/`add.rp.f64` for directed interval adds; the full interval hadamard-accumulate (`mul.rm/.rp`, `min/max.f64`, `add.rm/.rp`, ~18 islands/element) | elementwise normal forms, prod(shape) % 128 == 0 |
| interval family (elementwise) | `interval.py` + the same `.cu` | per-instruction directed rounding (PTX `.rm`/`.rp`), FFI-hosted; exact-rational oracle on the host; the shared IEEE-total directed scalar ops | the VCARM demonstration |
| **interval contraction references** | `ivl_reference.py` | three strata: exact rational interval semantics (the oracle), its directed rounding (the ideal — tightness's hard boundary), and the op-for-op directed mirror of the device fold (the bitwise parity anchor); plus the three-state classifier + verdict reduction | every row; exact/ideal need finite endpoints, the directed mirror is total on f64 |
| interval contraction, FFI | `ivl_ffi_lane.py` + the same `.cu` | the rows-as-data kernel over the enclosure semiring: lo/hi buffer pairs, every op directed per instruction; eval / aliased accumulate / **check** (on-device three-state verdict reduction: atomicMin over conclusively-failing indices + a straddle flag — the host reads two scalars, not an array) | rows with m <= 8, factors <= 4; conclusive-capable inside the `fits_f64` gate, verdict-sound everywhere |
| interval hadamard-accum, fused | `pallas_lane.py` (`interval_hadamard_accum_pallas`) | the enclosure semiring's streaming-update normal form as one fused Mosaic-GPU kernel, operation-identical to the FFI accumulate path (including its fresh-from-zero fold, mirrored as +0.0-immediate islands — bitwise parity by construction, signed zeros included) | elementwise, prod % 128 == 0, <= `IVL_HADAMARD_MAX_ELEMENTS` |
| host directed lane | `ivl_host_lane.py` | `fesetround` hardware rounding modes around plain scalar ops — a third mechanism computing the same fold (exact-emulation vs silicon vs device: a three-cornered witness); self-checks that directed rounding actually took effect, typed refusal where it didn't | any host whose libm honours C99 rounding modes; gates in CI with no GPU |
| **checker embed** | `ivl_screen.py` | the consumer boundary packaged: `RowChecker` (a per-row jit-cached program around `IvlRowCheck` — descriptor baked in static, verdict scalars read late, retrace-free candidate screening) + `screen_row` (the DATERWI ladder: float screen → exact-rational authority) + the contract-shaped seam (`row_from_bindings` / `screen_bound_contract`, diagonal-embedded outputs included) | every row; the device lane inside the descriptor bounds, host lanes anywhere |

Battery: `tests/` (property-based; bit-equality on s64 at full i64
range including wraparound, diagonals, broadcast axes, and the empty
tensor product; bitwise oracle-equality + the enclosure property for
the interval family; for the interval contraction: soundness against
the exact stratum, the ideal sandwich, bitwise three-corner parity,
and verdict truth — conclusive arms re-proven against exact
arithmetic, deliberate cancellation straddles deferring as
INCONCLUSIVE). Receipts: `../receipts/gpu-row-*.md`,
`../receipts/gpu-ivl-*.md`.

## The rounding law, worked (ENGINEERING #9–#10)

The engineering commitments' VCARM law has an architectural
consequence on GPU, and this directory is its demonstration:

- **Sound consumer float arithmetic cannot ride emitted XLA ops.**
  A fusion compiler's rewrite semantics are not contractual: no XLA
  op carries a rounding mode, and fusion may reassociate. For exact
  integer rows this is harmless (the battery proves bit-identity);
  for any float enclosure it is fatal.
- **It lives where rounding pins per instruction.** The custom-call
  lane (`ivl_addmul_f64`: PTX `add.rm.f64`/`add.rp.f64`/`mul.rm.f64`/
  `mul.rp.f64` — `.rm`/`.rp` are PTX's spellings of round-toward-minus/
  plus-infinity, the `_rd`/`_ru` of the CUDA intrinsics) and the
  Pallas `inline_mgpu` island (`interval_add_pallas`, same
  instructions inside a *fused* kernel). This upgrades the inline-PTX
  escape hatch from a speed mechanism to a **soundness mechanism**:
  it is how rounding-controlled arithmetic participates in fused
  programs at all.
- **Directed rounding has a jnp twin nowhere, and an oracle
  everywhere.** The interval battery checks device outputs bitwise
  against exact rational arithmetic followed by the unique correctly
  directed rounding (`interval.py`), and checks lo <= exact <= hi in
  exact arithmetic — outward rounding is how enclosures stay
  enclosures.
- **DATERWI is plumbed as a type.** `rowir.Verdict` is three-state —
  conclusive-pass / conclusive-fail(witness) / **inconclusive** — and
  inconclusive is never silently rounded into a decision: it routes
  to the caller's exact-rational path. For the s64 lanes the gate is
  `Row.fits_i64(operand_bounds)`: a conservative exact bound decides
  whether wrapped-64-bit equals exact (conclusive) or the row must
  defer. The architecture work inherits this as a requirement on the
  settled IR, not a suggestion.

## Interval multiplication under directed rounding, worked honestly

The enclosure semiring's product is the hull of the corner products:
for $[a,b] \otimes [c,d]$ (with $a \le b$, $c \le d$), the exact
result is $[\min P, \max P]$ over $P = \{ac, ad, bc, bd\}$ — a
bilinear function of each variable attains its extremes at box
corners, and the image of a connected box is an interval, so the
corner hull IS the product set, exactly.

The classical nine-case sign table picks the extremal corners without
computing all four (write $\downarrow$/$\uparrow$ for round-toward
$-\infty$/$+\infty$; N: $b \le 0$, M: $a < 0 < b$, P: $0 \le a$):

| $[a,b]$ | $[c,d]$ | lo ($\downarrow$) | hi ($\uparrow$) | products |
|---|---|---|---|---|
| P | P | $a c$ | $b d$ | 2 |
| P | M | $b c$ | $b d$ | 2 |
| P | N | $b c$ | $a d$ | 2 |
| M | P | $a d$ | $b d$ | 2 |
| M | M | $\min(a d,\, b c)$ | $\max(a c,\, b d)$ | 4 |
| M | N | $b c$ | $a c$ | 2 |
| N | P | $a d$ | $b c$ | 2 |
| N | M | $a d$ | $a c$ | 2 |
| N | N | $b d$ | $a c$ | 2 |

The kernels ship the **branch-free four-product form** instead: all
four corners rounded down folded by `fmin`, all four rounded up
folded by `fmax`. The two forms compute the same enclosure, and the
brute force loses NOTHING in tightness — directed rounding is
monotone ($x \le y \Rightarrow \downarrow x \le \downarrow y$), so

$$\min_i \downarrow p_i \;=\; \downarrow \min_i p_i
\qquad
\max_i \uparrow p_i \;=\; \uparrow \max_i p_i$$

i.e. the four-product form yields exactly the directed rounding of
the exact corner extreme — per-operation ideal, same as the table
with every selected product rounded in its direction. What the brute
force buys on GPU: **zero divergence** (no sign tests in a warp), no
boundary bookkeeping (a class boundary like $a = 0$ belongs to two
rows of the table, whose selected corners agree there — a fact the
table's user must prove and the brute force never needs), and
deterministic signed-zero behaviour (device `fmin`/`fmax` order
$-0 < +0$, so equal-valued zero corners resolve the same way in
every lane — the batteries assert this bitwise). The cost is 8
multiplies instead of 2–4; receipts price it. The sign-case table
remains the optimization ladder if a receipt ever demands it — this
section is the correctness argument it will be measured against.

Addition needs no cases: $[a,b] \oplus [c,d] =
[\downarrow(a+c),\, \uparrow(b+d)]$, endpointwise and directed.

## The verdict boundary (DATERWI at the checking edge)

`IvlRowCheck` classifies every output enclosure against a per-entry
demanded bound $[d_{lo}, d_{hi}]$ with two positive predicates —
pass: $d_{lo} \le lo \,\wedge\, hi \le d_{hi}$; fail: $hi < d_{lo}
\,\vee\, lo > d_{hi}$; otherwise INCONCLUSIVE — and reduces on
device (atomicMin over conclusively-failing indices, one flag for
straddles). Soundness facts, load-bearing and battery-checked:

- both conclusive predicates quantify over a superset of the possible
  values, so they are true of the exact semantics whenever the
  enclosure is sound (the battery re-proves each conclusive verdict
  in exact rational arithmetic, not merely against another lane);
- NaN endpoints (possible only outside the `fits_f64` gate, from
  $0 \cdot \infty$ corners of half-unbounded intermediates) fall
  through both positive predicates into INCONCLUSIVE — a voided
  enclosure claims nothing;
- saturated endpoints ($\pm\infty$ from overflow under directed
  rounding: roundTowardNegative never yields $+\infty$, dually for
  the other side) are honest half-unbounded enclosures, still sound;
- the fail witness is the LOWEST CONCLUSIVELY-failing index; entries
  before it may be inconclusive, so it is a genuine violation
  witness but not necessarily the first violation in ground truth —
  first-violation semantics live on the exact path
  (`check_row_interval_exact`, whose fail answers the universal
  property and is deliberately the weaker predicate).

`Row.fits_f64(bounds)` is the routing gate (the float sibling of
`fits_i64`): inside it no endpoint can leave the finite range, the
directed references mirror the device bitwise, and the lane is worth
dispatching before the exact path; outside it the lane stays
verdict-sound but conclusive answers thin out. A False is a routing
fact, not an error.

## The checker embed (`ivl_screen.py`)

The consumer packaging of the verdict boundary, for a checking
program that holds contraction rows and screens candidates:

- **`RowChecker`** — one jit-compiled program per row (the packed
  descriptor is a static constant; the FFI call sits inside
  `jax.jit`), so screening many candidates against one row is
  retrace-free and pays kernel price, with the two verdict scalars
  staying on device until `verdict()` reads them — the
  embed-jitted-read-late pattern the founding receipts priced.
- **`screen_row`** — the DATERWI ladder around any screen lane
  (device / `fesetround` host / directed reference): conclusive-fail
  is re-derived at the witness entry in exact rational arithmetic
  (cost one reduction domain — `exact_entry`); INCONCLUSIVE always
  defers to the full exact path; conclusive-pass rests on the VCARM
  enclosure argument, or is re-proven exactly under `confirm="all"`.
  The returned verdict is always conclusive, and a float fail that
  ever failed its exact confirmation would be flagged and replaced by
  the exact verdict, never propagated — soundness does not rest on
  float silicon.
- **the contract seam** — `row_from_bindings` accepts row data in
  contract shape `(extents, in_bindings, out_binding)`; repeated
  output axes (diagonal-embedded outputs) build the class-form row,
  and `screen_bound_contract` classifies the structurally-zero
  off-diagonal entries host-side in exact arithmetic — no float
  screen participates in those verdicts at all.

## Pipelines of record (ENGINEERING #6)

Two different instruments, named per lane: the FFI lane ships what
**offline nvcc** assembles (`ffi/build.sh`; the disassembly
branch-count is printed at build and recorded in receipts); the
Pallas lane ships what the **jaxlib Mosaic-GPU MLIR pipeline**
assembles. Acceptance timings run the deployed pipeline per lane —
the two demonstrably disagree on optimization behaviour in general.

## Measured substrate limits (jax 0.11.0, sm_120), for successors

- Mosaic-GPU layout inference on s64 handles full-array elementwise
  chains; it fails on matmul formulations (3D broadcast+sum, unrolled
  outer products with or without `layout_cast`, `fori_loop` +
  `dynamic_slice` hits an unimplemented lowering). Hence the Pallas
  lane's elementwise scope — a substrate fact, not a taste choice.
- `WG_STRIDED` needs prod(shape) divisible by the 128-lane warpgroup.
- jax >= 0.11 requires Python >= 3.12; a 3.11 image silently caps at
  jax 0.10.x (pip resolves the older release with no warning).

## Hosts for this pattern

The Python host here (via `jax.ffi`) is the pattern's first driver,
and its lanes are the reference semantics. A **Rust host** for the
same handler library — PJRT plugin loading, FFI handler registration,
StableHLO-text emission, no Python in the loop — is designed at
`crates/maitria-kernels-xla/DESIGN.md`; the freestanding NVRTC host
these custom calls do *not* need lives at
`crates/maitria-kernels-cuda::host` (the transitional pattern; the
module docs state the relation).

## Running

```
# on a CUDA box, in a py>=3.12 env with jax[cuda12] and hypothesis:
(cd ffi && bash build.sh)          # offline nvcc; prints SASS BRA count
python -m pytest tests/ -q         # the conformance battery
python -m gpu.bench                # the receipts table (run from repo root)
```
