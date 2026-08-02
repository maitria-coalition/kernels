# checker-embed receipts — 2026-07-20 (sm_120, RTX 5080)

Box: RunPod community RTX 5080 16 GB (sm_120), driver 580.159.03;
jax 0.11.0 (cuda12 wheels) on the image's own Python 3.12 via uv venv;
nvcc 12.9 offline for the FFI lane (`gpu/ffi/build.sh`, `-arch=native`).
SASS branch count at this build: **207** — against the founding
baseline of 391 built with nvcc **12.8**: the count is
toolchain-version-dependent (same `.cu`, same arch class, different
compiler release), so BRA baselines are per-toolchain facts; the
semantic gate is the battery, which is what actually re-verified.

## Battery (semantics)

`python -m pytest gpu/tests/ -q` on-box: **50 passed, 0 skipped, 100
hypothesis examples per property, 100.7 s** — the 33 prior properties
plus the checker-embed battery (`test_ivl_screen.py`), all
device-parametrized instruments live:

- ladder truth on generated rows with knife-edge demands, device lane
  included: final verdicts checked against exact rational ground
  truth; every fail witness re-derived exactly;
- `confirm="all"`: final verdict kind equals the full exact path's;
- device `RowChecker` bitwise-equal to the unjitted wrapper (outputs
  AND verdicts, pass and fail arms);
- **jit-cache instrument: one compiled entry across repeated checks**
  (`_cache_size() == 1`) — the retrace-free claim, asserted, and
  re-asserted by the bench below across 26 calls.

Instrument verification (per the family's standing lesson that the
instruments need adversarial review more than the kernels): two
deliberate mutations before the first commit — exact-deferral removed,
wrong witness planted — both caught by the battery's assertions.

## Embed economics (medians of 25, all-pass demands, ivl matmul 256^3)

| row | path | median ms |
|---|---|---:|
| ivl matmul 256^3 | jitted EVAL (context: the kernel floor) | 1.779 |
| ivl matmul 256^3 | wrapper (unjitted `check_row_interval_ffi`, sync per candidate) | 2.817 |
| ivl matmul 256^3 | `RowChecker` (jit-cached, sync per candidate) | 2.119 |
| ivl matmul 256^3 | `RowChecker` (batched `check_arrays`, verdict scalars read late) | 2.182 |

jit cache entries after 26 checks: **1**.

## Honest readings

- **The embed removes the python/trace overhead, ~0.70 ms/candidate
  here (25%)**: 2.817 → 2.119 ms. Check-over-eval prices at ~0.34 ms
  on this box (2.119 vs 1.779) — same order as the founding ~55 µs +
  sync, on a slower part with a different toolchain.
- **Late-read batching bought nothing further on this box** (2.182 ≈
  2.119): at 65K entries × red 256 the device kernel itself is the
  pipeline floor (1.78 ms), and jax's async dispatch already overlaps
  the per-candidate sync with the next kernel. The late-read pattern's
  headroom lives where kernels are SMALL relative to dispatch — the
  founding box measured that regime (wrapper 3.8 vs kernel 1.0); this
  box's regime is kernel-bound. Both receipts stand, each naming its
  regime.
- **The founding 5090 receipt and this 5080 receipt disagree on the
  wrapper's absolute overhead** (2.75 ms there, 0.70 ms here). Not
  reconciled on-box (different GPU, different nvcc, different host);
  the embed's structural claim — one compile, no retrace, verdict
  read decoupled from dispatch — is box-independent and is what the
  battery asserts.

## Spend

Pod `d1px2782j0vp4l` (rented GPU box), $0.39/hr, created 02:29 BST
(three 5090 draws refused — community stock; 5080 first draw landed),
destroyed 02:40 BST verified-by-poll (404). **~0.2 h ≈ $0.08.**
Registry row added at provision, pruned at teardown
(`boxes.tsv.bak-w506embed`).
