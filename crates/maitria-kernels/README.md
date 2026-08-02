# maitria-kernels

The crate. Kernel families live as modules; every family carries its
reference implementation, its lanes, and its conformance battery. The
repository-level README states the admission test and the lane law;
`ENGINEERING.md` states the commitments this crate is reviewed
against.

## `sweep` — first-violation sign sweeps and enclosure folds

The coefficient-sweep verdict shape: given a slab of exact integers,
find the first index violating a one-sided sign predicate
(`first_violation`), or fold the slab to its exact range with
first-occurrence extremum indices (`minmax`).

**The shared-denominator reduction** (why integer kernels decide
rational questions): the consumers' Bernstein-basis decision
procedures compare *rationals* against sign predicates and extract
rational ranges. Over a table cleared to one shared positive
denominator — the representation the exact-arithmetic batch
substrate produces natively (integer numerator planes over one
positive denominator per table) — the sign of every value is the sign
of its numerator, and the argmin/argmax of the values are the
argmin/argmax of the numerators. So the verdict-relevant loop is an
integer scan, exactly what these kernels are. Values whose numerators
exceed 64 bits are the caller's promotion ladder's business
(ENGINEERING #4): packing is fit-checked upstream; nothing here
rounds.

**Both sides use the same scan** (the admission test's clause 1,
witnessed): the checking side reads `first_violation == None` as the
accept verdict of a coefficient-sign side condition; the synthesis
side reads the same returned index as the counterexample witness that
steers refinement. One kernel, two readers.

**Not proof-coupled** (clause 2): the machine-checked soundness
results name their own certified implementations (in `qtsl`), not
this code. This kernel's assurance instruments are the in-tree
battery and the system-level cotest regime.

**VCARM/DATERWI status** (ENGINEERING #9-#10): vacuously clean. The
sweep family is exact-i64 end to end — no floating point exists on
any path, so there are no rounding modes to be careful about; and the
upstream fit-check refusal *is* the DATERWI deferral, implemented at
packing time: values that cannot be represented exactly never reach
these kernels, they ride the callers' promotion ladder instead.

### Differential partners (ENGINEERING #5)

- the scalar reference (`sweep::reference`), always compiled;
- an independently-derived formulation inside the battery itself
  (iterator-combinator min/max + position search — a different
  algorithm computing the same answers);
- downstream, the system-level cotest against the certified checking
  surface, operated by the consumers.

### Lanes

- `reference` — portable scalar, the semantics.
- `neon` (aarch64) — 8-wide unrolled block scans; NEON is baseline on
  aarch64, stable Rust.
- `avx2` (x86-64) — 4-wide block scans behind runtime feature
  detection, scalar fallback.

Dispatch is by `sweep::first_violation` / `sweep::minmax`;
`sweep::active_lane()` reports the choice (ENGINEERING #7). Per-lane
entry points stay public for battery use and explicit pinning.
Receipts: `../../receipts/`.
