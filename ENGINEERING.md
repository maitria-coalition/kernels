# Engineering commitments — binding on every kernel, every commit

These extend the sibling repositories' commitments (`geolog-alpha/
ENGINEERING.md`) with the obligations specific to a fast-path crate,
where the characteristic failure mode is not silent breakage but
*silent divergence*: an optimized lane that is almost always equal to
the reference. Any violation found is a bug regardless of whether the
code "works."

1. **Reference first.** Every kernel ships a scalar reference
   implementation: portable, dependency-free, written for obviousness
   over speed, always compiled on every target. The reference is the
   semantics; lanes are the economics.
2. **The lane law.** An acceleration lane may change cost, never
   verdicts. Every lane is conformance-gated against the reference by
   a property-based differential battery running in CI as a
   first-class gate — not a smoke test, a battery: boundary values,
   block-boundary positions for every vector width, degenerate and
   adversarial shapes. A lane that cannot pass its battery does not
   ship.
3. **Receipts before dispatch.** A lane becomes its architecture's
   default only with a measured receipt on real hardware, committed
   under `receipts/` with the machine, toolchain, shapes, and numbers.
   Passing the battery earns a lane *existence*; only a receipt earns
   it *dispatch*. A lane that loses to the reference on its own target
   stays available, un-dispatched, and marked — hand-vectorized code
   that must be faster by assumption is how fast-path crates rot.
4. **Fit-detection, never approximation.** Fixed-width paths detect
   inputs they cannot represent exactly and refuse them to the
   caller's promotion ladder. No rounding, no saturation, no
   wrap-and-hope exists on any verdict-relevant path. Exactness is a
   type-level fact where possible and a checked refusal everywhere
   else.
5. **Differential partners from the first commit.** Every kernel names
   its independent partners in its module documentation: at minimum
   the in-tree reference, plus an independently-derived formulation in
   the battery itself (a different algorithm computing the same
   answer), plus — where one exists — the system-level partner it is
   cotested against downstream. A kernel without named partners is not
   done, it is unverified.
6. **Pipeline-of-record for JIT-compiled lanes.** The same GPU kernel
   source can compile branch-free under an offline pipeline and
   branchy — an order of magnitude slower — under a runtime-JIT
   pipeline; the two are different instruments and demonstrably
   disagree. Every GPU kernel names its deployed compilation pipeline,
   acceptance benches run that pipeline, and a disassembly
   branch-count check is a standing acceptance item for JIT lanes.
7. **No silent anything.** An unsupported input is a typed refusal
   naming the input; `unwrap()` outside tests is a review flag; a lane
   selected by dispatch is observable (`active_lane()`), never
   inferred.
8. **Provenance of every claim.** Performance claims carry citations
   to committed receipts; design claims carry citations to in-repo
   documentation; deviations are marked DEVIATION with a reason.
9. **VCARM — Very Careful About Rounding Modes.** The checking-side
   engine must be sound even where it is not *verifiably* sound down
   to an ISA model — so its kernels are Very Careful About Rounding
   Modes (VCARM) and Defer Arithmetic To Exact Rationals When
   Inconclusive (DATERWI). The same properties are likely desirable
   for producer-side kernels, but are mandatory only on the checking
   side. Binding form: for any kernel a
   **consumer** (checking-side) lane evaluates, floating-point
   arithmetic is MANDATORILY rounding-controlled — directed or pinned
   rounding chosen per operation, no fast-math flags, no reassociating
   reductions, deterministic evaluation order. Outward rounding
   (toward minus infinity for lower bounds, toward plus infinity for
   upper bounds) is how enclosures stay enclosures. Architectural
   consequence, worked in `gpu/README.md`: rounding-controlled
   arithmetic cannot ride a fusion compiler whose rewrite semantics
   are not contractual; on GPU it lives where rounding pins
   per-instruction (hand kernels behind custom calls; inline-PTX
   islands inside fused kernels — which upgrades that escape hatch
   from a speed mechanism to a soundness mechanism). For
   **producer**-side lanes both laws are RECOMMENDED, not mandatory —
   producer-side unsoundness only wastes candidates, since the checker
   catches them; agreement buys replay-identical debugging — and any
   divergence is documented per kernel with that tradeoff argued.
10. **DATERWI — Defer Arithmetic To Exact Rationals When
    Inconclusive.** A floating-point (or fixed-width) verdict that the
    computed enclosure cannot decide is never rounded into a decision:
    the kernel's verdict type carries INCONCLUSIVE as a first-class
    outcome — conclusive-pass, conclusive-fail with witness, or
    inconclusive-defer — and the inconclusive arm routes to the exact
    arithmetic path (the callers' promotion ladder). Commitment 4's
    fit-detection is the special case for fixed-width integers; this
    commitment is the general law, and it is why refusal surfaces in
    this crate are typed rather than boolean.
