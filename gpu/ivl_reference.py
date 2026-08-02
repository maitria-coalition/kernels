"""Reference strata for interval-enclosure contraction rows.

The enclosure semiring is the consumer-side scalar carrier: elements
are closed intervals ``[lo, hi]`` of reals, addition is endpointwise,
multiplication is the product hull. A contraction row evaluated over
it yields, per output entry, an interval guaranteed to contain the
exact real value of the same contraction over any pointwise selection
from the operand intervals -- which is what a checking-side kernel
needs: a verdict read off a sound enclosure is a sound verdict.

Three reference strata, each a differential partner of the lanes
(ENGINEERING #5), from semantics down to economics:

1. **Exact** (``eval_row_interval_exact``): the interval contraction
   in exact rational arithmetic -- no rounding anywhere. This is the
   semantics. Order-independence is a theorem here: interval sum and
   product-hull are associative and commutative over the rationals
   (the product set of intervals is an interval -- connected image of
   a connected set -- so hull composition never loses points), hence
   the sum-of-products form needs no evaluation-order convention.
2. **Ideal** (``ideal_enclosure``): the directed f64 rounding of the
   exact endpoints -- the tightest representable enclosure any f64
   lane could produce. Lanes are measured against it (tightness);
   no lane can beat it (the sandwich ``lane_lo <= ideal_lo`` and
   ``ideal_hi <= lane_hi`` is asserted in the battery).
3. **Directed** (``eval_row_interval_ref``): the device fold mirrored
   op for op -- same operation sequence, same directed rounding per
   operation, same IEEE special cases (via ``interval.dir_sum`` /
   ``dir_prod`` / ``fmin_ieee`` / ``fmax_ieee``) -- the bitwise
   parity anchor for the FFI and Pallas lanes (the lane law,
   ENGINEERING #2).

The device fold, normative for stratum 3 and for every device lane
(deterministic evaluation order per VCARM -- ENGINEERING #9):

    entry(o):  acc = [+0.0, +0.0]
               for r in 0 .. red_total-1 (ascending):
                   term = [1.0, 1.0]
                   for k in 0 .. n_factors-1 (ascending):
                       term = term (*) F_k[off(k, o, r)]
                   acc = acc (+) term
    (+) : [a,b] (+) [c,d] = [add_rm(a,c), add_rp(b,d)]
    (*) : [a,b] (*) [c,d] = [min of the four rm-products,
                             max of the four rp-products]
          four products in the fixed order (a,c) (a,d) (b,c) (b,d),
          min/max folded pairwise: min(min(p1,p2), min(p3,p4)).
    accumulate variant: acc' = prev (+) entry(o)  -- one further
          directed add after the fresh fold, mirroring the s64
          lanes' ``acc + row`` composition.

The brute four-product multiply is exact-per-operation: directed
rounding is monotone, so the min of the four rounded-down products
equals the rounded-down min of the four exact products (and dually
for max) -- the branch-free form loses nothing against the sign-case
table (worked in ``gpu/README.md``).

Signed zeros and the exact stratum, the comparison convention (the
elementwise battery's founding lesson, spelled out): exact rationals
carry no signed zero, so every comparison against stratum 1 or 2 --
enclosure soundness, verdict truth, tightness -- is a VALUE
comparison in which -0.0 and +0.0 coincide (``Fraction(-0.0) ==
Fraction(0.0) == 0``). Bitwise discipline, signed zeros included,
applies exactly where two lanes of the SAME rounded algorithm meet:
device vs stratum 3, device vs device. Verdict classification is
value-level (float comparisons), so no verdict ever depends on the
sign of a zero.

Scope: stratum 1 and 2 require finite endpoints (typed refusal
otherwise -- rationals cannot represent infinities); stratum 3 is
total on f64 (IEEE-extended scalar ops). Within the ``Row.fits_f64``
gate the three strata and the device lanes are all finite and
stratum 3 is bitwise-total against the device; outside the gate,
device endpoints may saturate to +/-inf (still sound: an unbounded
side claims nothing false) or void to NaN (no claim at all), and the
classifier maps every such entry to INCONCLUSIVE -- never to a
conclusive verdict (DATERWI, ENGINEERING #10).
"""

from __future__ import annotations

import math
from fractions import Fraction

from .interval import dir_prod, dir_sum, fmax_ieee, fmin_ieee, round_down, round_up
from .rowir import Row, RowError, Verdict

# ── operand plumbing ─────────────────────────────────────────────────


def check_interval_operands(
    row: Row, operands: list[tuple[list[float], list[float]]]
) -> None:
    """Typed refusal unless ``operands`` -- one ``(lo_buf, hi_buf)``
    pair per factor, flat row-major -- match the row's induced shapes.
    Endpoint ORDER (lo <= hi) is the caller's contract and is not
    policed here: policing it would cost a full scan per call, and a
    mis-ordered interval yields a verdict-safe outcome anyway (the
    fold still computes min/max over the endpoint set; enclosure
    guarantees are stated for ordered inputs)."""
    if len(operands) != len(row.factors):
        raise RowError(
            f"row has {len(row.factors)} factors, got {len(operands)} operands"
        )
    for k, (lo_buf, hi_buf) in enumerate(operands):
        want = 1
        for e in row.factor_shape(k):
            want *= e
        for name, buf in (("lo", lo_buf), ("hi", hi_buf)):
            if len(buf) != want:
                raise RowError(
                    f"operand {k} ({name}): expected {want} entries for "
                    f"induced shape {row.factor_shape(k)}, got {len(buf)}"
                )


def _offsets(row: Row):
    """Shared index walk: yields, per output entry (in ``out`` row-major
    order), the list over reduction assignments of per-factor flat
    offsets -- the same addressing the device kernel derives from the
    packed descriptor."""
    strides = [row.strides(k) for k in range(len(row.factors))]
    for v in row.out_assignments():
        base = [
            sum(st.get(o, 0) * c for o, c in zip(row.out, v))
            for st in strides
        ]
        steps = []
        for vp in row.red_assignments():
            steps.append(
                [
                    base[k]
                    + sum(st.get(r, 0) * c for r, c in zip(row.red, vp))
                    for k, st in enumerate(strides)
                ]
            )
        yield steps


# ── stratum 1: exact rational semantics ──────────────────────────────


def eval_row_interval_exact(
    row: Row, operands: list[tuple[list[float], list[float]]]
) -> tuple[list[Fraction], list[Fraction]]:
    """The interval contraction in exact rational arithmetic: per
    output entry the exact endpoints ``[Lo, Hi]`` of the interval
    sum of interval product-hulls. Finite endpoints only (typed
    refusal names the offender)."""
    check_interval_operands(row, operands)
    for k, (lo_buf, hi_buf) in enumerate(operands):
        for name, buf in (("lo", lo_buf), ("hi", hi_buf)):
            for i, x in enumerate(buf):
                if not math.isfinite(x):
                    raise RowError(
                        f"operand {k} ({name})[{i}] = {x!r}: the exact "
                        "stratum requires finite endpoints"
                    )
    los: list[Fraction] = []
    his: list[Fraction] = []
    for steps in _offsets(row):
        acc_lo, acc_hi = Fraction(0), Fraction(0)
        for offs in steps:
            t_lo, t_hi = Fraction(1), Fraction(1)
            for k, (lo_buf, hi_buf) in enumerate(operands):
                a, b = Fraction(lo_buf[offs[k]]), Fraction(hi_buf[offs[k]])
                ps = (t_lo * a, t_lo * b, t_hi * a, t_hi * b)
                t_lo, t_hi = min(ps), max(ps)
            acc_lo += t_lo
            acc_hi += t_hi
        los.append(acc_lo)
        his.append(acc_hi)
    return los, his


def ideal_enclosure(
    los: list[Fraction], his: list[Fraction]
) -> tuple[list[float], list[float]]:
    """Stratum 2: the tightest representable f64 enclosure of exact
    endpoints -- lower endpoints rounded toward minus infinity, upper
    toward plus infinity."""
    return [round_down(fr) for fr in los], [round_up(fr) for fr in his]


# ── stratum 3: the directed op-for-op mirror ─────────────────────────


def _ivl_mul_ref(
    al: float, ah: float, bl: float, bh: float
) -> tuple[float, float]:
    """The device ``ivl_mul``, mirrored: four rm-products folded by
    device-fmin, four rp-products folded by device-fmax, fixed order."""
    lo = fmin_ieee(
        fmin_ieee(dir_prod(al, bl, False), dir_prod(al, bh, False)),
        fmin_ieee(dir_prod(ah, bl, False), dir_prod(ah, bh, False)),
    )
    hi = fmax_ieee(
        fmax_ieee(dir_prod(al, bl, True), dir_prod(al, bh, True)),
        fmax_ieee(dir_prod(ah, bl, True), dir_prod(ah, bh, True)),
    )
    return lo, hi


def _ivl_add_ref(
    al: float, ah: float, bl: float, bh: float
) -> tuple[float, float]:
    return dir_sum(al, bl, False), dir_sum(ah, bh, True)


def eval_row_interval_ref(
    row: Row,
    operands: list[tuple[list[float], list[float]]],
    acc: tuple[list[float], list[float]] | None = None,
) -> tuple[list[float], list[float]]:
    """The directed reference: the device fold transcribed (module
    docstring, normative order), total on f64. With ``acc`` the
    accumulate variant: ``prev (+) entry`` per entry."""
    check_interval_operands(row, operands)
    if acc is not None and (
        len(acc[0]) != row.n_out or len(acc[1]) != row.n_out
    ):
        raise RowError(
            f"accumulator has ({len(acc[0])}, {len(acc[1])}) entries, "
            f"output needs {row.n_out}"
        )
    los: list[float] = []
    his: list[float] = []
    for o, steps in enumerate(_offsets(row)):
        a_lo, a_hi = 0.0, 0.0
        for offs in steps:
            t_lo, t_hi = 1.0, 1.0
            for k, (lo_buf, hi_buf) in enumerate(operands):
                t_lo, t_hi = _ivl_mul_ref(
                    t_lo, t_hi, lo_buf[offs[k]], hi_buf[offs[k]]
                )
            a_lo, a_hi = _ivl_add_ref(a_lo, a_hi, t_lo, t_hi)
        if acc is not None:
            a_lo, a_hi = _ivl_add_ref(acc[0][o], acc[1][o], a_lo, a_hi)
        los.append(a_lo)
        his.append(a_hi)
    return los, his


# ── the verdict boundary (DATERWI, ENGINEERING #10) ──────────────────


def classify_entry(lo: float, hi: float, dlo: float, dhi: float) -> int:
    """Three-way classification of one enclosure against one demanded
    bound, NaN-safe by construction (the device kernel computes the
    identical predicate):

    - 0 (pass):  dlo <= lo  and  hi <= dhi -- every value the
      enclosure permits satisfies the demand.
    - 2 (fail):  hi < dlo  or  lo > dhi -- every value the enclosure
      permits violates the demand.
    - 1 (inconclusive): everything else -- the enclosure straddles a
      demand boundary, or an endpoint is NaN (every comparison with
      NaN is false, so NaN entries can reach neither conclusive arm).

    Both conclusive arms are sound for any sound enclosure: they
    quantify over a superset of the possible values. Infinite
    endpoints participate correctly (an entry with hi = +inf can pass
    only a demand with dhi = +inf, which every finite value
    satisfies). Value comparisons throughout: no verdict depends on
    the sign of a zero.
    """
    if dlo <= lo and hi <= dhi:
        return 0
    if hi < dlo or lo > dhi:
        return 2
    return 1


def verdict_from_classes(classes: list[int]) -> Verdict:
    """Reduce per-entry classes to one row Verdict.

    Precedence -- fail dominates: one conclusively-violating entry
    conclusively refutes ``all entries in bounds`` regardless of what
    the deferred entries would have said, and its index is a genuine
    violation witness. The witness is therefore the lowest
    CONCLUSIVELY-failing index: entries before it may be
    inconclusive, so it is not necessarily the first violation in
    ground truth -- a caller that needs first-violation semantics
    consults the enclosure arrays (or the exact path) rather than the
    witness. No fail and any straddle: inconclusive, defer (never
    rounded into a decision). Otherwise: pass.
    """
    fail_at = None
    saw_inconclusive = False
    for i, c in enumerate(classes):
        if c == 2:
            fail_at = i
            break
        if c == 1:
            saw_inconclusive = True
    if fail_at is not None:
        return Verdict.conclusive_fail(fail_at)
    if saw_inconclusive:
        return Verdict.inconclusive()
    return Verdict.conclusive_pass()


def check_row_interval_ref(
    row: Row,
    operands: list[tuple[list[float], list[float]]],
    dlo: list[float],
    dhi: list[float],
) -> tuple[list[float], list[float], Verdict]:
    """The directed reference of the checking kernel: evaluate
    (stratum 3), classify every entry, reduce. Returns the enclosure
    arrays alongside the Verdict -- the caller's promotion ladder
    wants both (the enclosure tells it WHICH entries to re-derive
    exactly)."""
    if len(dlo) != row.n_out or len(dhi) != row.n_out:
        raise RowError(
            f"demand arrays have ({len(dlo)}, {len(dhi)}) entries, "
            f"output needs {row.n_out}"
        )
    los, his = eval_row_interval_ref(row, operands)
    classes = [
        classify_entry(lo, hi, lo_d, hi_d)
        for lo, hi, lo_d, hi_d in zip(los, his, dlo, dhi)
    ]
    return los, his, verdict_from_classes(classes)


def check_row_interval_exact(
    row: Row,
    operands: list[tuple[list[float], list[float]]],
    dlo: list[float],
    dhi: list[float],
) -> Verdict:
    """The promotion ladder's terminal: the demand checked against
    the EXACT interval (stratum 1). Always conclusive -- with no
    rounding uncertainty left, the question ``does every possible
    value of entry i satisfy its demand?`` has a definite answer:
    pass iff every exact interval is contained in its demand; fail at
    the first entry whose exact interval is NOT contained (which
    covers both ``entirely outside`` and the genuine straddle, where
    the operand intervals themselves permit a violating selection --
    note this ``fail`` is therefore the negation of the universal
    property, deliberately weaker than the float classifier's
    ``entirely outside`` arm). NaN demands are refused; here the
    witness IS the first violation in ground truth."""
    for name, arr in (("dlo", dlo), ("dhi", dhi)):
        for i, d in enumerate(arr):
            if math.isnan(d):
                raise RowError(f"{name}[{i}] is NaN: not a demand")
    los, his = eval_row_interval_exact(row, operands)
    for i, (lo_fr, hi_fr) in enumerate(zip(los, his)):
        lo_ok = (
            True
            if dlo[i] == -math.inf
            else (not math.isinf(dlo[i])) and Fraction(dlo[i]) <= lo_fr
        )
        hi_ok = (
            True
            if dhi[i] == math.inf
            else (not math.isinf(dhi[i])) and hi_fr <= Fraction(dhi[i])
        )
        if not (lo_ok and hi_ok):
            return Verdict.conclusive_fail(i)
    return Verdict.conclusive_pass()
