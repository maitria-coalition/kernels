"""Battery for the interval-enclosure contraction lanes.

Instruments, per generated row (the differential-partner clique,
ENGINEERING #2/#5):

1. **Soundness** (the property the lane exists for) — the device
   enclosure contains the exact interval semantics, checked in exact
   rational arithmetic: ``lane_lo <= exact_lo`` and ``exact_hi <=
   lane_hi``, value comparisons (signed zeros coincide in the exact
   stratum by convention — see ``ivl_reference``).
2. **The ideal sandwich** (tightness's hard boundary) — no lane may
   beat the directed rounding of the exact endpoints:
   ``lane_lo <= ideal_lo`` and ``ideal_hi <= lane_hi``.
3. **Parity** (the lane law) — device outputs bitwise-equal to the
   directed op-for-op reference, signed zeros included; the hardware-
   fesetround host lane bitwise-equal to the same reference (a third
   corner, computed by a third mechanism).
4. **Verdict correctness** (DATERWI) — the device three-state verdict
   equals the reference classification; conclusive arms are TRUE
   against the exact stratum (pass: every exact interval inside its
   demand; fail: the witness entry's exact interval disjoint from its
   demand); deliberate cancellation straddles come out INCONCLUSIVE
   and the exact path resolves them.

Adversarial cancellation rides both the generated strategy (signed
values with exact mirror negations are one draw apart) and the
deterministic dot rows below, where terms cancel exactly in the
rationals while every float partial is inexact.
"""

from __future__ import annotations

import math
import os
import struct
from fractions import Fraction

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from gpu.ivl_reference import (
    check_row_interval_exact,
    check_row_interval_ref,
    classify_entry,
    eval_row_interval_exact,
    eval_row_interval_ref,
    ideal_enclosure,
    verdict_from_classes,
)
from gpu.rowir import Row, RowError, Verdict

MAX_EXAMPLES = int(os.environ.get("BATTERY_EXAMPLES", "100"))

SETTINGS = settings(
    max_examples=MAX_EXAMPLES,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.large_base_example,
    ],
)

# Endpoint magnitudes bounded at 1e12: with <= 3 factors and the small
# row strategy below (red_total <= 64), the fits_f64 bound holds by
# orders of magnitude, so every stratum is finite and parity is total.
F64_BOUNDED = st.one_of(
    st.floats(
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=True,
        min_value=-1e12,
        max_value=1e12,
    ),
    st.sampled_from(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            1 / 3,
            -1 / 3,
            0.1,
            2.0**-1074,
            -(2.0**-1074),
            1e-300,
            0.5,
            -2.5,
        ]
    ),
)


def _bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


@st.composite
def small_rows(draw, max_m=4, max_factors=3, max_extent=3, max_rank=2):
    """A well-formed Row, small enough that the exact stratum stays
    cheap (the s64 battery's generator, tightened)."""
    m = draw(st.integers(1, max_m))
    extents = tuple(draw(st.integers(1, max_extent)) for _ in range(m))
    nf = draw(st.integers(0, max_factors))
    factors = tuple(
        tuple(
            draw(
                st.lists(st.integers(0, m - 1), min_size=0, max_size=max_rank)
            )
        )
        for _ in range(nf)
    )
    hit = sorted({a for f in factors for a in f})
    unhit = [a for a in range(m) if a not in hit]
    extra = (
        draw(st.lists(st.sampled_from(hit), unique=True, max_size=len(hit)))
        if hit
        else []
    )
    out = tuple(draw(st.permutations(unhit + extra)))
    return Row(extents=extents, factors=factors, out=out)


@st.composite
def interval_operands(draw, row):
    """Per factor: two draws per entry, min/max'd into (lo, hi) —
    with a mirror-negation pass so exact cancellations are one draw
    away (adversarial-cancellation pressure inside the generator)."""
    ops = []
    for k in range(len(row.factors)):
        n = _prod(row.factor_shape(k))
        a = draw(st.lists(F64_BOUNDED, min_size=n, max_size=n))
        b = draw(st.lists(F64_BOUNDED, min_size=n, max_size=n))
        if draw(st.booleans()) and n > 1:
            # Mirror the first half negated onto the second: dot-like
            # rows then cancel exactly in the rationals.
            half = n // 2
            for i in range(half):
                a[n - 1 - i] = -a[i]
                b[n - 1 - i] = -b[i]
        lo = [min(x, y) for x, y in zip(a, b)]
        hi = [max(x, y) for x, y in zip(a, b)]
        ops.append((lo, hi))
    return ops


@st.composite
def rows_with_interval_operands(draw, **kw):
    row = draw(small_rows(**kw))
    return row, draw(interval_operands(row))


def _assert_sound_and_sandwiched(row, ops, los, his):
    """Instruments 1 + 2 against a lane's endpoint lists."""
    ex_lo, ex_hi = eval_row_interval_exact(row, ops)
    id_lo, id_hi = ideal_enclosure(ex_lo, ex_hi)
    for i in range(row.n_out):
        lo, hi = los[i], his[i]
        assert not math.isnan(lo) and not math.isnan(hi), (i, lo, hi)
        # Soundness, exact comparisons (infinite endpoints trivially
        # sound on their side).
        if not math.isinf(lo):
            assert Fraction(lo) <= ex_lo[i], (i, lo, ex_lo[i])
        else:
            assert lo == -math.inf
        if not math.isinf(hi):
            assert ex_hi[i] <= Fraction(hi), (i, hi, ex_hi[i])
        else:
            assert hi == math.inf
        # The ideal sandwich (value comparisons).
        assert lo <= id_lo[i], (i, lo, id_lo[i])
        assert id_hi[i] <= hi, (i, hi, id_hi[i])


# ---------------------------------------------------------------------------
# Reference strata: deterministic edges (run anywhere).
# ---------------------------------------------------------------------------


class TestReferenceEdges:
    def test_exact_matmul_point_intervals(self):
        row = Row(extents=(2, 2, 2), factors=((0, 2), (2, 1)), out=(0, 1))
        a = [1.0, 2.0, 3.0, 4.0]
        b = [5.0, 6.0, 7.0, 8.0]
        los, his = eval_row_interval_exact(row, [(a, a), (b, b)])
        assert los == his == [19, 22, 43, 50]
        # Small integers: every float op exact, so the directed
        # reference collapses to the same points.
        rlo, rhi = eval_row_interval_ref(row, [(a, a), (b, b)])
        assert rlo == rhi == [19.0, 22.0, 43.0, 50.0]

    def test_exact_interval_matmul_mixed_signs(self):
        # One entry, one reduction step, mixed-sign intervals: the
        # product hull's corner analysis in miniature.
        row = Row(extents=(1,), factors=((0,), (0,)), out=())
        los, his = eval_row_interval_exact(
            row, [([-2.0], [3.0]), ([-5.0], [4.0])]
        )
        # hull{[-2,3] * [-5,4]} = [min(10,-8,-15,12), max(...)] = [-15, 12]
        assert (los[0], his[0]) == (-15, 12)

    def test_empty_tensor_product_is_unit_intervals(self):
        row = Row(extents=(3,), factors=(), out=(0,))
        los, his = eval_row_interval_ref(row, [])
        assert los == his == [1.0, 1.0, 1.0]

    def test_cancellation_dot_straddles_zero(self):
        # a = (x, -x) point intervals, b = (y, y): exact 0, but x*y is
        # inexact in f64, so the directed fold must straddle 0 strictly.
        x, y = 1 / 3, 0.1
        row = Row(extents=(2,), factors=((0,), (0,)), out=())
        ops = [([x, -x], [x, -x]), ([y, y], [y, y])]
        ex_lo, ex_hi = eval_row_interval_exact(row, ops)
        assert ex_lo == ex_hi == [0]
        rlo, rhi = eval_row_interval_ref(row, ops)
        assert rlo[0] < 0.0 < rhi[0]
        # The enclosure is sound and one-or-two ulps wide, not junk.
        assert rhi[0] - rlo[0] < 1e-16

    def test_zero_fold_pins_upper_zero_sign(self):
        # The (-0, -0) corner the pallas lane's zero-fold exists for:
        # fresh upper endpoint -0.0 is rewritten to +0.0 by the
        # from-zero fold, so the accumulate's upper add sees (+0, -0)
        # -> +0.0 in every lane. Pinned here at the reference level.
        row = Row(extents=(1,), factors=((0,), (0,)), out=(0,))
        ops = [([-1.0], [-0.0]), ([0.0], [5.0])]
        # fresh = [+0,+0] (+) ([1,1](*)X(*)Y); t_hi = -0.0:
        rlo, rhi = eval_row_interval_ref(row, ops)
        assert _bits(rhi[0]) == _bits(0.0)  # +0.0, not -0.0
        # accumulate onto prev_hi = -0.0 keeps +0.0:
        alo, ahi = eval_row_interval_ref(
            row, ops, acc=([-3.0], [-0.0])
        )
        assert _bits(ahi[0]) == _bits(-0.0) or _bits(ahi[0]) == _bits(0.0)
        # the exact expectation: ru(-0.0 + +0.0) = +0.0
        assert _bits(ahi[0]) == _bits(0.0)

    def test_classify_entry(self):
        assert classify_entry(1.0, 2.0, 0.0, 3.0) == 0
        assert classify_entry(1.0, 2.0, 1.0, 2.0) == 0  # closed bounds
        assert classify_entry(1.0, 2.0, 2.5, 3.0) == 2  # entirely below
        assert classify_entry(1.0, 2.0, -1.0, 0.5) == 2  # entirely above
        assert classify_entry(1.0, 2.0, 1.5, 3.0) == 1  # straddle
        assert classify_entry(math.nan, 2.0, 0.0, 3.0) == 1  # NaN defers
        assert classify_entry(1.0, math.inf, 0.0, math.inf) == 0
        assert classify_entry(-math.inf, 2.0, 0.0, 3.0) == 1

    def test_verdict_precedence_fail_dominates(self):
        v = verdict_from_classes([0, 1, 2, 2])
        assert v.kind == "fail" and v.witness == 2  # lowest conclusive
        assert verdict_from_classes([0, 1, 0]).kind == "inconclusive"
        assert verdict_from_classes([0, 0]).kind == "pass"
        assert verdict_from_classes([]).kind == "pass"

    def test_exact_checker_answers_the_universal_property(self):
        # Genuine straddle (wide operand interval): float lane defers,
        # exact checker conclusively FAILS the universal claim.
        row = Row(extents=(1,), factors=((0,),), out=(0,))
        ops = [([-1.0], [1.0])]
        v = check_row_interval_exact(row, ops, [0.0], [2.0])
        assert v.kind == "fail" and v.witness == 0
        v2 = check_row_interval_exact(row, ops, [-1.0], [1.0])
        assert v2.kind == "pass"

    def test_fits_f64_gate(self):
        row = Row(extents=(4,), factors=((0,), (0,)), out=())
        assert row.fits_f64([1e12, 1e12])
        assert not row.fits_f64([1e200, 1e200])
        with pytest.raises(RowError):
            row.fits_f64([1e12])  # arity mismatch
        with pytest.raises(RowError):
            row.fits_f64([-1.0, 1.0])  # negative bound


# ---------------------------------------------------------------------------
# Reference strata: properties (run anywhere).
# ---------------------------------------------------------------------------


@SETTINGS
@given(rows_with_interval_operands())
def test_directed_reference_sound_and_sandwiched(rw):
    row, ops = rw
    los, his = eval_row_interval_ref(row, ops)
    _assert_sound_and_sandwiched(row, ops, los, his)


@SETTINGS
@given(rows_with_interval_operands(), st.data())
def test_reference_verdict_truth(rw, data):
    row, ops = rw
    los, his = eval_row_interval_ref(row, ops)
    dlo, dhi = _demands_for(data, los, his)
    _, _, verdict = check_row_interval_ref(row, ops, dlo, dhi)
    _assert_verdict_truth(row, ops, dlo, dhi, verdict)


@SETTINGS
@given(rows_with_interval_operands(), st.data())
def test_hw_lane_matches_directed_reference(rw, data):
    from gpu import ivl_host_lane

    if not ivl_host_lane.supported():
        pytest.skip(
            f"hardware directed rounding: {ivl_host_lane.why_unsupported()}"
        )
    row, ops = rw
    want_lo, want_hi = eval_row_interval_ref(row, ops)
    got_lo, got_hi = ivl_host_lane.eval_row_interval_hw(row, ops)
    assert [_bits(x) for x in got_lo] == [_bits(x) for x in want_lo]
    assert [_bits(x) for x in got_hi] == [_bits(x) for x in want_hi]
    if data.draw(st.booleans()):
        n = row.n_out
        a = data.draw(st.lists(F64_BOUNDED, min_size=n, max_size=n))
        b = data.draw(st.lists(F64_BOUNDED, min_size=n, max_size=n))
        acc = ([min(x, y) for x, y in zip(a, b)],
               [max(x, y) for x, y in zip(a, b)])
        want = eval_row_interval_ref(row, ops, acc=acc)
        got = ivl_host_lane.eval_row_interval_hw(row, ops, acc=acc)
        assert [[_bits(x) for x in side] for side in got] == [
            [_bits(x) for x in side] for side in want
        ]


# ---------------------------------------------------------------------------
# Demand construction + verdict truth (shared by host and device tests).
# ---------------------------------------------------------------------------


def _demands_for(data, los, his):
    """Per-entry demands engineered from a lane's enclosure to hit all
    three classes: containing (pass), disjoint (fail, either side),
    straddling (inconclusive), and the closed-bound edges."""
    dlo, dhi = [], []
    for lo, hi in zip(los, his):
        kind = data.draw(
            st.sampled_from(
                ["pass_wide", "pass_tight", "fail_above", "fail_below",
                 "straddle_hi", "straddle_lo", "wild"]
            )
        )
        if kind == "pass_wide":
            dlo.append(math.nextafter(lo, -math.inf) if math.isfinite(lo) else lo)
            dhi.append(math.nextafter(hi, math.inf) if math.isfinite(hi) else hi)
        elif kind == "pass_tight":
            dlo.append(lo)
            dhi.append(hi)
        elif kind == "fail_above":
            # demand strictly above the enclosure: hi < dlo
            d = math.nextafter(hi, math.inf)
            dlo.append(d)
            dhi.append(math.inf)
        elif kind == "fail_below":
            d = math.nextafter(lo, -math.inf)
            dlo.append(-math.inf)
            dhi.append(d)
        elif kind == "straddle_hi":
            # boundary inside (lo, hi) when the enclosure is wide
            # enough; degenerates to pass on point enclosures, which
            # classify_entry adjudicates — the assertion is vs the
            # reference classifier either way.
            dlo.append(hi)
            dhi.append(math.inf)
        elif kind == "straddle_lo":
            dlo.append(-math.inf)
            dhi.append(lo)
        else:
            a = data.draw(F64_BOUNDED)
            b = data.draw(F64_BOUNDED)
            dlo.append(min(a, b))
            dhi.append(max(a, b))
    return dlo, dhi


def _assert_verdict_truth(row, ops, dlo, dhi, verdict):
    """Conclusive arms checked against the exact stratum (soundness of
    the verdict, not merely agreement between lanes)."""
    ex_lo, ex_hi = eval_row_interval_exact(row, ops)
    if verdict.kind == "pass":
        for i in range(row.n_out):
            if math.isfinite(dlo[i]):
                assert Fraction(dlo[i]) <= ex_lo[i], i
            else:
                assert dlo[i] == -math.inf
            if math.isfinite(dhi[i]):
                assert ex_hi[i] <= Fraction(dhi[i]), i
            else:
                assert dhi[i] == math.inf
    elif verdict.kind == "fail":
        w = verdict.witness
        # The witness entry's exact interval is disjoint from its
        # demand: entirely below dlo or entirely above dhi.
        below = math.isfinite(dlo[w]) and ex_hi[w] < Fraction(dlo[w])
        above = math.isfinite(dhi[w]) and ex_lo[w] > Fraction(dhi[w])
        assert below or above, (w, dlo[w], dhi[w], ex_lo[w], ex_hi[w])


# ---------------------------------------------------------------------------
# Device lanes (skip loudly off-GPU).
# ---------------------------------------------------------------------------


def _gpu_available():
    try:
        import jax

        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


gpu_only = pytest.mark.skipif(
    not _gpu_available(), reason="no CUDA device; GPU lanes gate on-box"
)


def _to_dev(row, ops):
    import jax.numpy as jnp

    return [
        (
            jnp.asarray(lo, dtype=jnp.float64).reshape(row.factor_shape(k)),
            jnp.asarray(hi, dtype=jnp.float64).reshape(row.factor_shape(k)),
        )
        for k, (lo, hi) in enumerate(ops)
    ]


@gpu_only
@SETTINGS
@given(rows_with_interval_operands())
def test_ffi_interval_row_matches_reference_and_encloses(rw):
    row, ops = rw
    from gpu.ivl_ffi_lane import eval_row_interval_ffi

    got_lo, got_hi = eval_row_interval_ffi(row, _to_dev(row, ops))
    got_lo = got_lo.reshape(-1).tolist()
    got_hi = got_hi.reshape(-1).tolist()
    want_lo, want_hi = eval_row_interval_ref(row, ops)
    assert [_bits(x) for x in got_lo] == [_bits(x) for x in want_lo]
    assert [_bits(x) for x in got_hi] == [_bits(x) for x in want_hi]
    _assert_sound_and_sandwiched(row, ops, got_lo, got_hi)


@gpu_only
@SETTINGS
@given(rows_with_interval_operands(), st.data())
def test_ffi_interval_accum_matches_reference(rw, data):
    row, ops = rw
    import jax.numpy as jnp

    from gpu.ivl_ffi_lane import eval_row_interval_accum_ffi

    n = row.n_out
    a = data.draw(st.lists(F64_BOUNDED, min_size=n, max_size=n))
    b = data.draw(st.lists(F64_BOUNDED, min_size=n, max_size=n))
    acc = ([min(x, y) for x, y in zip(a, b)],
           [max(x, y) for x, y in zip(a, b)])
    acc_lo = jnp.asarray(acc[0], dtype=jnp.float64).reshape(row.out_shape)
    acc_hi = jnp.asarray(acc[1], dtype=jnp.float64).reshape(row.out_shape)
    got_lo, got_hi = eval_row_interval_accum_ffi(
        row, acc_lo, acc_hi, _to_dev(row, ops)
    )
    want_lo, want_hi = eval_row_interval_ref(row, ops, acc=acc)
    assert [_bits(x) for x in got_lo.reshape(-1).tolist()] == [
        _bits(x) for x in want_lo
    ]
    assert [_bits(x) for x in got_hi.reshape(-1).tolist()] == [
        _bits(x) for x in want_hi
    ]


@gpu_only
@SETTINGS
@given(rows_with_interval_operands(), st.data())
def test_ffi_check_verdict_matches_reference_and_truth(rw, data):
    row, ops = rw
    import jax.numpy as jnp

    from gpu.ivl_ffi_lane import check_row_interval_ffi

    ref_lo, ref_hi = eval_row_interval_ref(row, ops)
    dlo, dhi = _demands_for(data, ref_lo, ref_hi)
    dlo_dev = jnp.asarray(dlo, dtype=jnp.float64).reshape(row.out_shape)
    dhi_dev = jnp.asarray(dhi, dtype=jnp.float64).reshape(row.out_shape)
    got_lo, got_hi, got_v = check_row_interval_ffi(
        row, _to_dev(row, ops), dlo_dev, dhi_dev
    )
    want_lo, want_hi, want_v = check_row_interval_ref(row, ops, dlo, dhi)
    assert [_bits(x) for x in got_lo.reshape(-1).tolist()] == [
        _bits(x) for x in want_lo
    ]
    assert [_bits(x) for x in got_hi.reshape(-1).tolist()] == [
        _bits(x) for x in want_hi
    ]
    assert (got_v.kind, got_v.witness) == (want_v.kind, want_v.witness)
    _assert_verdict_truth(row, ops, dlo, dhi, got_v)


@gpu_only
def test_cancellation_straddle_defers_and_exact_resolves():
    """The DATERWI story end to end on device: exact cancellation, a
    demand the float enclosure cannot decide, INCONCLUSIVE from the
    kernel, conclusive PASS from the exact path."""
    import jax.numpy as jnp

    from gpu.ivl_ffi_lane import check_row_interval_ffi

    x, y = 1 / 3, 0.1
    row = Row(extents=(2,), factors=((0,), (0,)), out=())
    ops = [([x, -x], [x, -x]), ([y, y], [y, y])]
    dev_ops = _to_dev(row, ops)
    zero = jnp.zeros((), dtype=jnp.float64)

    lo, hi, v = check_row_interval_ffi(row, dev_ops, zero, zero)
    assert v.kind == "inconclusive"
    assert float(lo) < 0.0 < float(hi)
    assert check_row_interval_exact(row, ops, [0.0], [0.0]).kind == "pass"

    # A demand the enclosure CAN decide fails conclusively, witness 0.
    neg = jnp.full((), -1.0, dtype=jnp.float64)
    neg2 = jnp.full((), -0.5, dtype=jnp.float64)
    _, _, v2 = check_row_interval_ffi(row, dev_ops, neg, neg2)
    assert v2.kind == "fail" and v2.witness == 0

    # And a generous demand passes conclusively.
    wide_lo = jnp.full((), -1.0, dtype=jnp.float64)
    wide_hi = jnp.full((), 1.0, dtype=jnp.float64)
    _, _, v3 = check_row_interval_ffi(row, dev_ops, wide_lo, wide_hi)
    assert v3.kind == "pass"


# Pallas cross-lane: the fused interval hadamard-accumulate vs the FFI
# lane on the same row — operation-identical by design (the zero-fold
# mirroring), so the assertion is bitwise on both endpoints.

_IVL_PALLAS_SHAPES = [(128,), (256,), (16, 16), (2, 64), (4, 32, 4)]


@st.composite
def interval_hadamard_cases(draw):
    shape = draw(st.sampled_from(_IVL_PALLAS_SHAPES))
    n = _prod(shape)

    def pair():
        a = draw(st.lists(F64_BOUNDED, min_size=n, max_size=n))
        b = draw(st.lists(F64_BOUNDED, min_size=n, max_size=n))
        return ([min(x, y) for x, y in zip(a, b)],
                [max(x, y) for x, y in zip(a, b)])

    return shape, pair(), pair(), pair()


@gpu_only
@SETTINGS
@given(interval_hadamard_cases())
def test_pallas_interval_hadamard_matches_ffi_accum(case):
    import jax.numpy as jnp

    from gpu.ivl_ffi_lane import eval_row_interval_accum_ffi
    from gpu.pallas_lane import interval_hadamard_accum_pallas
    from gpu.rowir import RowError as PallasRowError

    shape, acc, xi, yi = case
    axes = tuple(range(len(shape)))
    row = Row(extents=shape, factors=(axes, axes), out=axes)

    def dev(pair):
        return (
            jnp.asarray(pair[0], dtype=jnp.float64).reshape(shape),
            jnp.asarray(pair[1], dtype=jnp.float64).reshape(shape),
        )

    acc_lo, acc_hi = dev(acc)
    x_lo, x_hi = dev(xi)
    y_lo, y_hi = dev(yi)
    try:
        p_lo, p_hi = interval_hadamard_accum_pallas(
            acc_lo, acc_hi, x_lo, x_hi, y_lo, y_hi
        )
    except PallasRowError as e:
        pytest.skip(f"pallas interval core unavailable: {e}")
    f_lo, f_hi = eval_row_interval_accum_ffi(
        row, acc_lo, acc_hi, [(x_lo, x_hi), (y_lo, y_hi)]
    )
    assert [_bits(v) for v in p_lo.reshape(-1).tolist()] == [
        _bits(v) for v in f_lo.reshape(-1).tolist()
    ]
    assert [_bits(v) for v in p_hi.reshape(-1).tolist()] == [
        _bits(v) for v in f_hi.reshape(-1).tolist()
    ]
