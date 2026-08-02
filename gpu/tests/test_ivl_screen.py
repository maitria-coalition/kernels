"""Battery for the checker embed (``gpu/ivl_screen.py``).

Ground truth throughout is the exact rational stratum: the ladder's
final verdicts are checked against exact containment per entry, never
merely against another float lane. Instruments:

1. **Per-entry exact arithmetic** — ``exact_entry`` agrees with the
   whole-row exact evaluator at every entry (the witness-confirmation
   economics rest on this function).
2. **Ladder truth** — ``screen_row``'s final verdict matches ground
   truth: pass iff every exact interval is contained in its demand;
   any fail's witness entry exactly violates. The ladder never
   returns INCONCLUSIVE (the exact path resolves every deferral).
3. **DATERWI end to end** — the mirrored-cancellation dot straddles
   its point demand in every float lane and the ladder resolves it
   conclusively via the exact path.
4. **Confirm modes** — "all" re-proves passes exactly; "none" still
   defers INCONCLUSIVE (DATERWI is not optional).
5. **The contract seam** — injective bindings reproduce direct row
   construction; diagonal-embedded outputs (repeated ``out_binding``
   axes) are screened correctly against a brute-force full-output
   exact ground truth in which off-diagonal entries are exact zero.
6. **Lane agreement ride-along** — hardware-fesetround and directed-
   reference screens produce identical final verdicts (skipped where
   fesetround is unsupported).

Device-lane instruments (jit-cache reuse, device/ladder agreement)
ride the same properties via the ``lane="device"`` parametrization and
skip cleanly off-GPU; the retrace-free receipt lives in the repo's
receipts, produced by ``gpu/bench.py``-style runs on a CUDA box.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from gpu.ivl_reference import (
    check_row_interval_exact,
    eval_row_interval_exact,
    eval_row_interval_ref,
)
from gpu.ivl_screen import (
    RowChecker,
    device_available,
    entry_exactly_violates,
    exact_entry,
    get_checker,
    row_from_bindings,
    screen_bound_contract,
    screen_row,
)
from gpu.rowir import Row, RowError, Verdict
from gpu.tests.test_ivl_contraction import (
    F64_BOUNDED,
    MAX_EXAMPLES,
    SETTINGS,
    small_rows,
)

# ── shared generators ────────────────────────────────────────────────


@st.composite
def rows_with_operands(draw):
    """A small row plus well-ordered interval operands (lo <= hi
    entrywise, finite endpoints — the exact stratum's domain)."""
    row = draw(small_rows())
    operands = []
    for k in range(len(row.factors)):
        n = 1
        for e in row.factor_shape(k):
            n *= e
        a = [draw(F64_BOUNDED) for _ in range(n)]
        b = [draw(F64_BOUNDED) for _ in range(n)]
        lo = [min(x, y) for x, y in zip(a, b)]
        hi = [max(x, y) for x, y in zip(a, b)]
        operands.append((lo, hi))
    return row, operands


@st.composite
def rows_operands_demands(draw):
    """Row + operands + a demand field mixing sure-passes, sure-fails,
    and knife-edge demands (the float straddle pressure)."""
    row, operands = draw(rows_with_operands())
    los, his = eval_row_interval_ref(row, operands)
    dlo, dhi = [], []
    for lo, hi in zip(los, his):
        mode = draw(st.integers(0, 3))
        if mode == 0:  # generous: contains the enclosure
            dlo.append(-math.inf if draw(st.booleans()) else lo - 1.0)
            dhi.append(math.inf if draw(st.booleans()) else hi + 1.0)
        elif mode == 1:  # disjoint above: sure fail
            dlo.append(hi + 1.0)
            dhi.append(hi + 2.0)
        elif mode == 2:  # disjoint below: sure fail
            dlo.append(lo - 2.0)
            dhi.append(lo - 1.0)
        else:  # knife edge: demand equals the float enclosure
            dlo.append(lo)
            dhi.append(hi)
    return row, operands, dlo, dhi


def exact_ground_truth(row, operands, dlo, dhi):
    """(all_contained, first_violation_index) in exact arithmetic."""
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
            return False, i
    return True, None


# ── 1. per-entry exact arithmetic ────────────────────────────────────


@SETTINGS
@given(rows_with_operands())
def test_exact_entry_matches_whole_row(rw):
    row, operands = rw
    los, his = eval_row_interval_exact(row, operands)
    for o in range(row.n_out):
        lo_fr, hi_fr = exact_entry(row, operands, o)
        assert lo_fr == los[o] and hi_fr == his[o]


# ── 2. ladder truth (ref + hw lanes; device parametrized below) ──────


def _lanes():
    from gpu import ivl_host_lane

    lanes = ["ref"]
    if ivl_host_lane.supported():
        lanes.append("hw")
    if device_available():
        lanes.append("device")
    return lanes


@pytest.mark.parametrize("lane", _lanes())
@SETTINGS
@given(rows_operands_demands())
def test_ladder_truth(lane, rod):
    row, operands, dlo, dhi = rod
    report = screen_row(row, operands, dlo, dhi, lane=lane)
    assert report.verdict.kind != "inconclusive", "ladder must resolve"
    assert not report.screen_disagreed, (
        "well-ordered operands must never produce a refuted fail"
    )
    all_ok, _first = exact_ground_truth(row, operands, dlo, dhi)
    if report.verdict.kind == "pass":
        assert all_ok, "ladder passed a row with an exact violation"
    else:
        w = report.verdict.witness
        assert entry_exactly_violates(row, operands, dlo[w], dhi[w], w), (
            "fail witness does not exactly violate"
        )


@pytest.mark.parametrize("lane", _lanes())
@SETTINGS
@given(rows_operands_demands())
def test_confirm_all_is_exact(lane, rod):
    row, operands, dlo, dhi = rod
    report = screen_row(row, operands, dlo, dhi, lane=lane, confirm="all")
    exact_v = check_row_interval_exact(row, operands, dlo, dhi)
    assert report.verdict.kind == exact_v.kind
    if report.verdict.kind == "fail":
        w = report.verdict.witness
        assert entry_exactly_violates(row, operands, dlo[w], dhi[w], w)


# ── 3. DATERWI end to end (the cancellation dot) ─────────────────────


def _cancellation_case(n=64):
    """The mirrored-cancellation dot: exact value 0 per entry, every
    float partial inexact — the row every float lane must straddle on
    a point demand."""
    row = Row((n,), ((0,), (0,)), ())
    xs = []
    for i in range(n // 2):
        v = (1.0 + i * 0.1) / 3.0
        xs += [v, -v]
    a = (xs, list(xs))  # point intervals
    b = ([1 / 3] * n, [1 / 3] * n)
    return row, [a, b]


def test_cancellation_resolves_via_exact():
    row, operands = _cancellation_case()
    report = screen_row(row, operands, [0.0], [0.0], lane="ref")
    assert report.screen_verdict is not None
    assert report.screen_verdict.kind == "inconclusive", (
        "the cancellation dot must straddle the point demand in float"
    )
    assert report.exact_used == "full"
    assert report.verdict.kind == "pass"
    # And a shifted demand fails conclusively with an exact witness.
    report2 = screen_row(row, operands, [1.0], [2.0], lane="ref")
    assert report2.verdict.kind == "fail" and report2.verdict.witness == 0


def test_confirm_none_still_defers_inconclusive():
    row, operands = _cancellation_case()
    report = screen_row(row, operands, [0.0], [0.0], lane="ref", confirm="none")
    assert report.exact_used == "full"
    assert report.verdict.kind == "pass"


def test_fits_gate_routes_to_exact():
    row, operands = _cancellation_case()
    report = screen_row(
        row,
        operands,
        [0.0],
        [0.0],
        lane="ref",
        operand_bounds=[2.0**600, 2.0**600],
    )
    assert report.fits is False
    assert report.screen_lane == "none" and report.exact_used == "full"
    assert report.verdict.kind == "pass"


# ── 4. lane agreement ride-along ─────────────────────────────────────


@SETTINGS
@given(rows_operands_demands())
def test_hw_and_ref_agree(rod):
    from gpu import ivl_host_lane

    if not ivl_host_lane.supported():
        pytest.skip(ivl_host_lane.why_unsupported())
    row, operands, dlo, dhi = rod
    r_ref = screen_row(row, operands, dlo, dhi, lane="ref")
    r_hw = screen_row(row, operands, dlo, dhi, lane="hw")
    assert r_ref.verdict == r_hw.verdict
    assert r_ref.screen_verdict == r_hw.screen_verdict


# ── 5. the contract seam ─────────────────────────────────────────────


def test_bindings_injective_roundtrip():
    row, embed = row_from_bindings((2, 3, 4), ((0, 2), (2, 1)), (0, 1))
    assert embed is None
    assert row == Row((2, 3, 4), ((0, 2), (2, 1)), (0, 1))


def test_bindings_gamma_refusal_inherited():
    with pytest.raises(RowError):
        row_from_bindings((2, 3), ((0,),), ())  # axis 1 unread


def test_bindings_diagonal_class_form():
    row, embed = row_from_bindings((3,), ((0,), (0,)), (0, 0))
    assert row.out == (0,)
    assert embed == ([0], [0, 0])


@st.composite
def diagonal_cases(draw):
    """A contract with a repeated out_binding axis, plus operands and
    full-output demands."""
    m = draw(st.integers(1, 3))
    extents = tuple(draw(st.integers(1, 3)) for _ in range(m))
    nf = draw(st.integers(1, 2))
    in_bindings = tuple(
        tuple(
            draw(st.lists(st.integers(0, m - 1), min_size=0, max_size=2))
        )
        for _ in range(nf)
    )
    hit = {a for f in in_bindings for a in f}
    # out_binding: every unhit axis present, plus one deliberate repeat.
    base = sorted(set(range(m)) - hit | {draw(st.integers(0, m - 1))})
    rep = draw(st.sampled_from(base))
    out_binding = tuple(base) + (rep,)
    operands = []
    for k in range(nf):
        n = 1
        for a in in_bindings[k]:
            n *= extents[a]
        a_ = [draw(F64_BOUNDED) for _ in range(n)]
        b_ = [draw(F64_BOUNDED) for _ in range(n)]
        operands.append(
            (
                [min(x, y) for x, y in zip(a_, b_)],
                [max(x, y) for x, y in zip(a_, b_)],
            )
        )
    full_n = 1
    for b in out_binding:
        full_n *= extents[b]
    dlo, dhi = [], []
    for _ in range(full_n):
        x = draw(F64_BOUNDED)
        y = draw(F64_BOUNDED)
        wide = draw(st.integers(0, 2))
        if wide == 0:
            dlo.append(-math.inf)
            dhi.append(math.inf)
        else:
            dlo.append(min(x, y))
            dhi.append(max(x, y))
    return extents, in_bindings, out_binding, operands, dlo, dhi


@SETTINGS
@given(diagonal_cases())
def test_contract_seam_diagonal_truth(case):
    extents, in_bindings, out_binding, operands, dlo, dhi = case
    report = screen_bound_contract(
        extents, in_bindings, out_binding, operands, dlo, dhi, lane="ref"
    )
    assert report.verdict.kind != "inconclusive"

    # Brute-force exact ground truth over the FULL output space.
    row, embed = row_from_bindings(extents, in_bindings, out_binding)
    class_order = list(dict.fromkeys(out_binding))
    axis_class = [class_order.index(b) for b in out_binding]
    class_extents = [extents[c] for c in class_order]
    full_shape = [extents[b] for b in out_binding]
    los, his = eval_row_interval_exact(row, operands)

    def full_entry(coords):
        cls = [None] * len(class_order)
        for t, c in enumerate(coords):
            if cls[axis_class[t]] is None:
                cls[axis_class[t]] = c
            elif cls[axis_class[t]] != c:
                return Fraction(0), Fraction(0)
        ci = 0
        for cc, e in zip(cls, class_extents):
            ci = ci * e + cc
        return los[ci], his[ci]

    def contained(lo_fr, hi_fr, d_lo, d_hi):
        lo_ok = (
            True
            if d_lo == -math.inf
            else (not math.isinf(d_lo)) and Fraction(d_lo) <= lo_fr
        )
        hi_ok = (
            True
            if d_hi == math.inf
            else (not math.isinf(d_hi)) and hi_fr <= Fraction(d_hi)
        )
        return lo_ok and hi_ok

    import itertools

    all_ok = True
    violations = set()
    for j, coords in enumerate(
        itertools.product(*(range(e) for e in full_shape))
    ):
        lo_fr, hi_fr = full_entry(coords)
        if not contained(lo_fr, hi_fr, dlo[j], dhi[j]):
            all_ok = False
            violations.add(j)

    if report.verdict.kind == "pass":
        assert all_ok
    else:
        assert report.verdict.witness in violations, (
            "diagonal-seam fail witness is not an exact violation"
        )


# ── 6. device lane specifics (skip off-GPU) ──────────────────────────

needs_device = pytest.mark.skipif(
    not device_available(), reason="no CUDA device / FFI library"
)


@needs_device
def test_device_checker_matches_wrapper():
    import jax.numpy as jnp

    from gpu.ivl_ffi_lane import check_row_interval_ffi

    row = Row((8, 8, 8), ((0, 2), (2, 1)), (0, 1))
    import numpy as np

    rng = np.random.default_rng(7)
    ops = []
    for k in range(2):
        a = rng.normal(size=row.factor_shape(k))
        w = np.abs(rng.normal(size=row.factor_shape(k))) * 0.1
        ops.append(
            (
                jnp.asarray(a - w, dtype=jnp.float64),
                jnp.asarray(a + w, dtype=jnp.float64),
            )
        )
    lo0, hi0, _v = check_row_interval_ffi(
        row,
        ops,
        jnp.full(row.out_shape, -jnp.inf, dtype=jnp.float64),
        jnp.full(row.out_shape, jnp.inf, dtype=jnp.float64),
    )
    dlo = jnp.nextafter(lo0, -jnp.inf)
    dhi = jnp.nextafter(hi0, jnp.inf)

    checker = get_checker(row)
    w_lo, w_hi, w_v = check_row_interval_ffi(row, ops, dlo, dhi)
    c_lo, c_hi, c_v = checker.check(ops, dlo, dhi)
    assert (
        jnp.array_equal(w_lo, c_lo, equal_nan=True)
        and jnp.array_equal(w_hi, c_hi, equal_nan=True)
    )
    assert w_v == c_v == Verdict.conclusive_pass()
    # A shifted demand fails with the same witness both ways.
    dlo2 = jnp.asarray(hi0 + 1.0)
    dhi2 = jnp.asarray(hi0 + 2.0)
    _, _, wv2 = check_row_interval_ffi(row, ops, dlo2, dhi2)
    _, _, cv2 = checker.check(ops, dlo2, dhi2)
    assert wv2 == cv2 and cv2.kind == "fail"


@needs_device
def test_device_jit_cache_reuses():
    import jax.numpy as jnp
    import numpy as np

    row = Row((4, 4), ((0, 1), (0, 1)), (0, 1))
    checker = RowChecker(row)
    rng = np.random.default_rng(11)
    for _ in range(3):
        a = rng.normal(size=row.factor_shape(0))
        ops = [
            (
                jnp.asarray(a - 0.1, dtype=jnp.float64),
                jnp.asarray(a + 0.1, dtype=jnp.float64),
            )
            for _ in range(2)
        ]
        dlo = jnp.full(row.out_shape, -jnp.inf, dtype=jnp.float64)
        dhi = jnp.full(row.out_shape, jnp.inf, dtype=jnp.float64)
        _, _, v = checker.check(ops, dlo, dhi)
        assert v.kind == "pass"
    size = checker.cache_size()
    if size is not None:
        assert size == 1, f"expected one compiled entry, saw {size}"
