"""Checker embed for interval-enclosure contraction rows.

The consumer boundary of record, packaged: a geolog-side checking
program holds contraction rows (index map + output set as data — the
house tensor-contraction row form) and wants each candidate's outputs
screened against demanded bounds fast, with soundness never resting on
float silicon alone. This module is that embed, in three pieces:

1. **``RowChecker``** — the jitted device lane. One instance per row
   caches a ``jax.jit``-compiled program around the ``IvlRowCheck``
   custom call (the row descriptor is baked in as a static constant),
   so screening MANY candidates against ONE row pays kernel price
   (~1 ms at 65K entries × red 256 on the founding box) instead of
   wrapper price (~3.8 ms): dispatch is retrace-free, and the verdict
   scalars stay on device until ``verdict()`` reads them — the
   "embed jitted, read the two scalars late" pattern the founding
   receipts named. Built lazily; importing this module needs no jax.

2. **``screen_row``** — the DATERWI ladder around any one screen lane
   (device / fesetround host / directed reference):

   - a **conclusive-fail** from the float screen is confirmed on the
     exact-rational path at the witness entry only (cost one entry's
     reduction domain, not the whole row) — the float lane finds the
     witness, exact arithmetic owns the verdict;
   - a **conclusive-pass** is sound by the enclosure argument (VCARM:
     both conclusive predicates quantify over a superset of the
     possible values) and is accepted as-is by default; callers whose
     assurance story wants every accepting verdict exactly derived
     pass ``confirm="all"`` and the pass is re-proven on the full
     exact path — the float lane is then purely a witness-finder;
   - **INCONCLUSIVE** always defers to the full exact path — never
     rounded into a decision (ENGINEERING #10).

   The returned verdict is therefore always conclusive, and every
   ``fail`` witness has been re-derived in exact rational arithmetic
   regardless of ``confirm`` mode.

3. **``row_from_bindings`` / ``screen_bound_contract``** — the seam to
   contract-shaped row data ``(extents, in_bindings, out_binding)`` as
   the geolog plan IR carries it. ``out_binding`` may repeat axes (an
   Eq-identified kappa lowers to a diagonal-embedded output); the row
   is built in CLASS form (first-occurrence-deduped output) and the
   off-diagonal entries — structurally exact zero, never evaluated —
   are classified host-side against their demands in exact arithmetic.
   Broadcast axes (hit by no factor) ride the row form natively.

Soundness inventory, explicit: conclusive-fail → exact (witness
re-derived); inconclusive → exact (full row); conclusive-pass → the
enclosure argument (or exact under ``confirm="all"``); off-diagonal
zeros → exact by construction. The one screen outcome trusted without
exact re-derivation is the default-mode pass, whose warrant is the
VCARM enclosure property — and ``confirm="all"`` removes even that.

If a float screen's conclusive-fail ever FAILS its exact confirmation
(possible only via an unsound enclosure — mis-ordered operand
endpoints, or a lane bug), the ladder does not propagate it: the row
falls through to the full exact path and the report flags
``screen_disagreed`` — the discrepancy is cargo, never a verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import product as _cartesian
from math import isfinite, isnan, prod

from .ivl_reference import (
    check_interval_operands,
    check_row_interval_exact,
    classify_entry,
    eval_row_interval_exact,
    eval_row_interval_ref,
    verdict_from_classes,
)
from .rowir import Row, RowError, Verdict

# ── per-entry exact arithmetic (the witness-confirmation economics) ──


def exact_entry(
    row: Row, operands: list[tuple[list[float], list[float]]], o: int
) -> tuple[Fraction, Fraction]:
    """The exact rational interval of ONE output entry (flat index
    ``o``, row-major in ``out``): cost proportional to ``red_total``,
    not ``n_out * red_total`` — what makes exact confirmation of a
    single fail witness cheap. Same fold as the exact stratum
    (order-independent over the rationals, so the shared order is a
    convenience, not a convention this result depends on)."""
    check_interval_operands(row, operands)
    if not (0 <= o < row.n_out):
        raise RowError(f"entry {o} out of range for {row.n_out} outputs")
    for k, (lo_buf, hi_buf) in enumerate(operands):
        for name, buf in (("lo", lo_buf), ("hi", hi_buf)):
            for i, x in enumerate(buf):
                if not isfinite(x):
                    raise RowError(
                        f"operand {k} ({name})[{i}] = {x!r}: the exact "
                        "stratum requires finite endpoints"
                    )
    # Decompose the flat output index, row-major over out_shape.
    coords = []
    rem = o
    for e in reversed(row.out_shape):
        coords.append(rem % e)
        rem //= e
    coords.reverse()
    strides = [row.strides(k) for k in range(len(row.factors))]
    base = [
        sum(st.get(ax, 0) * c for ax, c in zip(row.out, coords))
        for st in strides
    ]
    acc_lo, acc_hi = Fraction(0), Fraction(0)
    for vp in _cartesian(*(range(row.extents[r]) for r in row.red)):
        t_lo, t_hi = Fraction(1), Fraction(1)
        for k, (lo_buf, hi_buf) in enumerate(operands):
            off = base[k] + sum(
                strides[k].get(r, 0) * c for r, c in zip(row.red, vp)
            )
            a, b = Fraction(lo_buf[off]), Fraction(hi_buf[off])
            ps = (t_lo * a, t_lo * b, t_hi * a, t_hi * b)
            t_lo, t_hi = min(ps), max(ps)
        acc_lo += t_lo
        acc_hi += t_hi
    return acc_lo, acc_hi


def entry_exactly_violates(
    row: Row,
    operands: list[tuple[list[float], list[float]]],
    dlo: float,
    dhi: float,
    o: int,
) -> bool:
    """Whether entry ``o``'s EXACT interval fails containment in
    ``[dlo, dhi]`` — the exact stratum's fail predicate (negation of
    the universal property: some permitted value violates), at one
    entry. NaN demands are refused, as everywhere on the exact path."""
    if isnan(dlo) or isnan(dhi):
        raise RowError("NaN is not a demand")
    lo_fr, hi_fr = exact_entry(row, operands, o)
    lo_ok = (
        True
        if dlo == -math.inf
        else (not math.isinf(dlo)) and Fraction(dlo) <= lo_fr
    )
    hi_ok = (
        True
        if dhi == math.inf
        else (not math.isinf(dhi)) and hi_fr <= Fraction(dhi)
    )
    return not (lo_ok and hi_ok)


# ── the jitted device lane ───────────────────────────────────────────


class RowChecker:
    """A per-row, jit-cached handle on the ``IvlRowCheck`` device
    kernel. Construction is cheap and jax-free; the first
    ``check_arrays`` call imports jax, registers the FFI targets, and
    compiles ONE program with the packed descriptor as a baked-in
    constant. Every later call with the same operand shapes reuses the
    compiled executable (jit cache) — the retrace-free screening loop.

    ``check_arrays`` returns device arrays only (no host sync): the
    enclosure pair plus the two verdict scalars. ``verdict`` performs
    the deferred device-to-host read. ``check`` is the convenience
    composition of the two for callers that want the answer now.
    """

    def __init__(self, row: Row):
        if not row.fits_descriptor():
            raise RowError(
                f"row (m={row.m}, {len(row.factors)} factors) exceeds the "
                "descriptor bounds; no wider interval device lane exists — "
                "defer to the exact path"
            )
        self.row = row
        self._run = None

    def _build(self):
        import jax
        import jax.numpy as jnp
        import numpy as np

        from .ivl_ffi_lane import _ensure_registered

        _ensure_registered()
        row = self.row
        desc = tuple(row.pack_descriptor())
        n_factors = len(row.factors)
        spec = jax.ShapeDtypeStruct(row.out_shape, jnp.float64)
        from .rowir import MAX_FACTORS

        @jax.jit
        def run(dlo, dhi, *flat_ops):
            desc_arr = jnp.asarray(desc, dtype=jnp.int64)
            dummy = jnp.zeros((1,), dtype=jnp.float64)
            padded = list(flat_ops) + [dummy] * (
                2 * (MAX_FACTORS - n_factors)
            )
            call = jax.ffi.ffi_call(
                "maitria_ivl_row_check",
                (
                    spec,
                    spec,
                    jax.ShapeDtypeStruct((1,), jnp.uint64),
                    jax.ShapeDtypeStruct((1,), jnp.int32),
                ),
            )
            return call(
                desc_arr, dlo, dhi, *padded, n_out=np.int64(row.n_out)
            )

        self._run = run

    def check_arrays(self, operands, dlo, dhi):
        """Jitted device check, NO host synchronization. ``operands``
        is a list of ``(lo, hi)`` jax arrays in the row's induced
        shapes; returns ``(out_lo, out_hi, fail_idx, incon)`` — all
        device arrays; the verdict scalars are read late via
        ``verdict``."""
        if self._run is None:
            self._build()
        if len(operands) != len(self.row.factors):
            raise RowError(
                f"row has {len(self.row.factors)} factors, "
                f"got {len(operands)} operands"
            )
        flat = [x for pair in operands for x in pair]
        return self._run(dlo, dhi, *flat)

    @staticmethod
    def verdict(fail_idx, incon) -> Verdict:
        """The deferred verdict read: two scalars cross the boundary
        here and nowhere else."""
        no_fail = (1 << 64) - 1
        fail = int(fail_idx[0])
        if fail != no_fail:
            return Verdict.conclusive_fail(fail)
        if int(incon[0]) != 0:
            return Verdict.inconclusive()
        return Verdict.conclusive_pass()

    def check(self, operands, dlo, dhi):
        """Convenience: jitted check + immediate verdict read."""
        out_lo, out_hi, fail_idx, incon = self.check_arrays(
            operands, dlo, dhi
        )
        return out_lo, out_hi, self.verdict(fail_idx, incon)

    def cache_size(self) -> int | None:
        """The underlying jit cache's entry count, where the jax
        version exposes it (an instrument for the retrace-free claim;
        ``None`` means unavailable, not zero)."""
        if self._run is None:
            return 0
        try:
            return self._run._cache_size()
        except AttributeError:
            return None


_checker_cache: dict[Row, RowChecker] = {}


def get_checker(row: Row) -> RowChecker:
    """The per-row ``RowChecker``, memoized module-wide: every caller
    screening the same row shares one compiled program (``Row`` is
    frozen and hashes by its defining data). The direct constructor
    remains available for callers managing their own lifetimes."""
    ch = _checker_cache.get(row)
    if ch is None:
        ch = _checker_cache[row] = RowChecker(row)
    return ch


def device_available() -> bool:
    """Whether the jitted device lane can run here: jax imports, a
    CUDA backend is live, and the compiled FFI library exists. False
    is a routing fact (fall to the host lanes), never an error."""
    try:
        import jax

        if not any(d.platform == "gpu" for d in jax.devices()):
            return False
    except Exception:
        return False
    import os

    from . import ivl_ffi_lane

    return os.path.exists(ivl_ffi_lane._LIB_PATH)


# ── the screening ladder (DATERWI at the embed boundary) ─────────────


@dataclass(frozen=True)
class ScreenReport:
    """One screened row: the final (always conclusive) verdict plus
    the provenance a promotion ladder wants.

    - ``verdict`` — the sound answer. ``fail`` witnesses are always
      exactly re-derived; ``pass`` rests on the enclosure argument
      unless ``exact_used`` says otherwise.
    - ``screen_lane`` — "device" | "hw" | "ref" | "none" (no float
      screen ran: routed straight to exact).
    - ``screen_verdict`` — the float lane's raw three-state verdict,
      before the ladder resolved it (None when no screen ran).
    - ``exact_used`` — "none" | "witness" (one entry) | "full".
    - ``fits`` — the ``fits_f64`` routing gate's answer when operand
      bounds were supplied; None when they weren't.
    - ``screen_disagreed`` — True iff a float conclusive-fail failed
      its exact confirmation (unsound enclosure input or lane bug;
      the ladder fell through to full exact — see module docstring).
    """

    verdict: Verdict
    screen_lane: str
    screen_verdict: Verdict | None
    exact_used: str
    fits: bool | None
    screen_disagreed: bool = False


def _screen_lane_eval(
    lane: str,
    row: Row,
    operands: list[tuple[list[float], list[float]]],
    dlo: list[float],
    dhi: list[float],
):
    """Run one float screen lane, returning (los, his, Verdict)."""
    if lane == "device":
        import jax.numpy as jnp

        checker = get_checker(row)
        ops = [
            (
                jnp.asarray(lo, dtype=jnp.float64).reshape(
                    row.factor_shape(k)
                ),
                jnp.asarray(hi, dtype=jnp.float64).reshape(
                    row.factor_shape(k)
                ),
            )
            for k, (lo, hi) in enumerate(operands)
        ]
        dlo_a = jnp.asarray(dlo, dtype=jnp.float64).reshape(row.out_shape)
        dhi_a = jnp.asarray(dhi, dtype=jnp.float64).reshape(row.out_shape)
        out_lo, out_hi, v = checker.check(ops, dlo_a, dhi_a)
        return (
            [float(x) for x in out_lo.reshape(-1)],
            [float(x) for x in out_hi.reshape(-1)],
            v,
        )
    if lane == "hw":
        from . import ivl_host_lane

        los, his = ivl_host_lane.eval_row_interval_hw(row, operands)
    elif lane == "ref":
        los, his = eval_row_interval_ref(row, operands)
    else:
        raise RowError(f"unknown screen lane {lane!r}")
    classes = [
        classify_entry(lo, hi, d_lo, d_hi)
        for lo, hi, d_lo, d_hi in zip(los, his, dlo, dhi)
    ]
    return los, his, verdict_from_classes(classes)


def _pick_lane(row: Row) -> str:
    if row.fits_descriptor() and device_available():
        return "device"
    from . import ivl_host_lane

    if ivl_host_lane.supported():
        return "hw"
    return "ref"


def screen_row(
    row: Row,
    operands: list[tuple[list[float], list[float]]],
    dlo: list[float],
    dhi: list[float],
    *,
    lane: str = "auto",
    confirm: str = "fail",
    operand_bounds: list | None = None,
) -> ScreenReport:
    """Screen one row's outputs against per-entry demands and resolve
    to a conclusive verdict (module docstring has the ladder). Operand
    and demand buffers use the reference convention: flat row-major
    python lists. Repeated screening of MANY candidates against ONE
    row at device speed should hold a ``RowChecker`` directly and run
    this ladder's exact arms on the (rare) non-pass outcomes; this
    function is the one-shot orchestrator.

    ``lane``: "auto" (device if live, else hardware-directed host,
    else directed reference), or one of "device" | "hw" | "ref" |
    "exact" (skip the float screen entirely).
    ``confirm``: "fail" (default — exact-confirm fail witnesses),
    "all" (additionally re-prove passes on the full exact path),
    "none" (trust the enclosure argument for both conclusive arms;
    INCONCLUSIVE still defers — DATERWI is not optional).
    ``operand_bounds``: per-operand endpoint magnitude bounds; when
    given, ``Row.fits_f64`` gates the float screen — a False routes
    straight to exact (economics, not safety: outside the gate the
    float lanes stay verdict-sound but conclusive answers thin out).
    """
    if confirm not in ("fail", "all", "none"):
        raise RowError(f"unknown confirm mode {confirm!r}")
    if len(dlo) != row.n_out or len(dhi) != row.n_out:
        raise RowError(
            f"demand arrays have ({len(dlo)}, {len(dhi)}) entries, "
            f"output needs {row.n_out}"
        )
    fits: bool | None = None
    if operand_bounds is not None:
        fits = row.fits_f64(operand_bounds)
    if lane == "exact" or fits is False:
        v = check_row_interval_exact(row, operands, dlo, dhi)
        return ScreenReport(v, "none", None, "full", fits)
    if lane == "auto":
        lane = _pick_lane(row)

    _los, _his, sv = _screen_lane_eval(lane, row, operands, dlo, dhi)

    if sv.kind == "fail":
        if confirm == "none":
            return ScreenReport(sv, lane, sv, "none", fits)
        assert sv.witness is not None
        if entry_exactly_violates(
            row, operands, dlo[sv.witness], dhi[sv.witness], sv.witness
        ):
            return ScreenReport(sv, lane, sv, "witness", fits)
        # The float screen's fail did not survive exact arithmetic:
        # unsound enclosure input or a lane defect. Never propagate —
        # the exact path owns the row now, and the flag is the cargo.
        v = check_row_interval_exact(row, operands, dlo, dhi)
        return ScreenReport(v, lane, sv, "full", fits, True)
    if sv.kind == "pass":
        if confirm == "all":
            v = check_row_interval_exact(row, operands, dlo, dhi)
            return ScreenReport(v, lane, sv, "full", fits, v.kind != "pass")
        return ScreenReport(sv, lane, sv, "none", fits)
    # INCONCLUSIVE: the exact path is the authority (DATERWI).
    v = check_row_interval_exact(row, operands, dlo, dhi)
    return ScreenReport(v, lane, sv, "full", fits)


# ── the contract-shaped seam (geolog plan-IR row data) ───────────────


def row_from_bindings(
    extents: tuple[int, ...],
    in_bindings: tuple[tuple[int, ...], ...],
    out_binding: tuple[int, ...],
) -> tuple[Row, tuple[list[int], list[int]] | None]:
    """Build a ``Row`` from contract-shaped data as the geolog plan IR
    carries it. ``out_binding`` may REPEAT axes (a diagonal-embedded
    output); the returned row is the CLASS form — first-occurrence-
    deduped output — plus ``embed = (class_order, axis_class)`` when
    the binding repeats (None when injective, the common case). The
    row constructor's γ discipline (an axis nobody reads is refused)
    is inherited unchanged."""
    class_order = list(dict.fromkeys(out_binding))
    row = Row(
        tuple(extents),
        tuple(tuple(b) for b in in_bindings),
        tuple(class_order),
    )
    if len(class_order) == len(out_binding):
        return row, None
    axis_class = [class_order.index(b) for b in out_binding]
    return row, (class_order, axis_class)


def screen_bound_contract(
    extents: tuple[int, ...],
    in_bindings: tuple[tuple[int, ...], ...],
    out_binding: tuple[int, ...],
    operands: list[tuple[list[float], list[float]]],
    dlo: list[float],
    dhi: list[float],
    **screen_kw,
) -> ScreenReport:
    """``screen_row`` for contract-shaped row data, demands given over
    the FULL output shape (``out_binding`` order, row-major, flat).
    Injective bindings pass straight through. For diagonal-embedded
    outputs the checker splits exactly:

    - off-diagonal entries hold EXACT zero by construction (never
      evaluated), so their classification against their demands is
      exact arithmetic on the spot — a conclusive-fail there needs no
      float screen and no confirmation;
    - diagonal entries carry the class-form row's values; their
      demands are gathered to class shape and the ordinary ladder
      runs.

    Fail witnesses index the FULL output space. The witness is a
    genuine violation but not necessarily the lowest-index one (the
    off-diagonal walk is checked first — it is free)."""
    row, embed = row_from_bindings(extents, in_bindings, out_binding)
    if embed is None:
        return screen_row(row, operands, dlo, dhi, **screen_kw)
    class_order, axis_class = embed
    full_shape = tuple(extents[b] for b in out_binding)
    n_full = prod(full_shape)
    if len(dlo) != n_full or len(dhi) != n_full:
        raise RowError(
            f"demand arrays have ({len(dlo)}, {len(dhi)}) entries, "
            f"full output needs {n_full}"
        )
    # Walk the full output space once, host-side: split diagonal
    # positions (gather demands to class shape) from off-diagonal ones
    # (classify exact zero now).
    class_extents = [extents[c] for c in class_order]
    dlo_diag = [0.0] * prod(class_extents)
    dhi_diag = [0.0] * prod(class_extents)
    off_fail: int | None = None
    for j, coords in enumerate(_cartesian(*(range(e) for e in full_shape))):
        for d in (dlo[j], dhi[j]):
            if isnan(d):
                raise RowError(f"demand[{j}] is NaN: not a demand")
        cls_coords = [None] * len(class_order)
        diagonal = True
        for t, c in enumerate(coords):
            cc = cls_coords[axis_class[t]]
            if cc is None:
                cls_coords[axis_class[t]] = c
            elif cc != c:
                diagonal = False
                break
        if diagonal:
            ci = 0
            for cc, e in zip(cls_coords, class_extents):
                ci = ci * e + cc
            dlo_diag[ci] = dlo[j]
            dhi_diag[ci] = dhi[j]
        else:
            # For the exact value 0 and non-NaN demands the classifier
            # is a trichotomy with no inconclusive arm: pass iff
            # dlo <= 0 <= dhi, else 0 < dlo or 0 > dhi — a fail.
            if classify_entry(0.0, 0.0, dlo[j], dhi[j]) == 2 and (
                off_fail is None
            ):
                off_fail = j
    if off_fail is not None:
        # Exact by construction: the entry IS zero; zero violates.
        return ScreenReport(
            Verdict.conclusive_fail(off_fail), "none", None, "none", None
        )
    report = screen_row(row, operands, dlo_diag, dhi_diag, **screen_kw)
    if report.verdict.kind == "fail":
        # Map the class-space witness back to its full-space index.
        w = report.verdict.witness
        cls_coords = []
        rem = w
        for e in reversed(class_extents):
            cls_coords.append(rem % e)
            rem //= e
        cls_coords.reverse()
        full_idx = 0
        for t, e in zip(range(len(out_binding)), full_shape):
            full_idx = full_idx * e + cls_coords[axis_class[t]]
        return ScreenReport(
            Verdict.conclusive_fail(full_idx),
            report.screen_lane,
            report.screen_verdict,
            report.exact_used,
            report.fits,
            report.screen_disagreed,
        )
    return report
