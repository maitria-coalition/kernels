"""Host directed-rounding lane: hardware rounding modes via fesetround.

A scalar CPU lane computing the SAME directed fold as the device
kernels and the directed reference — but by a genuinely different
mechanism: the C runtime's ``fesetround`` switches the hardware FPU
rounding mode around plain float operations, so every directed result
comes from silicon rather than from exact-rational emulation. That
makes it an independently-derived differential partner (ENGINEERING
#5) for ``ivl_reference.eval_row_interval_ref``: the two agree
bitwise iff glibc's mode switching, the host FPU, and the
Fraction-then-round emulation all implement the same IEEE 754 —
a three-cornered witness costing ~nothing, and one that runs (and
gates in CI) on any host, no GPU required.

Fragility, named and self-checked: the ``FE_*`` constants are
per-architecture ABI facts (x86-64 and aarch64 glibc values are
carried below); CPython must execute float arithmetic live on the
FPU (it does — but its peephole folder computes CONSTANT float
expressions at compile time under round-nearest, which is why the
self-check routes every operand through function arguments); and a
platform is free to ignore ``fesetround`` (C99 allows a stub).
``supported()`` therefore VERIFIES the mechanism at first use —
directed division must actually produce direction-dependent results —
and every entry point raises a typed refusal on hosts where it
doesn't, rather than computing round-nearest and calling it directed.

Scope: finite-range folds (the ``Row.fits_f64`` gate). The special
cases the exact-emulation ops spell by hand (signed zeros, inf, NaN)
are exactly what the hardware does natively — inside the gate none of
the non-finite cases arise, and the signed-zero cases are covered by
the battery's strategy floor.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import platform
from contextlib import contextmanager

from .interval import fmax_ieee, fmin_ieee
from .ivl_reference import _offsets, check_interval_operands
from .rowir import Row, RowError

# glibc <fenv.h> rounding-mode constants, per architecture ABI.
_FE_CONSTANTS: dict[str, tuple[int, int, int]] = {
    # machine: (FE_TONEAREST, FE_DOWNWARD, FE_UPWARD)
    "x86_64": (0, 0x400, 0x800),
    "aarch64": (0, 0x800000, 0x400000),
}

_state: dict | None = None


def _init() -> dict:
    global _state
    if _state is not None:
        return _state
    st: dict = {"ok": False, "why": "uninitialized"}
    machine = platform.machine()
    consts = _FE_CONSTANTS.get(machine)
    if consts is None:
        st["why"] = f"no FE_* constants recorded for machine {machine!r}"
        _state = st
        return st
    try:
        libm = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
    except OSError as e:
        st["why"] = f"libm unavailable: {e}"
        _state = st
        return st
    nearest, down, up = consts
    fesetround = libm.fesetround
    fesetround.argtypes = [ctypes.c_int]
    fesetround.restype = ctypes.c_int

    def div(x: float, y: float) -> float:
        # Function boundary keeps the peephole folder blind.
        return x / y

    try:
        ok_dn = fesetround(down) == 0
        lo = div(1.0, 3.0)
        ok_up = fesetround(up) == 0
        hi = div(1.0, 3.0)
    finally:
        fesetround(nearest)
    if not (ok_dn and ok_up and lo < hi):
        st["why"] = (
            f"fesetround did not take effect on {machine} "
            f"(rc down/up ok: {ok_dn}/{ok_up}, 1/3 down {lo!r} vs up {hi!r})"
        )
        _state = st
        return st
    st.update(
        ok=True, why="", libm=libm, fesetround=fesetround,
        nearest=nearest, down=down, up=up,
    )
    _state = st
    return st


def supported() -> bool:
    """Whether hardware directed rounding is live on this host (the
    self-check has actually run)."""
    return _init()["ok"]


def why_unsupported() -> str:
    return _init()["why"]


@contextmanager
def _mode(code: int):
    st = _init()
    st["fesetround"](code)
    try:
        yield
    finally:
        st["fesetround"](st["nearest"])


def eval_row_interval_hw(
    row: Row,
    operands: list[tuple[list[float], list[float]]],
    acc: tuple[list[float], list[float]] | None = None,
) -> tuple[list[float], list[float]]:
    """The normative fold (``ivl_reference`` module docstring), every
    directed operation performed by the host FPU under the matching
    rounding mode. Typed refusal off supported hosts."""
    st = _init()
    if not st["ok"]:
        raise RowError(f"hardware directed rounding unavailable: {st['why']}")
    check_interval_operands(row, operands)
    if acc is not None and (
        len(acc[0]) != row.n_out or len(acc[1]) != row.n_out
    ):
        raise RowError(
            f"accumulator has ({len(acc[0])}, {len(acc[1])}) entries, "
            f"output needs {row.n_out}"
        )
    down, up = st["down"], st["up"]

    def mul_dir(al, ah, bl, bh):
        with _mode(down):
            lo = fmin_ieee(
                fmin_ieee(al * bl, al * bh), fmin_ieee(ah * bl, ah * bh)
            )
        with _mode(up):
            hi = fmax_ieee(
                fmax_ieee(al * bl, al * bh), fmax_ieee(ah * bl, ah * bh)
            )
        return lo, hi

    def add_dir(al, ah, bl, bh):
        with _mode(down):
            lo = al + bl
        with _mode(up):
            hi = ah + bh
        return lo, hi

    los: list[float] = []
    his: list[float] = []
    for o, steps in enumerate(_offsets(row)):
        a_lo, a_hi = 0.0, 0.0
        for offs in steps:
            t_lo, t_hi = 1.0, 1.0
            for k, (lo_buf, hi_buf) in enumerate(operands):
                t_lo, t_hi = mul_dir(
                    t_lo, t_hi, lo_buf[offs[k]], hi_buf[offs[k]]
                )
            a_lo, a_hi = add_dir(a_lo, a_hi, t_lo, t_hi)
        if acc is not None:
            a_lo, a_hi = add_dir(acc[0][o], acc[1][o], a_lo, a_hi)
        los.append(a_lo)
        his.append(a_hi)
    return los, his
