"""Directed-rounding interval family: the VCARM demonstration
(ENGINEERING #9-#10), on-device and against its exact oracle.

Elementwise interval add and multiply on f64 where every lower bound
rounds toward minus infinity and every upper bound toward plus
infinity, per instruction, as literal PTX (``add.rm.f64`` /
``add.rp.f64`` / ``mul.rm.f64`` / ``mul.rp.f64`` in
``ffi/rowkernel.cu`` — ``.rm``/``.rp`` are PTX's spellings of the
directed modes; ptxas rejects the CUDA-intrinsic-style ``.rd``/``.ru``
suffixes, a lesson this line exists to keep). Outward rounding is how
enclosures stay enclosures; arithmetic like this cannot ride a fusion
compiler whose rewrite semantics are not contractual, which is why it
lives in the custom-call lane (and, fused, in the inline-PTX island of
the Pallas lane) rather than in emitted XLA ops.

The host reference here is the DATERWI exact path run in reverse as
an oracle: exact rational arithmetic (``fractions.Fraction``), then
the unique correctly-directed f64 rounding of the exact value. The
battery asserts the device outputs equal the oracle's and that the
enclosure property holds exactly: lo <= exact <= hi.

The directed scalar ops (``dir_sum``, ``dir_prod``, ``fmin_ieee``,
``fmax_ieee``, ``round_down``, ``round_up``) are TOTAL on f64: finite
arguments ride the exact-rational path; infinities, NaNs, overflow
(saturating per the directed mode: roundTowardNegative never produces
+inf from a finite value, roundTowardPositive never -inf), and
signed-zero rules follow IEEE 754 explicitly. They are shared
vocabulary for every interval lane in this package — the
interval-contraction references (``ivl_reference``) mirror the device
kernels op for op with exactly these functions.

Battery scope for THIS module's elementwise demo: finite inputs with
|x| <= 1e150 (keeps every intermediate finite). The contraction
battery carries its own scope via the ``Row.fits_f64`` gate.
"""

from __future__ import annotations

import ctypes
import math
import os
from fractions import Fraction

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

# ── the exact oracle ─────────────────────────────────────────────────

#: Largest finite f64 (2^1024 * (1 - 2^-53)).
MAX_F64 = math.nextafter(math.inf, 0.0)
_MAX_FR = Fraction(MAX_F64)


def round_down(fr: Fraction) -> float:
    """The largest f64 <= fr. Total: values beyond the finite range
    saturate per roundTowardNegative (+huge -> MAX_F64, never +inf;
    -huge -> -inf)."""
    if fr > _MAX_FR:
        return MAX_F64
    if fr < -_MAX_FR:
        return -math.inf
    f = float(fr)  # correctly rounded (round-nearest-even)
    return math.nextafter(f, -math.inf) if Fraction(f) > fr else f


def round_up(fr: Fraction) -> float:
    """The smallest f64 >= fr. Total: saturates per
    roundTowardPositive (+huge -> +inf; -huge -> -MAX_F64)."""
    if fr > _MAX_FR:
        return math.inf
    if fr < -_MAX_FR:
        return -MAX_F64
    f = float(fr)
    return math.nextafter(f, math.inf) if Fraction(f) < fr else f


def fmin_ieee(x: float, y: float) -> float:
    """Device-``fmin`` semantics: a single NaN operand is dropped (the
    other operand returns); equal-valued zeros order -0.0 below
    +0.0."""
    if math.isnan(x):
        return y
    if math.isnan(y):
        return x
    if x < y:
        return x
    if y < x:
        return y
    # equal values: prefer the negative-signed one
    return x if math.copysign(1.0, x) < 0 else y


def fmax_ieee(x: float, y: float) -> float:
    if math.isnan(x):
        return y
    if math.isnan(y):
        return x
    if x > y:
        return x
    if y > x:
        return y
    return x if math.copysign(1.0, x) > 0 else y


def dir_sum(x: float, y: float, up: bool) -> float:
    """IEEE-correct directed rounding of x + y, total on f64.

    Signed zeros (the sign of an exact zero sum): like-signed zero
    operands keep their sign in every direction; every OTHER
    exactly-zero sum is +0 — except in roundTowardNegative, where it
    is -0. The battery's first run caught the naive Fraction oracle
    missing this (exact rationals carry no signed zero); the device
    was right. Non-finite arguments follow IEEE addition (inf + -inf
    is NaN); overflow saturates per the directed mode (round_down /
    round_up handle it).
    """
    if math.isnan(x) or math.isnan(y):
        return math.nan
    if math.isinf(x) or math.isinf(y):
        if math.isinf(x) and math.isinf(y) and (x > 0) != (y > 0):
            return math.nan
        return x if math.isinf(x) else y
    fr = Fraction(x) + Fraction(y)
    if fr == 0:
        sx, sy = math.copysign(1.0, x), math.copysign(1.0, y)
        if x == 0.0 and y == 0.0 and sx == sy:
            return math.copysign(0.0, x)
        return 0.0 if up else -0.0
    return round_up(fr) if up else round_down(fr)


def dir_prod(x: float, y: float, up: bool) -> float:
    """IEEE-correct directed rounding of x * y, total on f64: an exact
    zero product (a finite factor is zero) carries the XOR of the
    operand signs in every rounding direction; 0 * inf is NaN; other
    non-finite products carry the sign XOR; nonzero finite products
    ride the generic path (which also gets underflow-to-zero signs
    right), with overflow saturating per the directed mode."""
    if math.isnan(x) or math.isnan(y):
        return math.nan
    if math.isinf(x) or math.isinf(y):
        if x == 0.0 or y == 0.0:
            return math.nan
        neg = (math.copysign(1.0, x) < 0) != (math.copysign(1.0, y) < 0)
        return -math.inf if neg else math.inf
    fr = Fraction(x) * Fraction(y)
    if fr == 0:
        neg = (math.copysign(1.0, x) < 0) != (math.copysign(1.0, y) < 0)
        return -0.0 if neg else 0.0
    return round_up(fr) if up else round_down(fr)


# Underscore aliases retained for in-repo history readability; the
# public names above are the maintained surface.
_fmin = fmin_ieee
_fmax = fmax_ieee
_dir_sum = dir_sum
_dir_prod = dir_prod


def ivl_addmul_ref(
    al: float, ah: float, bl: float, bh: float
) -> tuple[float, float, float, float]:
    """One element of the reference: (sum_lo, sum_hi, prod_lo, prod_hi),
    mirroring the device kernel operation for operation."""
    sum_lo = dir_sum(al, bl, up=False)
    sum_hi = dir_sum(ah, bh, up=True)
    pairs = ((al, bl), (al, bh), (ah, bl), (ah, bh))
    lo = [dir_prod(x, y, up=False) for x, y in pairs]
    hi = [dir_prod(x, y, up=True) for x, y in pairs]
    prod_lo = fmin_ieee(fmin_ieee(lo[0], lo[1]), fmin_ieee(lo[2], lo[3]))
    prod_hi = fmax_ieee(fmax_ieee(hi[0], hi[1]), fmax_ieee(hi[2], hi[3]))
    return sum_lo, sum_hi, prod_lo, prod_hi


# ── the device lane (FFI custom call) ────────────────────────────────

_LIB_PATH = os.path.join(os.path.dirname(__file__), "ffi", "librowkernel.so")
_registered = False


def _ensure_registered() -> None:
    global _registered
    if _registered:
        return
    lib = ctypes.cdll.LoadLibrary(_LIB_PATH)
    jax.ffi.register_ffi_target(
        "maitria_ivl_addmul", jax.ffi.pycapsule(lib.IvlAddMul), platform="CUDA"
    )
    _registered = True


def ivl_addmul_ffi(al, ah, bl, bh):
    """Device interval add+mul; returns (sum_lo, sum_hi, prod_lo, prod_hi)
    arrays of the common shape."""
    _ensure_registered()
    import jax.numpy as jnp

    shape = al.shape
    n = int(np.prod(shape)) if shape else 1
    spec = jax.ShapeDtypeStruct(shape, jnp.float64)
    call = jax.ffi.ffi_call(
        "maitria_ivl_addmul", (spec, spec, spec, spec)
    )
    return call(al, ah, bl, bh, n=np.int64(n))
