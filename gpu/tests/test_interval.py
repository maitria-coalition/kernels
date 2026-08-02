"""Battery for the directed-rounding interval family (VCARM demo).

Three instruments per element:

1. **Oracle parity** — the device outputs equal the exact-rational
   oracle's directed roundings (``gpu/interval.py``): exact Fraction
   arithmetic, then the unique correctly-rounded-toward-the-bound f64.
   Compared bitwise (a directed rounding has exactly one correct
   answer).
2. **Enclosure** — lo <= exact <= hi, checked in exact arithmetic.
   This is the property outward rounding exists to preserve.
3. **Cross-lane** — the fused Pallas inline-PTX interval add equals
   the FFI kernel's, bitwise.
"""

from __future__ import annotations

import os
from fractions import Fraction

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

MAX_EXAMPLES = int(os.environ.get("BATTERY_EXAMPLES", "100"))

SETTINGS = settings(
    max_examples=MAX_EXAMPLES,
    deadline=None,
    # large_base_example: a batch of 128 IS the smallest natural input
    # here — one warpgroup is the pallas lane's hard floor, so the
    # strategy cannot shrink below it.
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.large_base_example,
    ],
)

F64 = st.one_of(
    st.floats(
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=True,
        min_value=-1e150,
        max_value=1e150,
    ),
    st.sampled_from(
        [0.0, -0.0, 1.0, -1.0, 2.0**-1074, -(2.0**-1074), 1 / 3, 0.1, 1e150]
    ),
)


def _gpu_available():
    try:
        import jax

        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


gpu_only = pytest.mark.skipif(
    not _gpu_available(), reason="no CUDA device; GPU lanes gate on-box"
)

N = 128  # one warpgroup's worth per example


@st.composite
def interval_batches(draw):
    a1 = draw(st.lists(F64, min_size=N, max_size=N))
    a2 = draw(st.lists(F64, min_size=N, max_size=N))
    b1 = draw(st.lists(F64, min_size=N, max_size=N))
    b2 = draw(st.lists(F64, min_size=N, max_size=N))
    al = [min(x, y) for x, y in zip(a1, a2)]
    ah = [max(x, y) for x, y in zip(a1, a2)]
    bl = [min(x, y) for x, y in zip(b1, b2)]
    bh = [max(x, y) for x, y in zip(b1, b2)]
    return al, ah, bl, bh


def _bits(x: float) -> int:
    import struct

    return struct.unpack("<Q", struct.pack("<d", x))[0]


@gpu_only
@SETTINGS
@given(interval_batches())
def test_ffi_interval_matches_oracle_and_encloses(batch):
    import jax.numpy as jnp

    from gpu.interval import ivl_addmul_ffi, ivl_addmul_ref

    al, ah, bl, bh = batch
    dev = [jnp.asarray(v, dtype=jnp.float64) for v in (al, ah, bl, bh)]
    slo, shi, plo, phi = (x.tolist() for x in ivl_addmul_ffi(*dev))
    for i in range(N):
        rlo, rhi, rplo, rphi = ivl_addmul_ref(al[i], ah[i], bl[i], bh[i])
        assert _bits(slo[i]) == _bits(rlo), (i, slo[i], rlo)
        assert _bits(shi[i]) == _bits(rhi), (i, shi[i], rhi)
        assert _bits(plo[i]) == _bits(rplo), (i, plo[i], rplo)
        assert _bits(phi[i]) == _bits(rphi), (i, phi[i], rphi)
        # Enclosure, in exact arithmetic.
        s_exact = Fraction(al[i]) + Fraction(bl[i])
        assert Fraction(slo[i]) <= s_exact
        assert Fraction(ah[i]) + Fraction(bh[i]) <= Fraction(shi[i])
        prods = [
            Fraction(x) * Fraction(y)
            for x, y in (
                (al[i], bl[i]),
                (al[i], bh[i]),
                (ah[i], bl[i]),
                (ah[i], bh[i]),
            )
        ]
        assert Fraction(plo[i]) <= min(prods)
        assert max(prods) <= Fraction(phi[i])


@gpu_only
@SETTINGS
@given(interval_batches())
def test_pallas_interval_add_matches_ffi(batch):
    import jax.numpy as jnp

    from gpu.interval import ivl_addmul_ffi
    from gpu.pallas_lane import interval_add_pallas
    from gpu.rowir import RowError

    al, ah, bl, bh = batch
    dev = [jnp.asarray(v, dtype=jnp.float64) for v in (al, ah, bl, bh)]
    try:
        lo_p, hi_p = interval_add_pallas(*dev)
    except RowError as e:
        pytest.skip(f"pallas interval core unavailable: {e}")
    slo, shi, _, _ = ivl_addmul_ffi(*dev)
    assert [_bits(x) for x in lo_p.tolist()] == [
        _bits(x) for x in slo.tolist()
    ]
    assert [_bits(x) for x in hi_p.tolist()] == [
        _bits(x) for x in shi.tolist()
    ]
