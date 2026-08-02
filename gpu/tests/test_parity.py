"""Conformance battery for the GPU/XLA contraction-row lanes.

The lane law, applied: every device lane must be *bit-identical* to
the definitional pure-Python reference on s64 -- including
two's-complement wraparound at full i64 range, diagonal index maps,
broadcast output axes, and the empty tensor product. Property-based
row generation covers the structural space; deterministic edges pin
the corners; an independently-derived formulation (the XLA einsum
lowering) diffs against the definitional loop, so a shared bug would
need to appear in two unrelated algorithms *and* two substrates.

GPU-lane tests skip (loudly) off-GPU; the receipts record the box and
seed of the run that gates dispatch.
"""

from __future__ import annotations

import os

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from gpu.reference import eval_row, eval_row_accumulate, wrap_i64
from gpu.rowir import MAX_FACTORS, MAX_M, Row, RowError

MAX_EXAMPLES = int(os.environ.get("BATTERY_EXAMPLES", "100"))

SETTINGS = settings(
    max_examples=MAX_EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

I64 = st.one_of(
    st.integers(-(2**63), 2**63 - 1),
    st.sampled_from(
        [0, 1, -1, 2, -2, 2**63 - 1, -(2**63), 2**32, -(2**32), 10**18]
    ),
)


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


@st.composite
def rows(draw, max_m=5, max_factors=3, max_extent=4, max_rank=3):
    """A random well-formed Row (γ-condition satisfied by construction)."""
    m = draw(st.integers(1, max_m))
    extents = tuple(
        draw(st.integers(1, max_extent)) for _ in range(m)
    )
    nf = draw(st.integers(0, max_factors))
    factors = tuple(
        tuple(
            draw(
                st.lists(
                    st.integers(0, m - 1), min_size=0, max_size=max_rank
                )
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
def rows_with_operands(draw, **kw):
    row = draw(rows(**kw))
    operands = [
        draw(
            st.lists(
                I64,
                min_size=_prod(row.factor_shape(k)),
                max_size=_prod(row.factor_shape(k)),
            )
        )
        for k in range(len(row.factors))
    ]
    return row, operands


# ---------------------------------------------------------------------------
# Reference-only: definitional edges (run anywhere, no jax needed).
# ---------------------------------------------------------------------------


class TestReferenceEdges:
    def test_empty_tensor_product_is_ones(self):
        # No factors: the multiplicative unit broadcast over out.
        row = Row(extents=(3, 2), factors=(), out=(0, 1))
        assert eval_row(row, []) == [1] * 6

    def test_diagonal(self):
        # One factor naming the same axis twice: the diagonal.
        row = Row(extents=(3,), factors=(((0, 0)),), out=(0,))
        buf = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert eval_row(row, [buf]) == [1, 5, 9]

    def test_broadcast_axis_constant(self):
        # Output axis hit by no factor: result constant along it.
        row = Row(extents=(2, 3), factors=((0,),), out=(0, 1))
        assert eval_row(row, [[7, 9]]) == [7, 7, 7, 9, 9, 9]

    def test_matmul_2x2(self):
        row = Row(extents=(2, 2, 2), factors=((0, 2), (2, 1)), out=(0, 1))
        a = [1, 2, 3, 4]  # [[1,2],[3,4]]
        b = [5, 6, 7, 8]  # [[5,6],[7,8]]
        assert eval_row(row, [a, b]) == [19, 22, 43, 50]

    def test_wraparound(self):
        row = Row(extents=(1,), factors=((0,), (0,)), out=())
        big = 2**62
        # big * 4 == 2^124 mod 2^64 ... exact: (2^62)*(2^62) = 2^124;
        # 2^124 mod 2^64 = 0.
        assert eval_row(row, [[big], [big]]) == [0]
        assert eval_row(row, [[2**63 - 1], [2]]) == [wrap_i64((2**63 - 1) * 2)]

    def test_axis_nobody_reads_refused(self):
        with pytest.raises(RowError):
            Row(extents=(2, 2), factors=((0,),), out=(0,))

    def test_scalar_factor(self):
        # Rank-0 factor: a scalar multiplier.
        row = Row(extents=(2,), factors=((), (0,)), out=(0,))
        assert eval_row(row, [[3], [10, 20]]) == [30, 60]

    def test_accumulate(self):
        row = Row(extents=(2,), factors=((0,),), out=(0,))
        assert eval_row_accumulate(row, [100, 2**63 - 1], [[1, 1]]) == [
            101,
            wrap_i64(2**63),
        ]


@SETTINGS
@given(rows_with_operands())
def test_reference_total(rw):
    """The reference is total on well-formed rows (and output length
    always matches the induced out shape)."""
    row, operands = rw
    out = eval_row(row, operands)
    assert len(out) == row.n_out
    assert all(-(2**63) <= x < 2**63 for x in out)


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


def _to_jnp(row, operands):
    import jax.numpy as jnp

    return [
        jnp.asarray(buf, dtype=jnp.int64).reshape(row.factor_shape(k))
        for k, buf in enumerate(operands)
    ]


@gpu_only
@SETTINGS
@given(rows_with_operands())
def test_xla_matches_reference(rw):
    row, operands = rw
    from gpu.xla_lane import eval_row_xla

    got = eval_row_xla(row, _to_jnp(row, operands))
    assert got.reshape(-1).tolist() == eval_row(row, operands)


@gpu_only
@SETTINGS
@given(rows_with_operands(max_factors=MAX_FACTORS, max_m=MAX_M))
def test_ffi_matches_reference(rw):
    row, operands = rw
    from gpu.ffi_lane import eval_row_ffi

    got = eval_row_ffi(row, _to_jnp(row, operands))
    assert got.reshape(-1).tolist() == eval_row(row, operands)


@gpu_only
@SETTINGS
@given(rows_with_operands(max_factors=MAX_FACTORS, max_m=MAX_M), st.data())
def test_ffi_accum_matches_reference(rw, data):
    row, operands = rw
    import jax.numpy as jnp

    from gpu.ffi_lane import eval_row_accum_ffi

    acc = data.draw(
        st.lists(I64, min_size=row.n_out, max_size=row.n_out)
    )
    acc_dev = jnp.asarray(acc, dtype=jnp.int64).reshape(row.out_shape)
    got = eval_row_accum_ffi(row, acc_dev, _to_jnp(row, operands))
    assert got.reshape(-1).tolist() == eval_row_accumulate(
        row, acc, operands
    )


# Pallas lane: the elementwise (hadamard-accumulate) normal form.
# Shapes obey the warpgroup constraint: prod(shape) % 128 == 0.

_PALLAS_SHAPES = [(128,), (256,), (16, 16), (2, 64), (4, 32, 4)]


@st.composite
def hadamard_rows_with_operands(draw):
    shape = draw(st.sampled_from(_PALLAS_SHAPES))
    axes = tuple(range(len(shape)))
    row = Row(extents=shape, factors=(axes, axes), out=axes)
    n = _prod(shape)
    x = draw(st.lists(I64, min_size=n, max_size=n))
    y = draw(st.lists(I64, min_size=n, max_size=n))
    acc = draw(st.lists(I64, min_size=n, max_size=n))
    return row, acc, [x, y]


def _hadamard_case(rw, use_inline_ptx):
    import jax.numpy as jnp

    from gpu.pallas_lane import eval_hadamard_accum_pallas

    row, acc, operands = rw
    acc_dev = jnp.asarray(acc, dtype=jnp.int64).reshape(row.out_shape)
    got = eval_hadamard_accum_pallas(
        row, acc_dev, _to_jnp(row, operands), use_inline_ptx=use_inline_ptx
    )
    assert got.reshape(-1).tolist() == eval_row_accumulate(
        row, acc, operands
    )


@gpu_only
@SETTINGS
@given(hadamard_rows_with_operands())
def test_pallas_hadamard_jnp_twin_matches_reference(rw):
    _hadamard_case(rw, use_inline_ptx=False)


@gpu_only
@SETTINGS
@given(hadamard_rows_with_operands())
def test_pallas_hadamard_inline_ptx_matches_reference(rw):
    _hadamard_case(rw, use_inline_ptx=True)
