"""Definitional reference evaluator for contraction rows.

This is the semantics: a direct transcription of the contraction
definition (see ``rowir`` module docstring), written for obviousness
over speed, in pure Python with no third-party dependencies -- so it
is independent of every accelerated lane it gates, including the XLA
emission lane (a reference that rode the library under test would not
be a differential partner).

Arithmetic: Python integers are exact, so the inner sum-product is
computed in exact integer arithmetic and reduced to two's-complement
64-bit once per output entry. Reduction mod 2^64 is a ring
homomorphism, so this equals wrapping at every intermediate step --
which is what the device lanes (XLA s64, CUDA 64-bit integer ops) do.
"""

from __future__ import annotations

from .rowir import Row, RowError

_MOD = 1 << 64
_HALF = 1 << 63


def wrap_i64(x: int) -> int:
    """Reduce an exact integer to signed two's-complement 64-bit."""
    x &= _MOD - 1
    return x - _MOD if x >= _HALF else x


def check_operands(row: Row, operands: list[list[int]]) -> None:
    """Typed refusal unless ``operands`` match the row's induced shapes."""
    if len(operands) != len(row.factors):
        raise RowError(
            f"row has {len(row.factors)} factors, got {len(operands)} operands"
        )
    for k, buf in enumerate(operands):
        want = 1
        for e in row.factor_shape(k):
            want *= e
        if len(buf) != want:
            raise RowError(
                f"operand {k}: expected {want} entries for induced shape "
                f"{row.factor_shape(k)}, got {len(buf)}"
            )


def eval_row(row: Row, operands: list[list[int]]) -> list[int]:
    """Evaluate the row over flat row-major i64 operand buffers.

    Returns the output as a flat row-major list (``row.out`` order),
    each entry wrapped to signed 64-bit.
    """
    check_operands(row, operands)
    strides = [row.strides(k) for k in range(len(row.factors))]
    out: list[int] = []
    for v in row.out_assignments():
        # Per-factor flat offset contributed by the output assignment.
        base = [
            sum(st.get(o, 0) * c for o, c in zip(row.out, v))
            for st in strides
        ]
        acc = 0
        for vp in row.red_assignments():
            term = 1
            for k, st in enumerate(strides):
                off = base[k] + sum(
                    st.get(r, 0) * c for r, c in zip(row.red, vp)
                )
                term *= operands[k][off]
            acc += term
        out.append(wrap_i64(acc))
    return out


def eval_row_accumulate(
    row: Row, acc: list[int], operands: list[list[int]]
) -> list[int]:
    """``acc + eval_row(...)`` entrywise, wrapped -- the in-place update
    shape the aliased device lane implements without materializing a
    fresh output."""
    fresh = eval_row(row, operands)
    if len(acc) != len(fresh):
        raise RowError(
            f"accumulator has {len(acc)} entries, output needs {len(fresh)}"
        )
    return [wrap_i64(a + f) for a, f in zip(acc, fresh)]
