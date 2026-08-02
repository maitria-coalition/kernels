"""XLA emission lane: evaluate a contraction row as a fusible fragment.

Lowers the row to ``jnp.einsum`` plus explicit broadcast insertion --
an *independently derived* formulation relative to the definitional
loop in ``reference`` (different algorithm, same answers), which is
exactly what the conformance battery wants from a lane (a lane that
re-implemented the reference's loop would correlate its bugs).

Two deliberate properties:

- **s64 end to end.** XLA's signed 64-bit arithmetic wraps mod 2^64,
  matching the reference's semantics exactly; the battery asserts
  bit-equality, not closeness.
- **einsum, never eager sum-product.** The row IS the einsum program;
  repeated indices within one factor lower to einsum's diagonal
  semantics, which coincide with the stride-sum semantics of the
  descriptor lanes (one linear form per factor).

Broadcast axes (output axes hit by no factor) are outside einsum's
vocabulary; they are inserted after the contraction by expand+broadcast,
which XLA fuses into the same fragment.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .rowir import Row, RowError

jax.config.update("jax_enable_x64", True)

_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def eval_row_xla(row: Row, operands: list[jax.Array]) -> jax.Array:
    """Evaluate the row over s64 device arrays of the induced shapes.

    Returns the s64 output array of shape ``row.out_shape``. Jit-safe;
    the row is trace-time data (baked into the program), the operands
    are runtime data.
    """
    if row.m > len(_LETTERS):
        raise RowError(f"XLA lane covers m<={len(_LETTERS)} axes, got {row.m}")
    if len(operands) != len(row.factors):
        raise RowError(
            f"row has {len(row.factors)} factors, got {len(operands)} operands"
        )
    for k, op in enumerate(operands):
        if tuple(op.shape) != row.factor_shape(k):
            raise RowError(
                f"operand {k}: expected induced shape {row.factor_shape(k)}, "
                f"got {tuple(op.shape)}"
            )
        if op.dtype != jnp.int64:
            raise RowError(f"operand {k}: expected int64, got {op.dtype}")

    hit = {a for f in row.factors for a in f}
    out_hit = [o for o in row.out if o in hit]

    if row.factors:
        subs = ",".join(
            "".join(_LETTERS[a] for a in f) for f in row.factors
        )
        out_sub = "".join(_LETTERS[o] for o in out_hit)
        res = jnp.einsum(f"{subs}->{out_sub}", *operands)
    else:
        # The empty tensor product: the multiplicative unit, broadcast
        # over the (necessarily all-broadcast) output axes.
        res = jnp.ones((), dtype=jnp.int64)

    # Insert broadcast axes at their positions in `out` order, then
    # broadcast to full extents. out_hit preserves relative order, so
    # positional insertion is exact.
    for j, o in enumerate(row.out):
        if o not in hit:
            res = jnp.expand_dims(res, axis=j)
    return jnp.broadcast_to(res, row.out_shape).astype(jnp.int64)
