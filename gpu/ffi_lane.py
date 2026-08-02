"""FFI lane: the hand CUDA microkernel behind a jax.ffi custom call.

The workhorse GPU embedding: the row rides as data (the packed
descriptor, a trace-time constant array), so one compiled kernel
serves every row within the descriptor bounds and changing rows never
touches device code. Rows beyond the bounds get a typed refusal --
the caller's fallback is the XLA emission lane (``xla_lane``).

Two entry points:

- ``eval_row_ffi`` -- fresh output.
- ``eval_row_accum_ffi`` -- ``acc + row``, declared with
  ``input_output_aliases`` so XLA may hand the kernel coincident
  pointers and the update happens in place: the free-boundary pattern
  for applying a delta batch to resident state. Semantics do not
  depend on the alias being honored (the handler falls back to a
  device copy), only the boundary cost does.

The library is loaded and the targets registered lazily on first use;
``build.sh`` in ``gpu/ffi/`` produces ``librowkernel.so`` (offline
nvcc -- the pipeline of record for this lane).
"""

from __future__ import annotations

import ctypes
import os

import jax
import jax.numpy as jnp
import numpy as np

from .rowir import MAX_FACTORS, Row, RowError

jax.config.update("jax_enable_x64", True)

_LIB_PATH = os.path.join(os.path.dirname(__file__), "ffi", "librowkernel.so")
_registered = False


def _ensure_registered() -> None:
    global _registered
    if _registered:
        return
    if not os.path.exists(_LIB_PATH):
        raise RowError(
            f"{_LIB_PATH} not built; run gpu/ffi/build.sh on a CUDA box"
        )
    lib = ctypes.cdll.LoadLibrary(_LIB_PATH)
    jax.ffi.register_ffi_target(
        "maitria_row_eval", jax.ffi.pycapsule(lib.RowEval), platform="CUDA"
    )
    jax.ffi.register_ffi_target(
        "maitria_row_accum", jax.ffi.pycapsule(lib.RowAccum), platform="CUDA"
    )
    _registered = True


def _padded(row: Row, operands: list[jax.Array]) -> list[jax.Array]:
    """Operands padded to the kernel's fixed arity with 1-entry dummies
    (never dereferenced: the descriptor's factor count bounds the loop).
    The dummy is created per call, never cached at module scope — a
    cached array minted inside a jit trace would leak its tracer."""
    dummy = jnp.zeros((1,), dtype=jnp.int64)
    pads = MAX_FACTORS - len(operands)
    return list(operands) + [dummy] * pads


def _check(row: Row, operands: list[jax.Array]) -> None:
    if not row.fits_descriptor():
        raise RowError(
            f"row (m={row.m}, {len(row.factors)} factors) exceeds the "
            "descriptor bounds; use the XLA emission lane"
        )
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


def eval_row_ffi(row: Row, operands: list[jax.Array]) -> jax.Array:
    """Evaluate the row on device via the hand CUDA kernel."""
    _ensure_registered()
    _check(row, operands)
    desc = jnp.asarray(row.pack_descriptor(), dtype=jnp.int64)
    call = jax.ffi.ffi_call(
        "maitria_row_eval",
        jax.ShapeDtypeStruct(row.out_shape, jnp.int64),
    )
    out = call(desc, *_padded(row, operands), n_out=np.int64(row.n_out))
    return out


def eval_row_accum_ffi(
    row: Row, acc: jax.Array, operands: list[jax.Array]
) -> jax.Array:
    """``acc + row`` with the output aliased onto ``acc`` (in-place
    update of resident state; free at the custom-call boundary when the
    alias is honored)."""
    _ensure_registered()
    _check(row, operands)
    if tuple(acc.shape) != row.out_shape:
        raise RowError(
            f"accumulator shape {tuple(acc.shape)} != out shape {row.out_shape}"
        )
    desc = jnp.asarray(row.pack_descriptor(), dtype=jnp.int64)
    call = jax.ffi.ffi_call(
        "maitria_row_accum",
        jax.ShapeDtypeStruct(row.out_shape, jnp.int64),
        # Operand order: desc, acc, f0..f3 -> operand index 1 aliases
        # result 0.
        input_output_aliases={1: 0},
    )
    return call(desc, acc, *_padded(row, operands), n_out=np.int64(row.n_out))
