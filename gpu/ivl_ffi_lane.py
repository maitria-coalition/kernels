"""FFI lane for interval-enclosure contraction rows.

The consumer-side (checking) fast path: the same rows-as-data CUDA
embedding as ``ffi_lane``, over the enclosure semiring — operands and
outputs are f64 interval tensors carried as separate lo/hi arrays,
every operation directed-rounded per instruction on device (PTX
``.rm``/``.rp``; ``ffi/rowkernel.cu``). Deliberately NO XLA-emission
twin exists for this lane: no emitted XLA op carries a rounding mode
and fusion may reassociate, so a sound enclosure cannot ride the
emission path at all — the custom call IS the sound embedding
(VCARM, ENGINEERING #9; the architectural argument is worked in
``gpu/README.md``).

Three entry points:

- ``eval_row_interval_ffi`` — fresh enclosure per output entry.
- ``eval_row_interval_accum_ffi`` — ``prev (+) row`` with both
  endpoint buffers aliased in place (the resident-state update
  shape); semantics never depend on the alias being honored.
- ``check_row_interval_ffi`` — evaluate AND classify on device
  against per-entry demanded bounds, reducing to a three-state
  ``rowir.Verdict`` (DATERWI, ENGINEERING #10): conclusive-pass /
  conclusive-fail(witness) / inconclusive-defer. The verdict
  reduction happens on device (atomicMin over conclusively-failing
  indices, one flag for straddles), so the boundary cost is two
  scalars, not an array walk. The witness is the LOWEST
  conclusively-failing index — entries before it may be
  inconclusive; a caller needing ground-truth-first-violation
  consults the returned enclosure arrays or the exact path.

Conclusiveness scope: within the ``Row.fits_f64`` fit gate every
endpoint is finite and the directed reference mirrors the device
bitwise. Outside it the lane stays verdict-sound (endpoints saturate
to +/-inf as honest half-unbounded enclosures, or void to NaN, which
classifies as inconclusive) but conclusive verdicts get rarer — the
gate is the routing predicate for whether this lane is WORTH
dispatching before the exact path, not a safety wall.
"""

from __future__ import annotations

import ctypes
import os

import jax
import jax.numpy as jnp
import numpy as np

from .rowir import MAX_FACTORS, Row, RowError, Verdict

jax.config.update("jax_enable_x64", True)

_LIB_PATH = os.path.join(os.path.dirname(__file__), "ffi", "librowkernel.so")
_registered = False

_NO_FAIL = (1 << 64) - 1  # ULLONG_MAX sentinel from the device reduction


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
        "maitria_ivl_row_eval",
        jax.ffi.pycapsule(lib.IvlRowEval),
        platform="CUDA",
    )
    jax.ffi.register_ffi_target(
        "maitria_ivl_row_accum",
        jax.ffi.pycapsule(lib.IvlRowAccum),
        platform="CUDA",
    )
    jax.ffi.register_ffi_target(
        "maitria_ivl_row_check",
        jax.ffi.pycapsule(lib.IvlRowCheck),
        platform="CUDA",
    )
    _registered = True


def _check(row: Row, operands: list[tuple[jax.Array, jax.Array]]) -> None:
    if not row.fits_descriptor():
        raise RowError(
            f"row (m={row.m}, {len(row.factors)} factors) exceeds the "
            "descriptor bounds; no wider interval lane exists — defer to "
            "the exact path (an XLA-emission fallback cannot be sound for "
            "enclosures)"
        )
    if len(operands) != len(row.factors):
        raise RowError(
            f"row has {len(row.factors)} factors, got {len(operands)} operands"
        )
    for k, (lo, hi) in enumerate(operands):
        for name, arr in (("lo", lo), ("hi", hi)):
            if tuple(arr.shape) != row.factor_shape(k):
                raise RowError(
                    f"operand {k} ({name}): expected induced shape "
                    f"{row.factor_shape(k)}, got {tuple(arr.shape)}"
                )
            if arr.dtype != jnp.float64:
                raise RowError(
                    f"operand {k} ({name}): expected float64, got {arr.dtype}"
                )


def _padded_pairs(
    operands: list[tuple[jax.Array, jax.Array]],
) -> list[jax.Array]:
    """Flatten (lo, hi) pairs to the kernel's fixed 8-buffer arity,
    padding with 1-entry dummies (never dereferenced: the descriptor's
    factor count bounds the loop). Per call, never cached — a cached
    array minted inside a jit trace would leak its tracer."""
    dummy = jnp.zeros((1,), dtype=jnp.float64)
    flat: list[jax.Array] = []
    for lo, hi in operands:
        flat += [lo, hi]
    flat += [dummy] * (2 * (MAX_FACTORS - len(operands)))
    return flat


def eval_row_interval_ffi(
    row: Row, operands: list[tuple[jax.Array, jax.Array]]
) -> tuple[jax.Array, jax.Array]:
    """Fresh interval evaluation on device; returns ``(out_lo,
    out_hi)`` of the row's output shape."""
    _ensure_registered()
    _check(row, operands)
    desc = jnp.asarray(row.pack_descriptor(), dtype=jnp.int64)
    spec = jax.ShapeDtypeStruct(row.out_shape, jnp.float64)
    call = jax.ffi.ffi_call("maitria_ivl_row_eval", (spec, spec))
    return call(desc, *_padded_pairs(operands), n_out=np.int64(row.n_out))


def eval_row_interval_accum_ffi(
    row: Row,
    acc_lo: jax.Array,
    acc_hi: jax.Array,
    operands: list[tuple[jax.Array, jax.Array]],
) -> tuple[jax.Array, jax.Array]:
    """``prev (+) row`` with both endpoint outputs aliased onto the
    accumulator buffers (in-place update of resident enclosure state;
    free at the custom-call boundary when honored)."""
    _ensure_registered()
    _check(row, operands)
    for name, arr in (("acc_lo", acc_lo), ("acc_hi", acc_hi)):
        if tuple(arr.shape) != row.out_shape:
            raise RowError(
                f"{name} shape {tuple(arr.shape)} != out shape {row.out_shape}"
            )
    desc = jnp.asarray(row.pack_descriptor(), dtype=jnp.int64)
    spec = jax.ShapeDtypeStruct(row.out_shape, jnp.float64)
    call = jax.ffi.ffi_call(
        "maitria_ivl_row_accum",
        (spec, spec),
        # Operand order: desc, acc_lo, acc_hi, f0l.. -> results 0, 1.
        input_output_aliases={1: 0, 2: 1},
    )
    return call(
        desc, acc_lo, acc_hi, *_padded_pairs(operands),
        n_out=np.int64(row.n_out),
    )


def check_row_interval_ffi(
    row: Row,
    operands: list[tuple[jax.Array, jax.Array]],
    dlo: jax.Array,
    dhi: jax.Array,
) -> tuple[jax.Array, jax.Array, Verdict]:
    """Evaluate and classify on device; returns ``(out_lo, out_hi,
    verdict)``. Reading the verdict synchronizes with the device (two
    scalars) — this is the checking boundary, the caller wants the
    answer. Demand arrays are per-entry ``[dlo, dhi]`` in the row's
    output shape; the caller's contract is ``dlo <= dhi`` entrywise
    (a malformed demand degrades to fail-or-inconclusive, never to a
    false pass — the pass predicate cannot hold when the demand is
    empty). NaN demands classify as inconclusive, never conclusive.
    """
    _ensure_registered()
    _check(row, operands)
    for name, arr in (("dlo", dlo), ("dhi", dhi)):
        if tuple(arr.shape) != row.out_shape:
            raise RowError(
                f"{name} shape {tuple(arr.shape)} != out shape {row.out_shape}"
            )
        if arr.dtype != jnp.float64:
            raise RowError(f"{name}: expected float64, got {arr.dtype}")
    desc = jnp.asarray(row.pack_descriptor(), dtype=jnp.int64)
    spec = jax.ShapeDtypeStruct(row.out_shape, jnp.float64)
    call = jax.ffi.ffi_call(
        "maitria_ivl_row_check",
        (
            spec,
            spec,
            jax.ShapeDtypeStruct((1,), jnp.uint64),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        ),
    )
    out_lo, out_hi, fail_idx, incon = call(
        desc, dlo, dhi, *_padded_pairs(operands), n_out=np.int64(row.n_out)
    )
    fail = int(fail_idx[0])
    if fail != _NO_FAIL:
        return out_lo, out_hi, Verdict.conclusive_fail(fail)
    if int(incon[0]) != 0:
        return out_lo, out_hi, Verdict.inconclusive()
    return out_lo, out_hi, Verdict.conclusive_pass()
