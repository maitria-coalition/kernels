"""Provisional contraction-row IR.

A *contraction row* is the data form of one semiring contraction: an
index map and an output set, never an eagerly expanded sum-product.
Given extents ``B : [M] -> Nat``, a row holds

- ``factors`` -- the index map ``a``, factor by factor: each factor is
  a tuple of ``[M]``-indices, one per operand axis (repeated indices
  within one factor select the diagonal; the operand's shape is
  *induced* from the extents, so the map is well-typed by
  construction);
- ``out`` -- the ordered output axes ``O`` (a subset of ``[M]``).

Axes of ``[M]`` hit by no factor must appear in ``out`` (the broadcast
axes: shape-relevant, value-irrelevant -- the result is constant along
them). An axis that is neither hit nor an output would silently scale
results by its extent; it is refused at construction rather than
discouraged.

Semantics (the definitional evaluator transcribes this directly; see
``reference.py``): for each assignment ``v`` of the output axes, sum
over assignments ``v'`` of the remaining axes the product over factors
of the operand entry at the coordinates the index map selects. The
scalar carrier in this module family is the ring of signed 64-bit
integers with two's-complement wraparound -- equivalently exact
integer arithmetic reduced mod 2^64 at the end, since reduction is a
ring homomorphism. Callers that cannot tolerate wraparound fit-check
upstream and route wide values to their arbitrary-precision path;
nothing here rounds.

PROVISIONAL: this module is a deliberately thin stand-in. The compiler
wave's architecture work will settle the canonical row IR (richer
semiring vocabulary, plan-level metadata, digests); every consumer in
this directory reads rows only through this module's small surface
(`Row`, `pack_descriptor`), so re-targeting onto the settled IR is
expected to be one localized change, and nothing downstream may grow
dependencies on this exact encoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product as _cartesian
from math import prod

# Bounds of the fixed-width GPU descriptor (see pack_descriptor and
# ffi/rowkernel.cu, which must agree). The IR itself is unbounded; the
# device lanes refuse rows beyond these bounds to the caller's
# fallback lane (fit-detection, never silent truncation).
MAX_M = 8
MAX_FACTORS = 4
DESC_LEN = 4 + MAX_M + MAX_M + MAX_FACTORS * MAX_M * 2  # = 84


class RowError(ValueError):
    """A typed refusal: the row (or its operands) is malformed."""


@dataclass(frozen=True)
class Row:
    """One contraction row: index map + output set as data.

    ``extents[m]`` is the extent of axis ``m``; ``factors[k][i]`` is
    the ``[M]``-index that the ``i``-th axis of operand ``k`` maps to;
    ``out`` is the ordered tuple of output axes.
    """

    extents: tuple[int, ...]
    factors: tuple[tuple[int, ...], ...]
    out: tuple[int, ...]

    # Derived, filled by __post_init__.
    red: tuple[int, ...] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        m = len(self.extents)
        for e in self.extents:
            if not (isinstance(e, int) and e >= 1):
                raise RowError(f"extent {e!r}: extents must be integers >= 1")
        for k, f in enumerate(self.factors):
            for a in f:
                if not (0 <= a < m):
                    raise RowError(f"factor {k} names axis {a}, but [M] has {m} axes")
        seen = set()
        for o in self.out:
            if not (0 <= o < m):
                raise RowError(f"out names axis {o}, but [M] has {m} axes")
            if o in seen:
                raise RowError(f"out repeats axis {o}")
            seen.add(o)
        hit = {a for f in self.factors for a in f}
        for a in range(m):
            if a not in hit and a not in self.out:
                raise RowError(
                    f"axis {a} is hit by no factor and is not an output; "
                    "an axis nobody reads must not exist (drop it or add it to out)"
                )
        object.__setattr__(
            self, "red", tuple(a for a in range(m) if a not in seen)
        )

    # -- induced shapes ------------------------------------------------

    @property
    def m(self) -> int:
        """Number of axes in ``[M]``."""
        return len(self.extents)

    def factor_shape(self, k: int) -> tuple[int, ...]:
        """The induced shape of operand ``k`` (row-major)."""
        return tuple(self.extents[a] for a in self.factors[k])

    @property
    def out_shape(self) -> tuple[int, ...]:
        """The output shape, in ``out`` order."""
        return tuple(self.extents[o] for o in self.out)

    @property
    def n_out(self) -> int:
        """Total number of output entries."""
        return prod(self.out_shape)

    @property
    def red_shape(self) -> tuple[int, ...]:
        """Extents of the reduction axes, in ascending-axis order."""
        return tuple(self.extents[r] for r in self.red)

    @property
    def red_total(self) -> int:
        """Size of the reduction domain."""
        return prod(self.red_shape)

    # -- descriptor packing (the GPU lanes' wire form) -----------------

    def strides(self, k: int) -> dict[int, int]:
        """Per-``[M]``-axis flat-index stride of operand ``k``.

        Row-major within the operand; axes the factor names twice
        (diagonals) contribute the *sum* of their position strides, so
        one linear form covers diagonals with no special case.
        """
        f = self.factors[k]
        shape = self.factor_shape(k)
        pos_stride = [0] * len(f)
        s = 1
        for i in range(len(f) - 1, -1, -1):
            pos_stride[i] = s
            s *= shape[i]
        acc: dict[int, int] = {}
        for i, a in enumerate(f):
            acc[a] = acc.get(a, 0) + pos_stride[i]
        return acc

    def fits_descriptor(self) -> bool:
        """Whether the fixed-width GPU descriptor can represent this row."""
        return self.m <= MAX_M and len(self.factors) <= MAX_FACTORS

    def pack_descriptor(self) -> list[int]:
        """Pack the row into the flat i64 descriptor the CUDA kernel reads.

        Layout (all i64, length DESC_LEN; ffi/rowkernel.cu must agree):
        [0] n_out_axes  [1] n_red_axes  [2] n_factors  [3] red_total
        [4..4+MAX_M)              out_extent, out order, zero-padded
        [4+MAX_M..4+2*MAX_M)      red_extent, ascending-axis order, zero-padded
        then MAX_FACTORS blocks of MAX_M out-axis strides (factor-major),
        then MAX_FACTORS blocks of MAX_M red-axis strides.
        """
        if not self.fits_descriptor():
            raise RowError(
                f"row has m={self.m} axes / {len(self.factors)} factors; the "
                f"descriptor lane covers m<={MAX_M}, factors<={MAX_FACTORS} "
                "(fall back to the XLA-emission lane)"
            )
        d = [0] * DESC_LEN
        d[0] = len(self.out)
        d[1] = len(self.red)
        d[2] = len(self.factors)
        d[3] = self.red_total
        for j, o in enumerate(self.out):
            d[4 + j] = self.extents[o]
        for j, r in enumerate(self.red):
            d[4 + MAX_M + j] = self.extents[r]
        base_o = 4 + 2 * MAX_M
        base_r = base_o + MAX_FACTORS * MAX_M
        for k in range(len(self.factors)):
            st = self.strides(k)
            for j, o in enumerate(self.out):
                d[base_o + k * MAX_M + j] = st.get(o, 0)
            for j, r in enumerate(self.red):
                d[base_r + k * MAX_M + j] = st.get(r, 0)
        return d

    # -- iteration helpers (shared by the reference evaluator) ---------

    def out_assignments(self):
        """Iterate assignments of the output axes, row-major in ``out``."""
        return _cartesian(*(range(self.extents[o]) for o in self.out))

    def red_assignments(self):
        """Iterate assignments of the reduction axes, ascending-axis order."""
        return _cartesian(*(range(self.extents[r]) for r in self.red))

    # -- DATERWI gate (ENGINEERING #10) --------------------------------

    def magnitude_bound(self, operand_bounds: list[int]) -> int:
        """Conservative bound on |output entry| given per-operand
        magnitude bounds: red_total times the product of the bounds.
        Exact-integer arithmetic; never rounds."""
        if len(operand_bounds) != len(self.factors):
            raise RowError(
                f"row has {len(self.factors)} factors, "
                f"got {len(operand_bounds)} bounds"
            )
        b = self.red_total
        for x in operand_bounds:
            if x < 0:
                raise RowError(f"magnitude bound {x} < 0")
            b *= x
        return b

    def fits_i64(self, operand_bounds: list[int]) -> bool:
        """Whether the s64 lanes are CONCLUSIVE for these operands.

        True: every intermediate and output is exactly representable;
        the wrapped result equals the exact result and any verdict a
        consumer reads from it is conclusive. False: INCONCLUSIVE --
        the caller defers this row to its exact-rational path
        (DATERWI). A False here is a routing fact, not an error.
        """
        return self.magnitude_bound(operand_bounds) < (1 << 63)

    def fits_f64(self, operand_bounds: list) -> bool:
        """Whether the f64 interval-enclosure lanes stay FINITE for
        these operands -- the float sibling of ``fits_i64``, and the
        DATERWI fit gate for the enclosure-semiring kernels.

        ``operand_bounds[k]`` bounds the magnitude of every endpoint of
        operand ``k`` (for interval operands: max over entries of
        max(|lo|, |hi|)); int, float, or Fraction, compared exactly.

        The bound argument (all arithmetic here exact, never rounded):
        every exact intermediate of the device fold -- any corner
        product of a term and any partial sum -- has magnitude at most
        ``B = red_total * prod(bounds)``. Each directed rounding
        multiplies magnitude by at most ``1 + 2^-52`` in the normal
        range, and the fold performs fewer than ``red_total * 5`` such
        roundings per entry, so with ``red_total <= 2^50`` the
        accumulated float factor stays below 2; ``B <= 2^1022``
        therefore keeps every float intermediate at or below 2^1023,
        strictly inside the finite range. True: no endpoint can
        overflow to infinity, no NaN can arise, and the directed
        references mirror the device bitwise. False: INCONCLUSIVE-
        capable only -- endpoints may saturate to +/-inf (still sound
        enclosures) or void to NaN (classified as deferrals, never as
        conclusive verdicts); route the row to the exact path for a
        conclusive answer. A False is a routing fact, not an error.
        """
        if len(operand_bounds) != len(self.factors):
            raise RowError(
                f"row has {len(self.factors)} factors, "
                f"got {len(operand_bounds)} bounds"
            )
        if self.red_total > (1 << 50):
            return False
        b = Fraction(self.red_total)
        for x in operand_bounds:
            fx = Fraction(x)
            if fx < 0:
                raise RowError(f"magnitude bound {x} < 0")
            b *= fx
        return b <= (1 << 1022)


@dataclass(frozen=True)
class Verdict:
    """Three-state kernel verdict (ENGINEERING #10, DATERWI).

    A decision procedure over row outputs answers exactly one of:

    - ``conclusive_pass()`` -- the property holds, exactly.
    - ``conclusive_fail(witness)`` -- the property fails, and
      ``witness`` is the first violating output index (the producer
      side reads the same index as its refinement witness).
    - ``inconclusive()`` -- the fixed-width lane could not decide
      (fit-bound exceeded, or a float enclosure straddles the
      predicate boundary); the caller MUST defer to the exact path.
      Never silently rounded into a pass or fail.
    """

    kind: str  # "pass" | "fail" | "inconclusive"
    witness: int | None = None

    @staticmethod
    def conclusive_pass() -> "Verdict":
        return Verdict("pass")

    @staticmethod
    def conclusive_fail(witness: int) -> "Verdict":
        return Verdict("fail", witness)

    @staticmethod
    def inconclusive() -> "Verdict":
        return Verdict("inconclusive")

    @property
    def must_defer(self) -> bool:
        return self.kind == "inconclusive"
