"""Pallas/Mosaic-GPU lane: literal PTX inside a fused kernel.

What this lane proves is the *mechanism*: a Pallas kernel is one
custom call to XLA, and inside it, array logic and literal PTX
compose in a single fused launch via ``inline_mgpu`` — so hand PTX is
available without leaving the fused program. Under the engineering
commitments' rounding law (VCARM, ENGINEERING #9) this escape hatch
is a *soundness* mechanism, not just a speed one: emitted XLA ops
offer no contractual rounding control, while a PTX island pins the
rounding mode on each instruction.

Scope, stated plainly: this lane covers the **elementwise normal
forms** of a contraction row — the shapes whose layouts Mosaic's
inference handles today, and, not coincidentally, the shapes of the
streaming update path (applying a delta batch to resident state):

- ``hadamard_accum`` — ``acc' = acc + x * y`` on s64: a real
  producer-and-consumer row (two factors on the output axes, empty
  reduction) fused with its accumulate; its inline-PTX core is ONE
  instruction, ``mad.lo.s64``, applied per element. The jnp-ops twin
  computes the identical verdict and the battery diffs the two.
- ``interval_add`` — f64 interval addition with per-instruction
  directed rounding (``add.rm.f64`` / ``add.rp.f64``). This one HAS
  no jnp twin, and that is the point (no XLA op carries a rounding
  mode): its differential partners are the FFI interval kernel and
  the exact-rational oracle in ``gpu/interval.py``.

General contractions (matmul form and up) stay the FFI lane's
business: rows there are runtime data and layout inference never
enters. Measured fact from this box, recorded for successors: s64
matmul formulations (3D broadcast+sum; unrolled outer products with
and without layout casts; fori_loop with dynamic slices) all fail
Mosaic-GPU layout inference or hit unimplemented lowerings on jax
0.11.0 — the elementwise scope is what the substrate supports today,
not a taste choice.

Constraint inherited from the substrate: a warpgroup is 128 lanes, so
array extents must satisfy ``prod(shape) % 128 == 0`` (typed refusal
otherwise; callers fall back to the FFI or XLA lane).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .rowir import Row, RowError

jax.config.update("jax_enable_x64", True)


def hadamard_form(row: Row) -> bool:
    """Whether the row is ``out = f0 * f1`` elementwise: exactly two
    factors, each factor's axis tuple equal to ``out``, no reduction."""
    return (
        len(row.factors) == 2
        and len(row.red) == 0
        and row.factors[0] == row.out
        and row.factors[1] == row.out
    )


#: Ceiling on elements per call. The inline-PTX island unrolls one asm
#: op per register at trace time (a register = one lane-element at
#: vec_size=1), so per-thread register counts beyond a few hundred
#: explode MLIR size and compile time — measured: a 4M-element call
#: (32768 regs/thread in one warpgroup) tripped XLA's slow-compile
#: alarm and never returned. Tiling with a grid is the growth path;
#: until a consumer needs it, the lane refuses instead.
MAX_ELEMENTS = 32768


def _check_shape(shape: tuple[int, ...]) -> None:
    n = 1
    for e in shape:
        n *= e
    if n % 128 != 0:
        raise RowError(
            f"pallas lane needs prod(shape) % 128 == 0 (a warpgroup is "
            f"128 lanes); got {shape} ({n} elements) — use the FFI lane"
        )
    if n > MAX_ELEMENTS:
        raise RowError(
            f"pallas lane covers up to {MAX_ELEMENTS} elements per call "
            f"(trace-time asm unroll; see MAX_ELEMENTS); got {n} — "
            "use the FFI lane"
        )


def _supported() -> bool:
    try:
        from jax.experimental.pallas import mosaic_gpu as plgpu  # noqa: F401

        return True
    except ImportError:
        return False


# ── inline-PTX cores ─────────────────────────────────────────────────
#
# Scalarization follows the idiom Mosaic GPU itself uses for
# instruction-granular ops over register vectors: extract each lane,
# apply the instruction, insert into an undef-seeded result vector.


def _scalarize(op_scalar):
    """Lift a scalar-MLIR-op into a register-level op that handles both
    scalar and vector registers."""
    from jax._src.lib.mlir import ir
    from jax._src.lib.mlir.dialects import llvm as llvm_dialect
    from jax._src.lib.mlir.dialects import vector as vector_dialect

    def op(*regs):
        ty = regs[0].type
        if not isinstance(ty, ir.VectorType):
            return op_scalar(*regs)
        result = llvm_dialect.mlir_undef(ty)
        [vec_len] = ir.VectorType(ty).shape
        for i in range(vec_len):
            pos = ir.DenseI64ArrayAttr.get([i])
            scalars = [
                vector_dialect.extract(
                    r, dynamic_position=[], static_position=pos
                )
                for r in regs
            ]
            result = vector_dialect.insert(
                op_scalar(*scalars),
                result,
                dynamic_position=[],
                static_position=pos,
            )
        return result

    return op


def _asm1(ptx: str, constraints: str, result_ty):
    """A scalar op emitting one PTX instruction via llvm.inline_asm."""
    from jax._src.lib.mlir.dialects import llvm as llvm_dialect

    def op_scalar(*xs):
        return llvm_dialect.inline_asm(
            result_ty(), list(xs), ptx, constraints, has_side_effects=False
        )

    return _scalarize(op_scalar)


# ── hadamard-accumulate (s64, mad.lo.s64) ────────────────────────────


def eval_hadamard_accum_pallas(
    row: Row,
    acc: jax.Array,
    operands: list[jax.Array],
    *,
    use_inline_ptx: bool = True,
) -> jax.Array:
    """``acc + f0 * f1`` elementwise on s64, as one fused Mosaic-GPU
    kernel; the multiply-accumulate is literal ``mad.lo.s64`` when
    ``use_inline_ptx`` (the battery diffs it against the jnp twin and
    the definitional reference)."""
    if not hadamard_form(row):
        raise RowError(
            "pallas hadamard lane covers out = f0 * f1 rows only"
        )
    if not _supported():
        raise RowError("installed jax lacks the Mosaic-GPU Pallas backend")
    _check_shape(row.out_shape)
    if tuple(acc.shape) != row.out_shape:
        raise RowError(
            f"accumulator shape {tuple(acc.shape)} != {row.out_shape}"
        )
    for k, op in enumerate(operands):
        if tuple(op.shape) != row.factor_shape(k):
            raise RowError(
                f"operand {k}: expected {row.factor_shape(k)}, "
                f"got {tuple(op.shape)}"
            )
    from jax.experimental.pallas import mosaic_gpu as plgpu

    shape = row.out_shape
    layout = plgpu.Layout.WG_STRIDED(shape, vec_size=1)

    if use_inline_ptx:

        def make_mad():
            from jax._src.lib.mlir import ir

            core = _asm1(
                "mad.lo.s64 $0, $1, $2, $3;",
                "=l,l,l,l",
                lambda: ir.IntegerType.get_signless(64),
            )

            @plgpu.inline_mgpu(
                arg_types=(layout, layout, layout),
                return_type=plgpu.ShapeDtypeStruct(
                    shape, jnp.int64, layout=layout
                ),
            )
            def mad(ctx, xv, yv, cv):
                del ctx
                return xv._pointwise(core, yv, cv, output_is_signed=True)

            return mad

        def kernel(x_ref, y_ref, c_ref, o_ref):
            xv = plgpu.layout_cast(x_ref[...], layout)
            yv = plgpu.layout_cast(y_ref[...], layout)
            cv = plgpu.layout_cast(c_ref[...], layout)
            o_ref[...] = make_mad()(xv, yv, cv)

    else:

        def kernel(x_ref, y_ref, c_ref, o_ref):
            o_ref[...] = c_ref[...] + x_ref[...] * y_ref[...]

    run = plgpu.kernel(
        kernel, out_type=jax.ShapeDtypeStruct(shape, jnp.int64)
    )
    return run(operands[0], operands[1], acc)


# ── interval hadamard-accumulate (f64, the enclosure semiring's
#    elementwise normal form) ─────────────────────────────────────────

#: Ceiling on elements per call for the interval hadamard-accumulate:
#: its inline-PTX core is ~18 asm ops per element (8 directed
#: products, 6 min/max folds, 2 zero-folds, 2 directed adds), versus
#: 1 for ``hadamard_accum`` — trace-time unroll scales with the
#: product, so the cap sits below MAX_ELEMENTS pending an on-box
#: measurement (receipts note the compile-time observations).
IVL_HADAMARD_MAX_ELEMENTS = 8192


def interval_hadamard_accum_pallas(
    acc_lo: jax.Array,
    acc_hi: jax.Array,
    xl: jax.Array,
    xh: jax.Array,
    yl: jax.Array,
    yh: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """``acc (+) (X (*) Y)`` elementwise over the enclosure semiring,
    as one fused Mosaic-GPU kernel with every floating-point operation
    a directed-rounding PTX island: the four corner products per bound
    (``mul.rm.f64`` / ``mul.rp.f64``) folded by ``min.f64`` /
    ``max.f64``, then the accumulate adds (``add.rm.f64`` /
    ``add.rp.f64``). The streaming-update normal form of an interval
    contraction row (two factors on the output axes, empty reduction),
    fused with its accumulate — the enclosure sibling of
    ``hadamard_accum``, and like ``interval_add`` it can have no jnp
    twin (VCARM: no emitted op carries a rounding mode).

    Operation-exact parity with the FFI lane (its differential
    partner, alongside the directed reference): the FFI accumulate
    path computes ``fresh = [+0,+0] (+) ([1,1] (*) X (*) Y)`` and then
    ``prev (+) fresh``. The unit fold ``[1,1] (*) X`` is bitwise-
    invisible (directed products by 1.0 are exact, and the corner
    min/max reproduces the endpoints, signed zeros included) — but the
    zero fold is NOT: ``ru(+0 + (-0)) = +0`` rewrites a ``-0.0`` upper
    endpoint, and dropping it would diverge exactly on the
    ``(prev_hi = -0, fresh_hi = -0)`` corner (like-signed zeros keep
    their sign; mixed-signed go mode-determined). So this kernel
    mirrors the zero fold explicitly — two islands adding a +0.0 PTX
    double immediate (``0d0000000000000000``) in each direction —
    making every lane of the family the same operation sequence,
    bitwise, everywhere.
    """
    if not _supported():
        raise RowError("installed jax lacks the Mosaic-GPU Pallas backend")
    shape = tuple(acc_lo.shape)
    for name, arr in (
        ("acc_hi", acc_hi),
        ("xl", xl),
        ("xh", xh),
        ("yl", yl),
        ("yh", yh),
    ):
        if tuple(arr.shape) != shape:
            raise RowError(
                f"{name} shape {tuple(arr.shape)} != acc_lo shape {shape}"
            )
    _check_shape(shape)
    n = 1
    for e in shape:
        n *= e
    if n > IVL_HADAMARD_MAX_ELEMENTS:
        raise RowError(
            f"interval hadamard-accumulate covers up to "
            f"{IVL_HADAMARD_MAX_ELEMENTS} elements per call (trace-time asm "
            f"unroll, ~18 ops/element; see IVL_HADAMARD_MAX_ELEMENTS); got "
            f"{n} — use the FFI lane"
        )
    from jax.experimental.pallas import mosaic_gpu as plgpu

    layout = plgpu.Layout.WG_STRIDED(shape, vec_size=1)

    def make_bin(ptx):
        from jax._src.lib.mlir import ir

        core = _asm1(ptx, "=d,d,d", lambda: ir.F64Type.get())

        @plgpu.inline_mgpu(
            arg_types=(layout, layout),
            return_type=plgpu.ShapeDtypeStruct(
                shape, jnp.float64, layout=layout
            ),
        )
        def op(ctx, xv, yv):
            del ctx
            return xv._pointwise(core, yv)

        return op

    def make_zero_fold(ptx_op):
        """rd/ru fold of x with a +0.0 immediate — the FFI accumulate
        path's fresh-from-zero add, mirrored as a unary island."""
        from jax._src.lib.mlir import ir

        core = _asm1(
            f"{ptx_op} $0, $1, 0d0000000000000000;",
            "=d,d",
            lambda: ir.F64Type.get(),
        )

        @plgpu.inline_mgpu(
            arg_types=(layout,),
            return_type=plgpu.ShapeDtypeStruct(
                shape, jnp.float64, layout=layout
            ),
        )
        def op(ctx, xv):
            del ctx
            return xv._pointwise(core)

        return op

    def kernel(al_ref, ah_ref, xl_ref, xh_ref, yl_ref, yh_ref, lo_ref, hi_ref):
        cast = lambda ref: plgpu.layout_cast(ref[...], layout)  # noqa: E731
        av_lo, av_hi = cast(al_ref), cast(ah_ref)
        xv_lo, xv_hi = cast(xl_ref), cast(xh_ref)
        yv_lo, yv_hi = cast(yl_ref), cast(yh_ref)

        mul_rm, mul_rp = make_bin("mul.rm.f64 $0, $1, $2;"), make_bin(
            "mul.rp.f64 $0, $1, $2;"
        )
        fmin_, fmax_ = make_bin("min.f64 $0, $1, $2;"), make_bin(
            "max.f64 $0, $1, $2;"
        )
        add_rm, add_rp = make_bin("add.rm.f64 $0, $1, $2;"), make_bin(
            "add.rp.f64 $0, $1, $2;"
        )
        zf_rm = make_zero_fold("add.rm.f64")
        zf_rp = make_zero_fold("add.rp.f64")

        # (*): four corner products per bound, fixed order, pairwise
        # min/max — the device ivl_mul, island for island.
        t_lo = fmin_(
            fmin_(mul_rm(xv_lo, yv_lo), mul_rm(xv_lo, yv_hi)),
            fmin_(mul_rm(xv_hi, yv_lo), mul_rm(xv_hi, yv_hi)),
        )
        t_hi = fmax_(
            fmax_(mul_rp(xv_lo, yv_lo), mul_rp(xv_lo, yv_hi)),
            fmax_(mul_rp(xv_hi, yv_lo), mul_rp(xv_hi, yv_hi)),
        )
        # The FFI path's fresh-from-[+0,+0] fold, mirrored (docstring).
        f_lo, f_hi = zf_rm(t_lo), zf_rp(t_hi)
        # prev (+) fresh.
        lo_ref[...] = add_rm(av_lo, f_lo)
        hi_ref[...] = add_rp(av_hi, f_hi)

    spec = jax.ShapeDtypeStruct(shape, jnp.float64)
    run = plgpu.kernel(kernel, out_type=(spec, spec))
    return run(acc_lo, acc_hi, xl, xh, yl, yh)


# ── interval add (f64, add.rm/.rp — PTX's directed-rounding names) ──


def interval_add_pallas(
    al: jax.Array, ah: jax.Array, bl: jax.Array, bh: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Elementwise interval sum with directed rounding pinned per
    instruction, inside a fused Mosaic-GPU kernel: lower bounds by
    ``add.rm.f64``, upper bounds by ``add.rp.f64``. Differential
    partners: the FFI interval kernel and the exact-rational oracle
    (``gpu/interval.py``); a jnp twin cannot exist, which is the
    architectural point (VCARM, ENGINEERING #9)."""
    if not _supported():
        raise RowError("installed jax lacks the Mosaic-GPU Pallas backend")
    shape = tuple(al.shape)
    _check_shape(shape)
    from jax.experimental.pallas import mosaic_gpu as plgpu

    layout = plgpu.Layout.WG_STRIDED(shape, vec_size=1)

    def make_core(ptx):
        from jax._src.lib.mlir import ir

        core = _asm1(ptx, "=d,d,d", lambda: ir.F64Type.get())

        @plgpu.inline_mgpu(
            arg_types=(layout, layout),
            return_type=plgpu.ShapeDtypeStruct(
                shape, jnp.float64, layout=layout
            ),
        )
        def add_dir(ctx, xv, yv):
            del ctx
            return xv._pointwise(core, yv)

        return add_dir

    def kernel(al_ref, ah_ref, bl_ref, bh_ref, lo_ref, hi_ref):
        alv = plgpu.layout_cast(al_ref[...], layout)
        ahv = plgpu.layout_cast(ah_ref[...], layout)
        blv = plgpu.layout_cast(bl_ref[...], layout)
        bhv = plgpu.layout_cast(bh_ref[...], layout)
        lo_ref[...] = make_core("add.rm.f64 $0, $1, $2;")(alv, blv)
        hi_ref[...] = make_core("add.rp.f64 $0, $1, $2;")(ahv, bhv)

    spec = jax.ShapeDtypeStruct(shape, jnp.float64)
    run = plgpu.kernel(kernel, out_type=(spec, spec))
    return run(al, ah, bl, bh)
