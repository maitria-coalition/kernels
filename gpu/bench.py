"""Receipts generator for the GPU contraction-row lanes.

Prints the receipt table (medians over repeated timed runs, warmup
excluded) plus a cross-lane bit-equality check per shape: at receipt
sizes the definitional Python reference is out of reach by design, so
the battery (small shapes, full reference parity) carries semantics
and this tool carries economics -- with the cross-lane diff as the
in-run witness that the timed lanes computed the same verdict.

Run on the target box:  python3 -m gpu.bench
"""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp

from .rowir import Row
from .xla_lane import eval_row_xla

jax.config.update("jax_enable_x64", True)

REPS = 25


def _key(i):
    return jax.random.PRNGKey(i)


def _rand_i64(key, shape):
    # Full-width random bits, viewed as s64.
    return jax.random.bits(key, shape=shape, dtype=jnp.uint64).astype(
        jnp.int64
    )


def _med_ms(fn, *args):
    jax.block_until_ready(fn(*args))  # warmup + compile (pytree-safe)
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main():
    from .ffi_lane import eval_row_accum_ffi, eval_row_ffi

    dev = jax.devices()[0]
    print(f"# device: {dev.device_kind}; jax {jax.__version__}; reps={REPS}")
    print()
    print("| row | lane | median ms |")
    print("|---|---|---:|")

    # No 1024^3 row: on jax 0.11.0/sm_120 the s64 1024^3 einsum's
    # autotuning compile ran 20+ minutes and then died to the memory
    # ceiling of a 32 GB box — the compile cliff at large s64 dots is
    # itself a receipt (enable the persistent compilation cache before
    # attempting it; see the repository receipts).
    shapes = [
        ("matmul 512^3", Row((512, 512, 512), ((0, 2), (2, 1)), (0, 1))),
        (
            "3-factor (256,256,16,16)",
            Row(
                (256, 256, 16, 16),
                ((0, 2), (2, 1, 3), (3,)),
                (0, 1),
            ),
        ),
        (
            "hadamard 4M (delta-apply shape)",
            Row((2048, 2048), ((0, 1), (0, 1)), (0, 1)),
        ),
        (
            # In pallas scope (<= MAX_ELEMENTS): the fused inline-PTX
            # lanes bench here; xla/ffi run it too for comparison.
            "hadamard 32K (pallas scope)",
            Row((256, 128), ((0, 1), (0, 1)), (0, 1)),
        ),
    ]

    for name, row in shapes:
        ops = [
            _rand_i64(_key(17 * k + 1), row.factor_shape(k))
            for k in range(len(row.factors))
        ]

        xla_j = jax.jit(lambda *o, _r=row: eval_row_xla(_r, list(o)))
        ffi_j = jax.jit(lambda *o, _r=row: eval_row_ffi(_r, list(o)))

        ref_dev = xla_j(*ops)
        got_ffi = ffi_j(*ops)
        parity = bool(jnp.array_equal(ref_dev, got_ffi))
        print(f"| {name} | xla | {_med_ms(xla_j, *ops):.3f} |")
        print(
            f"| {name} | ffi (fresh) | {_med_ms(ffi_j, *ops):.3f} |"
            + ("" if parity else "  <-- PARITY FAIL")
        )

        # Aliased accumulate: donate the accumulator so XLA may honour
        # the alias end to end (the free-boundary pattern).
        accum_j = jax.jit(
            lambda a, *o, _r=row: eval_row_accum_ffi(_r, a, list(o)),
            donate_argnums=0,
        )
        acc0 = _rand_i64(_key(999), row.out_shape)
        expect = (acc0 + ref_dev).block_until_ready()
        got = accum_j(acc0, *ops)
        ap = bool(jnp.array_equal(expect, got))

        def timed_accum(*o, _r=row, _j=accum_j):
            a = _rand_i64(_key(998), _r.out_shape)
            return _j(a, *o)

        print(
            f"| {name} | ffi (accum, aliased) | {_med_ms(timed_accum, *ops):.3f} |"
            + ("" if ap else "  <-- PARITY FAIL")
        )

        from .pallas_lane import eval_hadamard_accum_pallas, hadamard_form

        if hadamard_form(row):
            for label, ptx in (("jnp twin", False), ("inline PTX", True)):
                try:
                    hp_j = jax.jit(
                        lambda a, x, y, _r=row, _p=ptx: eval_hadamard_accum_pallas(
                            _r, a, [x, y], use_inline_ptx=_p
                        )
                    )
                    acc1 = _rand_i64(_key(777), row.out_shape)
                    got_h = hp_j(acc1, *ops)
                    hp = bool(jnp.array_equal(acc1 + ref_dev, got_h))
                    print(
                        f"| {name} | pallas accum ({label}) | "
                        f"{_med_ms(hp_j, acc1, *ops):.3f} |"
                        + ("" if hp else "  <-- PARITY FAIL")
                    )
                except Exception as e:  # noqa: BLE001 - receipts tool
                    print(
                        f"| {name} | pallas accum ({label}) | n/a "
                        f"({type(e).__name__}) |"
                    )
        print()

    # Interval family (VCARM demo): FFI vs fused-Pallas directed adds.
    from .interval import ivl_addmul_ffi

    n = 1 << 22
    al = -jnp.abs(_rand_i64(_key(5), (n,))).astype(jnp.float64) / 2**32
    ah = al + jnp.abs(_rand_i64(_key(6), (n,))).astype(jnp.float64) / 2**40
    bl = -jnp.abs(_rand_i64(_key(7), (n,))).astype(jnp.float64) / 2**32
    bh = bl + jnp.abs(_rand_i64(_key(8), (n,))).astype(jnp.float64) / 2**40
    ffi_iv = jax.jit(lambda *xs: ivl_addmul_ffi(*xs))
    print(f"| interval addmul 4M | ffi (.rm/.rp PTX) | {_med_ms(ffi_iv, al, ah, bl, bh):.3f} |")
    try:
        from .pallas_lane import MAX_ELEMENTS, interval_add_pallas

        m = MAX_ELEMENTS
        als, ahs, bls, bhs = al[:m], ah[:m], bl[:m], bh[:m]
        pl_iv = jax.jit(lambda *xs: interval_add_pallas(*xs))
        lo_p, hi_p = pl_iv(als, ahs, bls, bhs)
        slo, shi, _, _ = ffi_iv(als, ahs, bls, bhs)
        ivp = bool(
            jnp.array_equal(lo_p, slo) and jnp.array_equal(hi_p, shi)
        )
        print(
            f"| interval add 32K | pallas fused (inline .rm/.rp) | "
            f"{_med_ms(pl_iv, als, ahs, bls, bhs):.3f} |"
            + ("" if ivp else "  <-- PARITY FAIL")
        )
    except Exception as e:  # noqa: BLE001
        print(f"| interval add 32K | pallas fused | n/a ({type(e).__name__}) |")

    _interval_contraction_section()


def _rand_ivl_pair(seed, shape, scale=1.0):
    """A random interval tensor: two draws, endpointwise min/max."""
    import jax.numpy as jnp

    a = jax.random.normal(_key(seed), shape, dtype=jnp.float64) * scale
    b = jax.random.normal(_key(seed + 1), shape, dtype=jnp.float64) * scale
    return jnp.minimum(a, b), jnp.maximum(a, b)


def _interval_contraction_section():
    """Receipts for the enclosure-semiring contraction lanes: eval /
    aliased accumulate / on-device check economics, plus the
    tightness receipt (device width vs the ideal — the directed
    rounding of the exact endpoints — on rows small enough for exact
    rational evaluation, cancellation-adversarial included)."""
    import jax.numpy as jnp

    from .ivl_ffi_lane import (
        check_row_interval_ffi,
        eval_row_interval_accum_ffi,
        eval_row_interval_ffi,
    )

    print()
    ivl_shapes = [
        ("ivl matmul 256^3", Row((256, 256, 256), ((0, 2), (2, 1)), (0, 1))),
        (
            "ivl hadamard 4M",
            Row((2048, 2048), ((0, 1), (0, 1)), (0, 1)),
        ),
    ]
    for name, row in ivl_shapes:
        ops = [
            _rand_ivl_pair(29 * k + 3, row.factor_shape(k))
            for k in range(len(row.factors))
        ]
        flat = [x for pair in ops for x in pair]
        ev = jax.jit(
            lambda *xs, _r=row: eval_row_interval_ffi(
                _r, [(xs[2 * i], xs[2 * i + 1]) for i in range(len(xs) // 2)]
            )
        )
        lo0, hi0 = ev(*flat)
        ok = bool(jnp.all(lo0 <= hi0))
        print(
            f"| {name} | ivl ffi (fresh) | {_med_ms(ev, *flat):.3f} |"
            + ("" if ok else "  <-- ORDER FAIL")
        )

        acc_lo, acc_hi = _rand_ivl_pair(997, row.out_shape)
        accum = jax.jit(
            lambda a_lo, a_hi, *xs, _r=row: eval_row_interval_accum_ffi(
                _r, a_lo, a_hi,
                [(xs[2 * i], xs[2 * i + 1]) for i in range(len(xs) // 2)],
            ),
            donate_argnums=(0, 1),
        )

        def timed_accum(*xs, _r=row, _j=accum):
            a_lo, a_hi = _rand_ivl_pair(995, _r.out_shape)
            return _j(a_lo, a_hi, *xs)

        print(
            f"| {name} | ivl ffi (accum, aliased) | "
            f"{_med_ms(timed_accum, *flat):.3f} |"
        )

        # Check = eval + on-device classify/reduce + a 2-scalar read.
        # Demands widened one ulp around the fresh enclosure: the
        # all-pass path (the common checking case).
        dlo = jnp.nextafter(lo0, -jnp.inf)
        dhi = jnp.nextafter(hi0, jnp.inf)

        def checked(*xs, _r=row, _dlo=dlo, _dhi=dhi):
            return check_row_interval_ffi(
                _r,
                [(xs[2 * i], xs[2 * i + 1]) for i in range(len(xs) // 2)],
                _dlo,
                _dhi,
            )

        _, _, v = checked(*flat)
        print(
            f"| {name} | ivl ffi (check, verdict={v.kind}) | "
            f"{_med_ms(checked, *flat):.3f} |"
        )
        print()

    # Fused-Pallas interval hadamard-accumulate at its scope.
    try:
        from .pallas_lane import (
            IVL_HADAMARD_MAX_ELEMENTS,
            interval_hadamard_accum_pallas,
        )

        n = IVL_HADAMARD_MAX_ELEMENTS
        shape = (n,)
        row = Row((n,), ((0,), (0,)), (0,))
        xlo, xhi = _rand_ivl_pair(41, shape)
        ylo, yhi = _rand_ivl_pair(43, shape)
        alo, ahi = _rand_ivl_pair(45, shape)
        pl = jax.jit(interval_hadamard_accum_pallas)
        p_lo, p_hi = pl(alo, ahi, xlo, xhi, ylo, yhi)
        from .ivl_ffi_lane import eval_row_interval_accum_ffi as _accum

        f_lo, f_hi = _accum(row, alo, ahi, [(xlo, xhi), (ylo, yhi)])
        par = bool(
            jnp.array_equal(p_lo, f_lo) and jnp.array_equal(p_hi, f_hi)
        )
        print(
            f"| ivl hadamard-accum {n} | pallas fused (18 islands/elt) | "
            f"{_med_ms(pl, alo, ahi, xlo, xhi, ylo, yhi):.3f} |"
            + ("" if par else "  <-- PARITY FAIL")
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"| ivl hadamard-accum | pallas fused | n/a ({type(e).__name__}) |"
        )

    _tightness_receipt()


def _tightness_receipt():
    """Width of the device enclosure against the ideal (directed
    rounding of exact endpoints), in exact rational arithmetic, on
    rows small enough to afford the exact stratum. Two regimes: a
    random small matmul (generic accumulation), and the mirrored-
    cancellation dot (exact answer 0, every float partial inexact —
    the adversarial case the straddle verdict exists for)."""
    import statistics as stats

    import jax.numpy as jnp

    from .ivl_ffi_lane import eval_row_interval_ffi
    from .ivl_reference import eval_row_interval_exact, ideal_enclosure

    print()
    print("tightness (device width vs ideal width, exact-arithmetic):")

    cases = []
    row_mm = Row((16, 16, 16), ((0, 2), (2, 1)), (0, 1))
    cases.append(("matmul 16^3 (generic)", row_mm, None))
    n = 128
    row_dot = Row((n,), ((0,), (0,)), ())
    cases.append(("mirrored-cancellation dot 128", row_dot, "mirror"))

    for name, row, mode in cases:
        ops_dev = [
            _rand_ivl_pair(61 * k + 7, row.factor_shape(k), scale=1.0)
            for k in range(len(row.factors))
        ]
        if mode == "mirror":
            # POINT intervals with mirrored terms: X = (x, -x point-
            # wise mirrored), Y = (y, y duplicated), so term(i+half) =
            # -term(i) and the exact interval is exactly [0, 0] —
            # ideal width 0, and the device width IS the accumulated
            # directed-rounding cost, isolated. (Mirroring only one
            # operand: negating both would double products, not cancel
            # them; genuine-width intervals would cancel midpoints but
            # sum widths, burying the rounding signal.)
            half = n // 2
            x = ops_dev[0][0][:half]
            y = ops_dev[1][0][:half]
            xs = jnp.concatenate([x, -x])
            ys = jnp.concatenate([y, y])
            ops_dev = [(xs, xs), (ys, ys)]
        got_lo, got_hi = eval_row_interval_ffi(row, ops_dev)
        ops_host = [
            (lo.reshape(-1).tolist(), hi.reshape(-1).tolist())
            for lo, hi in ops_dev
        ]
        ex_lo, ex_hi = eval_row_interval_exact(row, ops_host)
        id_lo, id_hi = ideal_enclosure(ex_lo, ex_hi)
        ratios, dev_only, exact_hits = [], [], 0
        for i, (glo, ghi) in enumerate(
            zip(got_lo.reshape(-1).tolist(), got_hi.reshape(-1).tolist())
        ):
            dw = ghi - glo
            iw = id_hi[i] - id_lo[i]
            if iw > 0:
                ratios.append(dw / iw)
            elif dw > 0:
                dev_only.append(dw)
            else:
                exact_hits += 1
        line = f"  {name}: n={row.n_out}"
        if ratios:
            line += (
                f"; width ratio median {stats.median(ratios):.3f} "
                f"max {max(ratios):.3f} (over {len(ratios)} ideal-positive)"
            )
        if dev_only:
            line += (
                f"; ideal-width-0 entries: {len(dev_only)}, device width "
                f"median {stats.median(dev_only):.3e} max {max(dev_only):.3e}"
            )
        if exact_hits:
            line += f"; exact (width 0 both): {exact_hits}"
        print(line)


if __name__ == "__main__":
    main()
