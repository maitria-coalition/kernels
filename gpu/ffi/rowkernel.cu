// rowkernel.cu — contraction-row evaluation microkernel (s64), hosted
// behind XLA FFI custom calls.
//
// One compiled kernel serves every row within the descriptor bounds:
// the row arrives as DATA (the i64 descriptor packed by
// gpu/rowir.py::Row.pack_descriptor — the two encodings must agree),
// so changing the row never recompiles device code. Layout:
//
//   [0] n_out_axes  [1] n_red_axes  [2] n_factors  [3] red_total
//   [4..12)   out_extent (out order, zero-padded to MAX_M)
//   [12..20)  red_extent (ascending-axis order, zero-padded)
//   [20..52)  out-axis strides, factor-major [MAX_FACTORS][MAX_M]
//   [52..84)  red-axis strides, factor-major [MAX_FACTORS][MAX_M]
//
// Each thread owns output entries in a grid-stride loop: decode the
// output multi-index (row-major), fold it through each factor's
// stride row to a base offset, then walk the reduction domain
// accumulating the product of factor entries. Diagonals need no
// special case: a factor axis named twice contributes the sum of its
// position strides to one linear form.
//
// Arithmetic is performed in unsigned 64-bit (well-defined wraparound
// mod 2^64) and reinterpreted as signed at the boundary — the same
// two's-complement semantics the reference evaluator and XLA s64
// define; the conformance battery asserts bit-equality.
//
// Two FFI entry points share the kernel:
//   RowEval  — fresh output (accumulate = 0).
//   RowAccum — out = acc + row; registered for use with
//              input_output_aliases so XLA hands coincident pointers
//              and the update is in place (the free-boundary pattern).
//              If a caller does NOT alias, the handler first copies
//              acc into out and accumulates there — semantics never
//              depend on the alias being honoured, only the cost does.
//
// Pipeline of record: OFFLINE nvcc (see build.sh), per the engineering
// commitment on JIT-compiled lanes; the disassembly branch count is
// recorded beside the receipts.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

#define MAX_M 8
#define MAX_FACTORS 4

extern "C" __global__ void row_eval_s64(
    long long *__restrict__ out, const long long *__restrict__ desc,
    const long long *f0, const long long *f1, const long long *f2,
    const long long *f3, long long n_out, int accumulate) {
  const long long n_out_axes = desc[0];
  const long long n_red_axes = desc[1];
  const long long n_factors = desc[2];
  const long long red_total = desc[3];
  const long long *out_ext = desc + 4;
  const long long *red_ext = desc + 4 + MAX_M;
  const long long *ostr = desc + 4 + 2 * MAX_M;
  const long long *rstr = desc + 4 + 2 * MAX_M + MAX_FACTORS * MAX_M;
  const long long *fac[MAX_FACTORS] = {f0, f1, f2, f3};

  for (long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       o < n_out; o += (long long)gridDim.x * blockDim.x) {
    long long base[MAX_FACTORS] = {0, 0, 0, 0};
    long long rem = o;
    for (long long j = n_out_axes - 1; j >= 0; --j) {
      long long c = rem % out_ext[j];
      rem /= out_ext[j];
      for (long long k = 0; k < n_factors; ++k)
        base[k] += c * ostr[k * MAX_M + j];
    }
    unsigned long long acc = 0;
    for (long long r = 0; r < red_total; ++r) {
      long long off[MAX_FACTORS] = {base[0], base[1], base[2], base[3]};
      long long rrem = r;
      for (long long j = n_red_axes - 1; j >= 0; --j) {
        long long c = rrem % red_ext[j];
        rrem /= red_ext[j];
        for (long long k = 0; k < n_factors; ++k)
          off[k] += c * rstr[k * MAX_M + j];
      }
      unsigned long long term = 1ull;
      for (long long k = 0; k < n_factors; ++k)
        term *= (unsigned long long)fac[k][off[k]];
      acc += term;
    }
    unsigned long long prev =
        accumulate ? (unsigned long long)out[o] : 0ull;
    out[o] = (long long)(prev + acc);
  }
}

// ── directed-rounding interval kernel (VCARM demo, ENGINEERING #9) ──
//
// Elementwise interval add and multiply on f64 with PER-INSTRUCTION
// directed rounding, written as literal PTX: lower bounds round
// toward minus infinity (PTX .rm; the CUDA intrinsics call it _rd),
// upper bounds toward plus infinity (PTX .rp / _ru) — outward
// rounding is how enclosures stay enclosures. This is
// arithmetic that CANNOT ride a fusion compiler whose rewrite
// semantics are not contractual; it lives here, where every rounding
// mode is pinned on the instruction itself. The battery checks each
// output value-equal to an exact-rational host reference applying the
// same directed roundings (DATERWI's exact path, run in reverse as an
// oracle) and checks the enclosure property against exact arithmetic.

__device__ __forceinline__ double dadd_rd(double x, double y) {
  double r;
  asm("add.rm.f64 %0, %1, %2;" : "=d"(r) : "d"(x), "d"(y));
  return r;
}
__device__ __forceinline__ double dadd_ru(double x, double y) {
  double r;
  asm("add.rp.f64 %0, %1, %2;" : "=d"(r) : "d"(x), "d"(y));
  return r;
}
__device__ __forceinline__ double dmul_rd(double x, double y) {
  double r;
  asm("mul.rm.f64 %0, %1, %2;" : "=d"(r) : "d"(x), "d"(y));
  return r;
}
__device__ __forceinline__ double dmul_ru(double x, double y) {
  double r;
  asm("mul.rp.f64 %0, %1, %2;" : "=d"(r) : "d"(x), "d"(y));
  return r;
}

extern "C" __global__ void ivl_addmul_f64(
    double *__restrict__ sum_lo, double *__restrict__ sum_hi,
    double *__restrict__ prod_lo, double *__restrict__ prod_hi,
    const double *a_lo, const double *a_hi, const double *b_lo,
    const double *b_hi, long long n) {
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (long long)gridDim.x * blockDim.x) {
    const double al = a_lo[i], ah = a_hi[i], bl = b_lo[i], bh = b_hi[i];
    sum_lo[i] = dadd_rd(al, bl);
    sum_hi[i] = dadd_ru(ah, bh);
    prod_lo[i] = fmin(fmin(dmul_rd(al, bl), dmul_rd(al, bh)),
                      fmin(dmul_rd(ah, bl), dmul_rd(ah, bh)));
    prod_hi[i] = fmax(fmax(dmul_ru(al, bl), dmul_ru(al, bh)),
                      fmax(dmul_ru(ah, bl), dmul_ru(ah, bh)));
  }
}

// ── interval-enclosure contraction kernel (VCARM consumer fast-path,
//    ENGINEERING #9-#10) ──
//
// The same contraction-row machinery over the enclosure semiring:
// operands, accumulator, and outputs are f64 interval tensors carried
// as separate lo/hi buffers; every lower-bound operation rounds toward
// minus infinity (PTX .rm) and every upper-bound operation toward plus
// infinity (PTX .rp), per instruction, so the computed interval always
// CONTAINS the exact interval semantics — outward rounding is how
// enclosures stay enclosures, and it is what makes a verdict read off
// this kernel sound.
//
// Interval multiply is the branch-free four-product form: the four
// corner products each rounded down (resp. up), folded with device
// fmin/fmax. Directed rounding is monotone, so min over rounded-down
// corners equals the rounded-down exact corner-min — the branch-free
// form is exactly as tight as the nine-case sign table, with zero
// divergence (case table + equivalence argument: gpu/README.md).
// fmin/fmax drop a single NaN operand, which handles the 0 * inf
// corners of half-unbounded operands conservatively-correctly; an
// all-NaN corner set yields NaN endpoints, which the classifier maps
// to INCONCLUSIVE (never a conclusive verdict).
//
// Normative fold order (the directed references mirror it op for op;
// deterministic evaluation order is a VCARM obligation):
//   acc = [+0.0, +0.0]; for r ascending: term = [1,1];
//   for k ascending: term = term (*) F_k; acc = acc (+) term.
// Accumulate variant: one further (+) of the previous output AFTER
// the fresh fold (matching the s64 lanes' acc + row composition).
//
// The checking entry point (IvlRowCheck) classifies each entry
// against a demanded bound [dlo, dhi] — pass: enclosure inside;
// fail: enclosure entirely outside (witness: lowest conclusively-
// failing index, atomicMin); inconclusive: straddle or NaN — and
// reduces on-device to two scalars, so the host reads a Verdict, not
// an array. NaN-safe by construction: both conclusive predicates are
// positive comparisons, false under NaN.

struct Ivl {
  double lo, hi;
};

__device__ __forceinline__ Ivl ivl_mul(Ivl a, Ivl b) {
  double lo = fmin(fmin(dmul_rd(a.lo, b.lo), dmul_rd(a.lo, b.hi)),
                   fmin(dmul_rd(a.hi, b.lo), dmul_rd(a.hi, b.hi)));
  double hi = fmax(fmax(dmul_ru(a.lo, b.lo), dmul_ru(a.lo, b.hi)),
                   fmax(dmul_ru(a.hi, b.lo), dmul_ru(a.hi, b.hi)));
  return {lo, hi};
}

__device__ __forceinline__ Ivl ivl_add(Ivl a, Ivl b) {
  return {dadd_rd(a.lo, b.lo), dadd_ru(a.hi, b.hi)};
}

// One output entry's fresh fold. The descriptor walk deliberately
// mirrors row_eval_s64 line for line (same wire format, same
// addressing); the s64 kernel is left untouched so its SASS receipt
// stands.
__device__ __forceinline__ Ivl ivl_row_entry(
    long long o, const long long *__restrict__ desc,
    const double *const *fl, const double *const *fh) {
  const long long n_out_axes = desc[0];
  const long long n_red_axes = desc[1];
  const long long n_factors = desc[2];
  const long long red_total = desc[3];
  const long long *out_ext = desc + 4;
  const long long *red_ext = desc + 4 + MAX_M;
  const long long *ostr = desc + 4 + 2 * MAX_M;
  const long long *rstr = desc + 4 + 2 * MAX_M + MAX_FACTORS * MAX_M;

  long long base[MAX_FACTORS] = {0, 0, 0, 0};
  long long rem = o;
  for (long long j = n_out_axes - 1; j >= 0; --j) {
    long long c = rem % out_ext[j];
    rem /= out_ext[j];
    for (long long k = 0; k < n_factors; ++k)
      base[k] += c * ostr[k * MAX_M + j];
  }
  Ivl acc = {0.0, 0.0};
  for (long long r = 0; r < red_total; ++r) {
    long long off[MAX_FACTORS] = {base[0], base[1], base[2], base[3]};
    long long rrem = r;
    for (long long j = n_red_axes - 1; j >= 0; --j) {
      long long c = rrem % red_ext[j];
      rrem /= red_ext[j];
      for (long long k = 0; k < n_factors; ++k)
        off[k] += c * rstr[k * MAX_M + j];
    }
    Ivl term = {1.0, 1.0};
    for (long long k = 0; k < n_factors; ++k)
      term = ivl_mul(term, Ivl{fl[k][off[k]], fh[k][off[k]]});
    acc = ivl_add(acc, term);
  }
  return acc;
}

extern "C" __global__ void ivl_row_eval_f64(
    double *__restrict__ out_lo, double *__restrict__ out_hi,
    const long long *__restrict__ desc, const double *f0l, const double *f0h,
    const double *f1l, const double *f1h, const double *f2l, const double *f2h,
    const double *f3l, const double *f3h, long long n_out, int accumulate) {
  const double *fl[MAX_FACTORS] = {f0l, f1l, f2l, f3l};
  const double *fh[MAX_FACTORS] = {f0h, f1h, f2h, f3h};
  for (long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       o < n_out; o += (long long)gridDim.x * blockDim.x) {
    Ivl v = ivl_row_entry(o, desc, fl, fh);
    if (accumulate) v = ivl_add(Ivl{out_lo[o], out_hi[o]}, v);
    out_lo[o] = v.lo;
    out_hi[o] = v.hi;
  }
}

extern "C" __global__ void ivl_row_check_f64(
    double *__restrict__ out_lo, double *__restrict__ out_hi,
    unsigned long long *__restrict__ fail_idx, int *__restrict__ inconclusive,
    const long long *__restrict__ desc, const double *__restrict__ dlo,
    const double *__restrict__ dhi, const double *f0l, const double *f0h,
    const double *f1l, const double *f1h, const double *f2l, const double *f2h,
    const double *f3l, const double *f3h, long long n_out) {
  const double *fl[MAX_FACTORS] = {f0l, f1l, f2l, f3l};
  const double *fh[MAX_FACTORS] = {f0h, f1h, f2h, f3h};
  for (long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       o < n_out; o += (long long)gridDim.x * blockDim.x) {
    Ivl v = ivl_row_entry(o, desc, fl, fh);
    out_lo[o] = v.lo;
    out_hi[o] = v.hi;
    bool pass = (dlo[o] <= v.lo) && (v.hi <= dhi[o]);
    bool fail = !pass && ((v.hi < dlo[o]) || (v.lo > dhi[o]));
    if (fail)
      atomicMin(fail_idx, (unsigned long long)o);
    else if (!pass)
      atomicOr(inconclusive, 1);
  }
}

namespace {

const long long *typed(const ffi::Buffer<ffi::S64> &b) {
  return reinterpret_cast<const long long *>(b.typed_data());
}

ffi::Error launch(cudaStream_t stream, int64_t n_out,
                  const ffi::Buffer<ffi::S64> &desc,
                  const ffi::Buffer<ffi::S64> &f0,
                  const ffi::Buffer<ffi::S64> &f1,
                  const ffi::Buffer<ffi::S64> &f2,
                  const ffi::Buffer<ffi::S64> &f3, long long *out,
                  int accumulate) {
  if (n_out <= 0) return ffi::Error::InvalidArgument("n_out must be >= 1");
  unsigned block = 256;
  unsigned grid = (unsigned)std::min<int64_t>((n_out + block - 1) / block,
                                              65535);
  row_eval_s64<<<grid, block, 0, stream>>>(
      out, typed(desc), typed(f0), typed(f1), typed(f2), typed(f3),
      (long long)n_out, accumulate);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return ffi::Error::Internal(cudaGetErrorString(err));
  return ffi::Error::Success();
}

ffi::Error RowEvalImpl(cudaStream_t stream, int64_t n_out,
                       ffi::Buffer<ffi::S64> desc, ffi::Buffer<ffi::S64> f0,
                       ffi::Buffer<ffi::S64> f1, ffi::Buffer<ffi::S64> f2,
                       ffi::Buffer<ffi::S64> f3,
                       ffi::ResultBuffer<ffi::S64> out) {
  return launch(stream, n_out, desc, f0, f1, f2, f3,
                reinterpret_cast<long long *>(out->typed_data()),
                /*accumulate=*/0);
}

ffi::Error RowAccumImpl(cudaStream_t stream, int64_t n_out,
                        ffi::Buffer<ffi::S64> desc, ffi::Buffer<ffi::S64> acc,
                        ffi::Buffer<ffi::S64> f0, ffi::Buffer<ffi::S64> f1,
                        ffi::Buffer<ffi::S64> f2, ffi::Buffer<ffi::S64> f3,
                        ffi::ResultBuffer<ffi::S64> out) {
  long long *dst = reinterpret_cast<long long *>(out->typed_data());
  if (dst != typed(acc)) {
    // Alias not honoured: materialize acc into out, then accumulate in
    // out. Verdict-identical either way; only the boundary cost moves.
    cudaError_t err =
        cudaMemcpyAsync(dst, typed(acc), (size_t)n_out * sizeof(long long),
                        cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess)
      return ffi::Error::Internal(cudaGetErrorString(err));
  }
  return launch(stream, n_out, desc, f0, f1, f2, f3, dst, /*accumulate=*/1);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(RowEval, RowEvalImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("n_out")
                                  .Arg<ffi::Buffer<ffi::S64>>()  // desc
                                  .Arg<ffi::Buffer<ffi::S64>>()  // f0
                                  .Arg<ffi::Buffer<ffi::S64>>()  // f1
                                  .Arg<ffi::Buffer<ffi::S64>>()  // f2
                                  .Arg<ffi::Buffer<ffi::S64>>()  // f3
                                  .Ret<ffi::Buffer<ffi::S64>>());

namespace {

ffi::Error IvlAddMulImpl(cudaStream_t stream, int64_t n,
                         ffi::Buffer<ffi::F64> a_lo, ffi::Buffer<ffi::F64> a_hi,
                         ffi::Buffer<ffi::F64> b_lo, ffi::Buffer<ffi::F64> b_hi,
                         ffi::ResultBuffer<ffi::F64> sum_lo,
                         ffi::ResultBuffer<ffi::F64> sum_hi,
                         ffi::ResultBuffer<ffi::F64> prod_lo,
                         ffi::ResultBuffer<ffi::F64> prod_hi) {
  if (n <= 0) return ffi::Error::InvalidArgument("n must be >= 1");
  unsigned block = 256;
  unsigned grid = (unsigned)std::min<int64_t>((n + block - 1) / block, 65535);
  ivl_addmul_f64<<<grid, block, 0, stream>>>(
      sum_lo->typed_data(), sum_hi->typed_data(), prod_lo->typed_data(),
      prod_hi->typed_data(), a_lo.typed_data(), a_hi.typed_data(),
      b_lo.typed_data(), b_hi.typed_data(), (long long)n);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return ffi::Error::Internal(cudaGetErrorString(err));
  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(IvlAddMul, IvlAddMulImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("n")
                                  .Arg<ffi::Buffer<ffi::F64>>()  // a_lo
                                  .Arg<ffi::Buffer<ffi::F64>>()  // a_hi
                                  .Arg<ffi::Buffer<ffi::F64>>()  // b_lo
                                  .Arg<ffi::Buffer<ffi::F64>>()  // b_hi
                                  .Ret<ffi::Buffer<ffi::F64>>()  // sum_lo
                                  .Ret<ffi::Buffer<ffi::F64>>()  // sum_hi
                                  .Ret<ffi::Buffer<ffi::F64>>()  // prod_lo
                                  .Ret<ffi::Buffer<ffi::F64>>());  // prod_hi

XLA_FFI_DEFINE_HANDLER_SYMBOL(RowAccum, RowAccumImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("n_out")
                                  .Arg<ffi::Buffer<ffi::S64>>()  // desc
                                  .Arg<ffi::Buffer<ffi::S64>>()  // acc
                                  .Arg<ffi::Buffer<ffi::S64>>()  // f0
                                  .Arg<ffi::Buffer<ffi::S64>>()  // f1
                                  .Arg<ffi::Buffer<ffi::S64>>()  // f2
                                  .Arg<ffi::Buffer<ffi::S64>>()  // f3
                                  .Ret<ffi::Buffer<ffi::S64>>());

// ── interval-enclosure handlers ──────────────────────────────────────

namespace {

struct IvlPtrs {
  const double *fl[MAX_FACTORS];
  const double *fh[MAX_FACTORS];
};

IvlPtrs ivl_ptrs(const ffi::Buffer<ffi::F64> &f0l,
                 const ffi::Buffer<ffi::F64> &f0h,
                 const ffi::Buffer<ffi::F64> &f1l,
                 const ffi::Buffer<ffi::F64> &f1h,
                 const ffi::Buffer<ffi::F64> &f2l,
                 const ffi::Buffer<ffi::F64> &f2h,
                 const ffi::Buffer<ffi::F64> &f3l,
                 const ffi::Buffer<ffi::F64> &f3h) {
  return IvlPtrs{{f0l.typed_data(), f1l.typed_data(), f2l.typed_data(),
                  f3l.typed_data()},
                 {f0h.typed_data(), f1h.typed_data(), f2h.typed_data(),
                  f3h.typed_data()}};
}

ffi::Error ivl_launch(cudaStream_t stream, int64_t n_out,
                      const ffi::Buffer<ffi::S64> &desc, const IvlPtrs &p,
                      double *out_lo, double *out_hi, int accumulate) {
  if (n_out <= 0) return ffi::Error::InvalidArgument("n_out must be >= 1");
  unsigned block = 256;
  unsigned grid =
      (unsigned)std::min<int64_t>((n_out + block - 1) / block, 65535);
  ivl_row_eval_f64<<<grid, block, 0, stream>>>(
      out_lo, out_hi, typed(desc), p.fl[0], p.fh[0], p.fl[1], p.fh[1],
      p.fl[2], p.fh[2], p.fl[3], p.fh[3], (long long)n_out, accumulate);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
  return ffi::Error::Success();
}

ffi::Error IvlRowEvalImpl(
    cudaStream_t stream, int64_t n_out, ffi::Buffer<ffi::S64> desc,
    ffi::Buffer<ffi::F64> f0l, ffi::Buffer<ffi::F64> f0h,
    ffi::Buffer<ffi::F64> f1l, ffi::Buffer<ffi::F64> f1h,
    ffi::Buffer<ffi::F64> f2l, ffi::Buffer<ffi::F64> f2h,
    ffi::Buffer<ffi::F64> f3l, ffi::Buffer<ffi::F64> f3h,
    ffi::ResultBuffer<ffi::F64> out_lo, ffi::ResultBuffer<ffi::F64> out_hi) {
  return ivl_launch(stream, n_out, desc,
                    ivl_ptrs(f0l, f0h, f1l, f1h, f2l, f2h, f3l, f3h),
                    out_lo->typed_data(), out_hi->typed_data(),
                    /*accumulate=*/0);
}

ffi::Error IvlRowAccumImpl(
    cudaStream_t stream, int64_t n_out, ffi::Buffer<ffi::S64> desc,
    ffi::Buffer<ffi::F64> acc_lo, ffi::Buffer<ffi::F64> acc_hi,
    ffi::Buffer<ffi::F64> f0l, ffi::Buffer<ffi::F64> f0h,
    ffi::Buffer<ffi::F64> f1l, ffi::Buffer<ffi::F64> f1h,
    ffi::Buffer<ffi::F64> f2l, ffi::Buffer<ffi::F64> f2h,
    ffi::Buffer<ffi::F64> f3l, ffi::Buffer<ffi::F64> f3h,
    ffi::ResultBuffer<ffi::F64> out_lo, ffi::ResultBuffer<ffi::F64> out_hi) {
  // Un-honoured aliases fall back to a device copy, exactly as in
  // RowAccum: semantics never depend on the alias, only the cost does.
  double *dst_lo = out_lo->typed_data();
  double *dst_hi = out_hi->typed_data();
  if (dst_lo != acc_lo.typed_data()) {
    cudaError_t err = cudaMemcpyAsync(
        dst_lo, acc_lo.typed_data(), (size_t)n_out * sizeof(double),
        cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess)
      return ffi::Error::Internal(cudaGetErrorString(err));
  }
  if (dst_hi != acc_hi.typed_data()) {
    cudaError_t err = cudaMemcpyAsync(
        dst_hi, acc_hi.typed_data(), (size_t)n_out * sizeof(double),
        cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess)
      return ffi::Error::Internal(cudaGetErrorString(err));
  }
  return ivl_launch(stream, n_out, desc,
                    ivl_ptrs(f0l, f0h, f1l, f1h, f2l, f2h, f3l, f3h), dst_lo,
                    dst_hi, /*accumulate=*/1);
}

ffi::Error IvlRowCheckImpl(
    cudaStream_t stream, int64_t n_out, ffi::Buffer<ffi::S64> desc,
    ffi::Buffer<ffi::F64> dlo, ffi::Buffer<ffi::F64> dhi,
    ffi::Buffer<ffi::F64> f0l, ffi::Buffer<ffi::F64> f0h,
    ffi::Buffer<ffi::F64> f1l, ffi::Buffer<ffi::F64> f1h,
    ffi::Buffer<ffi::F64> f2l, ffi::Buffer<ffi::F64> f2h,
    ffi::Buffer<ffi::F64> f3l, ffi::Buffer<ffi::F64> f3h,
    ffi::ResultBuffer<ffi::F64> out_lo, ffi::ResultBuffer<ffi::F64> out_hi,
    ffi::ResultBuffer<ffi::U64> fail_idx,
    ffi::ResultBuffer<ffi::S32> inconclusive) {
  if (n_out <= 0) return ffi::Error::InvalidArgument("n_out must be >= 1");
  // Verdict scalars: fail_idx = ULLONG_MAX sentinel (all-ones bytes,
  // memset-expressible; atomicMin narrows it), inconclusive = 0.
  cudaError_t err = cudaMemsetAsync(fail_idx->typed_data(), 0xFF,
                                    sizeof(unsigned long long), stream);
  if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
  err = cudaMemsetAsync(inconclusive->typed_data(), 0, sizeof(int), stream);
  if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
  IvlPtrs p = ivl_ptrs(f0l, f0h, f1l, f1h, f2l, f2h, f3l, f3h);
  unsigned block = 256;
  unsigned grid =
      (unsigned)std::min<int64_t>((n_out + block - 1) / block, 65535);
  ivl_row_check_f64<<<grid, block, 0, stream>>>(
      out_lo->typed_data(), out_hi->typed_data(),
      reinterpret_cast<unsigned long long *>(fail_idx->typed_data()),
      inconclusive->typed_data(), typed(desc), dlo.typed_data(),
      dhi.typed_data(), p.fl[0], p.fh[0], p.fl[1], p.fh[1], p.fl[2], p.fh[2],
      p.fl[3], p.fh[3], (long long)n_out);
  err = cudaGetLastError();
  if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
  return ffi::Error::Success();
}

}  // namespace

#define IVL_FACTOR_ARGS                 \
  .Arg<ffi::Buffer<ffi::F64>>() /*f0l*/ \
      .Arg<ffi::Buffer<ffi::F64>>() /*f0h*/ \
      .Arg<ffi::Buffer<ffi::F64>>() /*f1l*/ \
      .Arg<ffi::Buffer<ffi::F64>>() /*f1h*/ \
      .Arg<ffi::Buffer<ffi::F64>>() /*f2l*/ \
      .Arg<ffi::Buffer<ffi::F64>>() /*f2h*/ \
      .Arg<ffi::Buffer<ffi::F64>>() /*f3l*/ \
      .Arg<ffi::Buffer<ffi::F64>>() /*f3h*/

XLA_FFI_DEFINE_HANDLER_SYMBOL(IvlRowEval, IvlRowEvalImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("n_out")
                                  .Arg<ffi::Buffer<ffi::S64>>()  // desc
                                  IVL_FACTOR_ARGS
                                  .Ret<ffi::Buffer<ffi::F64>>()   // out_lo
                                  .Ret<ffi::Buffer<ffi::F64>>());  // out_hi

XLA_FFI_DEFINE_HANDLER_SYMBOL(IvlRowAccum, IvlRowAccumImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("n_out")
                                  .Arg<ffi::Buffer<ffi::S64>>()  // desc
                                  .Arg<ffi::Buffer<ffi::F64>>()  // acc_lo
                                  .Arg<ffi::Buffer<ffi::F64>>()  // acc_hi
                                  IVL_FACTOR_ARGS
                                  .Ret<ffi::Buffer<ffi::F64>>()   // out_lo
                                  .Ret<ffi::Buffer<ffi::F64>>());  // out_hi

XLA_FFI_DEFINE_HANDLER_SYMBOL(IvlRowCheck, IvlRowCheckImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("n_out")
                                  .Arg<ffi::Buffer<ffi::S64>>()  // desc
                                  .Arg<ffi::Buffer<ffi::F64>>()  // dlo
                                  .Arg<ffi::Buffer<ffi::F64>>()  // dhi
                                  IVL_FACTOR_ARGS
                                  .Ret<ffi::Buffer<ffi::F64>>()  // out_lo
                                  .Ret<ffi::Buffer<ffi::F64>>()  // out_hi
                                  .Ret<ffi::Buffer<ffi::U64>>()  // fail_idx
                                  .Ret<ffi::Buffer<ffi::S32>>());  // inconclusive
