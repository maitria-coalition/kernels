// rnsfold.cu — the CUDA lane of the rnsfold family (see the core
// crate's `rnsfold` module documentation for semantics and the
// soundness argument; this file implements exactly the reference
// lane's predicate, in Montgomery form).
//
// Thread mapping: one thread per (acol, channel) — blockIdx.y is the
// channel, blockDim.x-strided x covers acols. Per-thread state is
// three u64 registers (accumulator + scratch); no local arrays, no
// carries, no capacity refusals anywhere on device (channel-count
// selection happened on the host, from the descriptor's own planes).
//
// Montgomery conventions (R = 2^64, p an odd 63-bit prime):
//   montmul(a, b) = a*b*R^{-1} mod p     valid for a < 2^64, b < p
//   powr2[ch*k+l] = 2^{64(l+2)} mod p    so montmul(limb_l, powr2_l)
//                                        = limb_l * 2^{64 l} * R mod p
//   lamres / multres are Montgomery-form (one R factor) residues,
//   host-computed. Products of two one-R operands under montmul keep
//   exactly one R factor, so both sides of the final compare carry R
//   and equality is form-independent.
//
// Value operands arrive RAW (limb planes + sign + multiplier id) and
// are decomposed in registers per use — each slot is read by exactly
// one (acol) thread per channel, so nothing is ever materialized.

typedef unsigned long long u64;
typedef unsigned int u32;

__device__ __forceinline__ u64 addmod(u64 a, u64 b, u64 p) {
    // a, b < p < 2^63: the sum cannot wrap u64.
    u64 s = a + b;
    return s >= p ? s - p : s;
}

// Montgomery REDC product. Precondition: a < 2^64, b < p < 2^63.
// Result: a*b*R^{-1} mod p, canonical (< p).
__device__ __forceinline__ u64 montmul(u64 a, u64 b, u64 p, u64 pinv) {
    u64 lo = a * b;
    u64 hi = __umul64hi(a, b);
    u64 m = lo * pinv; // mod 2^64
    u64 mp_hi = __umul64hi(m, p);
    // low half of (lo + m*p) is zero by REDC construction; its carry
    // out is 1 exactly when lo != 0.
    u64 u = hi + mp_hi + (lo != 0);
    return u >= p ? u - p : u;
}

// Decompose slot s: signed raw magnitude times its multiplier, into
// Montgomery form. mag is limb-plane SoA: limb l of slot s at
// mag[l*n_slots + s].
__device__ __forceinline__ u64 slot_res(
    u32 s, u32 k, u32 n_slots, u32 n_mults, u32 ch,
    const u64 *__restrict__ mag, const int *__restrict__ sign,
    const u32 *__restrict__ mult_id, const u64 *__restrict__ powr2,
    const u64 *__restrict__ multres, u64 p, u64 pinv) {
    u64 v = 0;
    const u64 *pw = powr2 + (size_t)ch * k;
    for (u32 l = 0; l < k; ++l)
        v = addmod(v, montmul(mag[(size_t)l * n_slots + s], pw[l], p, pinv), p);
    if (sign[s] < 0 && v != 0) v = p - v;
    return montmul(v, multres[(size_t)ch * n_mults + mult_id[s]], p, pinv);
}

extern "C" __global__ void rnsfold_fold(
    u32 n_acols, u32 n_slots, u32 k, u32 n_lams, u32 n_mults,
    const u64 *__restrict__ primes,       // [C]
    const u64 *__restrict__ pinvs,        // [C]
    const u64 *__restrict__ powr2,        // [C*k]
    const u64 *__restrict__ lamres,       // [C*n_lams]
    const u64 *__restrict__ multres,      // [C*n_mults]
    const int *__restrict__ sign,         // [n_slots]
    const u64 *__restrict__ mag,          // [k*n_slots], limb planes
    const u32 *__restrict__ mult_id,      // [n_slots]
    const u32 *__restrict__ acol_attempt, // [n_acols]
    const u32 *__restrict__ csc_ptr,      // [n_acols+1]
    const u32 *__restrict__ csc_lam,      // [nnz]
    const u32 *__restrict__ csc_slot,     // [nnz]
    const u32 *__restrict__ concl_slot,   // [n_acols]; 0xFFFFFFFF = absent
    u32 *flags)                           // [n_attempts]; bit 0 = mismatch
{
    u32 acol = blockIdx.x * blockDim.x + threadIdx.x;
    u32 ch = blockIdx.y;
    if (acol >= n_acols) return;
    u64 p = primes[ch];
    u64 pinv = pinvs[ch];

    u64 acc = 0;
    u32 i0 = csc_ptr[acol], i1 = csc_ptr[acol + 1];
    for (u32 i = i0; i < i1; ++i) {
        u64 v = slot_res(csc_slot[i], k, n_slots, n_mults, ch, mag, sign,
                         mult_id, powr2, multres, p, pinv);
        u64 lam = lamres[(size_t)ch * n_lams + csc_lam[i]];
        acc = addmod(acc, montmul(lam, v, p, pinv), p);
    }

    u32 cs = concl_slot[acol];
    u64 c = (cs == 0xFFFFFFFFu)
                ? 0
                : slot_res(cs, k, n_slots, n_mults, ch, mag, sign, mult_id,
                           powr2, multres, p, pinv);

    if (acc != c) atomicOr(&flags[acol_attempt[acol]], 1u);
}
