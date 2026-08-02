// roweq.cu — the CUDA lane of the roweq family (see the core crate's
// `roweq` module documentation for semantics; this file implements
// exactly the reference lane's predicate: structural row equality,
// first-match pool scan).
//
// Thread mapping: one thread per query row; each thread scans its
// attempt's pool rows in order and stops at the first structural
// match — the reference's own access pattern, parallel over queries.
// No arithmetic exists in this kernel at all (compares only): no
// rounding, no overflow, no capacity refusals; verdict identity with
// the reference is comparison-for-comparison.
//
// Memory shape: mag is limb-plane SoA (limb l of slot s at
// mag[l*n_slots + s], matching the rnsfold family); per-position slot
// compares read k strided words per side. Volume is small relative to
// the sibling fold kernel; a coalesced-layout variant is a measured
// successor, not assumed.

typedef unsigned long long u64;
typedef unsigned int u32;

extern "C" __global__ void roweq_member(
    u32 n_queries, u32 n_slots, u32 k,
    const u32 *__restrict__ q_row,    // [n_queries] row id of each query
    const u32 *__restrict__ p_lo,     // [n_queries] pool row range lo
    const u32 *__restrict__ p_hi,     // [n_queries] pool row range hi
    const u32 *__restrict__ row_ptr,  // [n_rows+1]
    const u32 *__restrict__ nnz_col,  // [nnz]
    const u32 *__restrict__ nnz_slot, // [nnz]
    const int *__restrict__ sign,     // [n_slots]
    const u64 *__restrict__ mag,      // [k*n_slots], limb planes
    const u32 *__restrict__ den_id,   // [n_slots]
    u32 *__restrict__ matched)        // [n_queries]; 1 = some pool row equal
{
    u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_queries) return;
    u32 q = q_row[t];
    u32 q0 = row_ptr[q];
    u32 len = row_ptr[q + 1] - q0;

    u32 ok = 0;
    for (u32 p = p_lo[t]; p < p_hi[t] && !ok; ++p) {
        u32 p0 = row_ptr[p];
        if (row_ptr[p + 1] - p0 != len) continue;
        u32 eq = 1;
        for (u32 i = 0; i < len && eq; ++i) {
            u32 ia = q0 + i, ib = p0 + i;
            if (nnz_col[ia] != nnz_col[ib]) { eq = 0; break; }
            u32 sa = nnz_slot[ia], sb = nnz_slot[ib];
            if (sa == sb) continue;
            if (sign[sa] != sign[sb] || den_id[sa] != den_id[sb]) { eq = 0; break; }
            for (u32 l = 0; l < k; ++l) {
                if (mag[(size_t)l * n_slots + sa] != mag[(size_t)l * n_slots + sb]) {
                    eq = 0;
                    break;
                }
            }
        }
        ok = eq;
    }
    matched[t] = ok;
}
