//! The sparse lane: fill-reducing mod-$p$ elimination for giant
//! inputs.
//!
//! Two phases, one optional dense core:
//!
//! 1. **Row-space reduction, free pivot order.** Pivots are chosen by
//!    the Dumas–Villard linear-pivoting heuristic (sparsest remaining
//!    row; within it, the column populating the fewest active rows —
//!    the local Markowitz cost that accounts for exact cancellation),
//!    eliminating each pivot column from every other *active* row and
//!    retiring the pivot row untouched thereafter. Retired rows are
//!    triangular against each other (each is the unique holder of its
//!    pivot column among rows retired no earlier), so their count is
//!    the rank and their original indices are the rank witness.
//!    When the active submatrix is small and full enough (see
//!    [`Params`]), it is materialized and finished by dense echelon —
//!    the classic hybrid, cheaper than fighting fill sparsely.
//! 2. **Leftmost canonicalization.** The surviving rows (rank many)
//!    are reduced to *reduced* row echelon form scanning columns left
//!    to right, which yields the leftmost column rank profile and the
//!    canonical basis — bit-identical to the dense reference's, since
//!    both are intrinsic to the matrix mod $p$.
//!
//! Arithmetic is Montgomery form on the axpy hot path (one conversion
//! per eliminated row, plain-residue storage). Fill is metered: if
//! live entries ever exceed the caller's budget the lane refuses by
//! typed outcome ([`NullspaceError::FillBudget`]) — for inputs whose
//! fill defeats elimination the documented successor is a black-box
//! (Wiedemann-family) lane, per the elimination-vs-iteration verdict
//! of the finite-field linear algebra literature.
//!
//! Single-threaded by design: the core crate is dependency-free (see
//! the crate manifest), and per-pivot update sets are the natural
//! parallel seam — a `rayon`-feature lane over the same merge kernel
//! is the named next lane if receipts demand one.

use super::{prime_ok, ModpNullspace, NullspaceError, SparseVec, Triplets};

/// Tuning knobs. Every field changes cost, never verdicts (the
/// battery runs degenerate values to enforce exactly that).
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Refuse (typed) when live sparse entries — plus the dense core,
    /// while one is materialized — exceed this. Default `1 << 27`
    /// entries (~2 GiB of row storage at 12 bytes/entry).
    pub max_entries: usize,
    /// Largest `active_rows * active_cols` the dense core may
    /// materialize. Default `1 << 24` (128 MiB of `u64`).
    pub dense_cap: usize,
    /// Switch to the dense core when
    /// `live_entries * dense_inv_density >= active_rows * active_cols`
    /// (i.e. density has reached `1 / dense_inv_density`) and the
    /// product fits `dense_cap`. Default 8 (switch at 12.5% full).
    pub dense_inv_density: usize,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            max_entries: 1 << 27,
            dense_cap: 1 << 24,
            dense_inv_density: 8,
        }
    }
}

// ---------------------------------------------------------------
// Montgomery arithmetic, R = 2^64, odd p < 2^63.
// ---------------------------------------------------------------

struct Mont {
    p: u64,
    /// `-p^{-1} mod 2^64`.
    ninv: u64,
    /// `R^2 mod p`.
    r2: u64,
}

impl Mont {
    fn new(p: u64) -> Mont {
        // Newton iteration doubles correct low bits each round:
        // 5 rounds from a seed exact mod 2^2 reach 2^64.
        let mut inv = p; // p * p ≡ 1 (mod 4) for odd p: exact mod 2^2… seed p is exact mod 2^3 in fact
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(p.wrapping_mul(inv)));
        }
        debug_assert_eq!(p.wrapping_mul(inv), 1);
        let r = ((1u128 << 64) % p as u128) as u64;
        let r2 = ((r as u128 * r as u128) % p as u128) as u64;
        Mont {
            p,
            ninv: inv.wrapping_neg(),
            r2,
        }
    }

    /// `a * b * R^{-1} mod p`. With `a` in Montgomery form and `b`
    /// plain, this is the plain product `(a/R) * b mod p`.
    #[inline]
    fn mulredc(&self, a: u64, b: u64) -> u64 {
        let t = a as u128 * b as u128;
        let m = (t as u64).wrapping_mul(self.ninv);
        let t = ((t + m as u128 * self.p as u128) >> 64) as u64;
        if t >= self.p {
            t - self.p
        } else {
            t
        }
    }

    /// Plain residue -> Montgomery form.
    #[inline]
    fn to_mont(&self, a: u64) -> u64 {
        self.mulredc(a, self.r2)
    }
}

#[inline]
fn submod(a: u64, b: u64, p: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a + p - b
    }
}

#[inline]
fn mulmod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

fn invmod(a: u64, p: u64) -> u64 {
    let mut e = p - 2;
    let mut base = a % p;
    let mut acc = 1u64;
    while e > 0 {
        if e & 1 == 1 {
            acc = mulmod(acc, base, p);
        }
        base = mulmod(base, base, p);
        e >>= 1;
    }
    acc
}

#[inline]
fn residue(w: i64, p: u64) -> u64 {
    if w >= 0 {
        w as u64 % p
    } else {
        (p - w.unsigned_abs() % p) % p
    }
}

// ---------------------------------------------------------------
// Sparse row state shared by both phases.
// ---------------------------------------------------------------

struct RowStore {
    cols: Vec<Vec<u32>>,
    vals: Vec<Vec<u64>>,
    /// Original (pre-dedup) row index per stored row.
    orig: Vec<u32>,
    live_entries: usize,
}

impl RowStore {
    fn nnz(&self, r: usize) -> usize {
        self.cols[r].len()
    }

    fn contains(&self, r: usize, c: u32) -> Option<usize> {
        self.cols[r].binary_search(&c).ok()
    }

    fn replace(&mut self, r: usize, cols: Vec<u32>, vals: Vec<u64>) {
        self.live_entries -= self.cols[r].len();
        self.live_entries += cols.len();
        self.cols[r] = cols;
        self.vals[r] = vals;
    }

    fn clear(&mut self, r: usize) {
        self.live_entries -= self.cols[r].len();
        self.cols[r] = Vec::new();
        self.vals[r] = Vec::new();
    }
}

/// `target - f * pivot` over sorted sparse rows (two-pointer merge).
/// `f_mont` is `f` in Montgomery form, so `mulredc(f_mont, x)` is the
/// plain product `f * x mod p`. Returns the new row and appends to
/// `added` the columns present in the result but absent from the
/// target (adjacency maintenance).
#[allow(clippy::too_many_arguments)]
fn axpy_merge(
    tc: &[u32],
    tv: &[u64],
    pc: &[u32],
    pv: &[u64],
    f_mont: u64,
    mont: &Mont,
    added: &mut Vec<u32>,
) -> (Vec<u32>, Vec<u64>) {
    let p = mont.p;
    let mut oc = Vec::with_capacity(tc.len() + pc.len());
    let mut ov = Vec::with_capacity(tc.len() + pc.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < tc.len() || j < pc.len() {
        if j == pc.len() || (i < tc.len() && tc[i] < pc[j]) {
            oc.push(tc[i]);
            ov.push(tv[i]);
            i += 1;
        } else if i == tc.len() || pc[j] < tc[i] {
            // Pivot-only column: -f * pv, never zero (both factors
            // nonzero mod a prime).
            let v = p - mont.mulredc(f_mont, pv[j]);
            debug_assert_ne!(v, 0);
            oc.push(pc[j]);
            ov.push(v);
            added.push(pc[j]);
            j += 1;
        } else {
            let v = submod(tv[i], mont.mulredc(f_mont, pv[j]), p);
            if v != 0 {
                oc.push(tc[i]);
                ov.push(v);
            }
            i += 1;
            j += 1;
        }
    }
    (oc, ov)
}

// ---------------------------------------------------------------
// The lane.
// ---------------------------------------------------------------

/// Canonical mod-$p$ nullspace of `a` — verdict-identical to
/// [`super::reference::nullspace_mod_p`], engineered for giant sparse
/// inputs. See the module documentation for the algorithm and the
/// refusal surface.
pub fn nullspace_mod_p(
    a: Triplets<'_>,
    p: u64,
    params: Params,
) -> Result<ModpNullspace, NullspaceError> {
    if !prime_ok(p) {
        return Err(NullspaceError::BadPrime { prime: p });
    }
    a.validate()?;
    let n = a.cols;
    let mont = Mont::new(p);

    let mut store = build_rows(a, p, params.max_entries)?;
    let done = phase1(&mut store, n, p, &mont, &params)?;
    let (pivot_cols, rref_rows) = phase2(&mut store, &done, n, p, &mont, &params)?;
    debug_assert_eq!(rref_rows.len(), done.len());

    let rank = pivot_cols.len();
    let witness_rows: Vec<u32> = done.iter().map(|&r| store.orig[r]).collect();
    let basis = extract_basis(&store, &pivot_cols, &rref_rows, n, p);

    Ok(ModpNullspace {
        prime: p,
        rank,
        pivot_cols,
        witness_rows,
        basis,
    })
}

/// Accumulate triplets into sorted sparse rows mod `p`, dropping zero
/// entries, zero rows, and duplicate rows (identical support and
/// values — row-space preserving, and common in contest-scale Petri
/// nets).
fn build_rows(a: Triplets<'_>, p: u64, max_entries: usize) -> Result<RowStore, NullspaceError> {
    let mut per_row: Vec<Vec<(u32, i64)>> = vec![Vec::new(); a.rows];
    for &(r, c, w) in a.entries {
        per_row[r as usize].push((c, w));
    }
    let mut store = RowStore {
        cols: Vec::new(),
        vals: Vec::new(),
        orig: Vec::new(),
        live_entries: 0,
    };
    let mut seen = std::collections::HashMap::new();
    for (r, mut ents) in per_row.into_iter().enumerate() {
        ents.sort_unstable_by_key(|&(c, _)| c);
        let mut cols: Vec<u32> = Vec::with_capacity(ents.len());
        let mut vals: Vec<u64> = Vec::with_capacity(ents.len());
        let mut k = 0usize;
        while k < ents.len() {
            let c = ents[k].0;
            let mut acc = 0u64;
            while k < ents.len() && ents[k].0 == c {
                acc = (acc + residue(ents[k].1, p)) % p;
                k += 1;
            }
            if acc != 0 {
                cols.push(c);
                vals.push(acc);
            }
        }
        if cols.is_empty() {
            continue;
        }
        // Row dedup: hash the exact (cols, vals) content.
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        cols.hash(&mut h);
        vals.hash(&mut h);
        let key = (h.finish(), cols.len());
        let entry = seen.entry(key).or_insert_with(Vec::new);
        let dup = entry
            .iter()
            .any(|&prev| store.cols[prev] == cols && store.vals[prev] == vals);
        if dup {
            continue;
        }
        entry.push(store.cols.len());
        store.live_entries += cols.len();
        if store.live_entries > max_entries {
            return Err(NullspaceError::FillBudget {
                live_entries: store.live_entries,
                max_entries,
                pivots_done: 0,
                phase: 1,
            });
        }
        store.cols.push(cols);
        store.vals.push(vals);
        store.orig.push(r as u32);
    }
    Ok(store)
}

/// Phase 1: free-order row-space reduction. Returns the retired
/// (pivot) row ids, in retirement order.
fn phase1(
    store: &mut RowStore,
    n: usize,
    p: u64,
    mont: &Mont,
    params: &Params,
) -> Result<Vec<usize>, NullspaceError> {
    let m = store.cols.len();
    let mut active: Vec<bool> = vec![true; m];
    let mut colcount: Vec<u32> = vec![0; n];
    let mut col_rows: Vec<Vec<u32>> = vec![Vec::new(); n];
    for r in 0..m {
        for &c in &store.cols[r] {
            colcount[c as usize] += 1;
            col_rows[c as usize].push(r as u32);
        }
    }
    let mut active_rows = m;
    let mut active_cols = colcount.iter().filter(|&&x| x > 0).count();

    // Lazy min-heap on (nnz, row): stale entries skipped on pop.
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let mut heap: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
    for r in 0..m {
        heap.push(Reverse((store.nnz(r), r)));
    }
    // Row-visit stamps dedup adjacency lists during a pivot step.
    let mut stamp: Vec<u32> = vec![0; m];
    let mut generation: u32 = 0;

    let mut done: Vec<usize> = Vec::new();

    loop {
        // Dense-core switch check.
        if active_rows > 0 {
            let prod = active_rows.checked_mul(active_cols);
            if let Some(prod) = prod {
                if prod <= params.dense_cap
                    && store.live_entries.saturating_mul(params.dense_inv_density) >= prod
                    && store.live_entries + prod <= params.max_entries
                {
                    dense_core(store, &mut active, &mut done, n, p);
                    break;
                }
            }
        }

        // Next pivot row: least-populated active row (lazy heap).
        let r = loop {
            match heap.pop() {
                None => break usize::MAX,
                Some(Reverse((nnz, r))) => {
                    if active[r] && store.nnz(r) == nnz && nnz > 0 {
                        break r;
                    }
                }
            }
        };
        if r == usize::MAX {
            break;
        }

        // Pivot column: least-populated among the row's columns
        // (ties to the smallest index — deterministic).
        let (mut best_c, mut best_cnt) = (store.cols[r][0], u32::MAX);
        for &c in &store.cols[r] {
            let cnt = colcount[c as usize];
            if cnt < best_cnt {
                best_cnt = cnt;
                best_c = c;
            }
        }
        let c = best_c;
        let pk = store.contains(r, c).expect("pivot col in pivot row");
        let winv = invmod(store.vals[r][pk], p);

        // Eliminate c from every other active row holding it.
        generation += 1;
        let holders = std::mem::take(&mut col_rows[c as usize]);
        let mut added: Vec<u32> = Vec::new();
        for &t32 in &holders {
            let t = t32 as usize;
            if t == r || !active[t] || stamp[t] == generation {
                continue;
            }
            stamp[t] = generation;
            let Some(tk) = store.contains(t, c) else {
                continue; // lazy adjacency: stale
            };
            let f = mulmod(store.vals[t][tk], winv, p);
            let f_mont = mont.to_mont(f);
            added.clear();
            let (nc, nv) = axpy_merge(
                &store.cols[t],
                &store.vals[t],
                &store.cols[r],
                &store.vals[r],
                f_mont,
                mont,
                &mut added,
            );
            // Column counts: out with the old support, in with the new.
            for &oc in &store.cols[t] {
                colcount[oc as usize] -= 1;
                if colcount[oc as usize] == 0 {
                    active_cols -= 1;
                }
            }
            for &ncol in &nc {
                if colcount[ncol as usize] == 0 {
                    active_cols += 1;
                }
                colcount[ncol as usize] += 1;
            }
            for &ac in &added {
                col_rows[ac as usize].push(t32);
            }
            store.replace(t, nc, nv);
            if store.live_entries > params.max_entries {
                return Err(NullspaceError::FillBudget {
                    live_entries: store.live_entries,
                    max_entries: params.max_entries,
                    pivots_done: done.len(),
                    phase: 1,
                });
            }
            if store.nnz(t) == 0 {
                active[t] = false;
                active_rows -= 1;
            } else {
                heap.push(Reverse((store.nnz(t), t)));
            }
        }

        // Retire the pivot row: it leaves the active set untouched.
        active[r] = false;
        active_rows -= 1;
        for &rc in &store.cols[r] {
            colcount[rc as usize] -= 1;
            if colcount[rc as usize] == 0 {
                active_cols -= 1;
            }
        }
        done.push(r);
    }
    Ok(done)
}

/// Finish the active submatrix densely: plain row echelon over the
/// compacted active block, retiring pivot rows into `done`.
fn dense_core(store: &mut RowStore, active: &mut [bool], done: &mut Vec<usize>, n: usize, p: u64) {
    let rows: Vec<usize> = (0..store.cols.len()).filter(|&r| active[r]).collect();
    if rows.is_empty() {
        return;
    }
    // Compact the populated columns, ascending (so compact order is
    // global order and re-sparsified rows come out sorted).
    let mut col_used = vec![false; n];
    for &r in &rows {
        for &c in &store.cols[r] {
            col_used[c as usize] = true;
        }
    }
    let cmap: Vec<u32> = (0..n as u32).filter(|&c| col_used[c as usize]).collect();
    let mut cinv = vec![u32::MAX; n];
    for (k, &c) in cmap.iter().enumerate() {
        cinv[c as usize] = k as u32;
    }
    let (mr, mc) = (rows.len(), cmap.len());
    let mut dm = vec![0u64; mr * mc];
    for (i, &r) in rows.iter().enumerate() {
        for (k, &c) in store.cols[r].iter().enumerate() {
            dm[i * mc + cinv[c as usize] as usize] = store.vals[r][k];
        }
    }
    // Row echelon, leftmost column, first-available row; `perm[i]` is
    // the store row id currently sitting at dense row `i` (swapped in
    // lock-step with the matrix so pivot identities survive).
    let mut perm: Vec<usize> = rows.clone();
    let mut prow = 0usize;
    let mut pivots: Vec<(usize, usize)> = Vec::new(); // (dense row, dense col)
    for col in 0..mc {
        let Some(pr) = (prow..mr).find(|&i| dm[i * mc + col] != 0) else {
            continue;
        };
        if pr != prow {
            for c in 0..mc {
                dm.swap(prow * mc + c, pr * mc + c);
            }
            perm.swap(prow, pr);
        }
        pivots.push((prow, col));
        let inv = invmod(dm[prow * mc + col], p);
        for i in (prow + 1)..mr {
            if dm[i * mc + col] != 0 {
                let f = mulmod(dm[i * mc + col], inv, p);
                for c in col..mc {
                    let d = mulmod(f, dm[prow * mc + c], p);
                    dm[i * mc + c] = submod(dm[i * mc + c], d, p);
                }
            }
        }
        prow += 1;
        if prow == mr {
            break;
        }
    }
    // Retire pivot rows carrying their echelon states; clear the rest
    // (eliminated to redundancy — they are spanned by the pivots).
    let pivot_row_set: std::collections::HashSet<usize> =
        pivots.iter().map(|&(i, _)| perm[i]).collect();
    for &r in &rows {
        active[r] = false;
        if !pivot_row_set.contains(&r) {
            store.clear(r);
        }
    }
    for &(i, col) in &pivots {
        let r = perm[i];
        let mut cols: Vec<u32> = Vec::new();
        let mut vals: Vec<u64> = Vec::new();
        for k in col..mc {
            let v = dm[i * mc + k];
            if v != 0 {
                cols.push(cmap[k]);
                vals.push(v);
            }
        }
        store.replace(r, cols, vals);
        done.push(r);
    }
}

/// Phase 2: reduce the retired rows to reduced row echelon form with
/// leftmost column scanning. Returns `(pivot_cols, rref_row_ids)` with
/// `rref_row_ids[j]` the row holding pivot `pivot_cols[j]`.
fn phase2(
    store: &mut RowStore,
    done: &[usize],
    n: usize,
    p: u64,
    mont: &Mont,
    params: &Params,
) -> Result<(Vec<u32>, Vec<usize>), NullspaceError> {
    let mut col_rows: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &r in done {
        for &c in &store.cols[r] {
            col_rows[c as usize].push(r as u32);
        }
    }
    let mut processed: Vec<bool> = vec![false; store.cols.len()];
    let mut pivot_cols: Vec<u32> = Vec::new();
    let mut rref_rows: Vec<usize> = Vec::new();
    let mut stamp: Vec<u32> = vec![0; store.cols.len()];
    let mut generation: u32 = 0;

    for col in 0..n as u32 {
        // Candidates: unprocessed retired rows holding `col`.
        let holders = std::mem::take(&mut col_rows[col as usize]);
        let mut pivot: Option<usize> = None;
        let mut best = usize::MAX;
        generation += 1;
        let mut fresh: Vec<u32> = Vec::new();
        for &t32 in &holders {
            let t = t32 as usize;
            if stamp[t] == generation {
                continue;
            }
            stamp[t] = generation;
            if store.contains(t, col).is_none() {
                continue; // stale adjacency
            }
            fresh.push(t32);
            if !processed[t] && store.nnz(t) < best {
                best = store.nnz(t);
                pivot = Some(t);
            }
        }
        let Some(r) = pivot else {
            // Free column. (Processed rows may still hold it — those
            // entries are exactly the basis data extracted later.)
            continue;
        };
        // Normalize the pivot row so its `col` coordinate is 1.
        let pk = store.contains(r, col).expect("checked above");
        let inv_mont = mont.to_mont(invmod(store.vals[r][pk], p));
        let vals = &mut store.vals[r];
        for v in vals.iter_mut() {
            *v = mont.mulredc(inv_mont, *v);
        }
        debug_assert_eq!(store.vals[r][pk], 1);
        // Eliminate `col` from every other retired row holding it.
        let mut added: Vec<u32> = Vec::new();
        for &t32 in &fresh {
            let t = t32 as usize;
            if t == r {
                continue;
            }
            let Some(tk) = store.contains(t, col) else {
                continue;
            };
            let f = store.vals[t][tk];
            let f_mont = mont.to_mont(f);
            added.clear();
            let (nc, nv) = axpy_merge(
                &store.cols[t],
                &store.vals[t],
                &store.cols[r],
                &store.vals[r],
                f_mont,
                mont,
                &mut added,
            );
            for &ac in &added {
                col_rows[ac as usize].push(t32);
            }
            store.replace(t, nc, nv);
            if store.live_entries > params.max_entries {
                return Err(NullspaceError::FillBudget {
                    live_entries: store.live_entries,
                    max_entries: params.max_entries,
                    pivots_done: pivot_cols.len(),
                    phase: 2,
                });
            }
        }
        processed[r] = true;
        pivot_cols.push(col);
        rref_rows.push(r);
    }
    Ok((pivot_cols, rref_rows))
}

/// Read the canonical basis off the RREF rows.
fn extract_basis(
    store: &RowStore,
    pivot_cols: &[u32],
    rref_rows: &[usize],
    n: usize,
    p: u64,
) -> Vec<SparseVec> {
    let mut is_pivot = vec![false; n];
    for &c in pivot_cols {
        is_pivot[c as usize] = true;
    }
    // Free-column -> dense slot map, ascending.
    let free_cols: Vec<u32> = (0..n as u32).filter(|&c| !is_pivot[c as usize]).collect();
    let mut slot = vec![u32::MAX; n];
    for (k, &f) in free_cols.iter().enumerate() {
        slot[f as usize] = k as u32;
    }
    let mut entries: Vec<Vec<(u32, u64)>> = vec![Vec::new(); free_cols.len()];
    for (j, &r) in rref_rows.iter().enumerate() {
        let pc = pivot_cols[j];
        for (k, &c) in store.cols[r].iter().enumerate() {
            if c == pc {
                continue; // the unit pivot coordinate itself
            }
            debug_assert!(!is_pivot[c as usize], "RREF rows touch only free columns");
            let s = slot[c as usize] as usize;
            entries[s].push((pc, p - store.vals[r][k]));
        }
    }
    free_cols
        .iter()
        .enumerate()
        .map(|(k, &f)| {
            let mut ent = std::mem::take(&mut entries[k]);
            ent.push((f, 1));
            ent.sort_unstable_by_key(|&(c, _)| c);
            let (idx, val): (Vec<u32>, Vec<u64>) = ent.into_iter().unzip();
            SparseVec { idx, val }
        })
        .collect()
}
