# Refactoring report: vectorized COO SpMM kernels

Period: 2026-10-05 – 2026-10-14. Platform for all measurements: AMD EPYC (24 threads, `OMP_NUM_THREADS=24`,
`GOMP_CPU_AFFINITY=0-23`), gcc 12.2, `-O3 -march=native -mavx2`, single precision.

## 1. Goal

Before the refactoring, every COO SpMM variant was a separate `.cpp` file. Each one contained its own copy of the basic
vectorized COO kernel (from `kernel_coo_vec.cpp`), its own partitioner, SDDMM and statistics code, and did its reordering
inside its constructor. A better base kernel (`kernel_coo_vec_row_reg_fixedk.cpp`) existed, but the reorderings could
not use it without copying it into every file again.

The refactoring separates the three independent parts of a variant:

- the **base kernel** (how the non-zeros of a thread are computed),
- the **reordering** (in which order the non-zeros are processed),
- the **options** (prefetching, zeroing, ...),

so that any reordering runs on any base kernel, chosen at compile time with `-D` flags, while keeping the structure of
the original code (`struct COOArrays : Matrix_Format`, global `thread_*` arrays, `compute_coo_*` / `subkernel_*`
functions, `#ifdef` variant selection, one executable per variant, existing executable names).

## 2. Code structure

| File | Content |
|---|---|
| `kernel_coo.cpp` | The single source of all vectorized COO SpMM variants: `COOArrays`, constructor pipeline, variant selection, `spmm()` / `sddmm()`, `csr_to_format()`, statistics |
| `coo_base_xrow.h` | Per-non-zero base kernels (from `kernel_coo_vec.cpp`): row_split, atomic, perfect nnz balance |
| `coo_base_row_reg_fixedk.h` | Register-blocked fixed-k base kernel (from `kernel_coo_vec_row_reg_fixedk.cpp`), STORE / ACCUMULATE modes |
| `coo_reorder_none.h` | No reordering; documents the reordering interface |
| `coo_reorder_rcm.h` | Reverse Cuthill-McKee (group A) |
| `coo_reorder_segment_common.h` | Common part of all group B reorderings (applying a per-segment order, SDDMM output permutation, env parameter helpers) |
| `coo_reorder_z_order.h` | Z-order (Morton) inside each thread segment |
| `coo_reorder_hilbert.h` | Hilbert curve order inside each thread segment |
| `coo_reorder_cross.h` | Greedy row / column "cross" walk |
| `coo_reorder_frontier.h` | Greedy scored-frontier walk |
| `coo_reorder_ringbuffer.h` | Frontier walk with a ring-buffer cache model, 3 variants (plain, all_seed, roulette) |

### Constructor pipeline (`COOArrays::COOArrays`)

1. Build the COO arrays from the CSR input (+ colind0 modifier).
2. `reorder_relabel()` – group A reorderings (RCM): renumber rows / columns, re-sort row-major.
3. Partition: nnz-balanced thread ranges, snapped to row boundaries for the row_split and fixedk bases.
4. Record each thread's row range (`thread_row_s` / `thread_row_e`) **before** any reordering of the segments.
5. `reorder_segments()` – group B reorderings: reorder the non-zeros inside each thread segment.
6. fixedk STORE mode only: check that every row is a single run, abort otherwise.
7. `base_preprocess()` – fixedk ACCUMULATE mode: first-run flags and list of empty rows.
8. Optional diagnostic: `COO_PRINT_ORDER_HASH=1` prints a hash of the final non-zero order.

`spmm()` calls `reorder_spmm_begin()` / `reorder_spmm_end()` around the kernel (RCM uses them to permute x / y),
`sddmm()` calls `reorder_sddmm_begin()` / `reorder_sddmm_end()` (group B reorderings scatter the output back to the
original non-zero order).

### Reordering interface

Every `coo_reorder_*.h` defines:

| Name | Meaning |
|---|---|
| `COO_REORDER_KEEPS_ROWS` | 1 if every row stays one contiguous run in its segment (none, RCM), 0 otherwise (all group B) |
| `COO_REORDER_NAME` | Suffix of the format name |
| `reorder_relabel(coo, csr_ia, csr_ja)` | Group A relabeling |
| `reorder_segments(coo)` | Group B per-segment reordering |
| `reorder_spmm_begin/end`, `reorder_sddmm_begin/end` | Dense operand / output handling |
| `reorder_free(coo)` | Cleanup |

## 3. Base kernels

### row_split (`CUSTOM_COO_VEC_XROW_ROW_SPLIT`) – the original kernel

From `kernel_coo_vec.cpp`. Every non-zero does `y[row] += val * x[col]` directly on y (load x, load y, FMA, store y).
Thread ranges are nnz-balanced and snapped to row boundaries, so every row is owned by one thread and no atomics are
needed. Each thread zeroes its row range before computing. Works with any non-zero order.

### atomic (`CUSTOM_COO_VEC_XROW_ATOMIC`)

Same per-non-zero kernel, plain nnz-balanced ranges (no row snapping), atomic adds to y.

### perfect nnz balance, PBV (`CUSTOM_COO_VEC_XROW_PERFECT_NNZ_BALANCE`)

Plain nnz-balanced ranges; the row shared with the next thread is accumulated in a private buffer and added to y
serially at the end.

### fixedk (`CUSTOM_COO_VEC_ROW_REG_FIXEDK`)

From `kernel_coo_vec_row_reg_fixedk.cpp`. The thread's non-zeros are split into **runs** (maximal stretches of
consecutive non-zeros of the same row). Each run is accumulated in a compile-time-sized array of vector registers
(templates for k = 16, 32, 64, 128, 256, 512) and written to y once. Other k (e.g. 100, 1024) use the generic path,
which accumulates in a stack buffer `local_buf[k]`. With AVX2 (8-wide `ymm`, 16 registers, as used by the build) a whole
row fits in registers only up to k = 128; at k = 256 / 512 the accumulators spill to the stack.

Two write modes, selected automatically from `COO_REORDER_KEEPS_ROWS`:

| Mode | Used for | Behaviour |
|---|---|---|
| **STORE** | none, RCM (row-sorted orders) | `y = acc` per run (= per row); only the empty rows are zeroed (gaps between consecutive runs), so every row of y is written exactly once |
| **ACCUMULATE** | z-order, Hilbert, cross, frontier, ring-buffers | the first run of a row stores (`y = acc`), later runs add (`y += acc`); only the empty rows are zeroed. First-run flags and the empty-row list are computed once in the constructor |

`CUSTOM_COO_VEC_ROW_REG_FIXEDK_ZERO_ALL` restores the previous zeroing (whole row range zeroed, ACCUMULATE adds every
run). `CUSTOM_COO_VEC_ROW_REG_FIXEDK_ACCUMULATE` forces ACCUMULATE mode on row-sorted input.

**k-slicing (default since 2026-10-27).** Every run is processed in slices of S = 128 columns (16 accumulators), plus
the remaining whole vectors and a scalar tail, so one kernel serves every k. This replaces the per-k templates and the
generic `local_buf` path, which spilled at k = 256 / 512 and kept everything in memory at k = 1024. Build with
`-D'CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE=0'` (target `spmm_coo_vector_row_reg_fixedk_slice0.exe`) for the previous
per-k kernels. Other sizes: `..._slice<S>.exe`.

## 4. Reorderings

| Group | Macro | Description | Parameters (env, default = run.sh value) |
|---|---|---|---|
| – | (none) | CSR order | – |
| A | `COO_REORDER_RCM` | Reverse Cuthill-McKee relabeling of rows and columns (square matrices only, A + Aᵀ for asymmetric ones). By default x / y are permuted only on the first `spmm()` call (the checked warmup); the timed calls run as if the application kept x / y in RCM order. `CUSTOM_COO_VEC_RCM_PERMUTE_IN_TIMING` permutes on every call (old behaviour) | – |
| B | `COO_REORDER_Z_ORDER` | Sort each segment by the Morton code of (row, col) | – |
| B | `COO_REORDER_HILBERT` | Sort each segment by the Hilbert-curve distance of (row, col) | – |
| B | `COO_REORDER_CROSS` | Greedy walk through the row / column "cross" of the last non-zero, simulated cache of recently used rows / columns | `SEED_ROW` (50), `CROSS_DISTANCE_LIMIT` (100), `CACHE_CAPACITY` (256) |
| B | `COO_REORDER_FRONTIER` | Greedy walk over a frontier scored by `ALPHA·recency + BETA·coverage`, sorted every step | `SEED_ROW`, `FRONTIER_SIZE` (10000), `CACHE_CAPACITY`, `ALPHA` (1.0), `BETA` (1.0) |
| B | `COO_REORDER_RINGBUFFER` | Frontier walk with a ring-buffer cache model, sorted frontier, seed-row fallback | as frontier |
| B | `COO_REORDER_RINGBUFFER_ALL_SEED` | Same, frontier pre-filled from the seed rows **and columns** | as frontier |
| B | `COO_REORDER_RINGBUFFER_ROULETTE` | Stochastic choice (70 % best tier, 20 % seed jump, 10 % random leap), random frontier eviction, fixed per-thread random seed | `SEED_ROW`, `FRONTIER_SIZE`, `CACHE_CAPACITY` |

Group B reorderings only reorder non-zeros inside each thread's segment, so every row stays owned by one thread.
All heuristic reorderings were ported unchanged (same algorithm, same tie-breaking), verified with the order hash
(section 9).

## 5. Options

| Macro | Bases | Effect |
|---|---|---|
| `CUSTOM_COO_VEC_XROW_COLIND0` | all | every column index = column 0 (upper bound of x reuse) |
| `CUSTOM_COO_VEC_PREFETCH` | row_split, atomic, fixedk | prefetch the rows of the non-zero `PREFETCH_DIST` (env, default 4) positions ahead: x and y for row_split / atomic, x for fixedk. Bounds-checked |
| `CUSTOM_COO_VEC_PREFETCH_FULL_ROW` | with `_PREFETCH` | prefetch every cache line of those rows instead of only the first |
| `CUSTOM_COO_VEC_ZERO_STATIC` | all | zero all of y with one static (row-balanced) loop before the kernel |
| `CUSTOM_COO_VEC_ROW_REG_FIXEDK_ZERO_ALL` | fixedk | previous zeroing (see section 3) |
| `CUSTOM_COO_VEC_ROW_REG_FIXEDK_ACCUMULATE` | fixedk | force ACCUMULATE mode |
| `CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE=<S>` | fixedk | k-slicing slice size. **Default S = 128** (2026-10-27); `S = 0` = no slicing (per-k templates + generic path). Every row processed in slices of S columns with S/8 accumulators, one kernel for any k (remainder chunks + scalar tail); S = 120 = largest slice without spills on AVX2. Targets `spmm_coo_vector_row_reg_fixedk_slice<S>.exe`, `spmm_coo_vector_fixedk_z_order_slice<S>.exe` |
| `CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE_OUTER=<W>` | fixedk | outer slicing (2026-10-27, under test): the strip loop is outside the run loop, every thread processes all its runs for columns [0, W), then [W, 2W), ..., so one pass reads only n × W × 4 bytes of x (cache blocking of x in k). W a multiple of 8; needs slicing on and the precomputed runs (not `RUN_SCAN`). Targets `spmm_coo_vector_row_reg_fixedk_outer<W>.exe`, `..._fixedk_rcm_outer<W>.exe`, `..._fixedk_z_order_outer<W>.exe` |
| `CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE_OUTER_PACK` | fixedk + outer slicing | packing (2026-10-28, under test): before every strip the threads copy x[:, strip] into a contiguous n × W buffer (two buffers used alternately, one barrier per strip). Without it the strip pieces are k × 4 bytes apart and, for power-of-two k, compete for a few cache sets. Targets `spmm_coo_vector_row_reg_fixedk_pack<W>.exe`, `..._fixedk_rcm_pack<W>.exe`, `..._fixedk_z_order_pack<W>.exe` |
| `CUSTOM_COO_VEC_ROW_REG_FIXEDK_RUN_SCAN` | fixedk | previous run detection: scan `row_ind` in the timed loop. Default since 2026-10-27: run boundaries precomputed in the constructor (like CSR row pointers), +6–13 % on DLMC at k = 64 / 128 |
| `CUSTOM_COO_VEC_RCM_PERMUTE_IN_TIMING` | with RCM | permute x / y on every call |

Invalid combinations are compile errors (e.g. more than one base, prefetch with PBV, ZERO_ALL with ZERO_STATIC).

Run-time diagnostics (env variables, no rebuild needed): `COO_PRINT_ORDER_HASH=1` prints a hash of the final non-zero
order; `COO_PRINT_RUNS=1` prints `RUNS nnz=.. runs=.. avg_run_length=.. nonempty_rows=.. avg_row_length=.. runs_per_row=..`
(runs = maximal stretches of consecutive non-zeros of the same row inside a thread segment).

## 6. Executables

All built from `kernel_coo.cpp` unless noted. Format names are `<base>[_Prefetch|_PrefetchRow][_ZeroStatic|_ZeroAll]<reordering>`,
e.g. `COO_RowSplit_Vec_RowReg_FixedK_ZOrder`.

### In `EXE` (built by `make`)

| Executable | Base | Reordering / options |
|---|---|---|
| `spmm_coo_vector_xrow_row_split.exe` | row_split | – |
| `spmm_coo_vector_xrow_atomic.exe` | atomic | – |
| `spmm_coo_vector_xrow_perfect_nnz_balance.exe` | PBV | – |
| `spmm_coo_vector_xrow_colind0.exe` | row_split | colind0 |
| `spmm_coo_vector_row_reg_fixedk.exe` | fixedk (STORE) | – |
| `spmm_coo_vector_rcm.exe` / `spmm_coo_vector_fixedk_rcm.exe` | row_split / fixedk (STORE) | RCM |
| `spmm_coo_vector_z_order.exe` / `spmm_coo_vector_fixedk_z_order.exe` | row_split / fixedk (ACC) | z-order |
| `spmm_coo_vector_z_order_prefetch.exe` | row_split | z-order + prefetch |
| `spmm_coo_vector_hilbert.exe` / `spmm_coo_vector_fixedk_hilbert.exe` | row_split / fixedk (ACC) | Hilbert |
| `spmm_coo_vector_cross.exe` / `spmm_coo_vector_fixedk_cross.exe` | row_split / fixedk (ACC) | cross |
| `spmm_coo_vector_frontier.exe` / `spmm_coo_vector_fixedk_frontier.exe` | row_split / fixedk (ACC) | frontier |
| `spmm_coo_vector_frontier_ringbuffer[_all_seed\|_roulette].exe` | row_split | ring-buffer variants |
| `spmm_coo_vector_fixedk_frontier_ringbuffer[_all_seed\|_roulette].exe` | fixedk (ACC) | ring-buffer variants |
| `spmm_coo_vector_row_reg.exe` | – | **unchanged** `kernel_coo_vec_row_reg.cpp` (local-buffer kernel) |
| `spmm_coo_vector_fixedk_block.exe` | – | **unchanged** `kernel_coo_vec_fixedk_block.cpp` |

### Experiment targets (not in `EXE`)

| Executable | Purpose |
|---|---|
| `spmm_coo_vector_fixedk_accum.exe` | fixedk forced to ACCUMULATE on row-sorted input |
| `spmm_coo_vector_rcm_permxy.exe`, `spmm_coo_vector_fixedk_rcm_permxy.exe` | RCM with x / y permutation in the timed region |
| `..._xrow_row_split_prefetch[_row].exe`, `..._z_order_prefetch_row.exe`, `..._fixedk_prefetch[_row].exe`, `..._fixedk_z_order_prefetch_row.exe` | prefetch experiments |
| `..._xrow_row_split_zero_static.exe`, `..._row_reg_fixedk_zero_static.exe` | balanced zeroing (kept for the later partition work) |
| `..._row_reg_fixedk_zero_all.exe`, `..._fixedk_z_order_zero_all.exe`, `..._fixedk_accum_zero_all.exe` | previous fixedk zeroing |

### Reference builds of the old files (not in `EXE`)

`spmm_coo_vector_xrow_row_split_old.exe`, `..._hilbert_old.exe`, `..._cross_old.exe`, `..._frontier_old.exe`,
`..._frontier_ringbuffer[_all_seed|_roulette]_old.exe`, built from `old_traversal/`. They are only for comparisons and
can be removed once no longer needed. After `make clean` they have to be built explicitly.

### Format name changes

| Old format name | New format name |
|---|---|
| `COO_RCM_Vec` | `COO_RowSplit_Vec_RCM` (`..._RCM_PermXY` with permutation in timing) |
| `COO_Zorder` / `COO_Zorder_Prefetch` | `COO_RowSplit_Vec_ZOrder` / `COO_RowSplit_Vec_Prefetch_ZOrder` |
| `COO_Hilbert` | `COO_RowSplit_Vec_Hilbert` |
| `COO_Cross` | `COO_RowSplit_Vec_Cross` |
| `COO_Frontier` | `COO_RowSplit_Vec_Frontier` |
| `COO_Frontier_Ringbuffer[_all_seed\|_Roulette]` | `COO_RowSplit_Vec_RingBuffer[AllSeed\|Roulette]` |

The base kernels keep their names (`COO_RowSplit_Vec`, `COO_Atomic_Vec`, `COO_Vec_PBV`, `COO_ColInd0_Vec`,
`COO_RowSplit_Vec_RowReg_FixedK`). Analysis scripts that filter on the old names need updating.

## 7. Moved and unchanged files

Moved with `git mv` to `old_traversal/` (history kept):

| Old file | Replaced by |
|---|---|
| `kernel_coo_vec.cpp` | `coo_base_xrow.h` (row_split, atomic, PBV, colind0) |
| `kernel_coo_vec_row_reg_fixedk.cpp` | `coo_base_row_reg_fixedk.h` |
| `kernel_coo_rcm.cpp` | `coo_reorder_rcm.h` |
| `kernel_coo_z_order.cpp` | `coo_reorder_z_order.h` (+ prefetch option) |
| `kernel_coo_z_order_new.cpp` | dropped (same order as `kernel_coo_z_order.cpp`, with a zero-init bug) |
| `kernel_coo_hilbert.cpp` | `coo_reorder_hilbert.h` |
| `kernel_coo_vec_cross.cpp` | `coo_reorder_cross.h` |
| `kernel_coo_vec_frontier.cpp` | `coo_reorder_frontier.h` |
| `kernel_coo_vec_ring_buffer[_all_seed\|_roulette].cpp` | `coo_reorder_ringbuffer.h` |

The only changes to the moved files are an opt-in order-hash print (`COO_PRINT_ORDER_HASH`) and `#include <stdint.h>`
in the cross / frontier / ring-buffer files, added for the verification. Dropped: the `rowind_limit` variant.

Unchanged: `kernel_coo_vec_row_reg.cpp`, `kernel_coo_vec_fixedk_block.cpp`, `fused_attention_coo_vector*.cpp` and all
non-COO kernels (MKL, AOCL, CSR, HC, ...), except for the shared build flag in section 8.

## 8. Changes outside the COO kernels

| File | Change |
|---|---|
| `Makefile` | COO targets rebuilt on `kernel_coo.cpp` (+ `COO_HDRS` prerequisites); new / experiment / reference targets; **`CFLAGS += -falign-loops=64` for all kernels** (see 9.5) |
| `spmm_bench.cpp` | `CheckAccuracy`: `(long) col_idx[j] * k` (int32 overflow, see 9.4); opt-in `CHECK_ACCURACY_AFTER_TIMING=1` re-checks the result after the timed calls |
| `cache_pollute.h` + `spmm_bench.cpp`, `sddmm_bench.cpp`, `attention_bench.cpp` | opt-in cache pollution before every timed iteration, outside the timing (2026-10-27): `CACHE_POLLUTE=1` (512 MB) or `CACHE_POLLUTE_MB=<n>`, `CACHE_POLLUTE_MODE=write` (default, dirty lines, pessimistic) / `read` (clean lines, optimistic) / `x_warm` (write, then the dense inputs read once). Same OpenMP threads, one value per cache line. On the EPYC 7402 the effect is complete from 256 MB on (= 2x the socket's L3); write-backs of the dirty pollution are about half of the slowdown on small inputs; x_warm ≈ write (per-CCX L3 slices) |

## 9. Bug fixes and behaviour changes

| # | Where | Problem | Fix |
|---|---|---|---|
| 1 | atomic base | Two threads share a boundary row; one could zero it after the other had started adding. **The old `spmm_coo_vector_xrow_atomic.exe` failed the accuracy check on every matrix** (relative error 0.77–0.999) | barrier after zero-init (no measurable cost) |
| 2 | PBV base | read `row_ind[thread_j_s[t+1]]` out of bounds when the next thread had no work | guarded |
| 3 | z_order_new | zero-init range taken from `row_ind[j_s]` / `row_ind[j_e-1]` after the Z-sort (not the min / max row) | row ranges recorded before reordering, for all group B |
| 4 | z-order, Hilbert, cross, frontier, ring-buffers | per-call min / max row scan inside `spmm()` (counted in the timing) | computed once in the constructor |
| 5 | z-order prefetch | prefetch read past the end of the arrays (bounds check commented out) | bounds-checked |
| 6 | Hilbert, z-order | `sddmm()` wrote the output in the reordered non-zero order / did nothing | output scattered back to the original order |
| 7 | cross, frontier, ring-buffers, z-order | `atoi(getenv(...))` crashed when an env variable was unset | defaults = run.sh values |
| 8 | cross | `#ifdef A \|\| B \|\| C` only tested `A` | explicit partition selection |
| 9 | RCM | x / y permutation counted in the timing (2–10× slower than the kernel itself) | first call only by default, macro for the old behaviour |
| 10 | RCM + colind0 | colind0 was a no-op | the single column is relabeled consistently |
| 11 | RCM | out-of-bounds on non-square matrices | clear error message (RCM needs a square matrix; all DLMC matrices are rectangular) |
| 12 | `spmm_bench.cpp` `CheckAccuracy` | `col_idx[j]*k` overflowed int32 for large n·k (**every kernel failed on wiki-Talk at k = 1024**) | `long` cast; same overflow fixed in the fixedk prefetch lines |
| 13 | build | hot loop placement (see 9.5) | `-falign-loops=64` for all kernels |

### 9.5 Loop alignment

After the refactoring, the new row_split executable was slower than the old one at k ≥ 256 although the kernel code
was identical (DLMC k = 512: 342 vs 517 GFLOPS, 10 alternating runs each). The disassembly showed the same instructions;
the only difference was the address of the 6-instruction inner loop: in the new build it straddled a 64-byte line
(`0x40fa30..0x40fa49`), in the old build it happened to fit in one. On AMD Zen a tight front-end-bound loop crossing a
fetch boundary loses throughput; the effect grows with k (more time in the inner loop) and is largest when the data is
cache-resident. The old code had no alignment flag, it was luck. With `-falign-loops=64`:

| | old | new (unaligned) | new + `-falign-loops=64` |
|---|---|---|---|
| DLMC, k = 256 | 336.0 | 303.9 | 336.7 |
| DLMC, k = 512 | 516.7 | 342.0 | 514.0 |
| DLMC, k = 1024 | 528.3 | 439.2 | 526.4 |
| com-Amazon, k = 256 | 17.48 | 16.18 | 17.52 |

The flag was added to the shared `CFLAGS` so that all kernels are built the same way. **COO results at k ≥ 256 measured
before 2026-10-13 are affected by this; other kernels may shift slightly after the rebuild.**

## 10. Verification method

1. **Old vs new executables** on the same inputs: graphs `cora, pubmed, com-Amazon, roadNet-CA, wiki-Talk` (`DATASET=GRAPH`),
   a sample of the DLMC list of `run.sh` (`DATASET=DLMC`), k = 64 / 100 at first, k = 256 / 512 / 1024 from cross onwards.
2. **Accuracy** after the warmup call (existing check) and, with `CHECK_ACCURACY_AFTER_TIMING=1`, after the 128 timed
   calls (catches missing zeroing between calls, important for ACCUMULATE mode).
3. **Order hash** (`COO_PRINT_ORDER_HASH=1`): hash of the final non-zero order, printed by the new code and by the old
   reference files. Identical hash = identical order.
4. Repeated / alternating runs (best of N, medians) where single runs were too noisy (small DLMC matrices vary by ±30 %).

| Step | Result |
|---|---|
| bases (row_split, atomic, PBV, colind0, fixedk) | same accuracy and GFLOPS within noise; atomic now correct |
| RCM | correct (incl. asymmetric wiki-Talk); permxy ≈ old RCM |
| z-order (+ prefetch) | correct, same GFLOPS; first ACCUMULATE runs correct after the timed calls |
| fixedk zeroing change (STORE + ACCUMULATE) | correct after the timed calls |
| Hilbert, cross | same GFLOPS as old |
| frontier, cross | order hash identical to old; remaining gap explained by loop alignment (9.5) |
| ring-buffer, all_seed, roulette | order hash identical to old on all matrices; same GFLOPS |

## 11. Performance findings

Numbers in GFLOPS; the k = 64 / 100 results were measured before the loop-alignment flag.

- **fixedk zeroing.** Writing every row of y once (only empty rows zeroed) instead of zero + store: roadNet-CA
  18.9 → 30.3 (+60 %), com-Amazon +21–25 %, pubmed +17–23 %, cora +5–7 %, wiki-Talk +3–10 % (k = 64 / 100). The largest
  single improvement of the refactoring; the bottleneck of roadNet-CA was writing y, not reading x.
- **ACCUMULATE mode** with "first run stores": forced on row-sorted input it is within 1–4 % of STORE (before: 10–40 %
  slower). The mode itself is now almost free; its cost comes only from the number of runs a reordering creates.
- **Zeroing imbalance.** nnz-balanced row ranges give some threads millions of empty rows to zero on skewed matrices.
  `CUSTOM_COO_VEC_ZERO_STATIC` gives +50–56 % on wiki-Talk but −20–28 % on cora (cache lines move between cores, extra
  barrier). Also present in the original row_split baseline. Planned fix: partition by non-zeros + rows (section 13).
- **RCM** (permutation outside the timing): com-Amazon row_split 17.0 → 30.1 (k = 64), fixedk 23.1 → 32.6 (old zeroing);
  worse on roadNet-CA (−5 to −11 %) and pubmed with fixedk (−23 %). With the permutation in the timing it loses on every
  matrix (2–10× slower).
- **Group B reorderings with fixedk.** With the new zeroing, z-order / Hilbert / cross no longer beat plain fixedk on the
  graphs at k = 64 (com-Amazon 28.9 vs 29.2 / 27.7). Hilbert is never better than z-order. At k = 256–1024, frontier and
  ring-buffer give +10–12 % on com-Amazon, but on DLMC every traversal reordering makes fixedk 30–50 % slower than plain
  fixedk (and slower than row_split): DLMC rows are long, the traversals split them into runs of 1–2 non-zeros, and
  fixedk pays a per-run overhead (accumulator zeroing, dispatch, y write, spilled accumulators at k ≥ 256).
- **Prefetch** (`PREFETCH_DIST` sweep 1–64): plateau for d = 1–16, worse for d ≥ 32, default 4 kept. No effect on the
  specialized fixedk paths (out-of-order execution already overlaps the x loads); +14–21 % with full-row prefetch on the
  generic fixedk path (k = 100) on com-Amazon / wiki-Talk; +10–20 % first-line prefetch for row_split / z-order at k = 64.
- **Large k.** fixedk keeps a whole row in registers only up to k = 128 (AVX2); at k = 256 / 512 it spills and at
  k = 1024 it uses the generic path, so its advantage over row_split shrinks. pubmed drops from ~250 to ~40 GFLOPS
  between k = 256 and 512 for all kernels (x no longer fits in L3).
- **k-slicing** (`diag_slice.sh`, S = 64 / 96 / 120 / 128 against unsliced). Warm: k = 128 no gain (S = 128 equals
  unsliced, S = 120 −10–15 %), k = 256 +5–14 %, k = 512 +10–18 %, k = 1024 +19–41 % (DLMC 0.5: 621 / 643 vs 462 / 457,
  1.18–1.23× FusedMM). With cache pollution the gains are smaller but never a loss. All 30 correctness checks pass;
  the S = 120 kernels have no stack traffic. S = 128 was best or within 5 % of best everywhere → made the default.
- **DRAM-bound graphs** (wiki-Talk, OVCAR-8H, REDDIT-BINARY, ogbl-collab, roadNet-CA, sx-stackoverflow, com-Youtube,
  com-Amazon): x is 170 MB–2.7 GB, rows are short (2–14 non-zeros), so nearly every non-zero reads its x row from DRAM.
  About 0.5 FLOP/byte × ~80 GB/s ≈ 40 GFLOPS ceiling, flat in k and across kernels; measured at 60–85 % of it. Only
  more locality (reordering, tiling, outer slicing) or fewer bytes (bf16, streaming stores for y) can move them.
- **Outer slicing, unpacked** (`diag_outer.sh`, W = 8 / 16 / 32 / 64): no gain at k = 256 / 1024 on any input (W = 8 / 16
  lose 10–90 %, W = 32 / 64 0.84–0.98×), also not with RCM. At k = 1000 instead of 1024 the same builds are much
  faster (DLMC 0.5 outer8 255 vs 55 GFLOPS, outer64 765 vs 499, i.e. +22 % over fixedk): with power-of-two k the strip
  pieces of the x rows are 1 / 4 KiB apart and map to a few cache sets only. On DLMC the gain is L2 blocking (a W = 64
  strip of 512 rows is 128 KiB). com-Amazon gains nothing either way (its strips are 10–85 MiB). → packing option.

## 12. Scripts

| Script | Purpose |
|---|---|
| `check_coo.sh` | Runs a list of variants on graphs + a DLMC sample (`DLMC_SAMPLES`), k = 256 / 512 / 1024, best of `REPEAT`; prints format, GFLOPS, order hash (`VARIES` if not deterministic), accuracy after timing, `FAIL`. Stops if an executable is missing |
| `sweep_prefetch.sh` | `PREFETCH_DIST` sweep of the prefetch variants |
| `diag_order.sh` | Alternating old / new runs (median / min / max) and hot-loop placement from the disassembly |
| `diag_asm.sh` | Extracts the OpenMP-outlined functions of two executables for comparison |
| `diag_slice.sh` | k-slicing: correctness (k = 100 / 256 / 1001), spill summary of the S = 120 build, timings S = 64 / 96 / 120 / 128 against unsliced and FusedMM |
| `diag_pack.sh` | outer slicing with packing: correctness (k = 100 / 256 / 1001, incl. RCM / z-order), timings pack W = 32 / 64 / 128 against unpacked outer64, fixedk, RCM and FusedMM at k = 256 / 1000 / 1024 on DLMC 0.5 / 0.9 + pubmed, com-Amazon, ogbl-collab |
| `diag_outer.sh` | outer slicing: correctness (k = 100 / 256 / 1001, incl. RCM / z-order), timings W = 8 / 16 / 32 / 64 (+ RCM) against default fixedk, S = 0 and FusedMM on DLMC 0.5 + pubmed, com-Amazon, ogbl-collab, roadNet-CA |
| `overnight.sh` | Long run (2026-10-14): tier 1 = row_split / fixedk / z-order / RCM + MKL / FusedMM on all graphs and DLMC matrices (k = 256 / 512 / 1024), tier 2 = the heuristic reorderings at k = 256; writes `overnight_<date>/results.csv` (joinable with `matrix_features/` CSVs, incl. average run length) |

## 13. Open items

1. Final comparison of all reorderings (none, RCM, z-order, Hilbert, cross, frontier, 3 ring-buffers) on both bases with
   the new defaults (alignment, zeroing), k = 256 / 512 / 1024, graphs + DLMC.
2. Order features per reordering and matrix (average run length, x / y reuse within a window, cache-model hits, column
   jumps) and the existing CSR features on the reordered matrices, in CSVs like `matrix_features/` (proposed, paused).
3. Partition balanced on non-zeros + rows (fixes the zeroing imbalance on skewed matrices).
4. Possibly: "first non-zero stores" zeroing for the row_split base; integration of `kernel_coo_vec_row_reg.cpp`, `kernel_coo_vec_fixedk_block.cpp`
   and the fused attention kernels; offline reordering files.
5. Remove the `*_old` reference targets and the diagnostic order-hash code in `old_traversal/` when no longer needed.
