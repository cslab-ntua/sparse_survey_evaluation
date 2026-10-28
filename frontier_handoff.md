# Handoff: frontier reordering work (vectorized COO SpMM)

## Project context
- Repo: Z:\SpMM\sparse_survey_evaluation (Windows checkout; NO local compiler). The user builds and runs
  everything on the server epyc7 (AMD EPYC 7402, Zen2, AVX2, 24 cores = 8 CCX x 3 cores, 32 KiB L1d,
  512 KiB L2 per core, 16 MiB L3 per CCX (a core can only use its own CCX's L3), threads 0-23 on NUMA node 0).
- The vectorized COO SpMM kernels have been refactored into ONE source, `kernel_coo.cpp`, plus headers:
  - bases: `coo_base_*.h` (row_split / atomic / PBV / fixedk = `coo_base_row_reg_fixedk.h`, `coo_base_xrow.h`)
  - reorderings: `coo_reorder_*.h`
  Variants are chosen with -D flags; one .exe per variant; the OLD exe names are kept.
- Your memory directory already has `coo-refactor-conventions.md` and `coo-refactor-roadmap.md`. Read them.
  Do NOT edit `coo-refactor-roadmap.md` (another chat owns it). Keep your own memory file `frontier-roadmap.md`.

## Read first
1. `refactoring_report.md` (repo root): structure, reordering interface, all options/targets, fixes, findings.
2. `coo_reorder_frontier.h`: the ported frontier reordering.
3. `coo_reorder_segment_common.h`: shared machinery for reorderings that split rows into runs
   (segment_apply_order, seg_nnz_perm, env helpers).
4. `kernel_coo.cpp`: constructor steps, reorder hooks, format names, row-snapped partition.
5. `coo_base_row_reg_fixedk.h` and `coo_base_xrow.h`: the base kernels frontier runs on.
6. `old_traversal/`: the original (pre-refactor) frontier code, for reference.

## Conventions (from the user, must follow)
- Discuss the design BEFORE big code changes; the user approves step by step.
- One path at a time: the user loses track when many experiments are open.
- Keep the original structure: `struct COOArrays`, global `thread_*` arrays, `compute_coo_*` / `subkernel_*`.
- Keep `#ifdef` variants, one exe per variant, existing exe names. Replaced files go to `old_traversal/` (git mv).
- Other kernels (non-COO) stay unchanged.
- Code is not compiled locally: give the user exact `make` + run commands; they paste back the results.
- Writing to Z: from tools can be flaky: write to the scratchpad and copy if needed (see conventions memory).
- Keep `refactoring_report.md` updated after changes (coordinate: another chat may also edit it).

## File ownership while two chats run in parallel
- This chat edits: `coo_reorder_frontier.h` (or new frontier headers), frontier Makefile targets,
  its own diag scripts (e.g. `diag_frontier*.sh`), its own memory file.
- Avoid editing without asking the user first: `coo_base_row_reg_fixedk.h`, `kernel_coo.cpp`, the shared
  Makefile sections, `coo_reorder_segment_common.h`, `refactoring_report.md`. The other chat is changing the
  fixedk base (outer slicing / packing experiments).

## Key concepts
- Run = consecutive nnz of one row inside a thread segment.
  - No reordering (or RCM): run = row (row-snapped partition, row-sorted data) -> fixedk STORE mode.
  - Traversal reorderings (z-order, Hilbert, cross, frontier, ring-buffer) split rows into several runs
    -> fixedk ACCUMULATE mode (first run of a row stores, later runs add).
- fixedk pays a cost per run (call + y load/store per k-slice), so short runs hurt it.
- fixedk now slices k by default into S=128 columns (CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE, 0 = old per-k kernels)
  and uses precomputed runs (RUN_PTR default; RUN_SCAN = old scan).
  Older frontier numbers were measured WITHOUT k-slicing: re-measure before comparing.
- VEC_LEN comes from lib/vectorization (AVX2 = 8 floats).
- CFLAGS has `-falign-loops=64` for all kernels (an alignment difference earlier caused a fake ~10% gap).

## Frontier status
- Ported; order hash identical to the old code (COO_PRINT_ORDER_HASH=1, diag_order.sh).
- The remaining gap vs the old build was loop alignment (fixed with -falign-loops=64) -> counts as verified.
- Overnight findings: traversal reorderings cost fixedk ~26-41% on DLMC (they split rows into very short
  runs); on graphs the gain over plain fixedk is small. Check exact frontier numbers in refactoring_report.md
  and overnight_20261014_2306/ (results.csv, plots).
- RCM (square matrices only; DLMC is 2048x512 so no RCM there) is the strongest reordering on graphs
  (com-Amazon ~1.5x fixedk). RCM's default mode leaves y in RCM order after the first call, so the
  after-timing accuracy check is meaningless for RCM (use CHECK_ACCURACY_AFTER_TIMING=0).

## Tools for verification and measurement
- `check_coo.sh`: correctness + timings (REPEAT env, best of N; DLMC_SAMPLES env).
- `diag_order.sh` + COO_PRINT_ORDER_HASH=1: proves an order is identical to a reference.
- COO_PRINT_RUNS=1: prints average run length.
- CACHE_POLLUTE=1 (also CACHE_POLLUTE_MB, CACHE_POLLUTE_MODE=write|read|x_warm): polluted-cache timings.
  Report warm AND polluted; they can rank variants differently.
- CHECK_ACCURACY_AFTER_TIMING=1: re-check after the timed calls.
- Inputs: DLMC list /various/itasou/dlmc/transformer_matrices_small_onlyencoder_onlymagnitude.txt
  (e.g. 0.5 and 0.9 conv1 samples), graphs in ./matrices/ (pubmed, com-Amazon, ogbl-collab, roadNet-CA, ...).
- Use k = 256 / 512 / 1024 (and a non-power-of-two such as 1000: power-of-two k causes cache-set aliasing).
- DLMC single runs vary +-30%: always use medians/best of several runs.

## Commit state
- The user should commit the current work before this chat starts, so both chats begin from a known state.

## My frontier ideas (fill in)
- ...
The SpMM traversal algorithm
Data set:
State:
Rows in cache
Columns in cache
Type 2 nodes in frontier (2 hits if visited)
Type 1 nodes in frontier (1 hit if visited)
Type 0 nodes in frontier (0 hits if visited)
Visited nodes
Unvisited stars
Parameters:
Frontier size
Number of Type 1 visits before exploitation to exploration switch
Initialization step:
Find and visit TBD0 stars 
Add to the frontier the visited stars’ neighbors
If frontier overflow, trim frontier with a TBD1 criterion
Update state
Repeat:
/* Exploitation phase */
if (there is a Type 2 node in frontier) visit a Type 2 node
else if (there is a Type 1 node in frontier) visit the Type 1 node with TBD2 criterion
else visit Type 0 node from frontier with TBD2 criterion
Add to the frontier the neighbors of the visited node
if (frontier overflow) trim frontier with a TBD3 criterion
Update state

if TBD4 criterion enter exploration phase
/* Exploration phase */
Add to the frontier TBD5 unvisited stars
Update state

If TBD6 criterion enter leftover phase
/* leftover phase */
visit remaining nodes and exit

Until all nodes have been visited


TBD0
Description: How many stars to visit at first
Initial take: so many that their neighbors fill in the frontier
TBD1
Description: How to trim the initial frontier from the stars’ neighbors
Initial take: A non-issue with the initial take of TBD0
TBD2
Description: Which of the Type 1 nodes to visit
Initial take: The one that creates the highest number of Type 2 nodes
TBD3
Description: Which nodes to kick out from the frontier if full
Initial take: Type 0 -> Type 1 -> from Type 1 the ones that create the least Type 2 -> the ones with the oldest rows/columns … (not very sure about this)
TBD4
Description: When to stop visiting (exploiting) and start expanding (exploring)
Initial take: After k (parameter) repetitions of type 1 visits, k = 1 or k = 10 or k = 100
TBD5
Description: How many unvisited stars to insert in the frontier in the exploration phase
Initial take: I do not know 😀 try just 1 or 10 or 100
TBD6
Description: when to stop and just visit the remaining nodes (in order to make sure that everything is visited
Initial take: I do not know 😀 maybe this is not necessary, it is just a safety phase to make sure that we did not forget any node (especially those that are not connected)
stars=strongly connected nodes that do not share rows or columns

