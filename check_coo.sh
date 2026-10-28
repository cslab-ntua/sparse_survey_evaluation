#!/bin/bash
export LD_LIBRARY_PATH=/various/dgal/gcc/gcc-12.2.0/gcc_bin/lib64/:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=24                       # same as run.sh
export GOMP_CPU_AFFINITY="0-23"
export OMP_WAIT_POLICY='active'
export OMP_DYNAMIC='false'
export SYSTEM='AMD-EPYC-24'
export GPU_KERNEL=0
export PREFETCH_DIST=4                          # same as run.sh (the old z_order file crashes without it)
export SEED_ROW=50                              # cross / frontier / ring-buffer parameters, same as run.sh
export CROSS_DISTANCE_LIMIT=100                 # (the old files crash without them)
export CACHE_CAPACITY=256
export FRONTIER_SIZE=10000
export ALPHA=1.0
export BETA=1.0
# Re-check the result after the 128 timed calls too (catches missing zero-init of y between calls).
# Not meaningful for spmm_coo_vector_rcm.exe / fixedk_rcm.exe: after the first call they leave y in RCM order.
export CHECK_ACCURACY_AFTER_TIMING=0

# Graphs (DATASET=GRAPH), same path as path_graph in run.sh.
# wiki-Talk is stored as "general" (asymmetric), so it exercises the A + A^T path of generate_rcm.
path_graph=./matrices
graph_matrices=(cora.mtx pubmed.mtx com-Amazon.mtx)
# graph_matrices=(cora.mtx pubmed.mtx com-Amazon.mtx roadNet-CA.mtx wiki-Talk.mtx)

# DLMC (DATASET=DLMC), same path and list file as in run.sh. DLMC_SAMPLES matrices are taken evenly spaced from the list
# (DLMC_SAMPLES=0 skips DLMC). DLMC matrices are rectangular, so RCM variants can't run on them.
path_dlmc=/various/itasou/dlmc
dlmc_list="$path_dlmc/transformer_matrices_small_onlyencoder_onlymagnitude.txt"
DLMC_SAMPLES=${DLMC_SAMPLES:-6}

ks=(256 512 1024)

# Every variant is run REPEAT times and the best GFLOPS is reported (single runs on the small DLMC matrices vary by +-30%).
REPEAT=${REPEAT:-3}

# Hash of the final non-zero order (kernel_coo.cpp and the old reference files): same hash = same order.
export COO_PRINT_ORDER_HASH=1

# Ring-buffer check: old reference vs new (row_split base) for the 3 variants, then the fixedk versions, each next to no
# reordering. The ringbuffer / all_seed variants sort the frontier every step (slow like frontier), roulette is fast.
variants_1=(
  frontier_ringbuffer_old frontier_ringbuffer
  frontier_ringbuffer_all_seed_old frontier_ringbuffer_all_seed
  frontier_ringbuffer_roulette_old frontier_ringbuffer_roulette
  xrow_row_split
)
variants_2=(
  fixedk_frontier_ringbuffer fixedk_frontier_ringbuffer_all_seed fixedk_frontier_ringbuffer_roulette row_reg_fixedk
)

# Stop early if an executable is missing (e.g. the *_old references, which are not in EXE and not rebuilt by "make").
missing=0
for v in "${variants_1[@]}" "${variants_2[@]}"; do
  if [ ! -x spmm_coo_vector_$v.exe ]; then
    echo "missing spmm_coo_vector_$v.exe (build it with: make spmm_coo_vector_$v.exe)"
    missing=1
  fi
done
[ $missing -eq 1 ] && exit 1

# One line per variant: name, format, best GFLOPS of REPEAT runs, order hash ("-" if not printed, "VARIES" if it differed
# between runs), max relative diff after the timed calls (last run), "FAIL" if the first check failed in any run.
run_variant() {   # run_variant <variant> <matrix file> <k>
  for r in $(seq $REPEAT); do ./spmm_coo_vector_$1.exe $2 $3; done | awk -v name="$1" -v rep=$REPEAT '
    /Test failed/        { fail = 1 }
    /ORDER_HASH/         { if (hash != "" && hash != $2) varies = 1; hash = $2 }
    /format:/            { match($0, /format: [^,]*/); fmt = substr($0, RSTART + 8, RLENGTH - 8)
                           match($0, /gflops: [0-9.]*/); g = substr($0, RSTART + 8, RLENGTH - 8) + 0
                           if (g > gf) gf = g }
    /Accuracy after/     { after = $NF }
    END                  { printf "  %-36s %-52s %8.2f GFLOPS (best of %d)   hash: %-16s   after: %s%s\n", name, fmt, gf, rep,
                                  (varies ? "VARIES" : (hash == "" ? "-" : hash)), after, (fail ? "   FAIL" : "") }'
}

run_matrix() {    # run_matrix <matrix file>
  for k in "${ks[@]}"; do
    echo "=== $(basename $1)  k=$k  ($DATASET)"
    for v in "${variants_1[@]}"; do run_variant $v $1 $k; done
    echo "  ---"
    for v in "${variants_2[@]}"; do run_variant $v $1 $k; done
  done
}

export DATASET='GRAPH'
for m in "${graph_matrices[@]}"; do
  run_matrix ${path_graph}/$m
done

if [ "$DLMC_SAMPLES" -gt 0 ]; then
  if [ ! -f "$dlmc_list" ]; then
    echo "DLMC list $dlmc_list not found, skipping DLMC"
  else
    export DATASET='DLMC'
    total=$(grep -c . "$dlmc_list")
    step=$(( total / DLMC_SAMPLES ))
    [ "$step" -lt 1 ] && step=1
    mapfile -t dlmc_matrices < <(grep . "$dlmc_list" | awk -v step=$step -v n=$DLMC_SAMPLES '(NR - 1) % step == 0 && c < n { print; c++ }')
    for m in "${dlmc_matrices[@]}"; do
      run_matrix ${path_dlmc}/$m
    done
  fi
fi
