#!/bin/bash
# Sweep of PREFETCH_DIST (read at run time by the CUSTOM_COO_VEC_PREFETCH executables, no rebuild needed).
# Prints one table per matrix and k: GFLOPS of every variant for every distance, plus the variant without prefetch.

export LD_LIBRARY_PATH=/various/dgal/gcc/gcc-12.2.0/gcc_bin/lib64/:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=24                       # same as run.sh
export GOMP_CPU_AFFINITY="0-23"
export OMP_WAIT_POLICY='active'
export OMP_DYNAMIC='false'
export SYSTEM='AMD-EPYC-24'
export GPU_KERNEL=0
export DATASET='GRAPH'
path=./matrices

# The matrices where prefetching had an effect (cora is too small / noisy, roadNet-CA showed no effect at all).
matrices=(pubmed.mtx com-Amazon.mtx wiki-Talk.mtx)
ks=(64 100)
distances=(1 2 4 8 16 32 64)

# prefetch variant : the same variant without prefetch (the reference column)
variants=(
  xrow_row_split_prefetch:xrow_row_split
  xrow_row_split_prefetch_row:xrow_row_split
  z_order_prefetch:z_order
  z_order_prefetch_row:z_order
  fixedk_prefetch:row_reg_fixedk
  fixedk_prefetch_row:row_reg_fixedk
  fixedk_z_order_prefetch_row:fixedk_z_order
)

gflops() {   # gflops <exe name> <matrix> <k>
  ./spmm_coo_vector_$1.exe ${path}/$2 $3 | grep -o 'gflops: [0-9.]*' | awk '{print $2}'
}

for m in "${matrices[@]}"; do
  for k in "${ks[@]}"; do
    echo "=== $m  k=$k"
    printf "  %-30s %8s" "variant" "no-pref"
    for d in "${distances[@]}"; do printf " %7s" "d=$d"; done
    echo
    for vr in "${variants[@]}"; do
      v=${vr%%:*}
      ref=${vr##*:}
      printf "  %-30s %8s" "$v" "$(gflops $ref $m $k)"
      for d in "${distances[@]}"; do
        printf " %7s" "$(PREFETCH_DIST=$d gflops $v $m $k)"
      done
      echo
    done
  done
done
