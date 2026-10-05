#!/bin/bash

MODE=server
# MODE=laptop

#####################################################################################################
#### No need to edit the following paths! ###########################################################
#####################################################################################################
# export AOCL_PATH=/various/pmpakos/spmv_paper/aocl-sparse/build/release/

if [[ "$MODE" == "laptop" ]]; then
    export AOCL_ROOT=/home/pmpakos/sparse_survey_evaluation/deps/epyc5_libs/aocl-5.0
    export AOCL_PATH=${AOCL_ROOT}/aocl-sparse/build/release/
else
    export AOCL_ROOT=/various/pmpakos/epyc5_libs/aocl-5.0
    export AOCL_PATH=${AOCL_ROOT}/aocl-sparse-dev/build/release/
fi

if [[ "$MODE" == "laptop" ]]; then
    export MKL_PATH=/home/pmpakos/sparse_survey_evaluation/deps/epyc5_libs/cslab_mkl
else
    # export MKL_PATH=/various/pmpakos/intel/oneapi/mkl/2024.1/
    export MKL_PATH=/various/common_tools/intel_parallel_studio/compilers_and_libraries/linux/mkl
fi
export CUDA_PATH=/usr/local/cuda/

export LD_LIBRARY_PATH=${MKL_PATH}/lib/intel64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${AOCL_PATH}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:$LD_LIBRARY_PATH
if [[ "$MODE" == "laptop" ]]; then
    export LD_LIBRARY_PATH=/home/pmpakos/sparse_survey_evaluation/deps/epyc5_libs/openssl-1.1.1o:${AOCL_ROOT}/amd-blis/lib/LP64:${AOCL_ROOT}/amd-libflame/lib/LP64:${AOCL_ROOT}/amd-utils/lib:$LD_LIBRARY_PATH                               # FOR LAPTOP
else
    export LD_LIBRARY_PATH=/various/pmpakos/epyc5_libs/openssl-1.1.1o/:${AOCL_ROOT}/amd-blis/lib/LP64:${AOCL_ROOT}/amd-libflame/lib/LP64:${AOCL_ROOT}/amd-utils/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/various/dgal/gcc/gcc-12.2.0/gcc_bin/lib64/:$LD_LIBRARY_PATH
fi
export LD_LIBRARY_PATH=`pwd`/deps/sputnik/build/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=`pwd`/deps/libtorch/lib/:$LD_LIBRARY_PATH
# lscpu | grep -q -i amd
# if (($? == 0)); then
#     export MKL_DEBUG_CPU_TYPE=5
# fi
# export MKL_ENABLE_INSTRUCTIONS=AVX512
# export MKL_VERBOSE=1
#####################################################################################################
#####################################################################################################

# Number of cores used for OpenMP parallelization.
if [[ "$MODE" == "laptop" ]]; then
    cores='8'
    max_cores=16
else
    cores='24'
    max_cores=96
fi
export OMP_NUM_THREADS="$cores"
# export AOCLSPARSE_NUM_THREADS="$cores"
export GOMP_CPU_AFFINITY="0-$((max_cores-1))"

# Encourages idle threads to spin rather than sleep.
export OMP_WAIT_POLICY='active'
# Don't let the runtime deliver fewer threads than those we asked for.
export OMP_DYNAMIC='false'

# Set cache sizes in bytes per core(L1,L2) for blocking.
export L1_CACHE_SIZE=$(getconf LEVEL1_DCACHE_SIZE)

export L1_FLOATS=$(( L1_CACHE_SIZE / 4 ))

export L2_CACHE_SIZE=$(getconf LEVEL2_CACHE_SIZE)

export L2_FLOATS=$(( L2_CACHE_SIZE / 4 ))

export L3_CACHE_SIZE=$(getconf LEVEL3_CACHE_SIZE)

export L3_FLOATS=$(( L3_CACHE_SIZE / 4 ))

export PREFETCH_DIST=4

export CROSS_DISTANCE_LIMIT=100

export SEED_ROW=50

export CACHE_CAPACITY=256

export FRONTIER_SIZE=10000

export ALPHA=1.0

export BETA=1.0

# Set cache line size in bytes.
export CACHELINE_SIZE=$(getconf LEVEL1_DCACHE_LINESIZE)

export CACHELINE_FLOATS=$(( CACHELINE_SIZE / 4 ))

# Head-dimension (d) block size for fused_attention_coo_vector_blocking.cpp: how many columns of
# K/V are touched at a time (by every thread in lockstep) before moving to the next block.
export ATTENTION_D_BLOCK_SIZE=256

# k-dimension block size for kernel_coo_vec_fixedk_block.cpp: how many columns of x[] every
# thread works on at once (in lockstep, via a barrier between blocks) before moving to the next
# block. Must be one of 32/64/128/256.
export COO_K_BLOCK_SIZE=128

# Select the dataset you want to run the benchmarks on.
# export DATASET='MATRIX_MARKET'
# export DATASET='GRAPH'
# export DATASET='MASKS'
export DATASET='DLMC'

path_validation='/various/pmpakos/SpMV-Research/validation_matrices'

path_graph='./matrices'

path_masks='/various/pmpakos/sparse_survey/artificial_big_bird_matrix_generator/matrices'


path_dlmc='/various/itasou/dlmc'
# path_dlmc='./matrices_dlmc'

# Select on which matrices from the following list you want to run the benchmarks.
# These matrices are located on the above defined path.
matrices_validation=(
    # these are the initial matrices that I experimented on, no need to run them again
    scircuit.mtx
    mac_econ_fwd500.mtx
    raefsky3.mtx
    rgg_n_2_17_s0.mtx
    bbmat.mtx
    appu.mtx
    mc2depi.mtx
    rma10.mtx
    cop20k_A.mtx
    thermomech_dK.mtx
    webbase-1M.mtx
    cant.mtx
    ASIC_680k.mtx
    roadNet-TX.mtx
    pdb1HYS.mtx
    TSOPF_RS_b300_c3.mtx
    Chebyshev4.mtx
    consph.mtx
    com-Youtube.mtx
    rajat30.mtx
    radiation.mtx
    Stanford_Berkeley.mtx
    shipsec1.mtx
    PR02R.mtx
    CurlCurl_2.mtx
    gupta3.mtx
    mip1.mtx
    rail4284.mtx
    pwtk.mtx
    crankseg_2.mtx
    Si41Ge41H72.mtx
    TSOPF_RS_b2383.mtx
    in-2004.mtx
    Ga41As41H72.mtx
    mac_econ_fwd500.mtx
    raefsky3.mtx
    rgg_n_2_17_s0.mtx
    bbmat.mtx
    appu.mtx
    mc2depi.mtx
    cop20k_A.mtx
    ASIC_680k.mtx
    webbase-1M.mtx
    TSOPF_RS_b300_c3.mtx
    roadNet-TX.mtx
    com-Youtube.mtx
    Stanford_Berkeley.mtx
    gupta3.mtx
    mip1.mtx
    crankseg_2.mtx
    rail4284.mtx
    Si41Ge41H72.mtx
    in-2004.mtx
)

matrices_graph=(
    # these are the sparse survey matrices (selected from other matrices that work on graphs etc...)
    citeseer.mtx
    cora.mtx
    pubmed.mtx
    PROTEINS.mtx
    ogbl-ddi.mtx
    ogbl-collab.mtx
    ogbn-arxiv.mtx
    harvard.mtx
    com-Amazon.mtx
    REDDIT-BINARY.mtx
    amazon0505.mtx
    OVCAR-8H.mtx
    wiki-Talk.mtx
    roadNet-CA.mtx
    com-Youtube.mtx
    web-BerkStan.mtx
    sx-stackoverflow.mtx
    ogbn-proteins.mtx

)

# These are the masks generated by Shane's project!
matrices_masks=(
    big_bird_512_0.50.mtx
    big_bird_512_0.60.mtx
    big_bird_512_0.70.mtx
    big_bird_512_0.80.mtx
    big_bird_512_0.90.mtx
    big_bird_512_0.95.mtx
    big_bird_512_0.98.mtx
    big_bird_1024_0.50.mtx
    big_bird_1024_0.60.mtx
    big_bird_1024_0.70.mtx
    big_bird_1024_0.80.mtx
    big_bird_1024_0.90.mtx
    big_bird_1024_0.95.mtx
    big_bird_1024_0.98.mtx
    big_bird_2048_0.50.mtx
    big_bird_2048_0.60.mtx
    big_bird_2048_0.70.mtx
    big_bird_2048_0.80.mtx
    big_bird_2048_0.90.mtx
    big_bird_2048_0.95.mtx
    big_bird_2048_0.98.mtx
    random_attention_512_0.50.mtx
    random_attention_512_0.60.mtx
    random_attention_512_0.70.mtx
    random_attention_512_0.80.mtx
    random_attention_512_0.90.mtx
    random_attention_512_0.95.mtx
    random_attention_512_0.98.mtx
    random_attention_1024_0.50.mtx
    random_attention_1024_0.60.mtx
    random_attention_1024_0.70.mtx
    random_attention_1024_0.80.mtx
    random_attention_1024_0.90.mtx
    random_attention_1024_0.95.mtx
    random_attention_1024_0.98.mtx
    random_attention_2048_0.50.mtx
    random_attention_2048_0.60.mtx
    random_attention_2048_0.70.mtx
    random_attention_2048_0.80.mtx
    random_attention_2048_0.90.mtx
    random_attention_2048_0.95.mtx
    random_attention_2048_0.98.mtx
    window_attention_512_0.50.mtx
    window_attention_512_0.60.mtx
    window_attention_512_0.70.mtx
    window_attention_512_0.80.mtx
    window_attention_512_0.90.mtx
    window_attention_512_0.95.mtx
    window_attention_512_0.98.mtx
    window_attention_1024_0.50.mtx
    window_attention_1024_0.60.mtx
    window_attention_1024_0.70.mtx
    window_attention_1024_0.80.mtx
    window_attention_1024_0.90.mtx
    window_attention_1024_0.95.mtx
    window_attention_1024_0.98.mtx
    window_attention_2048_0.50.mtx
    window_attention_2048_0.60.mtx
    window_attention_2048_0.70.mtx
    window_attention_2048_0.80.mtx
    window_attention_2048_0.90.mtx
    global_attention_512_0.50.mtx
    global_attention_512_0.60.mtx
    global_attention_512_0.70.mtx
    global_attention_512_0.80.mtx
    global_attention_512_0.90.mtx
    global_attention_512_0.95.mtx
    global_attention_512_0.98.mtx
    global_attention_1024_0.50.mtx
    window_attention_2048_0.95.mtx
    window_attention_2048_0.98.mtx
    global_attention_1024_0.60.mtx
    global_attention_1024_0.70.mtx
    global_attention_1024_0.80.mtx
    global_attention_1024_0.90.mtx
    global_attention_1024_0.95.mtx
    global_attention_1024_0.98.mtx
    global_attention_2048_0.50.mtx
    global_attention_2048_0.60.mtx
    global_attention_2048_0.70.mtx
    global_attention_2048_0.80.mtx
    global_attention_2048_0.90.mtx
    global_attention_2048_0.95.mtx
    global_attention_2048_0.98.mtx
)

dlmc_matrices_files=(
    # "$path_dlmc/transformer_matrices.txt"
    # "./transformer_matrices_custom.txt"
    "$path_dlmc/transformer_matrices_small_onlyencoder_onlymagnitude.txt"
    # "$path_dlmc/transformer_matrices_small.txt"

)
if [ "$DATASET" = "MATRIX_MARKET" ]; then
    path="$path_validation"
    matrices=(
        "${matrices_validation[@]}"
    )
elif [ "$DATASET" = "GRAPH" ]; then
    path="$path_graph"
    matrices=(
        "${matrices_graph[@]}"
    )
elif [ "$DATASET" = "MASKS" ]; then
    path="$path_masks"
    matrices=(
        "${matrices_masks[@]}"
    )
elif [ "$DATASET" = "DLMC" ]; then
    path="$path_dlmc"
    matrices=()

    for f in "${dlmc_matrices_files[@]}"; do
        mapfile -t dlmc_matrices < "$f"
        for a in "${dlmc_matrices[@]}"; do
            # if [[ "$a" == *"0.6"* ]]; then
            matrices+=("$a")
            # fi
        done
    done
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

# make clean;
# time make -j
export RUNS=2
# For CPU kernels, no need for 1000 extra iterations for warmup, just change the environment variable
echo "Running benchmarks on $DATASET dataset for $RUNS runs with APLHA=$ALPHA, BETA=$BETA, CROSS_DISTANCE_LIMIT=$CROSS_DISTANCE_LIMIT, SEED_ROW=$SEED_ROW, CACHE_CAPACITY=$CACHE_CAPACITY, FRONTIER_SIZE=$FRONTIER_SIZE, COO_K_BLOCK_SIZE=$COO_K_BLOCK_SIZE, ATTENTION_D_BLOCK_SIZE=$ATTENTION_D_BLOCK_SIZE"
echo "CPU kernels"
export GPU_KERNEL=0
if [[ "$MODE" == "laptop" ]]; then
    export SYSTEM='AMD-5800H'
else
    export SYSTEM='AMD-EPYC-24'
fi
for r in $(seq 1 $RUNS);
do
    echo "Run $r of $RUNS"
    for k in 16 32 64 128 256 512 1024 2048 4096;
    do
        # ./gemm_mkl.exe 1024 1024 1024
        for a in "${matrices[@]}"
        do
            echo '--------'
            echo ${path}/$a
            ./spmm_mkl.exe ${path}/$a $k
            # ./spmm_mkl_colind0.exe ${path}/$a $k
            # ./spmmm_mkl_coo.exe ${path}/$a $k
            # ./spmm_csr_naive.exe ${path}/$a $k
            # ./spmm_csr.exe ${path}/$a $k
            # ./spmm_csr_vector.exe ${path}/$a $k
            # ./spmm_csr_vector_xrow.exe ${path}/$a $k
            # ./spmm_csr_vector_xrow_row_split.exe ${path}/$a $k
            # ./spmm_csc_vector_xrow.exe ${path}/$a $k
            # ./spmm_csr_vector_xrow_blocked_l1.exe ${path}/$a $k
            # ./spmm_csr_vector_xrow_blocked_l1_j_stream.exe ${path}/$a $k
            # ./spmm_coo_vector_xrow_atomic.exe ${path}/$a $k
            ./spmm_coo_vector_xrow_row_split.exe ${path}/$a $k
            ./spmm_coo_vector_row_reg.exe ${path}/$a $k
            ./spmm_coo_vector_row_reg_fixedk.exe ${path}/$a $k
            ./spmm_coo_vector_fixedk_block.exe ${path}/$a $k
            # ./spmm_coo_vector_xrow_colind0.exe ${path}/$a $k
            # ./spmm_coo_vector_z_order.exe ${path}/$a $k
            # ./spmm_coo_vector_z_order_prefetch.exe ${path}/$a $k
            # ./spmm_coo_vector_hilbert.exe ${path}/$a $k
            # ./spmm_coo_vector_cross.exe ${path}/$a $k
            # ./spmm_coo_vector_frontier.exe ${path}/$a $k
            # ./spmm_coo_vector_frontier_ringbuffer.exe ${path}/$a $k
            # ./spmm_coo_vector_frontier_ringbuffer_roulette.exe ${path}/$a $k
            # ./spmm_coo_vector_frontier_ringbuffer_all_seed.exe ${path}/$a $k
            # ./spmm_coo_vector_rcm.exe ${path}/$a $k
            # ./spmm_coo_vector_z_order_new.exe ${path}/$a $k
            # ./spmm_coo_vector_xrow_perfect_nnz_balance.exe ${path}/$a $k
            # ./spmm_csr_vector_xrow_prefetch.exe ${path}/$a $k
            # ./spmm_csr_vector_perfect_nnz_balance.exe ${path}/$a $k
            # ./spmm_csr_vector_perfect_nnz_balance_prefetch.exe ${path}/$a $k
            # ./spmm_csr_kahan.exe ${path}/$a $k
            # ./spmm_csr_simd.exe ${path}/$a $k
            # ./spmm_csr_prefetch.exe ${path}/$a $k
            # ./spmm_csr_vector.exe ${path}/$a $k

            # ./spmm_csr_column.exe ${path}/$a $k
            # ./spmm_csr_kahan_column.exe ${path}/$a $k
            # ./spmm_csr_simd_column.exe ${path}/$a $k
            # ./spmm_csr_prefetch_column.exe ${path}/$a $k
            # ./spmm_csr_vector_column.exe ${path}/$a $k
            # ./spmm_csr_vector_perfect_nnz_balance_column.exe ${path}/$a $k
            # ./spmm_csr_vector_perfect_nnz_balance_prefetch_column.exe ${path}/$a $k

            # ./spmm_aocl.exe ${path}/$a $k
            # ./spmm_aspt_cpu.exe ${path}/$a $k
            ./spmm_fusedmm.exe ${path}/$a $k

            # ./sddmm_aspt_cpu.exe ${path}/$a $k
            # ./attention_coo_vector_row_split.exe ${path}/$a $k
            # ./attention_coo_vector_row_split_blocking.exe ${path}/$a $k
            # ./attention_coo_vector_row_split_blocking.exe ${path}/$a $k
        done
    done
done

# # For GPU kernels, we need to run 1000 extra iterations for warmup.
# echo "GPU kernels"
# export GPU_KERNEL=1
# if [[ "$MODE" == "laptop" ]]; then
#     export SYSTEM='NVIDIA-3070M'
# else
#     export SYSTEM='NVIDIA-A100'
# fi
# for k in 16 32 64 128 256 512 1024;
# do
#     for a in "${matrices[@]}"
#     do
#         echo '--------'
#         echo ${path}/$a
#         # ./spmm_cusparse.exe ${path}/$a $k
#         # # ./spmm_acc.exe ${path}/$a $k
#         # ./spmm_aspt_gpu.exe ${path}/$a $k
#         # ./spmm_rode.exe ${path}/$a $k
#         # # ./spmm_hc.exe ${path}/$a $k
#         # ./spmm_dgsparse_0.exe ${path}/$a $k # GESPMM_ALG_SEQREDUCE_ROWBALANCE
#         # ./spmm_dgsparse_1.exe ${path}/$a $k # GESPMM_ALG_PARREDUCE_ROWBALANCE
#         # ./spmm_dgsparse_2.exe ${path}/$a $k # GESPMM_ALG_SEQREDUCE_NNZBALANCE
#         # ./spmm_dgsparse_3.exe ${path}/$a $k # GESPMM_ALG_PARREDUCE_NNZBALANCE
#         # ./spmm_dgsparse_4.exe ${path}/$a $k # GESPMM_ALG_ROWCACHING_ROWBALANCE
#         # ./spmm_dgsparse_5.exe ${path}/$a $k # GESPMM_ALG_ROWCACHING_NNZBALANCE
#         # ./spmm_gnnpilot_1.exe ${path}/$a $k # BALANCE=1
#         # ./spmm_gnnpilot_2.exe ${path}/$a $k # BALANCE=2
#         # ./spmm_dtc_0.exe ${path}/$a $k # float_nonsplit
#         # ./spmm_dtc_1.exe ${path}/$a $k # float2_nonsplit
#         # ./spmm_dtc_2.exe ${path}/$a $k # float2_split
#         # ./spmm_dtc_3.exe ${path}/$a $k # float4_nonsplit
#         # ./spmm_dtc_4.exe ${path}/$a $k # float4_split
#         # ./spmm_dtc_5.exe ${path}/$a $k # v2 float_nonsplit
#         # ./spmm_dtc_6.exe ${path}/$a $k # v2 float4_split
#         # ./spmm_sputnik.exe ${path}/$a $k

#         # ./sddmm_cusparse.exe ${path}/$a $k
#         # ./sddmm_aspt_gpu.exe ${path}/$a $k
#         # # ./sddmm_rode.exe ${path}/$a $k
#         # ./sddmm_dgsparse.exe ${path}/$a $k
#         # ./sddmm_gnnpilot.exe ${path}/$a $k
#         # # ./sddmm_sputnik.exe ${path}/$a $k
#     done
# done

