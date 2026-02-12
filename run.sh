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

# Set cache line size in bytes.
export CACHELINE_SIZE=$(getconf LEVEL1_DCACHE_LINESIZE)

export CACHELINE_FLOATS=$(( CACHELINE_SIZE / 4 ))

# Select the dataset you want to run the benchmarks on.
# export DATASET='MATRIX_MARKET'
export DATASET='DLMC'

# path_validation='/various/pmpakos/SpMV-Research/validation_matrices'
path_validation='./matrices'

# path_dlmc='/various/itasou/dlmc'
path_dlmc='./dlmc_matrices'

# Select on which matrices from the following list you want to run the benchmarks.
# These matrices are located on the above defined path.
matrices_validation=(
    # these are the initial matrices that I experimented on, no need to run them again
    # scircuit.mtx
    # mac_econ_fwd500.mtx
    # raefsky3.mtx
    # rgg_n_2_17_s0.mtx
    # bbmat.mtx
    # appu.mtx
    # mc2depi.mtx
    # rma10.mtx
    # cop20k_A.mtx
    # thermomech_dK.mtx
    # webbase-1M.mtx
    # cant.mtx
    # ASIC_680k.mtx
    # roadNet-TX.mtx
    # pdb1HYS.mtx
    # TSOPF_RS_b300_c3.mtx
    # Chebyshev4.mtx
    # consph.mtx
    # com-Youtube.mtx
    # rajat30.mtx
    # radiation.mtx
    # Stanford_Berkeley.mtx
    # shipsec1.mtx
    # PR02R.mtx
    # CurlCurl_2.mtx
    # gupta3.mtx
    # mip1.mtx
    # rail4284.mtx
    # pwtk.mtx
    # crankseg_2.mtx
    # Si41Ge41H72.mtx
    # TSOPF_RS_b2383.mtx
    # in-2004.mtx
    # Ga41As41H72.mtx
    # mac_econ_fwd500.mtx
    # raefsky3.mtx
    # rgg_n_2_17_s0.mtx
    # bbmat.mtx
    # appu.mtx
    # mc2depi.mtx
    # cop20k_A.mtx
    # ASIC_680k.mtx
    # webbase-1M.mtx
    # TSOPF_RS_b300_c3.mtx
    # roadNet-TX.mtx
    # com-Youtube.mtx
    # Stanford_Berkeley.mtx
    # gupta3.mtx
    # mip1.mtx
    # crankseg_2.mtx
    # rail4284.mtx
    # Si41Ge41H72.mtx
    # in-2004.mtx

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

dlmc_matrices_files=(
    "$path_dlmc/transformer_matrices.txt"
    # "$path_dlmc/transformer_matrices_small.txt"

)
if [ "$DATASET" = "MATRIX_MARKET" ]; then
    path="$path_validation"
    matrices=(
        "${matrices_validation[@]}"
    )
elif [ "$DATASET" = "DLMC" ]; then
    path="$path_dlmc"
    matrices=()

    for f in "${dlmc_matrices_files[@]}"; do
        mapfile -t dlmc_matrices < "$f"
        for a in "${dlmc_matrices[@]}"; do
            if [[ "$a" != *"0.5"* ]]; then
            matrices+=("$a")
            fi
        done
    done
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

# make clean;
# time make -j

# For CPU kernels, no need for 1000 extra iterations for warmup, just change the environment variable
echo "CPU kernels"
export GPU_KERNEL=0
if [[ "$MODE" == "laptop" ]]; then
    export SYSTEM='AMD-5800H'
else
    export SYSTEM='AMD-EPYC-24'
fi
for k in 8 16 32 64 128 256 512 1024 2048 4096;
do
    for a in "${matrices[@]}"
    do
        echo '--------'
        echo ${path}/$a
        ./spmm_mkl.exe ${path}/$a $k
        ./spmmm_mkl_coo.exe ${path}/$a $k
        # ./spmm_csr_naive.exe ${path}/$a $k
        # ./spmm_csr.exe ${path}/$a $k
        # ./spmm_csr_vector.exe ${path}/$a $k
        ./spmm_csr_vector_xrow.exe ${path}/$a $k
        # ./spmm_csc_vector_xrow.exe ${path}/$a $k
        # ./spmm_csr_vector_xrow_blocked_l1.exe ${path}/$a $k
        # ./spmm_csr_vector_xrow_blocked_l1_j_stream.exe ${path}/$a $k
        # ./spmm_coo_vector_xrow_atomic.exe ${path}/$a $k
        ./spmm_coo_vector_xrow_row_split.exe ${path}/$a $k
        ./spmm_coo_vector_xrow_perfect_nnz_balance.exe ${path}/$a $k
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
        # ./spmm_fusedmm.exe ${path}/$a $k

        # ./sddmm_aspt_cpu.exe ${path}/$a $k
    done
done

# For GPU kernels, we need to run 1000 extra iterations for warmup.
echo "GPU kernels"
export GPU_KERNEL=1
if [[ "$MODE" == "laptop" ]]; then
    export SYSTEM='NVIDIA-3070M'
else
    export SYSTEM='NVIDIA-A100'
fi
for k in 16 32 64 128 256 512 1024;
do
    for a in "${matrices[@]}"
    do
        echo '--------'
        echo ${path}/$a
        # ./spmm_cusparse.exe ${path}/$a $k
        # # ./spmm_acc.exe ${path}/$a $k
        # ./spmm_aspt_gpu.exe ${path}/$a $k
        # ./spmm_rode.exe ${path}/$a $k
        # # ./spmm_hc.exe ${path}/$a $k
        # ./spmm_dgsparse_0.exe ${path}/$a $k # GESPMM_ALG_SEQREDUCE_ROWBALANCE
        # ./spmm_dgsparse_1.exe ${path}/$a $k # GESPMM_ALG_PARREDUCE_ROWBALANCE
        # ./spmm_dgsparse_2.exe ${path}/$a $k # GESPMM_ALG_SEQREDUCE_NNZBALANCE
        # ./spmm_dgsparse_3.exe ${path}/$a $k # GESPMM_ALG_PARREDUCE_NNZBALANCE
        # ./spmm_dgsparse_4.exe ${path}/$a $k # GESPMM_ALG_ROWCACHING_ROWBALANCE
        # ./spmm_dgsparse_5.exe ${path}/$a $k # GESPMM_ALG_ROWCACHING_NNZBALANCE
        # ./spmm_gnnpilot_1.exe ${path}/$a $k # BALANCE=1
        # ./spmm_gnnpilot_2.exe ${path}/$a $k # BALANCE=2
        # ./spmm_dtc_0.exe ${path}/$a $k # float_nonsplit
        # ./spmm_dtc_1.exe ${path}/$a $k # float2_nonsplit
        # ./spmm_dtc_2.exe ${path}/$a $k # float2_split
        # ./spmm_dtc_3.exe ${path}/$a $k # float4_nonsplit
        # ./spmm_dtc_4.exe ${path}/$a $k # float4_split
        # ./spmm_dtc_5.exe ${path}/$a $k # v2 float_nonsplit
        # ./spmm_dtc_6.exe ${path}/$a $k # v2 float4_split
        # ./spmm_sputnik.exe ${path}/$a $k

        # ./sddmm_cusparse.exe ${path}/$a $k
        # ./sddmm_aspt_gpu.exe ${path}/$a $k
        # # ./sddmm_rode.exe ${path}/$a $k
        # ./sddmm_dgsparse.exe ${path}/$a $k
        # ./sddmm_gnnpilot.exe ${path}/$a $k
        # # ./sddmm_sputnik.exe ${path}/$a $k
    done
done

