#!/bin/bash


path_validation='../matrices'
path_dlmc='/various/itasou/dlmc'


cores='48'
max_cores=96
export OMP_NUM_THREADS="$cores"
export GOMP_CPU_AFFINITY="23-$((max_cores-1))"

# Encourages idle threads to spin rather than sleep.
export OMP_WAIT_POLICY='active'
# Don't let the runtime deliver fewer threads than those we asked for.
export OMP_DYNAMIC='false'
export DATASET='MATRIX_MARKET'
# export DATASET='DLMC'

matrices_validation=(

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


for a in "${matrices[@]}"
do
    echo '--------'
    echo ${path_validation}/$a
    ./mat_feat.exe ${path_validation}/$a
done