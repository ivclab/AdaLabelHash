#!/bin/bash

CODE_LENS=$1
GPU_ID=$2


for CODE_LEN in $CODE_LENS; do
    echo GPU_ID: $GPU_ID, CODE_LEN: $CODE_LEN

    # Run test
    python test.py --gpu-id $GPU_ID \
                   --exp-dir experiments/cifar10_supB \
                   --num-classes 10 \
                   --code-len $CODE_LEN \
                   --sample-files sample_files/cifar10_supB_database.txt \
                   --sample-files sample_files/cifar10_supB_query.txt

    # Run evaluation
    (
    cd scripts
    MATLAB_COMMAND='run_eval({'${CODE_LEN}'}); exit'
    matlab -r "${MATLAB_COMMAND}" -nojvm -nodesktop -nosplash
    )
done
