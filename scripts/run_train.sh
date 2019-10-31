#!/bin/bash

CODE_LENS=$1
GPU_ID=$2


for CODE_LEN in $CODE_LENS; do
    echo GPU_ID: $GPU_ID, CODE_LEN: $CODE_LEN

    # Run train
    python train.py --gpu-id $GPU_ID \
                    --exp-dir experiments/cifar10_supB \
                    --sample-file sample_files/cifar10_supB_train.txt \
                    --code-len $CODE_LEN \
                    --num-classes 10 \
                    --config-file settings.cfg \
                    --config-opt ${CODE_LEN}bits
done
