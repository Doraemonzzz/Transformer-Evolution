#!/usr/bin/env bash
export NCCL_LL_THRESHOLD=0

GPUS=$1
batch_size=$2
ARCH=$3
LR=$4
DATASET=$5

PORT=$(( $RANDOM + 2000 ))
export MASTER_PORT=${MASTER_PORT:-$PORT}

OUTPUT_DIR=./checkpoints/$ARCH
RESUME=./checkpoints/$ARCH/checkpoint.pth

PROG=PATH_TO_Transformer-Evolution/trev_cv/main.py
DATA=PATH_TO_DATA

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    --use_env $PROG \
    --data-set $DATASET --data-path $DATA \
    --batch-size $batch_size --dist-eval --output_dir $OUTPUT_DIR \
    --resume $RESUME --model $ARCH --epochs 300 --fp32-resume --lr $LR \
    --clip-grad 5.0 \
    2>&1 | tee ${ARCH}.log
    