#! /usr/bin/bash

GPUS=$1
ARCH=$2
LR=$3
CLIP_NORM=$4

BATCH_SIZE=8
TOKENS_PER_SAMPLE=256
MAX_TOKEN=$((TOKENS_PER_SAMPLE*BATCH_SIZE))
DATA_DIR=PATH_TO_DATA/data-bin
prefix=lm
MAX_UPDATE=20000
WARM_UP=1600
UPDATE_FREQ=$(( 128 / $BATCH_SIZE / $GPUS ))
PORT=$(( $RANDOM + 2000 ))

trev-train --task language_modeling \
    $DATA_DIR \
    --save-dir checkpoints/$prefix/$ARCH \
    --distributed-world-size $GPUS  --distributed-port $PORT \
    --arch $ARCH --share-decoder-input-output-embed \
    --dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm $CLIP_NORM \
    --lr $LR --lr-scheduler inverse_sqrt --warmup-updates $WARM_UP --warmup-init-lr 1e-07 \
    --tokens-per-sample $TOKENS_PER_SAMPLE --sample-break-mode none \
    --max-tokens $MAX_TOKEN --update-freq $UPDATE_FREQ \
    --ddp-backend=legacy_ddp \
    --batch-size $BATCH_SIZE \
    --max-update $MAX_UPDATE \
    --log-format simple \
    --log-interval 1 \
    2>&1 | tee $ARCH.log