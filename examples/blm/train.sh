#! /usr/bin/bash

GPUS=$1
ARCH=$2
PEAK_LR=$3
CLIP_NORM=$4

TOTAL_UPDATES=20000    # Total number of training steps
WARMUP_UPDATES=1200    # Warmup the learning rate over this many updates
PEAK_LR=0.0001          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=256   # Max sequence length
MAX_POSITIONS=256       # Num. positional embeddings (usually same as above)
BATCH_SIZE=512
MAX_SENTENCES=128       
UPDATE_FREQ=$(( $BATCH_SIZE / $MAX_SENTENCES / $GPUS ))
PORT=$(( $RANDOM + 2000 ))
DATA_DIR=PATH_TO_DATA/data-bin
prefix=roberta
UPDATE_FREQ=1

trev-train $DATA_DIR \
    --task masked_lm --criterion masked_lm \
    --distributed-world-size $GPUS  --distributed-port $PORT \
    --save-dir checkpoints/$prefix/$ARCH \
    --arch $ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm $CLIP_NORM \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --ddp-backend=legacy_ddp \
    --find-unused-parameters \
    --skip-invalid-size-inputs-valid-test \
    --max-update $TOTAL_UPDATES \
    --log-format simple \
    --log-interval 1 \
    2>&1 | tee $ARCH.log