


TOTAL_UPDATES=50000    # Total number of training steps
WARMUP_UPDATES=3000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
PEAK_LR=0.0002
PEAK_LR=0.001
TOKENS_PER_SAMPLE=512   # Max sequence 
TOKENS_PER_SAMPLE=490
# TOKENS_PER_SAMPLE=20
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_POSITIONS=512
# MAX_SENTENCES=16        # Number of sequences per batch (batch size)
# UPDATE_FREQ=2          # Increase the batch size 16x

# 2卡update freq=16
# 4卡update freq=8

MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_POSITIONS=512
MAX_POSITIONS=512
# MAX_POSITIONS=20
# MAX_SENTENCES=32        # Number of sequences per batch (batch size)
MAX_POSITIONS=32
TOKENS_PER_SAMPLE=32
# TOKENS_PER_SAMPLE=16
# UPDATE_FREQ=16
#32 OOM
MAX_SENTENCES=32       # Number of sequences per batch (batch size)
MAX_SENTENCES=1
UPDATE_FREQ=1          # Increase the batch size 16x
# UPDATE_FREQ=1

DATA_DIR=/d/MOOC/fairseq/examples/transformer/roberta/data-bin/wikitext-103

#ARCH=transformer_lm_wiki103
prefix=roberta


TOTAL_UPDATES=4    # Total number of training steps
# TOTAL_UPDATES=30
WARMUP_UPDATES=1    # Warmup the learning rate over this many updates

	# -p MMG \
	# -x SH-IDC1-10-198-34-95 \
    # --ntasks-per-node $1 
# --sample-break-mode complete
# for w in 32 64; do
#     for h in 1 8; do
#         for i in 2 3; do
#             for j in 1 2 3; do
#  64 128 256
# for w in 32 64 128 256; do
for w in 64; do
    for h in 12; do
        for i in 1; do
            for j in 1; do
                for p in 1; do
                    # for ARCH in roberta_normtype_21_head_12_1 roberta_normtype_11_head_1_1 roberta_normtype_22_head_1_1; do
                    # for ARCH in roberta_glu_rms_layer roberta_glu_dropout_rms_layer roberta_glu_moreheads_rms_layer roberta_norm_type_stand_norm_33_h12 roberta_normtype_21_head_12_24 roberta_normtype_21_head_24_24; do
                    # for ARCH in roberta_spe roberta_1+elu_spe; do
                    # for ARCH in roberta_1+elu_per roberta_per
                    # for ARCH in roberta_urpe_1d_3_no_abs roberta_1+elu_1_1_no_abs roberta_1+elu_1d_3_no_abs roberta_urpe_1_1_no_abs
                    # for ARCH in roberta_1+elu_1d_3_no_abs
                    # for ARCH in roberta_glu_all_rms_layer
                    # for ARCH in roberta_t5
                    # for ARCH in roberta_glu_all_rms_layer roberta_glu_all_rms_layer_small roberta_glu_rms_layer_small
                    # for ARCH in roberta_rpaw
                    # for ARCH in roberta_glu_all_layernorm roberta_glu_all_layernorm_small
                    # for ARCH in roberta_1+elu roberta_1+elu_1_1 roberta_1+elu_1d_3 roberta_1+elu_spe roberta_1+elu_per
                    # for ARCH in roberta_base roberta_urpe_1_1 roberta_urpe_1d_3 roberta_spe roberta_t5
                    # for ARCH in roberta_base roberta_norm_type_11 roberta_norm_type_2_2_w64_h8 roberta_rms_layer_standard roberta_glu_rms_layer roberta_glu_all_layernorm roberta_glu_all_layernorm_small
                    # for ARCH in roberta_glu_all_rms_layer_ln_rms roberta_glu_all_rms_layer_small_ln_rms
                    # for ARCH in roberta_glu_all_rms_layer_ln_rms_urpe_1d3 roberta_glu_all_rms_layer_small_ln_rms_urpe_1d3 roberta_glu_all_layernorm_urpe_1d3 roberta_glu_all_layernorm_small_urpe_1d3
                    # for ARCH in roberta_glu_all_rms_layer_ln_rms
                    # for ARCH in roberta_glu_all_rms_layer roberta_glu_all_rms_layer_elu roberta_glu_all_rms_layer_small_elu roberta_glu_all_layernorm_elu roberta_glu_all_layernorm_small_elu
                    # for ARCH in roberta_ls
                    # for ARCH in roberta_glu_all_rms_layer_swish roberta_glu_all_rms_layer_small_swish roberta_glu_all_layernorm_swish roberta_glu_all_layernorm_small_swish
                    # for ARCH in roberta_performer roberta_ls
                    # for ARCH in roberta_ls_v2
                    # for ARCH in roberta_glu_all_rms_layer_ln_rms_urpe_1d3_dropout02 roberta_glu_all_rms_layer_small_ln_rms_urpe_1d3_dropout02 roberta_glu_all_layernorm_urpe_1d3_dropout02 roberta_glu_all_layernorm_small_urpe_1d3_dropout02
                    # for ARCH in roberta_glu_all_rms_layer_ln_rms
                    # for ARCH in roberta_ls
                    # for ARCH in roberta_glu_all_rms_layer_ln_rms_urpe_1d3_no_abs roberta_glu_all_rms_layer_small_ln_rms_urpe_1d3_no_abs roberta_glu_all_layernorm_urpe_1d3_no_abs roberta_glu_all_layernorm_small_urpe_1d3_no_abs
                    # for ARCH in roberta_glu_pure_rms_urpe_1d3_small_init roberta_glu_small_pure_rms_urpe_1d3_small_init
                    # for ARCH in roberta_glu_all_rms_layer_ln_rms_urpe_1d3 roberta_glu_pure_rms_urpe_1d3_small_init
                    # for ARCH in roberta_glu_pure_rms_urpe_1d3_small_init_no_abs roberta_glu_small_pure_rms_urpe_1d3_small_init_no_abs
                    # for ARCH in roberta_glu_pure_rms_urpe_1d3_geglu roberta_glu_small_pure_rms_urpe_1d3_geglu roberta_glu_pure_rms_urpe_1d3_geglu_small_init roberta_glu_small_pure_rms_urpe_1d3_geglu_small_init
                    # for ARCH in roberta_glu_pure_rms_urpe_1d3 roberta_glu_small_pure_rms_urpe_1d3 roberta_glu_pure_rms_urpe_1d3_laplace roberta_glu_small_pure_rms_urpe_1d3_laplace roberta_glu_pure_rms_urpe_1d3_gaussian roberta_glu_small_pure_rms_urpe_1d3_gaussian
                    # for ARCH in roberta_glu_pure_rms_urpe_1d3_laplace roberta_glu_small_pure_rms_urpe_1d3_laplace roberta_glu_pure_rms_urpe_1d3_gaussian roberta_glu_small_pure_rms_urpe_1d3_gaussian
                    # for ARCH in roberta_glu_pure_rms_urpe_1d3_final_dropout roberta_glu_small_pure_rms_urpe_1d3_final_dropout roberta_glu_pure_rms_urpe_1d3 roberta_glu_small_pure_rms_urpe_1d3
                    # for ARCH in roberta_glu_pure_rms_urpe_1d3_relu2 roberta_glu_small_pure_rms_urpe_1d3_relu2 
                    # for ARCH in roberta_glu_pure_rms_urpe_1d3_linear_chunk roberta_glu_small_pure_rms_urpe_1d3_linear_chunk
                    # for ARCH in roberta_glu_pure_rms_urpe_1d3_linear_chunk_32 roberta_glu_small_pure_rms_urpe_1d3_linear_chunk_32 roberta_glu_pure_rms_urpe_1d3_linear_chunk_16 roberta_glu_small_pure_rms_urpe_1d3_linear_chunk_16
                    # for ARCH in roberta_glu_pure_rms_urpe_1d3_linear_chunk_32
                    for ARCH in roberta_base
                    do
                    # ARCH=roberta_norm_type_${i}_${j}_w${w}_h${h}_p${p}
                    # ARCH=roberta_norm_type_stand_act2${i}
                    # ARCH=roberta_norm_type_w${w}_h12_13
                    # ARCH=roberta_norm_type_stand_norm_${i}${j}
                    # # ARCH=roberta_norm_type_stand_drop
                    # ARCH=roberta_norm_type_2_1_w64_h8
                    # ARCH=roberta_norm_type_stand_norm_11_h12
                    # ARCH=roberta_linear_standard_${i}${j}
                    # ARCH=roberta_norm_type_11
                    # ARCH=roberta_normtype_${i}${j}_glu_1
                    # ARCH=roberta_norm_type_2_1_w64_h12_p0.5
                    # ARCH=roberta_norm_type_stand_chunk_stl
                    echo $ARCH
                    fairseq-train $DATA_DIR \
                            --task masked_lm --criterion masked_lm \
                            --save-dir checkpoints/$prefix/$ARCH \
                            --arch $ARCH --sample-break-mode none --tokens-per-sample $TOKENS_PER_SAMPLE \
                            --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
                            --lr-scheduler polynomial_decay --lr $PEAK_LR --total-num-update $TOTAL_UPDATES \
                            --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
                            --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
                            --ddp-backend=legacy_ddp \
                            --find-unused-parameters \
                            --disable-validation \
                            --no-save \
                            --required-batch-size-multiple 1 \
                            --max-update $TOTAL_UPDATES --log-format simple --log-interval 1  2>&1 | tee $ARCH.train
                    done
                done
            done
        done
    done
done
# --warmup-updates $WARMUP_UPDATES
# --sample-break-mode none
