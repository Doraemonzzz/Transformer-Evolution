BATCH_SIZE=1
# 64 oom
BATCH_SIZE=64 
# 32 oom
BATCH_SIZE=32
BATCH_SIZE=24
BATCH_SIZE=2
TOKENS_PER_SAMPLE=120
MAX_TOKEN=$((TOKENS_PER_SAMPLE*BATCH_SIZE))
DATA_DIR=/d/MOOC/fairseq/examples/transformer/roberta/data-bin/wikitext-103
ARCH=rela_wiki_ada_v1
ARCH=flash_wiki_ada_v1
ARCH=mem_wiki_ada_has_out_gelu_init
ARCH=mem_wiki_ada_has_out_gelu_init_outnogelu
ARCH=mem_wiki_ada_has_out_gelu_init_rms_norm
ARCH=mem_wiki_ada_single_head_has_out_dropout
ARCH=gmu_wiki_ada_v1
ARCH=mem_wiki_ada_has_out_elu
ARCH=transformer_lm_cos
ARCH=mem_wiki_ada_has_out_elu_rms_norm
ARCH=mem_wiki_ada_has_out_elu_out_no_act
# ARCH=mem_wiki_ada_has_out_elu_lambda0
# ARCH=mem_wiki_ada_has_out_elu_lambda05
ARCH=transformer_lm_cos_type2
ARCH=mem_wiki_ada_has_out_elu_out_no_act_usek
ARCH=mem_wiki_ada_has_out_elu_out_no_act_sigmoid
ARCH=mem_wiki_ada_has_out_elu_out_no_act_exp
ARCH=mem_wiki_ada_has_out_elu_out_no_act_postnorm
ARCH=mem_wiki_ada_has_out_elu_out_no_act_rope
ARCH=rela_wiki_ada_relu2
ARCH=rela_wiki_ada_1+elu
ARCH=rela_wiki_ada_elu
ARCH=rela_wiki_ada_noact
ARCH=mem_wiki_ada_has_out_elu_out_no_act_gatednorm
ARCH=transformer_lm_rope
ARCH=rela_wiki_ada_1+relu
ARCH=rela_wiki_ada_2+elu
# ARCH=flash_linear_wiki_ada_v1
ARCH=1+elu_wiki
ARCH=1+elu_1_1_wiki
ARCH=1+elu_2_1_wiki
ARCH=1+elu_1_2_wiki
ARCH=1+elu_2_2_wiki
ARCH=transformer_lm_orpe_1_1
ARCH=transformer_lm_orpe_1_2
ARCH=transformer_lm_orpe_2_1
ARCH=transformer_lm_orpe_2_2
ARCH=mem_wiki_ada_has_out_1+elu_out_no_act_rope
# ARCH=mem_wiki_ada_has_out_relu_out_no_act_rope
ARCH=1+elu_3_1_wiki
ARCH=1+elu_1_1_wiki
ARCH=1+elu_1b_1_wiki
ARCH=transformer_lm_orpe_3_1
ARCH=transformer_lm_orpe_1b_1
ARCH=1+elu_1b_2_wiki
# ARCH=1+elu_3_2_wiki
ARCH=transformer_lm_orpe_1_2
ARCH=transformer_lm_orpe_1b_2
ARCH=transformer_lm_orpe_2_2
ARCH=transformer_lm_orpe_3_2
ARCH=mem_wiki_ada_has_out_elu_out_no_act_rope_c
ARCH=mem_wiki_ada_has_out_elu_out_no_act_rope_no_abs_pos
ARCH=mem_wiki_ada_has_out_elu_out_no_act_rope_multi_head
ARCH=mem_wiki_ada_has_out_elu_out_no_act_rope_use_v
ARCH=mem_wiki_ada_has_out_elu_out_no_act_rope_use_v_multi_head
ARCH=1+elu_1_3_wiki
# ARCH=transformer_lm_orpe_1_3

ARCH=transformer_lm_orpe_2_3
# ARCH=transformer_lm_orpe_3_3
# ARCH=1+elu_2_3_wiki
# ARCH=1+elu_3_3_wiki

ARCH=1+elu_1d_1_wiki
# ARCH=1+elu_1_3a_wiki
# ARCH=transformer_lm_orpe_1d_1
# ARCH=transformer_lm_orpe_1_3a

ARCH=mem_wiki_ada_has_out_leak_out_no_act
# ARCH=rela_wiki_ada_leak
ARCH=mem_wiki_ada_has_out_leak_out_no_act_0.01

ARCH=mem_wiki_ada_has_out_elu_out_no_act_4head
# ARCH=transformer_lm_wiki103_single_head
ARCH=norm_attention_lm_type1
ARCH=transformer_lm_orpe_1_1
ARCH=transformer_lm_orpe_1c_1

ARCH=1+elu_1_5_wiki
# ARCH=transformer_lm_orpe_1_5
# ARCH=norm_mix_attention_lm_type_1

# ARCH=1+elu_rope_wiki

ARCH=relu_1_5_wiki
# ARCH=relu_rope_wiki


ARCH=transformer_lm
TOKENS_PER_SAMPLE=40


ARCH=transformer_lm_base
ARCH=1+elu_1_1_wiki_base
ARCH=1+elu_wiki_base
# ARCH=transformer_lm_orpe_1_1_base


prefix=lm
MAX_UPDATE=2
# MAX_UPDATE=100
WARM_UP=1
UPDATE_FREQ=1
TOKENS_PER_SAMPLE=512
TOKENS_PER_SAMPLE=512
TOKENS_PER_SAMPLE=256
BATCH_SIZE=7
MAX_TOKEN=$((TOKENS_PER_SAMPLE*BATCH_SIZE))
# 调整
LR=1.0
# LR=0.1

# trev-train --task language_modeling \
#         $DATA_DIR \
#         --save-dir checkpoints/$prefix/512_$ARCH \
#         --arch $ARCH \
#         --max-update $MAX_UPDATE --lr $LR --lr-period-updates 60000 --lr-scheduler cosine --lr-shrink 0.75 \
#         --warmup-updates $WARM_UP --warmup-init-lr 1e-07 --stop-min-lr 1e-09 \
# 		--optimizer nag \
# 		--min-lr 0.0001 --clip-norm 0.1 \
#         --criterion adaptive_loss --max-tokens $MAX_TOKEN --update-freq 3 --tokens-per-sample $TOKENS_PER_SAMPLE --seed 1 \
#         --tensorboard-logdir log \
#         --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=legacy_ddp \
#         --batch-size $BATCH_SIZE --log-interval 1  2>&1 | tee $ARCH.train
# 1+elu_1d_3a_wiki_base
# for j in 2 3
# for j in 3a
# for j in 1 3 5

for j in 4
do
    for ARCH in norm_glu_lm_base_pure_rms_urpe_1d3_linear_chunk_32    
    do
      # ARCH=roberta_orpe_${i}_${j}
      # ARCH=1+elu_${i}_${j}_wiki_base
      # ARCH=1+elu_wiki_base
      # ARCH=transformer_lm_orpe_${i}_${j}_base
      echo $ARCH
      trev-train --task language_modeling \
        $DATA_DIR \
        --save-dir checkpoints/$prefix/512_$ARCH \
        --arch $ARCH --share-decoder-input-output-embed \
        --dropout 0.1 \
        --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
        --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 1 --warmup-init-lr 1e-07 \
        --tokens-per-sample $TOKENS_PER_SAMPLE --sample-break-mode none \
        --max-tokens $MAX_TOKEN --update-freq $UPDATE_FREQ \
        --batch-size $BATCH_SIZE \
        --disable-validation \
        --no-save \
        --validate-after-updates 1 \
        --max-update $MAX_UPDATE --log-interval 1  2>&1 | tee $ARCH.train
    done
done

# --disable-validation \
# --sample-break-mode none \
        # 