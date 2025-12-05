#!/bin/bash

# model_name  hidden_size  k
MODELS=(
"canelita_full_01 256 1024"
"canelita_full_02 256 2048"
"canelita_full_03 256 1024"
"canelita_full_04 256 256"
"canelita_full_05 128 256"
"canelita_full_06 256 512"
"canelita_full_07 256 1024"
"canelita_full_08 256 1024"
"canelita_full_09 128 1024"
"canelita_full_10 128 256"
"canelita_full_11 64 1024"
"canelita_full_12 64 128"  
)

TRAIN="tvtsv_trainval_2.npy"
TEST="tvtsv_test_2.npy"
ENC="tvtsv_test_label_encoder_2.npy"

for entry in "${MODELS[@]}"; do

    read MODEL H K <<< "$entry"

    echo "====================================="
    echo " Running $MODEL (hidden=$H, k=$K)"
    echo "====================================="

    python opti_approach.py \
      --trainval-file $TRAIN \
      --test-file $TEST \
      --label-encoder $ENC \
      --pretrained-model ./models/$MODEL/best.pt \
      --hidden-size $H \
      --k $K \
      --enc-layernorm \
      --device cuda \
      --output-folder ./results/$MODEL/boc_ista_results_p_l \
      --lambda-grid 0,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2 \
      --max-iter 5000 \
      --k-folds 5
done
