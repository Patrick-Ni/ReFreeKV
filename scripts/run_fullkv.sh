#!/bin/bash
# ============================================================
# Full KV Cache Baseline
# ============================================================

CUDA_VISIBLE_DEVICES=0
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"   # Change to your local model path
DATASET="gsm8k"
OUTPUT_PATH="outputs/${DATASET}/fullkv/"
METHOD="fullkv"
IMPLEMENTATION="eager"
BUDGET=1.0

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
    --model_path $MODEL_PATH \
    --dataset $DATASET \
    --output_path $OUTPUT_PATH \
    --method $METHOD \
    --attn_implementation $IMPLEMENTATION \
    --modify \
    --budget $BUDGET \
    --chat
