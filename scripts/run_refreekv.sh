#!/bin/bash
# ============================================================
# ReFreeKV: Threshold-Free KV Cache Pruning
# Example script for running ReFreeKV with dynamic budget
# ============================================================

# --- Configuration ---
CUDA_VISIBLE_DEVICES=0
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"   # Change to your local model path
DATASET="gsm8k"                                      # Dataset name
OUTPUT_PATH="outputs/${DATASET}/refreekv/"
METHOD="h2o"                                         # Base eviction method: h2o, snapkv, streamingllm
METRIC="last_token_attn"                             # Dynamic metric
IMPLEMENTATION="eager"                               # Attention implementation: eager, sdpa, flash_attention_2
BUDGET=1.0                                           # Initial budget ratio (1.0 = full cache)
THRESHOLD=0.05                                       # Threshold for dynamic budget adjustment

# --- Run ---
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
    --model_path $MODEL_PATH \
    --dataset $DATASET \
    --output_path $OUTPUT_PATH \
    --method $METHOD \
    --dynamic \
    --metric $METRIC \
    --normalize \
    --attn_implementation $IMPLEMENTATION \
    --modify \
    --budget $BUDGET \
    --threshold $THRESHOLD \
    --chat \
    --save_every 50
