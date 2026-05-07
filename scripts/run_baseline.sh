#!/bin/bash
# ============================================================
# Baseline Methods (H2O, StreamingLLM, SnapKV) with Fixed Budget
# ============================================================

CUDA_VISIBLE_DEVICES=0
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"   # Change to your local model path
DATASET="narrativeqa"
IMPLEMENTATION="eager"

# --- H2O with fixed budget ---
for BUDGET in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
        --model_path $MODEL_PATH \
        --dataset $DATASET \
        --output_path "outputs/${DATASET}/h2o/" \
        --method h2o \
        --attn_implementation $IMPLEMENTATION \
        --modify \
        --budget $BUDGET \
        --chat \
        --save_every 50
done

# --- StreamingLLM with fixed budget ---
for BUDGET in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
        --model_path $MODEL_PATH \
        --dataset $DATASET \
        --output_path "outputs/${DATASET}/streamingllm/" \
        --method streamingllm \
        --attn_implementation $IMPLEMENTATION \
        --modify \
        --budget $BUDGET \
        --chat \
        --save_every 50
done
