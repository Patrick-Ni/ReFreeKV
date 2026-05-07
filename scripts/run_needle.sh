#!/bin/bash
# ============================================================
# Needle-in-a-Haystack Evaluation
# ============================================================

MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"   # Change to your local model path

# Full KV
python needle_in_haystack.py \
    --model_name $MODEL_PATH \
    --model_provider LLaMA3 \
    --model_version llama3_full \
    --s_len 1000 \
    --e_len 8000 \
    --step 1000 \
    --method full \
    --attn_implementation eager

# ReFreeKV (H2O-based)
python needle_in_haystack.py \
    --model_name $MODEL_PATH \
    --model_provider LLaMA3 \
    --model_version llama3_h2o_128 \
    --s_len 1000 \
    --e_len 8000 \
    --step 1000 \
    --method h2o \
    --max_capacity_prompt 128 \
    --attn_implementation eager
