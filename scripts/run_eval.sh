#!/bin/bash
# ============================================================
# Evaluate results
# ============================================================

# Evaluate a single result file
python eval.py --path outputs/gsm8k/refreekv/ --dataset gsm8k --sort --key threshold

# Evaluate LongBench dataset
# python eval.py --path outputs/narrativeqa/refreekv/ --dataset narrativeqa

# Evaluate CoQA
# python eval.py --path outputs/coqa/refreekv/ --dataset coqa

# Evaluate with sorted budget values
# python eval.py --path outputs/gsm8k/h2o/ --dataset gsm8k --sort --key budget
