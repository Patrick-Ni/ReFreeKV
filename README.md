# ReFreeKV: Towards Threshold-Free KV Cache Pruning

This is the official implementation of the paper:

**[Towards Threshold-Free KV Cache Pruning](https://arxiv.org/abs/2502.16886)**

Xuanfan Ni, Liyan Xu, Chenyang Lyu, Longyue Wang, Mo Yu, Lemao Liu, Fandong Meng, Jie Zhou, Piji Li

## Overview

To reduce memory consumption during LLM inference, prior works have proposed numerous methods that focus on KV cache pruning based on various criteria. While these techniques often accomplish lossless memory reduction on many datasets, they often rely on an under-emphasized condition: a **dataset/domain-specific budget size threshold** needs to be pre-determined to achieve the optimal performance. However, such input-specific tuning may be considerably limited in real-world scenarios, as open-domain inputs span diverse domains, lengths and difficulty levels, without clear boundaries for pre-tuning.

In this work, we propose a new objective that lifts the threshold constraints for robust KV pruning, calling for **"threshold-free"** methods that automatically adjust budget sizes while ensuring full-cache performance. We then propose **ReFreeKV** as the first solution fulfilling this objective, validated by intensive experiments on 13 datasets of diverse context lengths, task types, and model sizes.

## Project Structure

```
ReFreeKV/
├── main.py                     # Main inference script
├── eval.py                     # Evaluation script
├── load_datasets.py            # Dataset loading utilities
├── utils.py                    # General utility functions
├── metrics.py                  # Evaluation metrics (F1, ROUGE, etc.)
├── theoremqa_utils.py          # TheoremQA-specific evaluation utilities
├── needle_in_haystack.py       # Needle-in-a-Haystack evaluation
├── modified_models/            # Core KV cache pruning implementations
│   ├── monkeypatch.py          # Model monkey-patching for attention replacement
│   ├── evict_utils.py          # KV cache eviction strategies
│   ├── dynamic_metrics.py      # Dynamic budget adjustment metrics
│   ├── llama_model.py          # Modified LLaMA attention modules
│   ├── mistral_model.py        # Modified Mistral attention modules
│   └── qwen_model.py           # Modified Qwen2 attention modules
├── scripts/                    # Example run scripts
│   ├── run_refreekv.sh         # Run ReFreeKV (dynamic budget)
│   ├── run_fullkv.sh           # Run Full KV baseline
│   ├── run_baseline.sh         # Run fixed-budget baselines (H2O, StreamingLLM, SnapKV)
│   ├── run_eval.sh             # Evaluate results
│   └── run_needle.sh           # Needle-in-a-Haystack evaluation
├── data/                       # Place datasets here
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/your-username/ReFreeKV.git
cd ReFreeKV
pip install -r requirements.txt
```

## Data Preparation

Download and place the datasets under the `data/` directory:

- **GSM8K**: Download `test.jsonl` from [GSM8K](https://github.com/openai/grade-school-math) and save as `data/gsm8k_test.jsonl`
- **LongBench**: Download from [LongBench](https://github.com/THUDM/LongBench) and place under `data/LongBench/`
- **CoQA**: Download from [CoQA](https://stanfordnlp.github.io/coqa/) and save as `data/coqa-dev.json`
- **TruthfulQA**: Download from [TruthfulQA](https://github.com/sylinrl/TruthfulQA) and save as `data/TruthfulQA.csv`
- **TheoremQA**: Save as `data/theoremqa.json`
- **GPQA**: Save as `data/gpqa_main.json`
- **Needle-in-a-Haystack**: Place Paul Graham essays under `data/PaulGrahamEssays/`

## Usage

### ReFreeKV (Dynamic Threshold-Free Pruning)

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset gsm8k \
    --output_path outputs/gsm8k/refreekv/ \
    --method h2o \
    --dynamic \
    --metric last_token_attn \
    --normalize \
    --attn_implementation eager \
    --modify \
    --budget 1.0 \
    --threshold 0.05 \
    --chat
```

### Full KV Baseline

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset gsm8k \
    --output_path outputs/gsm8k/fullkv/ \
    --method fullkv \
    --attn_implementation eager \
    --modify \
    --budget 1.0 \
    --chat
```

### Fixed-Budget Baselines

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset gsm8k \
    --output_path outputs/gsm8k/h2o/ \
    --method h2o \
    --attn_implementation eager \
    --modify \
    --budget 0.5 \
    --chat
```

### Evaluation

```bash
python eval.py --path outputs/gsm8k/refreekv/ --dataset gsm8k --sort --key threshold
```

### Needle-in-a-Haystack

```bash
python needle_in_haystack.py \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --model_provider LLaMA3 \
    --model_version llama3_refreekv \
    --s_len 1000 --e_len 8000 --step 1000 \
    --method h2o --max_capacity_prompt 128 \
    --attn_implementation eager
```

## Supported Models

- LLaMA-2 (7B, 13B, etc.)
- LLaMA-3 (8B, etc.)
- Mistral (7B, etc.)
- Qwen2.5

## Supported Datasets

| Category | Datasets |
|----------|----------|
| **Math Reasoning** | GSM8K, TheoremQA |
| **Question Answering** | CoQA, NarrativeQA, QAsper, HotpotQA, 2WikiMQA, MuSiQue, TriviaQA |
| **Summarization** | QMSum, Multi-News |
| **Classification** | TREC |
| **Multiple Choice** | QuALITY, GPQA, MMLU-STEM |
| **Factuality** | TruthfulQA |
| **Retrieval** | Needle-in-a-Haystack |

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Path to pretrained model | Required |
| `--dataset` | Dataset name | `gsm8k` |
| `--method` | KV pruning method (`fullkv`, `h2o`, `streamingllm`, `snapkv`, `pyramidkv`, `attn`) | `streamingllm` |
| `--dynamic` | Enable dynamic (threshold-free) budget adjustment | `False` |
| `--metric` | Metric for dynamic adjustment | `norm-1` |
| `--threshold` | Threshold value for dynamic budget | `0.05` |
| `--budget` | Budget ratio (fraction of KV cache to keep) | `1.0` |
| `--normalize` | Enable attention score normalization | `False` |
| `--modify` | Use modified generation pipeline | `False` |
| `--chat` | Use chat template for input formatting | `False` |

## Citation

If you find this work useful, please cite:

```bibtex
@article{ni2025towards,
  title={Towards Threshold-Free KV Cache Pruning},
  author={Ni, Xuanfan and Xu, Liyan and Lyu, Chenyang and Wang, Longyue and Yu, Mo and Liu, Lemao and Meng, Fandong and Zhou, Jie and Li, Piji},
  journal={arXiv preprint arXiv:2502.16886},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
