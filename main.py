import argparse
import os.path
import time
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.streamers import BaseStreamer

from load_datasets import load_dataset
from modified_models.monkeypatch import replace_llama, replace_mistral, replace_qwen2
from utils import *


class TimeStreamer(BaseStreamer):
    """Custom streamer to capture the time of the first generated token."""
    def __init__(self):
        self.first_token_time = None

    def put(self, value):
        if self.first_token_time is None:
            self.first_token_time = time.time()

    def end(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    # model settings
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--use_cache", type=bool, default=True)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--use_quantization", action="store_true")
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument("--modify", action="store_true")
    parser.add_argument("--model_max_len", type=int, default=-1)
    parser.add_argument("--min_length", type=int, default=-1)
    # dataset settings
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the dataset file")
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_gen_len", type=int, default=-1)
    # output settings
    parser.add_argument("--output_name", type=str, default="")
    parser.add_argument("--output_path", type=str, default="outputs")

    # method settings
    parser.add_argument("--method", type=str, default="streamingllm",
                        choices=["attn", "streamingllm", "h2o", "pyramidkv", "snapkv", "fullkv"])
    parser.add_argument("--attn_p", type=float, default=-1)
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--cache_size", type=int, default=-1)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--metric", type=str, default="norm-1")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--normalize", action="store_true")

    # other settings
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=-1)
    parser.add_argument("--save_rate", type=float, default=-1)
    parser.add_argument("--save_layer", type=str, default="")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--chat", action="store_true")

    args = parser.parse_args()
    print_opts(args)
    set_seed(args.seed)
    watch = args.watch
    if watch:
        torch.cuda.reset_peak_memory_stats()
        peak_memory = 0
    save_every = args.save_every if args.save_every != -1 else None
    budget = args.budget
    cache_size = args.cache_size
    use_quantization = args.use_quantization
    debug = args.debug
    method = args.method
    dataset_name = args.dataset
    save_layer = [int(k) for k in args.save_layer.split(",")] if args.save_layer != "" else []
    dynamic = args.dynamic
    metric = args.metric
    threshold = args.threshold
    batch_size = args.batch_size

    # Build output file name
    output_name = dataset_name + args.output_name + "_method_" + method
    output_name = output_name + "_budget_" + str(budget) if method != "fullkv" else output_name
    output_name = output_name + "_dynamic" if dynamic else output_name
    output_name = output_name + "_metric_" + metric if dynamic else output_name
    output_name = output_name + "_threshold_" + str(threshold)
    output_name = output_name + "_save_rate_" + str(args.save_rate) if dynamic and threshold < 0 else output_name
    output_name = output_name + "_attn_" + args.attn_implementation
    output_name = output_name + "_quantization" if use_quantization else output_name
    output_path = os.path.join(args.output_path, output_name + ".json")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not debug and os.path.exists(output_path) and not watch:
        result = read_json_file(output_path)
    else:
        result = []
    start = len(result) if args.start == -1 else args.start
    print("*" * 10, f"start: {start}", "*" * 10)

    model_path = args.model_path
    if "llama" in model_path.lower():
        replace_llama(method.lower(), args.modify)
    elif "mistral" in model_path.lower():
        replace_mistral(method.lower(), args.modify)
    elif "qwen2" in model_path.lower():
        replace_qwen2(method.lower(), args.modify)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ) if use_quantization else None

    # Set max_gen_len based on dataset
    dataset_to_max_gen_len = {
        "gsm8k": 500, "coqa": 20, "narrativeqa": 20, "qasper": 300,
        "quality": 20, "random_quality": 20, "fix_random_quality": 20,
        "trec": 64, "2wikimqa": 32, "qmsum": 200, "musique": 20,
        "multi_news": 600, "triviaqa": 32, "passage_count": 32, "lcc": 64,
        "gpqa": 32, "mmlu_stem": 32, "theoremqa": 512, "truthfulqa": 32,
    }
    if dataset_name.lower() not in dataset_to_max_gen_len:
        raise ValueError(f"Dataset {dataset_name} not supported")
    max_gen_len = dataset_to_max_gen_len[dataset_name.lower()]

    print(f"Model Using Chat Template: {args.chat}")
    batches = load_dataset(
        model_path, dataset_name, tokenizer, start, args.end, args.batch_size,
        model_max_len=args.model_max_len,
        max_gen_len=max_gen_len if args.max_gen_len == -1 else args.max_gen_len,
        chat=args.chat
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=args.use_cache,
        quantization_config=quantization_config,
        attn_implementation=args.attn_implementation
    )
    model.generation_config.top_p = None
    layers = len(model.model.layers)
    kernel_sizes = 7
    pooling = "maxpool"

    # KV Cache Method Configuration
    if method != "fullkv":
        if not isinstance(kernel_sizes, list):
            kernel_sizes = [kernel_sizes] * layers
        for i in range(layers):
            model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
            model.model.layers[i].self_attn.config.pooling = pooling
            model.model.layers[i].self_attn.config.debug = debug
            model.model.layers[i].self_attn.config.dynamic = dynamic
            model.model.layers[i].self_attn.config.threshold = threshold
            model.model.layers[i].self_attn.config.metric = metric
            model.model.layers[i].self_attn.config.attn_p = args.attn_p
            model.model.layers[i].self_attn.config.normalize = args.normalize
            model.model.layers[i].self_attn.config.save_rate = args.save_rate
            model.model.layers[i].self_attn.config.save_layer = save_layer

    model.eval()

    # Accumulators for timing statistics
    total_prefill_time_acc = 0
    total_decode_time_acc = 0
    total_input_tokens = 0
    total_gen_tokens = 0

    for idx, batch in enumerate(tqdm(batches)):
        tokenized_prompts = batch["tokenized_prompts"]
        batch_input_ids = tokenized_prompts["input_ids"]
        real_budget = 0

        # Budget calculation
        if cache_size != -1:
            max_capacity_prompts = cache_size
        elif budget != 0.0:
            max_capacity_prompts = round(batch_input_ids.shape[1] * budget)
        else:
            raise ValueError("cache_size or budget must be set")

        if method != "fullkv":
            if method.lower() in ["snapkv", "pyramidkv", "h2o", "attn"]:
                window_sizes = 8 if max_capacity_prompts > 8 else 1
            elif args.method.lower() in ["streamingllm"]:
                window_sizes = max_capacity_prompts - 4 if max_capacity_prompts > 4 else 1
            else:
                raise ValueError("method must be in [snapkv, pyramidkv, h2o, streamingllm]")

            if not isinstance(window_sizes, list):
                window_sizes = [window_sizes] * layers
            if not isinstance(max_capacity_prompts, list):
                max_capacity_prompts = [max_capacity_prompts] * layers
            for i in range(layers):
                model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]

        context_length = batch_input_ids.shape[-1]

        # Time measurement: Prefill vs Decode
        time_streamer = TimeStreamer()
        start_time = time.time()

        output = model.generate(
            **tokenized_prompts,
            max_new_tokens=max_gen_len if args.min_length == -1 else max(args.min_length, max_gen_len) + 1,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            min_length=context_length + 1 if args.min_length == -1 else args.min_length,
            eos_token_id=[tokenizer.eos_token_id],
            pad_token_id=tokenizer.eos_token_id,
            streamer=time_streamer
        )
        end_time = time.time()

        if time_streamer.first_token_time is None:
            time_streamer.first_token_time = end_time

        prefill_time = time_streamer.first_token_time - start_time
        decode_time = end_time - time_streamer.first_token_time
        total_time = end_time - start_time
        gen_len = output.shape[-1] - context_length

        # Accumulate statistics
        total_prefill_time_acc += prefill_time
        total_decode_time_acc += decode_time
        total_input_tokens += context_length
        total_gen_tokens += gen_len

        # Compute attention norm
        single_outputs = model(**tokenized_prompts, output_attentions=True)
        attentions = single_outputs.attentions
        norm_list = []
        for attn in attentions:
            last_rows = attn[..., -1, :]
            norms = torch.norm(last_rows, p='fro', dim=-1)
            norms = norms.mean().item()
            norm_list.append(norms)
        avg_norm = sum(norm_list) / len(norm_list)

        batch_outputs = tokenizer.batch_decode(output[..., context_length:].tolist(), skip_special_tokens=True)

        if args.watch:
            peak_memory = max(torch.cuda.max_memory_allocated(), peak_memory)
            layer_prefill_time_sum = 0
            if method != "fullkv":
                for i in range(layers):
                    if model.model.layers[i].self_attn.kv_cluster is not None:
                        layer_prefill_time_sum += model.model.layers[i].self_attn.layer_prefill_time

        if dynamic:
            for i in range(layers):
                if model.model.layers[i].self_attn.kv_cluster is None:
                    real_budget += 1
                else:
                    real_budget = real_budget + model.model.layers[i].self_attn.kv_cluster.real_capacity_ratio
            real_budget = real_budget / layers

        torch.cuda.empty_cache()

        # Save results
        for i in range(len(batch['prompt'])):
            result.append({
                "prompt": batch['prompt'][i],
                "predict": batch_outputs[i],
                "answer": batch['answers'][i],
                "norm": avg_norm,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "gen_len": gen_len,
            })
            if dataset_name not in ['gsm8k', 'coqa', "theoremqa", "random_quality",
                                     "fix_random_quality", "gpqa", "mmlu_stem", "truthfulqa"]:
                result[-1]["all_classes"] = batch['all_classes'][i]
                result[-1]["length"] = batch['length'][i]
            if dynamic:
                result[-1]["real_budget"] = real_budget

        if save_every and idx % save_every == 0:
            with open(output_path, 'w') as file:
                json.dump(result, file, indent=4)

    if not debug:
        with open(output_path, 'w') as file:
            json.dump(result, file, indent=4)

    if args.watch:
        print(f"Peak memory allocated: {peak_memory / (1024 ** 2):.2f} MB")
        avg_prefill = total_prefill_time_acc / len(batches)
        avg_decode = total_decode_time_acc / len(batches)
        print(f"Avg Prefill Time: {avg_prefill:.4f}s")
        print(f"Avg Decode Time: {avg_decode:.4f}s")


if __name__ == "__main__":
    main()
