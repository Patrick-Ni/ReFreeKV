import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamic_metrics import dynamic_attn


# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class PyramidKVCluster:
    def __init__(self, num_hidden_layers=32, window_size=64, max_capacity_prompt=256 + 64, kernel_size=5, pooling='avgpool', beta=20, num_layers=80,
                 debug=False, layer_idx=0, dynamic=False, metric="norm-fro", threshold=0.1):

        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        self.steps = -1
        self.beta = beta

        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.debug = debug
        self.layer_idx = layer_idx
        self.dynamic = dynamic
        self.metric = metric
        self.threshold = threshold
        self.real_capacity_ratio = 0

    def reset(self, window_size=64, max_capacity_prompt=256 + 64, kernel_size=5, pooling='avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):

        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # TODO
        # window_sizes = 32
        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num

        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num

        steps = (max_num - min_num) // self.num_hidden_layers
        max_capacity_prompt = max_num - self.layer_idx * steps

        if self.debug:
            print("-" * 10)
            print(f"Layer-{self.layer_idx} using PyramidKV and has max_capacity_prompt {max_capacity_prompt}")
        if q_len < self.max_capacity_prompt:
            if self.debug:
                print(f"Layer-{self.layer_idx} has key_states {key_states.shape} and value_states {value_states.shape}")
                print("-" * 10 + "\n")
            return key_states, value_states
        elif q_len < (self.max_capacity_prompt - self.window_size) * 2:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim=-2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            if self.dynamic:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                if attention_mask is not None:  # no matter the length, we just slice it
                    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                    attn_weights = attn_weights + causal_mask
                # print(attn_weights[0, :5, :100])
                # print(attn_weights[0, :6, :100])
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                cur_indices_list = []
                all_real_size = 0
                for i in range(bsz):
                    flip_indices = torch.flip(indices[i], dims=[1])
                    configs = dynamic_attn(attn_weights[i, :, :-self.window_size, :-self.window_size], flip_indices, self.metric, self.threshold)
                    real_size = len(configs[0])
                    if i == 0 and self.debug:
                        print("head 0:", configs[0])
                        print("real_size:", real_size)
                    cur_indices_list.append(configs)
                    all_real_size += real_size
                cur_indices = torch.stack(cur_indices_list, dim=0)
                cur_indices = cur_indices.unsqueeze(-1).repeat(1, 1, 1, head_dim)
                k_past_compress = key_states.gather(dim=2, index=cur_indices)
                v_past_compress = value_states.gather(dim=2, index=cur_indices)

                self.real_capacity_ratio = all_real_size / (q_len * bsz)
                if self.debug:
                    print("real capacity radio:", self.real_capacity_ratio)
            else:
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
                v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            if self.debug:
                print(f"Layer-{self.layer_idx} has key_states {key_states.shape} and value_states {value_states.shape}")
                print("-" * 10 + "\n")
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim=-2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(max_capacity_prompt, dim=-1).indices
            if self.dynamic:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                if attention_mask is not None:  # no matter the length, we just slice it
                    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                    attn_weights = attn_weights + causal_mask
                # print(attn_weights[0, :5, :100])
                # print(attn_weights[0, :6, :100])
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                cur_indices_list = []
                all_real_size = 0
                for i in range(bsz):
                    flip_indices = torch.flip(indices[i], dims=[1])
                    configs = dynamic_attn(attn_weights[i, :, :-self.window_size, :-self.window_size], flip_indices, self.metric, self.threshold)
                    real_size = len(configs[0])
                    if i == 0 and self.debug:
                        print("head 0:", configs[0])
                        print("real_size:", real_size)
                    cur_indices_list.append(configs)
                    all_real_size += real_size
                cur_indices = torch.stack(cur_indices_list, dim=0)
                cur_indices = cur_indices.unsqueeze(-1).repeat(1, 1, 1, head_dim)
                k_past_compress = key_states.gather(dim=2, index=cur_indices)
                v_past_compress = value_states.gather(dim=2, index=cur_indices)

                self.real_capacity_ratio = all_real_size / (q_len * bsz)
                if self.debug:
                    print("real capacity radio:", self.real_capacity_ratio)
            else:
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
                v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            if self.debug:
                print(f"Layer-{self.layer_idx} has key_states {key_states.shape} and value_states {value_states.shape}")
                print("-" * 10 + "\n")
            return key_states, value_states


class SnapKVCluster:
    def __init__(self, window_size=64, max_capacity_prompt=256 + 64, kernel_size=5, pooling='avgpool', debug=False, layer_idx=0, dynamic=False,
                 metric="norm-fro", threshold=0.1):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.debug = debug
        self.layer_idx = layer_idx
        self.dynamic = dynamic
        self.metric = metric
        self.threshold = threshold
        self.real_capacity_ratio = 0

    def reset(self, window_size=64, max_capacity_prompt=256 + 64, kernel_size=5, pooling='avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):

        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        if self.debug:
            print("-" * 10)
            print(f"Layer-{self.layer_idx} using SnapKV and has max_capacity_prompt {self.max_capacity_prompt}")

        if q_len < self.max_capacity_prompt:
            if self.debug:
                print(f"Layer-{self.layer_idx} has key_states {key_states.shape} and value_states {value_states.shape}")
                print("-" * 10 + "\n")
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask_with_window = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask_with_window

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim=-2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            if self.dynamic:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                if attention_mask is not None:  # no matter the length, we just slice it
                    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                    attn_weights = attn_weights + causal_mask
                # print(attn_weights[0, :5, :100])
                # print(attn_weights[0, :6, :100])
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                cur_indices_list = []

                for i in range(bsz):
                    flip_indices = torch.flip(indices[i], dims=[1])
                    configs = dynamic_attn(attn_weights[i, :, :-self.window_size, :-self.window_size], flip_indices, self.metric, self.threshold)
                    if i == 0 and self.debug:
                        print("head 0:", configs[0])

                    cur_indices_list.append(configs)

                cur_indices = torch.stack(cur_indices_list, dim=0)
                cur_indices = cur_indices.unsqueeze(-1).repeat(1, 1, 1, head_dim)
                k_past_compress = key_states.gather(dim=2, index=cur_indices)
                v_past_compress = value_states.gather(dim=2, index=cur_indices)

            else:
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
                v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            self.real_capacity_ratio = key_states.shape[2] / q_len
            if self.debug:
                print("real capacity radio:", self.real_capacity_ratio)
                print(f"Layer-{self.layer_idx} has key_states {key_states.shape} and value_states {value_states.shape}")
                print("-" * 10 + "\n")

            return key_states, value_states


class ATTNKVCluster:
    def __init__(self, window_size=64, max_capacity_prompt=256 + 64, debug=False, layer_idx=0, attn_p=-1, normalize=False):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.debug = debug
        self.layer_idx = layer_idx
        self.real_capacity_ratio = 0
        self.attn_p = attn_p
        self.normalize = normalize

    def reset(self, window_size=64, max_capacity_prompt=256 + 64, attn_p=-1):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.attn_p = attn_p
        self.normalize = False

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):

        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if self.debug:
            print("-" * 10)
            print(f"Layer-{self.layer_idx} using ATTN and has max_capacity_prompt {self.max_capacity_prompt}")

        if q_len < self.max_capacity_prompt:
            if self.debug:
                print(f"Layer-{self.layer_idx} has key_states {key_states.shape} and value_states {value_states.shape}")
                print("-" * 10 + "\n")
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
            # print(attn_weights[0, :5, :100])
            # print(attn_weights[0, :6, :100])
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            # attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            # mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            # mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            # mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            # mask = mask.to(attn_weights.device)
            # attention_mask_with_window = mask[None, None, :, :]
            #
            # attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask_with_window
            #
            # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            if self.attn_p == -1:
                attn_cache = attn_weights[:, :, -1, : -self.window_size]
            else:
                low_range = int(attn_weights.shape[-2] * self.attn_p)
                attn_cache = attn_weights[..., -low_range:, : -self.window_size].sum(dim=2)
                if self.normalize:
                    if self.debug:
                        print("attn weights normalized")
                    non_zero_counts = torch.count_nonzero(attn_weights[..., -low_range:, : -self.window_size], dim=2)
                    # print("non_zero_counts: ", non_zero_counts[0, 0, ...])
                    attn_cache = attn_cache / non_zero_counts.float()  # 使用float()确保除法是浮点数除法
            # print(attn_cache.shape)
            # print(attn_cache)
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            if self.debug:
                print("indices:")
                for x in indices[..., :20].tolist()[0]:
                    print(x)
            # print(indices)
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            if self.debug:
                print(f"Layer-{self.layer_idx} has key_states {key_states.shape} and value_states {value_states.shape}")
                print("-" * 10 + "\n")

            return key_states, value_states


class H2OKVCluster:
    def __init__(self, window_size=64, max_capacity_prompt=256 + 64, kernel_size=5, pooling='avgpool', debug=False, layer_idx=0, dynamic=False,
                 threshold=0.01, metric="norm-1", attn_p=-1, normalize=False, save_rate=0.5):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.debug = debug
        self.layer_idx = layer_idx
        self.dynamic = dynamic
        self.threshold = threshold
        self.metric = metric
        self.real_capacity_ratio = 0
        self.attn_p = attn_p
        self.normalize = normalize
        # self.save_train_data = save_train_data
        # self.save_path = save_path
        self.save_rate = save_rate

    def reset(self, window_size=64, max_capacity_prompt=256 + 64, kernel_size=5, pooling='avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):

        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if self.debug:
            print("-" * 10)
            print(f"Layer-{self.layer_idx} using H2O and has max_capacity_prompt {self.max_capacity_prompt}")

        if q_len < self.max_capacity_prompt:
            if self.debug:
                print(f"Layer-{self.layer_idx} has key_states {key_states.shape} and value_states {value_states.shape}")
                print("-" * 10 + "\n")
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask_with_window = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask_with_window

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim=-2)
            attn_cache = attn_weights_sum
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            if self.dynamic:
                # print("key", key_states.shape)
                # print("window", self.window_size)
                assert self.metric == "last_token_attn", f"Remember to fix this bug"
                # if query_states.shape[-2] > 14000:
                attn_weights = torch.matmul(query_states[:, :, -1:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
                # else:
                #     attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                # if attention_mask is not None:  # no matter the length, we just slice it
                #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                #     attn_weights = attn_weights + causal_mask
                # # print(attn_weights[0, :5, :100])
                # # print(attn_weights[0, :6, :100])
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                cur_indices_list = []
                for i in range(bsz):
                    flip_indices = torch.flip(indices[i], dims=[1])
                    # if query_states.shape[-2] > 14000:
                    #     configs = dynamic_attn(attn_weights[i], flip_indices, self.metric, self.threshold,
                    #                            attn_p=self.attn_p, normalize=self.normalize, save_rate=self.save_rate)
                    # else:
                    configs = dynamic_attn(attn_weights[i], flip_indices, self.metric, self.threshold,
                                           attn_p=self.attn_p, normalize=self.normalize, save_rate=self.save_rate)
                    if i == 0 and self.debug:
                        print("head 0:", configs[0])
                    cur_indices_list.append(configs)
                cur_indices = torch.stack(cur_indices_list, dim=0)
                cur_indices = cur_indices.unsqueeze(-1).repeat(1, 1, 1, head_dim)
                k_past_compress = key_states.gather(dim=2, index=cur_indices)
                v_past_compress = value_states.gather(dim=2, index=cur_indices)

            else:
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
                v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            self.real_capacity_ratio = key_states.shape[2] / q_len
            if self.debug:
                print("real capacity radio:", self.real_capacity_ratio)
                print(f"Layer-{self.layer_idx} has key_states {key_states.shape} and value_states {value_states.shape}")
                print("-" * 10 + "\n")
                # import os
                #
                # os.system("nvidia-smi")
            return key_states, value_states


class StreamingLLMKVCluster:
    def __init__(self, window_size=64, max_capacity_prompt=256 + 64, kernel_size=5, pooling='avgpool', debug=False, dynamic=False, layer_idx=0,
                 threshold=0.01, metric="norm-1", attn_p=-1, normalize=False, save_rate=0.5):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.real_capacity_radio = 0
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.debug = debug
        self.layer_idx = layer_idx
        self.dynamic = dynamic
        self.metric = metric
        self.threshold = threshold
        self.real_capacity_ratio = 0
        self.attn_p = attn_p
        self.normalize = normalize
        self.save_rate = save_rate

    def reset(self, window_size=64, max_capacity_prompt=256 + 64, kernel_size=5, pooling='avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if self.debug:
            print("-" * 10)
            print(f"Layer-{self.layer_idx} using StreamingLLM and has max_capacity_prompt {self.max_capacity_prompt}")

        if q_len < self.max_capacity_prompt:
            if self.debug:
                print(f"Layer-{self.layer_idx} has key_states {key_states.shape} and value_states {value_states.shape}")
                print("-" * 10 + "\n")
            return key_states, value_states
        else:
            indices = torch.tensor(range(self.max_capacity_prompt - self.window_size), dtype=torch.int64).to(key_states.device)
            indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
            if self.dynamic:
                # cur_indices = [i for i in range(q_len - self.window_size, q_len)]  # window_size = recent_size
                cur_indices = torch.tensor(range(q_len - self.window_size, q_len), dtype=torch.int64).to(key_states.device)
                cur_indices = cur_indices.unsqueeze(0).repeat(num_heads, 1)  # num_heads, window_size
                assert self.metric == "last_token_attn", f"Remember to fix this bug"
                if self.attn_p == -1:
                    # if save_train_data:
                    #     append_to_pickle_file(save_path, attention_score[:, -1, :])
                    attn_weights = torch.matmul(query_states[:, :, -1:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
                    input_attn_score = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                else:
                    low_range = int(q_len * self.attn_p)
                    attn_weights = torch.matmul(query_states[:, :, -low_range:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    input_attn_score = attn_weights.sum(dim=2)

                    if self.normalize:
                        non_zero_counts = torch.count_nonzero(attn_weights[..., -low_range:, :], dim=2)
                        # print(non_zero_counts)
                        # print(non_zero_counts.shape)
                        # print("non_zero_counts: ", non_zero_counts[0, 0, ...])
                        input_attn_score = input_attn_score / non_zero_counts.float()  # 使用float()确保除法是浮点数除法
                cur_indices_list = []
                for i in range(bsz):
                    configs = dynamic_attn(input_attn_score[i], cur_indices, self.metric, self.threshold, attn_p=self.attn_p,
                                           normalize=self.normalize, save_rate=self.save_rate)
                    if i == 0 and self.debug:
                        print("head 0:", configs[0])
                    cur_indices_list.append(configs)
                min_len = min(x.shape[-1] for x in cur_indices_list)
                cur_indices_list_trimmed = [x[..., :min_len] for x in cur_indices_list]
                # cur_indices = torch.stack(cur_indices_list, dim=0)
                cur_indices = torch.stack(cur_indices_list_trimmed, dim=0)
                cur_indices = cur_indices.unsqueeze(-1).repeat(1, 1, 1, head_dim)
                k_cur = key_states.gather(dim=2, index=cur_indices)
                v_cur = value_states.gather(dim=2, index=cur_indices)
            else:
                k_cur = key_states[:, :, -self.window_size:, :]
                v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            self.real_capacity_ratio = key_states.shape[2] / q_len
            if self.debug:
                print("real capacity radio:", self.real_capacity_ratio)
                print(f"Layer-{self.layer_idx} has key_states {key_states.shape} and value_states {value_states.shape}")
                print("-" * 10 + "\n")
            return key_states, value_states

    def get_real_capacity_ratio(self):
        return self.real_capacity_radio


def init_pyramidkv(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'debug'):
            self.config.debug = False

    self.kv_cluster = PyramidKVCluster(
        num_hidden_layers=num_hidden_layers,
        layer_idx=self.layer_idx,
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        kernel_size=self.config.kernel_size,
        pooling=self.config.pooling,
        debug=self.config.debug,
    )


def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 4096
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'debug'):
            self.config.debug = False

    # if self.layer_idx == 0:
    #     print("Layer_0 confirm using SnapKV")
    # if self.layer_idx == 31:
    #     print("Layer_31 confirm using SnapKV")
    self.kv_cluster = SnapKVCluster(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        kernel_size=self.config.kernel_size,
        pooling=self.config.pooling,
        debug=self.config.debug,
        layer_idx=self.layer_idx,
        dynamic=self.config.dynamic,
        threshold=self.config.threshold,
        metric=self.config.metric,
    )


def init_H2O(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'debug'):
            self.config.debug = False
        if not hasattr(self.config, "attn_p"):
            self.config.attn_p = -1
        if not hasattr(self.config, 'normalize'):
            self.config.normalize = False
        if not hasattr(self.config, 'save_rate'):
            self.config.save_rate = 0.5
        if not hasattr(self.config, "save_layer"):
            self.config.save_layer = [-1]
        # if not hasattr(self.config, 'dynamic_threshold'):
        #     self.config.get_train_data = False
        # if not hasattr(self.config, 'get_train_data'):
        #     self.config.get_train_data = False
        # if not hasattr(self.config, 'save_train_data_path'):
        #     self.config.save_train_data_path = ""
    # if self.config.dynamic_threshold and self.config.target_layer != self.layer_idx:
    #     self.kv_cluster = None
    # else:
    if self.layer_idx in self.config.save_layer:
        self.kv_cluster = None
    else:
        self.kv_cluster = H2OKVCluster(
            window_size=self.config.window_size,
            max_capacity_prompt=self.config.max_capacity_prompt,
            kernel_size=self.config.kernel_size,
            pooling=self.config.pooling,
            debug=self.config.debug,
            layer_idx=self.layer_idx,
            dynamic=self.config.dynamic,
            threshold=self.config.threshold,
            metric=self.config.metric,
            attn_p=self.config.attn_p,
            normalize=self.config.normalize,
            save_rate=self.config.save_rate
            # save_train_data=self.config.get_train_data,
            # save_path=self.config.save_train_data_path
        )


def init_ATTN(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'debug'):
            self.config.debug = False
        if not hasattr(self.config, 'attn_p'):
            self.config.attn_p = -1
        if not hasattr(self.config, 'normalize'):
            self.config.normalize = False

    # if self.layer_idx == 0:
    #     print("Layer_0 confirm using ATTN")
    # if self.layer_idx == 31:
    #     print("Layer_31 confirm using ATTN")
    self.kv_cluster = ATTNKVCluster(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        debug=self.config.debug,
        layer_idx=self.layer_idx,
        attn_p=self.config.attn_p,
        normalize=self.config.normalize
    )


def init_StreamingLLM(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'debug'):
            self.config.debug = False
        if not hasattr(self.config, "attn_p"):
            self.config.attn_p = -1
        if not hasattr(self.config, 'normalize'):
            self.config.normalize = False
        if not hasattr(self.config, 'save_rate'):
            self.config.save_rate = 0.5
        if not hasattr(self.config, "save_layer"):
            self.config.save_layer = [-1]
        # if not hasattr(self.config, 'dynamic_threshold'):
        #     self.config.dynamic = False
        #     self.config.get_train_data = False
        # if not hasattr(self.config, 'get_train_data'):
        #     self.config.get_train_data = False
        # if not hasattr(self.config, 'save_train_data_path'):
        #     self.config.save_train_data_path = ""
    if self.layer_idx in self.config.save_layer:
        self.kv_cluster = None
    else:
        self.kv_cluster = StreamingLLMKVCluster(
            window_size=self.config.window_size,
            max_capacity_prompt=self.config.max_capacity_prompt,
            kernel_size=self.config.kernel_size,
            pooling=self.config.pooling,
            debug=self.config.debug,
            dynamic=self.config.dynamic,
            layer_idx=self.layer_idx,
            threshold=self.config.threshold,
            metric=self.config.metric,
            attn_p=self.config.attn_p,
            normalize=self.config.normalize,
            save_rate=self.config.save_rate
            # save_train_data=self.config.get_train_data,
            # save_path=self.config.save_train_data_path
        )
