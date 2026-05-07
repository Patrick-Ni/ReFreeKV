import os
import pickle

import torch


def append_to_pickle_file(file_path, new_data):
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 读取现有的数据
        with open(file_path, 'rb') as file:
            try:
                existing_data = pickle.load(file)
            except EOFError:
                # 如果文件为空，创建一个新的空列表
                existing_data = []
    else:
        # 如果文件不存在，创建一个新的空列表
        existing_data = []

    # 追加新的数据
    existing_data.append(new_data)

    # 保存更新后的数据
    with open(file_path, 'wb') as file:
        pickle.dump(existing_data, file)


def dist(A, B):
    manhattan_distance = torch.dist(A, B, p=1)
    euclidean_distance = torch.dist(A, B, p=2)
    return manhattan_distance, euclidean_distance


def similarity(A, B):
    assert A.shape == B.shape and len(A.shape) == 1, f"{A.shape} {B.shape}"
    cosine_similarity = torch.nn.functional.cosine_similarity(A.float(), B.float(), dim=0)
    return cosine_similarity


def get_masked_attn_v2(attention_score, config_tensor):
    if config_tensor.device != attention_score.device:
        config_tensor = config_tensor.to(attention_score.device)
    config_tensor.to(attention_score.device)
    head_num, seq_len = attention_score.shape[0], attention_score.shape[-1]
    position_len = config_tensor.shape[-1]
    device = attention_score.device

    mask = torch.zeros(attention_score.shape, dtype=torch.bool).to(device)
    # 生成扩展的索引
    batch_indices = torch.arange(head_num).view(-1, 1).expand(-1, position_len).reshape(-1)
    position_flattened = config_tensor.reshape(-1)

    # 使用高级索引直接设置值
    mask[batch_indices, position_flattened, :] = True
    # mask[:, :, position_flattened] &= True  # 利用in-place操作来更新掩码
    # 利用广播机制生成行和列掩码
    # final_mask = row_mask & col_mask
    # 将掩码应用于attention_score，未被掩码标记的位置将被设置为零
    masked_attention_score = attention_score * mask
    return masked_attention_score


def get_masked_attn(attention_score, config_tensor):
    # 0.9
    # 31x13 +10 = 5 min / sample
    if config_tensor.device != attention_score.device:
        config_tensor = config_tensor.to(attention_score.device)
    head_num, seq_len = attention_score.shape[0], attention_score.shape[-1]
    position_len = config_tensor.shape[-1]
    device = attention_score.device

    # 创建一个全零的mask tensor
    row_mask = torch.zeros(attention_score.shape, dtype=torch.bool).to(device)
    col_mask = torch.zeros(attention_score.shape, dtype=torch.bool).to(device)

    # 生成扩展的索引
    batch_indices = torch.arange(head_num).view(-1, 1).expand(-1, position_len).reshape(-1)
    position_flattened = config_tensor.reshape(-1)

    # 使用高级索引直接设置值
    row_mask[batch_indices, position_flattened, :] = True
    col_mask[batch_indices, :, position_flattened] = True
    # 利用广播机制生成行和列掩码
    final_mask = row_mask & col_mask
    # 将掩码应用于attention_score，未被掩码标记的位置将被设置为零
    masked_attention_score = attention_score * final_mask
    return masked_attention_score


def compare_hidden_state(raw_hidden_state, new_hidden_state):
    dists = dist(raw_hidden_state, new_hidden_state)
    cosine = similarity(raw_hidden_state, new_hidden_state)
    return dists[0], dists[1], cosine


def dynamic_attn(attention_score, position, metric, threshold, return_idx=False, attn_p=-1, normalize=False, save_rate=0.5):
    # always the lowest, the least important
    # attention score: head_num, seq_len, seq_len
    # head_num = attention_score.shape[0]
    # position = [head_num, head_select_num]
    # print("*" * 100)
    # start = datetime.now()
    head_select_num = position.shape[-1]  # 32 x select_num

    if metric == "norm-1":
        p = 1
    elif metric == "norm-2":
        p = 2
    elif metric == "norm-inf":
        p = float("inf")
    elif metric == "norm-fro":
        p = "fro"
    elif metric == "row-square-sum":
        return dynamic_attn_v2(attention_score, position, threshold, 2, return_idx)
    elif metric == "col-square-sum":
        return dynamic_attn_v2(attention_score, position, threshold, 1, return_idx)
    elif metric == "row-sqrt-sum":
        return dynamic_attn_v3(attention_score, position, threshold, 2, return_idx)
    elif metric == "col-sqrt-sum":
        return dynamic_attn_v3(attention_score, position, threshold, 1, return_idx)
    elif metric == "last_token_attn":
        if attn_p == -1:
            # if save_train_data:
            #     append_to_pickle_file(save_path, attention_score[:, -1, :])
            # return last_token_attn(attention_score[:, -1, :], position, threshold, return_idx) if threshold > 0 else dynamic_last_token_attn(
            #     attention_score[:, -1, :], position, save_rate, return_idx)
            return last_token_attn(attention_score[:, -1, :], position, threshold, return_idx)
        else:
            # low_range = int(attention_score.shape[-2] * attn_p)
            # last_n_attention_score = attention_score[..., -low_range:, :].sum(dim=1)
            # if normalize:
            #     non_zero_counts = torch.count_nonzero(attention_score[..., -low_range:, :], dim=1)
            #     # print(non_zero_counts)
            #     # print(non_zero_counts.shape)
            #     # print("non_zero_counts: ", non_zero_counts[0, 0, ...])
            #     last_n_attention_score = last_n_attention_score / non_zero_counts.float()  # 使用float()确保除法是浮点数除法
            #     # print("final last n attention score:")
            #     # print(last_n_attention_score.shape)
            # # if save_train_data:
            # #     append_to_pickle_file(save_path, last_n_attention_score)
            # return last_token_attn(attention_score, position, threshold, return_idx) if threshold > 0 else dynamic_last_token_attn(
            #     attention_score, position, save_rate, return_idx)
            return last_token_attn(attention_score, position, threshold, return_idx)
    else:
        raise ValueError(f"metric {metric} not supported when using attn dynamic")
    # attention_score_masked = compare_attn_score(attention_score, layer_config, return_masked_attention_score=True)[-1]
    # print(layer_config[31])
    # print(attention_score_masked[0, -1, :])
    if head_select_num != attention_score.shape[-1]:
        attention_score = get_masked_attn(attention_score, position)  # h,i,j  i,j position[h]
    # end_get_mask_attention = datetime.now()
    # print(f"get masked attention time: {(end_get_mask_attention - start).total_seconds()}")
    # baseline = [torch.norm(attention_score[i], p=p).item() for i in range(head_num)]
    # baseline = sum(baseline) / len(baseline)
    baseline = torch.norm(attention_score, p=p, dim=(1, 2))
    baseline = torch.mean(baseline).item()
    torch.cuda.empty_cache()
    # baseline = torch.sum(baseline).item() / head_num
    threshold = baseline * threshold
    # end_get_baseline = datetime.now()
    # print(f"get baseline time: {(end_get_baseline - start).total_seconds()}")

    # scores = compare_attn_score([attention_score_masked[0]] * len(new_configs), new_configs, average=False)[metric_index]
    # config = layer_config
    # filtered_scores = [(index, score) for index, score in enumerate(scores) if score <= threshold]
    # if filtered_scores:
    #     highest_score_index, highest_score = max(filtered_scores, key=lambda x: x[1])
    #     config = new_configs[highest_score_index]
    # 0.8 1000 -> 800,
    # token Attention map+ [1x1000] = threshold -> metric (attention map)
    low, high = 1, head_select_num - 1
    best_idx = -1  # 如果没有找到任何一个小于m的字典，返回-1
    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            break
        # attention_score_masked = get_masked_attn(attention_score, new_configs[mid]) 4
        attention_score_masked = get_masked_attn(attention_score, position[:, mid:])
        # end_get_mask_attention = datetime.now()
        # print(f"get masked attention time: {(end_get_mask_attention - start).total_seconds()}")

        score = torch.norm((attention_score - attention_score_masked), p=p, dim=(1, 2))
        # end_get_score_time = datetime.now()
        # print(f"get score time: {(end_get_score_time - start).total_seconds()}")
        score = torch.mean(score).item()
        torch.cuda.empty_cache()
        # end_get_score_time = datetime.now()
        # print(f"get score mean time: {(end_get_score_time - start).total_seconds()}")
        if score <= threshold:
            best_idx = mid  # 记录下当前符合条件的字典的索引
            low = mid + 1  # 移动左边界，尝试找一个更大的、仍然小于m的值
        else:
            high = mid - 1  # 移动右边界，因为当前的k值已经不小于m了
    config = position[:, best_idx:] if best_idx != -1 else position
    if return_idx:
        return config, best_idx
    return config


def dynamic_attn_v2(attention_score, position, threshold, dim_direct=2, return_idx=False):
    # always the lowest, the least important
    # attention score: head_num, seq_len, seq_len
    # head_num = attention_score.shape[0]
    # position = [head_num, head_select_num]
    # print("*" * 100)
    # start = datetime.now()
    head_select_num = position.shape[-1]  # 32 x select_num
    if head_select_num != attention_score.shape[-1]:
        attention_score = get_masked_attn(attention_score, position)  # h,i,j  i,j position[h]
    sum_attention_score = torch.sum(attention_score ** 2, dim=dim_direct)
    torch.cuda.empty_cache()
    selected_attention_score = sum_attention_score.gather(1, position)
    baseline = torch.sum(selected_attention_score)
    # print("baseline:", baseline.item())
    csum_selected_attention_score = torch.flip(torch.cumsum(torch.flip(selected_attention_score, dims=[1]), dim=1), dims=[1])
    # print("csum_selected_attention_score:", csum_selected_attention_score, csum_selected_attention_score.shape)
    sum_selected_attention_score = torch.sum(csum_selected_attention_score, dim=0)  # 对所有样本求和
    # print("sum_selected_attention_score:", sum_selected_attention_score, sum_selected_attention_score.shape)
    ratio_diff = (baseline - sum_selected_attention_score) / baseline
    # print("ratio_diff:", ratio_diff)
    valid_indices = torch.where(ratio_diff <= threshold)[0]
    if len(valid_indices) > 0:
        max_position_index = torch.max(valid_indices)
    else:
        max_position_index = -1  # 如果没有满足条件的index，返回-1
    config = position[:, max_position_index:] if max_position_index != -1 else position
    if return_idx:
        return config, max_position_index
    return config


def expand(A, k):
    # 原始张量
    # 原始张量
    m, n = A.shape
    # 在每行的末尾添加三个4，以确保每个元素都可以形成一个长度为4的窗口
    padded_x = torch.cat((A, torch.full((m, n - 1), k).to(A.device)), dim=1)

    # 使用unfold创建滑动窗口
    result = padded_x.unfold(1, n, 1)
    return result


def dynamic_attn_v3(attention_score, position, threshold, dim_direct=2, return_idx=False):
    # always the lowest, the least important
    # attention score: head_num, seq_len, seq_len
    # head_num = attention_score.shape[0]
    # position = [head_num, head_select_num]
    # print("*" * 100)
    # start = datetime.now()
    head_select_num = position.shape[-1]  # 32 x select_num
    if head_select_num != attention_score.shape[-1]:
        attention_score = get_masked_attn(attention_score, position)  # h,i,j  i,j position[h]
    # print(attention_score)
    # end_get_mask_attention = datetime.now()
    # print(f"get masked attention time: {(end_get_mask_attention - start).total_seconds()}")
    # baseline = [torch.norm(attention_score[i], p=p).item() for i in range(head_num)]
    # baseline = sum(baseline) / len(baseline)
    baseline = torch.norm(attention_score, p="fro", dim=(1, 2))
    baseline = torch.sum(baseline).item()
    # print("baseline:", baseline)
    torch.cuda.empty_cache()

    square_sum_attention_score = torch.sum(attention_score ** 2, dim=dim_direct)
    # print("square_sum_attention_score", square_sum_attention_score)
    torch.cuda.empty_cache()
    selected_attention_score = square_sum_attention_score.gather(1, position)
    # print("selected_attention_score", selected_attention_score)
    csum_selected_attention_score = torch.flip(torch.cumsum(torch.flip(selected_attention_score, dims=[1]), dim=1), dims=[1])
    # print("csum_selected_attention_score", csum_selected_attention_score)
    sum_selected_attention_score = torch.sum(torch.sqrt(csum_selected_attention_score), dim=0)  # 对所有样本求和
    # print("sum_selected_attention_score", sum_selected_attention_score)
    ratio_diff = (baseline - sum_selected_attention_score) / baseline
    # print("ratio_diff", ratio_diff)
    valid_indices = torch.where(ratio_diff <= threshold)[0]
    if len(valid_indices) > 0:
        max_position_index = torch.max(valid_indices)
    else:
        max_position_index = -1  # 如果没有满足条件的index，返回-1
    config = position[:, max_position_index:] if max_position_index != -1 else position
    if return_idx:
        return config, max_position_index
    return config


def last_token_attn(last_token_attention_score, position, threshold, return_idx=False):
    # always the lowest, the least important
    # attention score: head_num, seq_len
    # head_num = attention_score.shape[0]
    # position = [head_num, head_select_num]
    # os.system("nvidia-smi")
    # print("*" * 100)
    # print(last_token_attention_score)
    last_token_attention_score = last_token_attention_score.float()  # avoid overflow, fp32
    # print("last_token_attn_score:", last_token_attention_score.shape)
    # os.system("nvidia-smi")
    head_num, head_select_num = position.shape  # 32 x select_num
    if threshold < 0:
        mask_portion = -threshold
        random_select_num = int(head_select_num * mask_portion)

        # 1. 生成随机索引，形状: [head_num, random_select_num]
        indices = torch.randint(0, head_select_num, (head_num, random_select_num)).to(position.device)

        # 2. 沿维度 1 (列) 提取数据
        # position 的形状是 [head_num, head_select_num]
        config = torch.gather(position, 1, indices)
        return config
    if head_select_num != last_token_attention_score.shape[-1]:
        mask = torch.zeros_like(last_token_attention_score, dtype=torch.bool)
        rows = torch.arange(head_num).unsqueeze(1).expand(head_num, head_select_num)
        mask[rows, position] = True
        last_token_attention_score = last_token_attention_score * mask
    # print("last_token_attention_score", last_token_attention_score)
    baseline = torch.norm(last_token_attention_score, p="fro", dim=1).mean()
    # print("baseline:", baseline)
    # os.system("nvidia-smi")

    square_last_token_attention_score = last_token_attention_score ** 2
    # print("square_last_token_attention_score", square_last_token_attention_score)
    # os.system("nvidia-smi")

    selected_attention_score = square_last_token_attention_score.gather(1, position)

    # print("selected_attention_score", selected_attention_score)
    # os.system("nvidia-smi")
    csum_selected_attention_score = torch.flip(torch.cumsum(torch.flip(selected_attention_score, dims=[1]), dim=1), dims=[1])

    # os.system("nvidia-smi")
    sum_selected_attention_score = torch.mean(torch.sqrt(csum_selected_attention_score), dim=0)  # 对所有样本求和

    # print("sum_selected_attention_score", sum_selected_attention_score)
    # os.system("nvidia-smi")
    ratio_diff = (baseline - sum_selected_attention_score) / baseline

    # print("ratio_diff", ratio_diff)
    # os.system("nvidia-smi")

    valid_indices = torch.where(ratio_diff <= threshold)[0]

    if len(valid_indices) > 0:
        max_position_index = torch.max(valid_indices)
    else:
        max_position_index = -1  # 如果没有满足条件的index，返回-1
    config = position[:, max_position_index:] if max_position_index != -1 else position
    if return_idx:
        return config, max_position_index

    return config


def dynamic_last_token_attn(last_token_attention_score, position, save_rate=0.5, return_idx=False):
    # always the lowest, the least important
    # attention score: head_num, seq_len
    # head_num = attention_score.shape[0]
    # position = [head_num, head_select_num]
    # os.system("nvidia-smi")
    # print("*" * 100)
    # print(last_token_attention_score)
    last_token_attention_score = last_token_attention_score.float()  # avoid overflow, fp32

    head_num, head_select_num = position.shape  # 32 x select_num
    if head_select_num != last_token_attention_score.shape[-1]:
        mask = torch.zeros_like(last_token_attention_score, dtype=torch.bool)
        rows = torch.arange(head_num).unsqueeze(1).expand(head_num, head_select_num)
        mask[rows, position] = True
        last_token_attention_score = last_token_attention_score * mask

    baseline = torch.sum(last_token_attention_score) * save_rate
    selected_attention_score = last_token_attention_score.gather(1, position)
    # csum_selected_attention_score = torch.flip(torch.cumsum(torch.flip(selected_attention_score, dims=[1]), dim=1), dims=[1])
    # csum_selected_attention_score = torch.cumsum(selected_attention_score, dim=1, reverse=True)
    csum = torch.cumsum(selected_attention_score, dim=1)
    csum_selected_attention_score = csum[:, -1:] - csum + selected_attention_score
    sum_selected_attention_score = torch.sum(csum_selected_attention_score, dim=0)  # 对所有样本求和

    valid_indices = torch.where(sum_selected_attention_score >= baseline)[0]
    if len(valid_indices) > 0:
        max_position_index = torch.max(valid_indices)
    else:
        max_position_index = -1  # 如果没有满足条件的index，返回-1

    config = position[:, max_position_index:] if max_position_index != -1 else position
    if return_idx:
        return config, max_position_index

    return config
