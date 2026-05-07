import transformers
from transformers.models.llama.modeling_llama import (
    LlamaAttention
)

from .llama_model import llama_attn_forward_PyramidKV, llama_attn_forward_H2O, llama_attn_forward_SnapKV, llama_attn_forward_StreamingLLM, \
    llama_attn_forward_ATTN, llama_flash_attn2_forward_StreamingLLM, llama_flash_attn2_forward_H2O, llama_flash_attn2_forward_SnapKV
from .llama_model import prepare_inputs_for_generation_llama, prepare_inputs_for_modified_generation_llama
from .mistral_model import mistral_attn_forward_H2O, mistral_attn_forward_SnapKV, mistral_attn_forward_StreamingLLM, \
    prepare_inputs_for_generation_mistral, prepare_inputs_for_modified_generation_mistral, mistral_flash_attn2_forward_StreamingLLM, \
    mistral_flash_attn2_forward_H2O, mistral_flash_attn2_forward_SnapKV, mistral_attn_forward_PyramidKV, mistral_flash_attn2_forward_PyramidKV
from .qwen_model import qwen2_flash_attn2_forward_StreamingLLM, qwen2_attn_forward_H2O, qwen2_attn_forward_Snapkv, qwen2_attn_forward_StreamingLLM, \
    qwen2_flash_attn2_forward_Snapkv, qwen2_flash_attn2_forward_H2O, prepare_inputs_for_modified_generation_qwen, prepare_inputs_for_generation_qwen


def replace_llama(method, modified=False):
    if method == "pyramidkv":
        print("Using PyramidKV!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_PyramidKV
    elif method == "streamingllm":
        print("Using StreamingLLM!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_StreamingLLM
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_StreamingLLM

    elif method == "h2o":
        print("Using H2O!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_H2O
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_H2O

    elif method == "snapkv":
        print("Using SnapKV!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_SnapKV
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SnapKV

    elif method == "attn":
        print("Using Attn!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_ATTN
    else:
        print("Using FullKV!")

    if method not in ["fullkv"]:
        print("Using prepare_inputs_for_generation_llama")
        if modified:
            print("modify generation llama")
            transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_modified_generation_llama
        else:
            transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_modified_generation_llama


def replace_mistral(method, modified=False):
    if method == "streamingllm":
        print("Using StreamingLLM!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_StreamingLLM
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_StreamingLLM

    elif method == "h2o":
        print("Using H2O!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_H2O
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_H2O

    elif method == "snapkv":
        print("Using SnapKV!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_SnapKV
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_SnapKV
    elif method == "pyramidkv":
        print("Using PyramidKV!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_PyramidKV
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_PyramidKV
    else:
        print("Using FullKV!")

    if method not in ["fullkv"]:
        if modified:
            print("modify generation mistral")
            transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_modified_generation_mistral
        else:
            transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_modified_generation_mistral


def replace_qwen2(method, modified=False):
    if method == "h2o":
        print("Using H2O!")
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_attn_forward_H2O
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = qwen2_flash_attn2_forward_H2O
    elif method == "snapkv":
        print("Using SnapKV!")
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_attn_forward_Snapkv
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = qwen2_flash_attn2_forward_Snapkv
    elif method == "streamingllm":
        print("Using StreamingLLM!")
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_attn_forward_StreamingLLM
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = qwen2_flash_attn2_forward_StreamingLLM
    else:
        print("Using FullKV")

    if method not in ["fullkv"]:
        if modified:
            print("modify generation qwen2")
            transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_modified_generation_qwen
        else:
            transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_modified_generation_qwen
