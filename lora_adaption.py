
import argparse
import json
import os
import sys
import numpy as np

import torch
import torch.nn.functional as F
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

from peft import load_peft_weights, LoraConfig
from safetensors.torch import save_file as safe_save_file
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    LlamaTokenizer, 
    AutoModel,
    get_linear_schedule_with_warmup,
    set_seed, 
) # noqa: F402
from scipy.optimize import linear_sum_assignment


def low_rank_decompose(matrix, rank, flag=False):
    """
    matrix: (n_vocab, n_vocab)
    """
    if not flag:
        U, S, V = torch.svd(matrix)
        Ur = U[:, :rank]
        Sr = torch.diag(torch.sqrt(S[:rank]))
        Vr = V[:, :rank]
        
        # 计算 B 和 A
        B = torch.matmul(Ur, Sr)
        A = torch.matmul(Sr, Vr.t())  # 使用转置 Vr.t()
        
        return B, A, S[:rank]
    else:
        U, S, V = torch.svd(matrix)
        Ur = U[:, :rank].contiguous()
        Sr = S[:rank].view(-1,1).contiguous()
        Vr = V[:, :rank].t().contiguous()
        
        return Ur, Sr, Vr


def cosine_similarity(qk1, qk2):
    # Ensure the tensors are 1D
    # qk1 = qk1.view(-1)
    # qk2 = qk2.view(-1)
    
    # Compute the dot product
    dot_product = torch.sum(qk1 * qk2)
    
    # Compute the L2 norms
    norm_qk1 = torch.norm(qk1, p=2)
    norm_qk2 = torch.norm(qk2, p=2)
    
    # Compute the cosine similarity
    cosine_sim = dot_product / (norm_qk1 * norm_qk2)
    return cosine_sim



def repeat_kv(project_matrix: torch.Tensor, head_dim: int, n_rep: int) -> torch.Tensor:
    """
    The project_matrix go from (num_key_value_heads * head_dim, hidden_size) to (num_attention_heads * head_dim, hidden_size)
    """
    num_kv_heads, hidden_size = project_matrix.shape
    num_key_value_heads = num_kv_heads // head_dim

    if n_rep == 1:
        return project_matrix
    project_matrix = project_matrix.view(-1, head_dim, hidden_size)
    # project_matrix = project_matrix.repeat(n_rep, 1, 1)
    project_matrix = torch.repeat_interleave(project_matrix, dim=0, repeats=n_rep)
    return project_matrix.view(-1, hidden_size)

def embedding_transformation(tokenizer1, tokenizer2, embedding1, embedding2, flag=True):
    vocab1 = tokenizer1.get_vocab()
    vocab2 = tokenizer2.get_vocab()

    # 找到交集的token
    common_tokens = set(vocab1.keys()).intersection(set(vocab2.keys()))

    # 提取交集词汇的索引
    indices1 = [vocab1[token] for token in common_tokens]
    indices2 = [vocab2[token] for token in common_tokens]

    # 从嵌入矩阵中提取交集词汇对应的嵌入
    embedding_common1 = embedding1[indices1]  # (num_common_tokens, hidden_size1)
    embedding_common2 = embedding2[indices2]  # (num_common_tokens, hidden_size2)

    # 计算伪逆
    embedding_common1_pinv = torch.linalg.pinv(embedding_common1)  # (hidden_size1, num_common_tokens)

    # 计算转化矩阵
    if flag:
        print("******Transform using matrix inverse.******")
        transformation_matrix = torch.matmul(embedding_common1_pinv, embedding_common2)  # (hidden_size1, hidden_size2)
    else:
        # 计算转化矩阵：使用最小二乘法求解
        transformation_matrix, _ = torch.lstsq(embedding_common2, embedding_common1)


    return transformation_matrix


def max_layer_similarity_mapping(A):
    # 获取矩阵A的大小
    n, m = A.shape
    if n > m:
        A = A.T
        n, m = m, n
    
    # 初始化dp数组，用于存储最大相似度之和
    dp = np.full((n, m), -float('inf'))
    # 初始化路径数组，用于存储路径信息，方便后续回溯找到具体的映射关系
    path = np.zeros((n, m), dtype=int)
    
    # 初始化dp第一层
    for j in range(m):
        if j <= m - n:
            dp[0][j] = A[0][j]
    
    # 填充dp数组
    for i in range(1, n):
        for j in range(i, m):  # j必须至少是i，才能保证顺序
            if j - i <= m - n:  # 确保总偏移不超过m-n
                max_value = -float('inf')
                max_index = -1
                # 寻找最大值和对应的索引
                for k in range(i-1, j):
                    if dp[i-1][k] + A[i][j] > max_value:
                        max_value = dp[i-1][k] + A[i][j]
                        max_index = k
                dp[i][j] = max_value
                path[i][j] = max_index
    
    # 回溯找到最大值路径
    max_sum = -float('inf')
    last_index = -1
    for j in range(n-1, m):
        if dp[n-1][j] > max_sum:
            max_sum = dp[n-1][j]
            last_index = j
    
    # 根据path数组找到具体的映射关系
    mapping = [0] * n
    for i in range(n-1, -1, -1):
        mapping[i] = last_index
        last_index = path[i][last_index]
    
    return mapping, max_sum


def max_head_similarity_mapping(A):
    # 获取矩阵A的大小
    n, m = A.shape
    
    # 检查n是否小于等于m
    if n > m:
        raise ValueError("行数不能大于列数，无法为每个attention head找到不重复的映射。")
    
    # 使用匈牙利算法求解最大加和匹配问题
    # 由于linear_sum_assignment求解的是最小化问题，因此需要对A取反
    cost_matrix = -A
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 计算最大相似度之和
    max_sum = A[row_ind, col_ind].sum()
    
    return col_ind, max_sum


def load_model(model_path, device_id) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = model_path

    tokenizer = AutoTokenizer.from_pretrained(base_model, device_map=device_id,)
    
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype= torch.float16 if "Llama" in base_model else "auto",
            device_map=device_id,
            trust_remote_code=True,
        ) 
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

    # Return layers
    num_layer = model.config.num_hidden_layers
    return tokenizer, model.state_dict(), num_layer

def tensor_inverse(A, type="left"):
    if type == "left":
        # A^T A
        AtA = torch.matmul(A.T, A)
        # (A^T A)^-1
        AtA_inv = torch.inverse(AtA)
        # (A^T A)^-1 A^T
        A_inv = torch.matmul(AtA_inv, A.T)
    elif type == "right":
        # A A^T
        AAt = torch.matmul(A, A.T)
        # (A A^T)^-1
        AAt_inv = torch.inverse(AAt)
        # A^T (A A^T)^-1
        A_inv = torch.matmul(A.T, AAt_inv)
    return A_inv

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    parser.add_argument('--new_model', type=str, required=True)
    parser.add_argument('--old_model', type=str, required=True)
    parser.add_argument('--old_lora_path', type=str, required=True)
    # parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--layer_method', type=str, default="xtransform")
    parser.add_argument('--attention_method', type=str, default="xtransform")
    parser.add_argument('--sim_metric', type=str, default="CKA")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    
    A = torch.load(f"./tmp/{args.old_model.split('/')[-2]}_{args.new_model.split('/')[-2]}_{args.sim_metric}.pt").cpu().numpy()
    # The following layer_methods only work for the MiniCPM.
    if args.layer_method == "xtransform":
        layer_mapping, _ = max_layer_similarity_mapping(A)
    elif args.layer_method == "first":
        layer_mapping = [i for i in range(A.shape[1])]
    elif args.layer_method == "medium":
        layer_mapping = [i for i in range(7, 47, 1)]
    elif args.layer_method == "last":
        layer_mapping = [i for i in range(A.shape[0]-A.shape[1], A.shape[0], 1)]

    old_lora_weights = load_peft_weights(args.old_lora_path)
    with open(args.old_model + "config.json", "r") as f:
        data = json.load(f)
        if args.old_model.split('/')[-2] == "bloom-560m":
            old_hidden_size, old_hidden_layers = data['n_embed'], data['n_layer']
        else:
            old_hidden_size, old_hidden_layers = data['hidden_size'], data['num_hidden_layers']
        old_num_attention_heads = data['num_attention_heads']
        old_head_size = old_hidden_size // old_num_attention_heads
        if "MiniCPM" in args.old_model.split('/')[-2]:
            old_num_key_value_heads = data["num_key_value_heads"]
            old_num_rep = old_num_attention_heads // old_num_key_value_heads

    with open(args.new_model + "config.json", "r") as f:
        data = json.load(f)
        if args.new_model.split('/')[-2] == "bloomz-1b1":
            new_hidden_size, new_hidden_layers = data['n_embed'], data['n_layer']
        else:
            new_hidden_size, new_hidden_layers = data['hidden_size'], data['num_hidden_layers']
        new_num_attention_heads = data['num_attention_heads']
        new_head_size = new_hidden_size // new_num_attention_heads
        if args.new_model.split('/')[-2] == "Qwen2.5-3B" or "Meta-Llama-3-8B" in args.new_model.split('/')[-2]:
            new_num_key_value_heads = data["num_key_value_heads"]
            new_num_rep = old_num_attention_heads // new_num_key_value_heads
            
    with open(args.old_lora_path + "adapter_config.json", "r") as f:
        data = json.load(f)
        old_rank = data['r']
        target_modules = data['target_modules']
    
    new_lora_weights = {}
        
    old_tokenizer, old_model_dic, _ = load_model(args.old_model, "cuda:0")
    new_tokenizer, new_model_dic, _ = load_model(args.new_model, "cuda:0")
    if (args.old_model.split('/')[-2] == "MiniCPM-S-1B-sft-llama-format" or 
         "Meta-Llama-3-8B" in args.new_model.split('/')[-2]
        ):    
        E1, E2 = old_model_dic["model.embed_tokens.weight"].to(device="cpu", dtype=torch.float32), new_model_dic["model.embed_tokens.weight"].to(device="cpu", dtype=torch.float32)
        W_x = embedding_transformation(old_tokenizer, new_tokenizer, E1, E2)
        del E1, E2
    elif args.old_model.split('/')[-2] == "bloom-560m":
        E1, E2 = old_model_dic["transformer.word_embeddings.weight"].to(device="cpu", dtype=torch.float32), new_model_dic["transformer.word_embeddings.weight"].to(device="cpu", dtype=torch.float32)
        W_x = embedding_transformation(old_tokenizer, new_tokenizer, E1, E2)
        del E1, E2
    elif args.old_model.split('/')[-2] == "pythia-1b" or args.new_model.split('/')[-2] == "Qwen2.5-3B":
        W_x = torch.eye(old_hidden_size).to(device="cpu", dtype=torch.float32)
    del old_tokenizer, new_tokenizer

    for i in range(min(old_hidden_layers, new_hidden_layers)):
        if args.old_model.split('/')[-2] == "pythia-1b":
            print(f"Layer {i} is mapping to Layer {layer_mapping[i]}.")
            W_old_QKV = old_model_dic[f"gpt_neox.layers.{i}.attention.query_key_value.weight"].to(device="cpu", dtype=torch.float32)
            W_old_O = old_model_dic[f"gpt_neox.layers.{i}.attention.dense.weight"].to(device="cpu", dtype=torch.float32)

            delta_W_QKV = old_lora_weights[f"base_model.model.gpt_neox.layers.{i}.attention.query_key_value.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.gpt_neox.layers.{i}.attention.query_key_value.lora_A.weight"].to(device="cpu", dtype=torch.float32)
            delta_W_O = old_lora_weights[f"base_model.model.gpt_neox.layers.{i}.attention.dense.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.gpt_neox.layers.{i}.attention.dense.lora_A.weight"].to(device="cpu", dtype=torch.float32)
        
        elif args.old_model.split('/')[-2] == "bloom-560m":
            print(f"Layer {i} is mapping to Layer {layer_mapping[i]}.")
            W_old_QKV = old_model_dic[f"transformer.h.{i}.self_attention.query_key_value.weight"].to(device="cpu", dtype=torch.float32)
            W_old_O = old_model_dic[f"transformer.h.{i}.self_attention.dense.weight"].to(device="cpu", dtype=torch.float32)
            if "dense_h_to_4h" in target_modules:
                W_old_U = old_model_dic[f"transformer.h.{i}.mlp.dense_h_to_4h.weight"].to(device="cpu", dtype=torch.float32)
                
            if "dense_4h_to_h" in target_modules:
                W_old_D = old_model_dic[f"transformer.h.{i}.mlp.dense_4h_to_h.weight"].to(device="cpu", dtype=torch.float32)
            
            delta_W_QKV = old_lora_weights[f"base_model.model.transformer.h.{i}.self_attention.query_key_value.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.transformer.h.{i}.self_attention.query_key_value.lora_A.weight"].to(device="cpu", dtype=torch.float32)
            delta_W_O = old_lora_weights[f"base_model.model.transformer.h.{i}.self_attention.dense.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.transformer.h.{i}.self_attention.dense.lora_A.weight"].to(device="cpu", dtype=torch.float32)
        
        elif args.old_model.split('/')[-2] == "MiniCPM-S-1B-sft-llama-format":
            print(f"Layer {layer_mapping[i]} is mapping to Layer {i}.")

            W_old_Q = old_model_dic[f"model.layers.{layer_mapping[i]}.self_attn.q_proj.weight"].to(device="cpu", dtype=torch.float32)
            # https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16/blob/main/modeling_minicpm.py#L286
            W_old_K = repeat_kv(old_model_dic[f"model.layers.{layer_mapping[i]}.self_attn.k_proj.weight"].to(device="cpu", dtype=torch.float32), old_head_size, old_num_rep)
            W_old_V = repeat_kv(old_model_dic[f"model.layers.{layer_mapping[i]}.self_attn.v_proj.weight"].to(device="cpu", dtype=torch.float32), old_head_size, old_num_rep)
            W_old_O = old_model_dic[f"model.layers.{layer_mapping[i]}.self_attn.o_proj.weight"].to(device="cpu", dtype=torch.float32)
            if "up_proj" in target_modules:    
                W_old_U = old_model_dic[f"model.layers.{layer_mapping[i]}.mlp.up_proj.weight"].to(device="cpu", dtype=torch.float32)
            if "down_proj" in target_modules:
                W_old_D = old_model_dic[f"model.layers.{layer_mapping[i]}.mlp.down_proj.weight"].to(device="cpu", dtype=torch.float32)

            if "adalora" in args.old_lora_path:
                print(f"Adalora Transformation")
                delta_W_Q = old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.q_proj.lora_B"].to(device="cpu", dtype=torch.float32) @ (old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.q_proj.lora_A"].to(device="cpu", dtype=torch.float32) * old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.q_proj.lora_E"].to(device="cpu", dtype=torch.float32))
                delta_W_K = repeat_kv(old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.k_proj.lora_B"].to(device="cpu", dtype=torch.float32) @ (old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.k_proj.lora_A"].to(device="cpu", dtype=torch.float32) * old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.k_proj.lora_E"].to(device="cpu", dtype=torch.float32)), old_head_size, old_num_rep)
                delta_W_V = repeat_kv(old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.v_proj.lora_B"].to(device="cpu", dtype=torch.float32) @ (old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.v_proj.lora_A"].to(device="cpu", dtype=torch.float32) * old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.v_proj.lora_E"].to(device="cpu", dtype=torch.float32)), old_head_size, old_num_rep)
                delta_W_O = old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.o_proj.lora_B"].to(device="cpu", dtype=torch.float32) @ (old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.o_proj.lora_A"].to(device="cpu", dtype=torch.float32) * old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.o_proj.lora_E"].to(device="cpu", dtype=torch.float32))
            else:
                delta_W_Q = old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.q_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.q_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32)
                delta_W_K = repeat_kv(old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.k_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.k_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32), old_head_size, old_num_rep)
                delta_W_V = repeat_kv(old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.v_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.v_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32), old_head_size, old_num_rep)
                delta_W_O = old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.o_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.o_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32)

        elif (args.old_model.split('/')[-2] == "Qwen1.5-1.8B") or ("Llama-2-7b" in args.old_model.split('/')[-2]):
            print(f"Layer {i} is mapping to Layer {layer_mapping[i]}.")

            W_old_Q = old_model_dic[f"model.layers.{i}.self_attn.q_proj.weight"].to(device="cpu", dtype=torch.float32)
            W_old_K = old_model_dic[f"model.layers.{i}.self_attn.k_proj.weight"].to(device="cpu", dtype=torch.float32)
            W_old_V = old_model_dic[f"model.layers.{i}.self_attn.v_proj.weight"].to(device="cpu", dtype=torch.float32)
            W_old_O = old_model_dic[f"model.layers.{i}.self_attn.o_proj.weight"].to(device="cpu", dtype=torch.float32)
            
            delta_W_Q = old_lora_weights[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32)
            delta_W_K = old_lora_weights[f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32)
            delta_W_V = old_lora_weights[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32)
            delta_W_O = old_lora_weights[f"base_model.model.model.layers.{i}.self_attn.o_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{i}.self_attn.o_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32)

            if "up_proj" in target_modules:    
                W_old_U = old_model_dic[f"model.layers.{i}.mlp.up_proj.weight"].to(device="cpu", dtype=torch.float32)
            if "down_proj" in target_modules:
                W_old_D = old_model_dic[f"model.layers.{i}.mlp.down_proj.weight"].to(device="cpu", dtype=torch.float32)



        if args.new_model.split('/')[-2] == "pythia-1.4b":
            W_new_QKV = new_model_dic[f"gpt_neox.layers.{layer_mapping[i]}.attention.query_key_value.weight"].to(device="cpu", dtype=torch.float32)
            W_new_O = new_model_dic[f"gpt_neox.layers.{layer_mapping[i]}.attention.dense.weight"].to(device="cpu", dtype=torch.float32)

            new_delta_W_QKV = torch.zeros(W_new_QKV.shape[0], W_new_QKV.shape[1]).to(device="cpu", dtype=torch.float32)
            new_delta_W_O = torch.zeros(W_new_O.shape[0], W_new_O.shape[1]).to(device="cpu", dtype=torch.float32)
        
        elif args.new_model.split('/')[-2] == "bloomz-1b1":
            W_new_QKV = new_model_dic[f"transformer.h.{layer_mapping[i]}.self_attention.query_key_value.weight"].to(device="cpu", dtype=torch.float32)
            W_new_O = new_model_dic[f"transformer.h.{layer_mapping[i]}.self_attention.dense.weight"].to(device="cpu", dtype=torch.float32)

            if "dense_h_to_4h" in target_modules:
                W_new_U = new_model_dic[f"transformer.h.{layer_mapping[i]}.mlp.dense_h_to_4h.weight"].to(device="cpu", dtype=torch.float32)

            if "dense_4h_to_h" in target_modules:
                W_new_D = new_model_dic[f"transformer.h.{layer_mapping[i]}.mlp.dense_4h_to_h.weight"].to(device="cpu", dtype=torch.float32)

            new_delta_W_QKV = torch.zeros(W_new_QKV.shape[0], W_new_QKV.shape[1]).to(device="cpu", dtype=torch.float32)
            new_delta_W_O = torch.zeros(W_new_O.shape[0], W_new_O.shape[1]).to(device="cpu", dtype=torch.float32)
        elif args.new_model.split('/')[-2] == "MiniCPM-2B-sft-fp32-llama-format":
            W_new_Q = new_model_dic[f"model.layers.{i}.self_attn.q_proj.weight"].to(device="cpu", dtype=torch.float32)
            W_new_K = new_model_dic[f"model.layers.{i}.self_attn.k_proj.weight"].to(device="cpu", dtype=torch.float32)
            W_new_V = new_model_dic[f"model.layers.{i}.self_attn.v_proj.weight"].to(device="cpu", dtype=torch.float32)
            W_new_O = new_model_dic[f"model.layers.{i}.self_attn.o_proj.weight"].to(device="cpu", dtype=torch.float32)
            if "up_proj" in target_modules:    
                W_new_U = new_model_dic[f"model.layers.{i}.mlp.up_proj.weight"].to(device="cpu", dtype=torch.float32)
            if "down_proj" in target_modules:
                W_new_D = new_model_dic[f"model.layers.{i}.mlp.down_proj.weight"].to(device="cpu", dtype=torch.float32)

            new_delta_W_Q = torch.zeros(W_new_Q.shape[0], W_new_Q.shape[1]).to(device="cpu", dtype=torch.float32)
            new_delta_W_K = torch.zeros(W_new_K.shape[0], W_new_K.shape[1]).to(device="cpu", dtype=torch.float32)
            new_delta_W_V = torch.zeros(W_new_V.shape[0], W_new_V.shape[1]).to(device="cpu", dtype=torch.float32)
            new_delta_W_O = torch.zeros(W_new_O.shape[0], W_new_O.shape[1]).to(device="cpu", dtype=torch.float32)
        elif args.new_model.split('/')[-2] == "Qwen2.5-3B" or  "Meta-Llama-3-8B" in args.new_model.split('/')[-2]:
            W_new_Q = new_model_dic[f"model.layers.{layer_mapping[i]}.self_attn.q_proj.weight"].to(device="cpu", dtype=torch.float32)
            W_new_K = repeat_kv(new_model_dic[f"model.layers.{layer_mapping[i]}.self_attn.k_proj.weight"].to(device="cpu", dtype=torch.float32), new_head_size, new_num_rep)
            W_new_V = repeat_kv(new_model_dic[f"model.layers.{layer_mapping[i]}.self_attn.v_proj.weight"].to(device="cpu", dtype=torch.float32), new_head_size, new_num_rep)
            W_new_O = new_model_dic[f"model.layers.{layer_mapping[i]}.self_attn.o_proj.weight"].to(device="cpu", dtype=torch.float32)

            new_delta_W_Q = torch.zeros(W_new_Q.shape[0], W_new_Q.shape[1]).to(device="cpu", dtype=torch.float32)
            new_delta_W_K = torch.zeros(W_new_K.shape[0], W_new_K.shape[1]).to(device="cpu", dtype=torch.float32)
            new_delta_W_V = torch.zeros(W_new_V.shape[0], W_new_V.shape[1]).to(device="cpu", dtype=torch.float32)
            new_delta_W_O = torch.zeros(W_new_O.shape[0], W_new_O.shape[1]).to(device="cpu", dtype=torch.float32)

            if "up_proj" in target_modules:    
                W_new_U = new_model_dic[f"model.layers.{layer_mapping[i]}.mlp.up_proj.weight"].to(device="cpu", dtype=torch.float32)
            if "down_proj" in target_modules:
                W_new_D = new_model_dic[f"model.layers.{layer_mapping[i]}.mlp.down_proj.weight"].to(device="cpu", dtype=torch.float32)

        # 计算指定层之间的head similarity
        head_sim_matrix = np.zeros((old_num_attention_heads, new_num_attention_heads))
        for a in range(old_num_attention_heads):
            for b in range(new_num_attention_heads):
                # if args.old_model.split('/')[-2] == "pythia-1b":
                #     # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L197
                #     Q1, K1 = W_old[a*old_head_size*3:a*old_head_size*3+old_head_size, :], W_old[a*old_head_size*3+old_head_size:a*old_head_size*3+2*old_head_size, :]
                #     Q2, K2 = W_new[b*new_head_size*3:b*new_head_size*3+new_head_size, :], W_new[b*new_head_size*3+new_head_size:b*new_head_size*3+2*new_head_size, :]
                #     qk1, qk2 = Q1.T @ K1, Q2.T @ K2
                #     head_sim_matrix[a, b] = cosine_similarity(qk1.view(-1), qk2.view(-1)).item()

                if args.old_model.split('/')[-2] == "pythia-1b" or args.old_model.split('/')[-2] == "bloom-560m":
                    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L197
                    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py#L214
                    # num_heads * 3 * head_dim
                    Q1, K1, V1 = W_old_QKV[(a*old_head_size*3):(a*old_head_size*3+old_head_size), :], W_old_QKV[(a*old_head_size*3+old_head_size):(a*old_head_size*3+2*old_head_size), :], W_old_QKV[(a*old_head_size*3+2*old_head_size):((a+1)*old_head_size*3), :]
                    Q2, K2, V2 = W_new_QKV[(b*new_head_size*3):(b*new_head_size*3+new_head_size), :], W_new_QKV[(b*new_head_size*3+new_head_size):(b*new_head_size*3+2*new_head_size), :], W_new_QKV[(b*new_head_size*3+2*new_head_size):((b+1)*new_head_size*3), :]
                    O1, O2 = W_old_O[:, a*old_head_size:(a+1)*old_head_size], W_new_O[:, b*new_head_size:(b+1)*new_head_size]
                    qk1, qk2 = W_x.T @ Q1.T @ K1 @ W_x, Q2.T @ K2
                    vo1, vo2 = W_x.T @ V1.T @ O1.T @ W_x, V2.T @ O2.T

                    vec1 = torch.cat([qk1.view(-1), vo1.view(-1)], dim=0)
                    vec2 = torch.cat([qk2.view(-1), vo2.view(-1)], dim=0)
                    head_sim_matrix[a, b] = cosine_similarity(vec1, vec2).item()
                
                else:
                    # GQA -> MHA
                    # https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16/blob/main/modeling_minicpm.py#L332
                    Q1, O1 = W_old_Q[a*old_head_size:(a+1)*old_head_size, :], W_old_O[:, a*old_head_size:(a+1)*old_head_size] 
                    K1, V1 = W_old_K[a*old_head_size:(a+1)*old_head_size, :], W_old_V[a*old_head_size:(a+1)*old_head_size, :]
                    qk1, vo1 = W_x.T @ Q1.T @ K1 @ W_x, W_x.T @ V1.T @ O1.T @ W_x
                    vec1 = torch.cat([qk1.view(-1), vo1.view(-1)], dim=0)

                    Q2, O2 = W_new_Q[b*new_head_size:(b+1)*new_head_size, :], W_new_O[:, b*new_head_size:(b+1)*new_head_size]
                    K2, V2 = W_new_K[b*new_head_size:(b+1)*new_head_size, :], W_new_V[b*new_head_size:(b+1)*new_head_size, :]
                    qk2, vo2 = Q2.T @ K2, V2.T @ O2.T
                    vec2 = torch.cat([qk2.view(-1), vo2.view(-1)], dim=0)
                    head_sim_matrix[a, b] = cosine_similarity(vec1, vec2).item()

        head_mapping, _ = max_head_similarity_mapping(head_sim_matrix)
        for a in range(old_num_attention_heads):
            # if args.old_model.split('/')[-2] == "pythia-1b":
            #     W_x = W_old[a*old_head_size*3:(a+1)*old_head_size*3, :] @ W_new[head_mapping[a]*new_head_size*3:(head_mapping[a]+1)*new_head_size*3, :].T
            #     new_delta_W[head_mapping[a]*new_head_size*3:(head_mapping[a]+1)*new_head_size*3, :] = (delta_W[a*old_head_size*3:(a+1)*old_head_size*3, :].T @ W_x).T
            if args.old_model.split('/')[-2] == "pythia-1b" or args.new_model.split('/')[-2] == "bloomz-1b1":
                W_Q_x = W_old_QKV[(a*old_head_size*3):(a*old_head_size*3+old_head_size), :] @ W_x @ W_new_QKV[(head_mapping[a]*new_head_size*3):(head_mapping[a]*new_head_size*3+new_head_size), :].T
                new_delta_W_QKV[(head_mapping[a]*new_head_size*3):(head_mapping[a]*new_head_size*3+new_head_size), :] = W_Q_x.T @ delta_W_QKV[(a*old_head_size*3):(a*old_head_size*3+old_head_size), :] @ W_x

                W_K_x = W_old_QKV[(a*old_head_size*3+old_head_size):(a*old_head_size*3+2*old_head_size), :] @ W_x @ W_new_QKV[(head_mapping[a]*new_head_size*3+new_head_size):(head_mapping[a]*new_head_size*3+2*new_head_size), :].T
                new_delta_W_QKV[(head_mapping[a]*new_head_size*3+new_head_size):(head_mapping[a]*new_head_size*3+2*new_head_size), :] = W_K_x.T @ delta_W_QKV[(a*old_head_size*3+old_head_size):(a*old_head_size*3+2*old_head_size), :] @ W_x

                W_V_x = W_old_QKV[(a*old_head_size*3+2*old_head_size):((a+1)*old_head_size*3), :] @ W_x @ W_new_QKV[(head_mapping[a]*new_head_size*3+2*new_head_size):((head_mapping[a]+1)*new_head_size*3), :].T
                new_delta_W_QKV[(head_mapping[a]*new_head_size*3+2*new_head_size):((head_mapping[a]+1)*new_head_size*3), :] = W_V_x.T @ delta_W_QKV[(a*old_head_size*3+2*old_head_size):((a+1)*old_head_size*3), :] @ W_x

                W_O_x = W_old_O[:, (a*old_head_size):((a+1)*old_head_size)].T @ W_x @ W_new_O[:, (head_mapping[a]*new_head_size):((head_mapping[a]+1)*new_head_size)]
                new_delta_W_O[:, (head_mapping[a]*new_head_size):((head_mapping[a]+1)*new_head_size)] = W_x.T @ delta_W_O[:, a*old_head_size:(a+1)*old_head_size] @ W_O_x
            
            else:
                if args.attention_method == "xtransform":
                    W_Q_x = W_old_Q[a*old_head_size:(a+1)*old_head_size, :] @ W_x @ W_new_Q[head_mapping[a]*new_head_size:(head_mapping[a]+1)*new_head_size, :].T
                    new_delta_W_Q[head_mapping[a]*new_head_size:(head_mapping[a]+1)*new_head_size, :] = W_Q_x.T @ delta_W_Q[a*old_head_size:(a+1)*old_head_size, :] @ W_x

                    W_K_x = W_old_K[a*old_head_size:(a+1)*old_head_size, :] @ W_x @ W_new_K[head_mapping[a]*new_head_size:(head_mapping[a]+1)*new_head_size, :].T
                    new_delta_W_K[head_mapping[a]*new_head_size:(head_mapping[a]+1)*new_head_size, :] = W_K_x.T @ delta_W_K[a*old_head_size:(a+1)*old_head_size, :] @ W_x

                    W_V_x = W_old_V[a*old_head_size:(a+1)*old_head_size, :] @ W_x @ W_new_V[head_mapping[a]*new_head_size:(head_mapping[a]+1)*new_head_size, :].T
                    new_delta_W_V[head_mapping[a]*new_head_size:(head_mapping[a]+1)*new_head_size, :] = W_V_x.T @ delta_W_V[a*old_head_size:(a+1)*old_head_size, :] @ W_x

                    W_O_x = W_old_O[:, a*old_head_size:(a+1)*old_head_size].T @ W_x @ W_new_O[:, head_mapping[a]*new_head_size:(head_mapping[a]+1)*new_head_size]
                    new_delta_W_O[:, head_mapping[a]*new_head_size:(head_mapping[a]+1)*new_head_size] = W_x.T @ delta_W_O[:, a*old_head_size:(a+1)*old_head_size] @ W_O_x
                else:
                    new_delta_W_Q = W_x.T @ delta_W_Q @ W_x
                    new_delta_W_K = W_x.T @ delta_W_K @ W_x
                    new_delta_W_V = W_x.T @ delta_W_V @ W_x
                    new_delta_W_O = W_x.T @ delta_W_O @ W_x


        if args.new_model.split('/')[-2] == "pythia-1.4b":
            B, A, _ = low_rank_decompose(new_delta_W_QKV, old_rank)
            
            new_lora_weights[f"base_model.model.gpt_neox.layers.{layer_mapping[i]}.attention.query_key_value.lora_B.weight"] = B
            new_lora_weights[f"base_model.model.gpt_neox.layers.{layer_mapping[i]}.attention.query_key_value.lora_A.weight"] = A

            B, A, _ = low_rank_decompose(new_delta_W_O, old_rank)
            new_lora_weights[f"base_model.model.gpt_neox.layers.{layer_mapping[i]}.attention.dense.lora_B.weight"] = B
            new_lora_weights[f"base_model.model.gpt_neox.layers.{layer_mapping[i]}.attention.dense.lora_A.weight"] = A

        elif args.new_model.split('/')[-2] == "bloomz-1b1":
            B, A, _ = low_rank_decompose(new_delta_W_QKV, old_rank)
            new_lora_weights[f"base_model.model.transformer.h.{layer_mapping[i]}.self_attention.query_key_value.lora_B.weight"] = B
            new_lora_weights[f"base_model.model.transformer.h.{layer_mapping[i]}.self_attention.query_key_value.lora_A.weight"] = A

            B, A, _ = low_rank_decompose(new_delta_W_O, old_rank)
            new_lora_weights[f"base_model.model.transformer.h.{layer_mapping[i]}.self_attention.dense.lora_B.weight"] = B
            new_lora_weights[f"base_model.model.transformer.h.{layer_mapping[i]}.self_attention.dense.lora_A.weight"] = A

            if "dense_h_to_4h" in target_modules:
                new_lora_weights[f"base_model.model.transformer.h.{layer_mapping[i]}.mlp.dense_h_to_4h.lora_B.weight"] = W_new_U @ torch.linalg.pinv(W_x) @ torch.linalg.pinv(W_old_U) @ old_lora_weights[f"base_model.model.transformer.h.{i}.mlp.dense_h_to_4h.lora_B.weight"].to(device="cpu", dtype=torch.float32) 
                new_lora_weights[f"base_model.model.transformer.h.{layer_mapping[i]}.mlp.dense_h_to_4h.lora_A.weight"] = old_lora_weights[f"base_model.model.transformer.h.{i}.mlp.dense_h_to_4h.lora_A.weight"].to(device="cpu", dtype=torch.float32) @ W_x
            if "dense_4h_to_h" in target_modules:
                new_lora_weights[f"base_model.model.transformer.h.{layer_mapping[i]}.mlp.dense_4h_to_h.lora_B.weight"] = torch.linalg.pinv(W_x) @ old_lora_weights[f"base_model.model.transformer.h.{i}.mlp.dense_4h_to_h.lora_B.weight"].to(device="cpu", dtype=torch.float32) 
                new_lora_weights[f"base_model.model.transformer.h.{layer_mapping[i]}.mlp.dense_4h_to_h.lora_A.weight"] = old_lora_weights[f"base_model.model.transformer.h.{i}.mlp.dense_4h_to_h.lora_A.weight"].to(device="cpu", dtype=torch.float32) @ torch.linalg.pinv(W_old_D) @ W_x @ W_new_D

        elif args.new_model.split('/')[-2] == "MiniCPM-2B-sft-fp32-llama-format":
            B, A, E = low_rank_decompose(new_delta_W_Q, old_rank, "adalora" in args.old_lora_path)
            if "adalora" in args.old_lora_path:
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_E"] = E
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_B"] = B
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_A"] = A
            else:
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight"] = B
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight"] = A

            B, A, E = low_rank_decompose(new_delta_W_K, old_rank, "adalora" in args.old_lora_path)
            if "adalora" in args.old_lora_path:
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_E"] = E
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_B"] = B
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_A"] = A
            else:
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_B.weight"] = B
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_A.weight"] = A

            B, A, E = low_rank_decompose(new_delta_W_V, old_rank, "adalora" in args.old_lora_path)
            if "adalora" in args.old_lora_path:
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_E"] = E
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_B"] = B
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_A"] = A
            else:
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_B.weight"] = B
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_A.weight"] = A

            B, A, E = low_rank_decompose(new_delta_W_O, old_rank, "adalora" in args.old_lora_path)
            if "adalora" in args.old_lora_path:
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.o_proj.lora_E"] = E
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.o_proj.lora_B"] = B
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.o_proj.lora_A"] = A
            else:
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.o_proj.lora_B.weight"] = B
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.o_proj.lora_A.weight"] = A

            if "up_proj" in target_modules:    
                new_lora_weights[f"base_model.model.model.layers.{i}.mlp.up_proj.lora_B.weight"] = W_new_U @ torch.linalg.pinv(W_x) @ torch.linalg.pinv(W_old_U) @ old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.mlp.up_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) 
                new_lora_weights[f"base_model.model.model.layers.{i}.mlp.up_proj.lora_A.weight"] = old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.mlp.up_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32) @ W_x
            if "down_proj" in target_modules:
                new_lora_weights[f"base_model.model.model.layers.{i}.mlp.down_proj.lora_B.weight"] = torch.linalg.pinv(W_x) @ old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.mlp.down_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32)
                new_lora_weights[f"base_model.model.model.layers.{i}.mlp.down_proj.lora_A.weight"] = old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.mlp.down_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32) @ torch.linalg.pinv(W_old_D) @ W_x @ W_new_D
        
        elif args.new_model.split('/')[-2] == "Qwen2.5-3B" or  "Meta-Llama-3-8B" in args.new_model.split('/')[-2]:
            B, A, _ = low_rank_decompose(new_delta_W_Q, old_rank)
            new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.q_proj.lora_B.weight"] = B
            new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.q_proj.lora_A.weight"] = A

            new_delta_W_K = new_delta_W_K.view(new_num_key_value_heads, new_num_rep, new_head_size, new_hidden_size).mean(dim=1) 
            B, A, _ = low_rank_decompose(new_delta_W_K.view(new_num_key_value_heads*new_head_size, new_hidden_size), old_rank)
            new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.k_proj.lora_B.weight"] = B
            new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.k_proj.lora_A.weight"] = A

            new_delta_W_V = new_delta_W_V.view(new_num_key_value_heads, new_num_rep, new_head_size, new_hidden_size).mean(dim=1) 
            B, A, _ = low_rank_decompose(new_delta_W_V.view(new_num_key_value_heads*new_head_size, new_hidden_size), old_rank)
            new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.v_proj.lora_B.weight"] = B
            new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.v_proj.lora_A.weight"] = A

            B, A, _ = low_rank_decompose(new_delta_W_O, old_rank)
            new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.o_proj.lora_B.weight"] = B
            new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.o_proj.lora_A.weight"] = A
            # if "up_proj" in target_modules:    
            #     new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.mlp.up_proj.lora_B.weight"] = W_new_U @ torch.linalg.pinv(W_x) @ torch.linalg.pinv(W_old_U) @ old_lora_weights[f"base_model.model.model.layers.{i}.mlp.up_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) 
            #     new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.mlp.up_proj.lora_A.weight"] = old_lora_weights[f"base_model.model.model.layers.{i}.mlp.up_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32) @ W_x
            # if "down_proj" in target_modules:
            #     new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.mlp.down_proj.lora_B.weight"] = torch.linalg.pinv(W_x) @ old_lora_weights[f"base_model.model.model.layers.{i}.mlp.down_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32)
            #     new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.mlp.down_proj.lora_A.weight"] = old_lora_weights[f"base_model.model.model.layers.{i}.mlp.down_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32) @ torch.linalg.pinv(W_old_D) @ W_x @ W_new_D
        


    new_lora_path = "./trained_models/xTransform/" + args.new_model.split('/')[-2] + f"_{args.old_lora_path.split('/')[-2]}/"
    if not os.path.exists(new_lora_path):
        os.makedirs(new_lora_path)

    peft_config = LoraConfig(
        r=data['r'],
        lora_alpha=data['lora_alpha'],
        lora_dropout=data['lora_dropout'],
        bias="none",
        target_modules=data['target_modules'],
        task_type="CAUSAL_LM",
    )
    peft_config.base_model_name_or_path = args.new_model
    peft_config.inference_mode = True
    peft_config.save_pretrained(new_lora_path, auto_mapping_dict=None)
    safe_save_file(
        new_lora_weights,
        os.path.join(new_lora_path, "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )


