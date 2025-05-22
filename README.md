# LoRASuite
[![Arxiv](https://img.shields.io/badge/arXiv-2505.13515-B21A1B)](https://arxiv.org/abs/2505.13515)

The Official PyTorch implementation of [**LoRASuite: Efficient LoRA Adaptation Across Large Language Model Upgrades**](https://arxiv.org/abs/2505.13515).

Authors: [Yanan Li](https://scholar.google.com/citations?user=fCk0tD8AAAAJ&hl=zh-CN) $^\dagger$, [Fanxu Meng](https://fxmeng.github.io/) $^\ddagger$, [Muhan Zhang](https://muhanzhang.github.io/) $^\ddagger$, Shiai Zhu $^\nmid$, [Shangguang Wang](https://wangshangguang.github.io/) $^\dagger$, [Mengwei Xu](https://xumengwei.github.io/) $^\dagger$  
$^\dagger$ Beijing University of Posts and Telecommunications, $^\ddagger$ Peking University, $^\nmid$ Unaffiliated

## Overview
As Large Language Models (LLMs) are frequently updated, LoRA weights trained on earlier versions quickly become obsolete. The conventional practice of retraining LoRA weights from scratch on the latest model is costly, time-consuming, and environmentally detrimental, particularly as the diversity of LLMs and downstream tasks expands. This motivates a critical question: "How can we efficiently leverage existing LoRA weights to adapt to newer model versions?" To address this, we propose LoRASuite, a modular approach tailored specifically to various types of LLM updates. First, we compute a transfer matrix utilizing known parameters from both old and new LLMs. Next, we allocate corresponding layers and attention heads based on centered kernel alignment and cosine similarity metrics, respectively. A subsequent small-scale, skillful fine-tuning step ensures numerical stability. Experimental evaluations demonstrate that LoRASuite consistently surpasses small-scale vanilla LoRA methods. Notably, on backbone LLMs such as MiniCPM and Qwen, LoRASuite even exceeds the performance of full-scale LoRA retraining, with average improvements of +1.4 and +6.6 points on math tasks, respectively. Additionally, LoRASuite significantly reduces memory consumption by 5.5 GB and computational time by 78.23%.

## Experiments

### Setup

You can create the environment from the environment.yml file:

```
conda env create -f environment.yml
```


We modify the Peft `LoraLayer` to enable weight initialization from a specified path instead of the default Kaiming initialization for matrix A and zero initialization for matrix B. Please refer to our provided Peft implementation for further details.
```
def xtransform_init(self, adapter_name, init_lora_path, current_key):
        from peft import load_peft_weights

        weight = self.get_base_layer().weight
        dtype = weight.dtype
        device = weight.device

        old_lora_weights = load_peft_weights(init_lora_path)
        if f"base_model.model.{current_key}.lora_A.weight" in old_lora_weights.keys():
            self.lora_A[adapter_name].weight.data = old_lora_weights[f"base_model.model.{current_key}.lora_A.weight"].to(device, dtype)
            self.lora_B[adapter_name].weight.data = old_lora_weights[f"base_model.model.{current_key}.lora_B.weight"].to(device, dtype)
        else:
            # nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            # nn.init.zeros_(self.lora_B[adapter_name].weight)
            nn.init.zeros_(self.lora_A[adapter_name].weight)
            nn.init.zeros_(self.lora_B[adapter_name].weight)
            self.lora_A[adapter_name].weight.requires_grad = False
            self.lora_B[adapter_name].weight.requires_grad = False
```


### Vanilla LoRA
```
CUDA_VISIBLE_DEVICES=0 python finetune.py --base_model ./modelzoo/MiniCPM-S-1B-sft-llama-format/ --data_path ./ft-training_set/math_10k.json --output_dir ./trained_models/MiniCPM-S-1B-sft-llama-format-lora-math-r32-qkvo-10k/ --init_lora_weights True --warmup_steps 100 --lora_r 32 --lora_alpha 32 --lora_dropout 0 --batch_size 16 --micro_batch_size 4 --num_epochs 3 --learning_rate 3e-4 --cutoff_len 256 --adapter_name lora --target_modules "['q_proj','k_proj', 'v_proj', 'o_proj']" 

bash evaluate.sh 0 MiniCPM-1B ./modelzoo/MiniCPM-S-1B-sft-llama-format/ ./trained_models/MiniCPM-S-1B-sft-llama-format-lora-math-r32-qkvo-10k/ 4 true 0 | tee ./logs/MiniCPM-S-1B-sft-llama-format-lora-math-r32-qkvo-10k-2025XXXX.log
```

### LoRASuite w/o LFT
```
CUDA_VISIBLE_DEVICES=0,1 python lora_adaption.py --new_model ./modelzoo/MiniCPM-2B-sft-fp32-llama-format/ --old_model ./modelzoo/MiniCPM-S-1B-sft-llama-format/ --old_lora_path ./trained_models/MiniCPM-S-1B-sft-llama-format-lora-math-r32-qkvo-10k/

bash evaluate.sh 0 MiniCPM-2B ./modelzoo/MiniCPM-2B-sft-fp32-llama-format/ ./trained_models/xTransform/MiniCPM-2B-sft-fp32-llama-format_MiniCPM-S-1B-sft-llama-format-lora-math-r32-qkvo-10k/ 4 true 0 | tee ./logs/xTransform/MiniCPM-2B-sft-fp32-llama-format_MiniCPM-S-1B-sft-llama-format-lora-math-r32-qkvo-10k-2025XXXX.log
```

### LoRASuite 
```
CUDA_VISIBLE_DEVICES=0 python finetune.py --base_model ./modelzoo/MiniCPM-2B-sft-fp32-llama-format/ --data_path ./ft-training_set/sampled_100_math_10k.json --output_dir ./trained_models/xTransform/MiniCPM-2B-sft-fp32-llama-format_MiniCPM-S-1B-sft-llama-format-lora-math-r32-qkvo-10k/ --init_lora_weights xtransform --warmup_steps 0 --lora_r 32 --lora_alpha 32 --lora_dropout 0 --batch_size 16 --micro_batch_size 4 --num_epochs 3 --learning_rate 1e-4 --cutoff_len 256 --val_set_size 0 --adapter_name lora --target_modules "['q_proj', 'k_proj', 'v_proj', 'o_proj']" 

bash evaluate.sh 0 MiniCPM-2B ./modelzoo/MiniCPM-2B-sft-fp32-llama-format/ ./trained_models/xTransform/MiniCPM-2B-sft-fp32-llama-format_MiniCPM-S-1B-sft-llama-format-lora-math-r32-qkvo-10k/xtransform/ 4 true 0 | tee ./logs/xTransform/MiniCPM-2B-sft-fp32-llama-format_MiniCPM-S-1B-sft-llama-format-lora-math-r32-qkvo-10k_100-2025XXXX.log
```

## Useful Links
- LLM-Adapters: https://github.com/AGI-Edgerunners/LLM-Adapters
- DoRA: https://github.com/NVlabs/DoRA

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=YananLi18/LoRASuite&type=Date)](https://star-history.com/#YananLi18/LoRASuite&Date)

## Citation

Please cite our paper if it's helpful to your work!
```
@article{li2025lorasuite,
  title={LoRASuite: Efficient LoRA Adaptation Across Large Language Model Upgrades},
  author={Li, Yanan and Meng, Fanxu and Zhang, Muhan and Zhu, Shia and Wang, Shangguang and Xu, Mengwei},
  journal={arXiv preprint arXiv:2505.13515},
  year={2025}
}
```
