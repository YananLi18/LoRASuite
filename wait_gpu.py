import subprocess
import os
import time
import datetime

def get_gpu_memory_usage():
    """
    获取所有 GPU 的显存使用情况，返回列表，其中每个元素表示对应 GPU 的显存占用（以 MB 为单位）。
    """
    try:
        # 运行 nvidia-smi 查询显存使用情况
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            print(f"Error running nvidia-smi: {result.stderr}")
            return []
        # 提取显存使用值，转换为整数列表
        memory_usage = [int(x.strip()) for x in result.stdout.strip().split("\n")]
        return memory_usage
    except Exception as e:
        print(f"Exception while getting GPU memory usage: {e}")
        return []

def find_available_gpus(threshold=5):
    """
    找到显存占用低于指定阈值（单位：MB）的 GPU，返回 GPU 索引列表。
    """
    memory_usage = get_gpu_memory_usage()
    available_gpus = [index for index, usage in enumerate(memory_usage) if usage < threshold]
    return available_gpus

def run_program_if_gpus_available(program_command, required_gpus=2, memory_threshold=5):
    """
    检测到至少两张 GPU 显存占用低于阈值时，运行指定程序。
    
    :param program_command: 要运行的程序命令（字符串）
    :param required_gpus: 至少需要的 GPU 数量
    :param memory_threshold: GPU 显存占用阈值（单位：MB）
    """
    while True:
        available_gpus = find_available_gpus(threshold=memory_threshold)
        if len(available_gpus) >= required_gpus:
            print(f"{datetime.datetime.now()}: Found {len(available_gpus)} GPUs with memory usage below {memory_threshold} MB: {available_gpus}")
            print(f"Running program: {program_command}")
            os.system(f"CUDA_VISIBLE_DEVICES={available_gpus[0]},{available_gpus[1]} {program_command}")
            break
        else:
            print(f"{datetime.datetime.now()}: Only {len(available_gpus)} GPUs available below {memory_threshold} MB. Retrying...")
            time.sleep(60*15)  # 等待 5 秒后重试

if __name__ == "__main__":
    # 要运行的命令（根据需要替换）
    program_to_run = "python lora_adaption.py --old_model /data/share/pythia-1b/ --new_model /data/share/pythia-1.4b/ --old_lora_path ./trained_models/pythia-1b-lora-common-r32-qkvo-170k/"
    # 开始检测并运行
    run_program_if_gpus_available(program_to_run)
