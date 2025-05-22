import copy
import json
import os
import re
import sys
import argparse

import fire

import torch

# sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, set_seed

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        share_gradio: bool = False,
):
    args = parse_args()
    set_seed(args.seed)

    def evaluate(
            instructions,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=256,
            base_model=None,
            **kwargs,
    ):
        # prompt = generate_prompt(instruction, input)
        prompts = [generate_prompt(instruction, input, base_model) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        outputs = [o.split("Answer:")[1].strip() for o in outputs]
        # print(outputs)
        return outputs
        

    """
    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
    """
    if args.with_lora:
        save_file = args.lora_weights + f"experiment/{args.model}-{args.adapter}-{args.dataset}.json"
        create_dir(args.lora_weights + "experiment/")
    else:
        save_file = f"logs/experiment/{args.model}/{args.dataset}.json"
        create_dir(f"logs/experiment/{args.model}/")

    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)
    tokenizer, model = load_model(args)
    # total = len(dataset)
    total = len(batches)
    correct = 0
    current = 0
    miss = 0.001
    output_data = []
    pbar = tqdm(total=total)
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get('instruction') for data in batch]

        outputs = evaluate(instructions, base_model=args.model)

        for data, output in zip(batch, outputs):
            label = data.get('answer')
            flag = False
            if args.dataset.lower() in ['aqua']:
                predict = extract_answer_letter(args, output)
                if label == predict:
                    correct += 1
                    flag = True
            else:
                if isinstance(label, str):
                    label = float(label)
                predict = extract_answer_number(args, output)
                if abs(label - predict) <= miss:
                    correct += 1
                    flag = True
            new_data = copy.deepcopy(data)
            new_data['output_pred'] = output
            new_data['pred'] = predict
            new_data['flag'] = flag
            output_data.append(new_data)
            print(data["instruction"])
            print(output)
            print('prediction:', predict)
            print('label:', label)
            print('---------------')
        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
        print('---------------')
        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=4)
        pbar.update(1)
    pbar.close()
    print('\n')
    print('test finished')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input=None, model=None):
    if input:
        return f"Context: {input}\nQuestion: {instruction}\nAnswer: "
    else:
        return f"Question: {instruction}\nAnswer: "


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'mawps', 'SVAMP'],
                        default='SVAMP')
    parser.add_argument('--model', type=str, choices=['LLaMA-7B', "LLaMA-13B",'LLaMA-2-7B','LLaMA-3-8B',
                                                      'Yi-6B','Yi-1.5-9B',
                                                      'Qwen1.5-1.8B', 'Qwen2.5-3B',
                                                      'phi-1_5', 'phi-2',
                                                      'pythia-1b', 'pythia-1.4b', 
                                                      'MiniCPM-1B', 'MiniCPM-2B',
                                                      'BLOOM-7B', 'bloomz-560m', 'bloom-1b1', 'bloom-560m', 'bloomz-1b1',
                                                      'GPT-j-6B'], required=True)
    parser.add_argument('--adapter', type=str, choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel', 'Prefix', 'Prompt'],
                        default='LoRA')
    parser.add_argument('--with_lora', action='store_true', default=False)
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--lora_weights', type=str, default=None)
    parser.add_argument('--load_8bit', action='store_true', default=False)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()

def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches

def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    lora_weights = args.lora_weights
    if not lora_weights and args.with_lora:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')

    load_8bit = args.load_8bit
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # Allow batched inference
    
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16 if args.model == "MiniCPM-1B" else "auto",
            device_map="auto",
            trust_remote_code=True,
        ) # fix zwq
        if args.with_lora:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype="auto",
                device_map={"":0},
                adapter_name="base"
            )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype="auto",
        )
        if args.with_lora:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype="auto",
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if args.with_lora:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )

        # unwind broken decapoda-research config
        # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        # model.config.bos_token_id = 1
        # model.config.eos_token_id = 2
        

        # if not load_8bit:
        #     model.half()  # seems to fix bugs for some users.

    model.eval()
        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     model = torch.compile(model)
    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer_number(args, sentence: str) -> float:
    dataset = args.dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp", "mawps"]:
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def extract_answer_letter(args, sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[0]
    else:
        return ''


if __name__ == "__main__":
    fire.Fire(main)
