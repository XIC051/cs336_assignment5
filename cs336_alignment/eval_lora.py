import json
import os
import argparse
import pandas as pd
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.evaluate_baseline import evaluate_vllm

def load_gsm8k(path: str):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_gsm8k_example(example: dict, prompt_template: str):
    question = example["question"]
    full_answer = example["answer"]
    answer = full_answer.split("####")[-1].strip()
    prompt = prompt_template.format(question=question)
    return {
        "prompt": prompt,
        "original_answer": answer
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="outputs/lora_sft/lora-full-5e-05-16/merged_model")
    parser.add_argument("--test_path", type=str, default="../data/gsm8k/test.jsonl")
    parser.add_argument("--prompt_template_path", type=str, default="prompts/r1_zero.prompt")
    parser.add_argument("--eval_max_new_tokens", type=int, default=512)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading prompt template from {args.prompt_template_path}")
    with open(args.prompt_template_path, "r") as f:
        prompt_template = f.read().strip()

    print(f"Loading test data from {args.test_path}")
    test_raw = load_gsm8k(args.test_path)
    formatted_test = [format_gsm8k_example(ex, prompt_template) for ex in test_raw]

    prompts = [ex["prompt"] for ex in formatted_test]
    ground_truths = [ex["original_answer"] for ex in formatted_test]

    print(f"Initializing vLLM with model: {args.model_path}")
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.eval_max_new_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    
    # Check gpu_memory_utilization if needed, but default usually works for single process
    vllm_model = LLM(model=args.model_path, tensor_parallel_size=1)
    
    avg_acc = evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_path=args.output_file,
    )
    
    # Write accuracy to a small file so the parent process can read it easily
    acc_file = args.output_file.replace(".jsonl", "_acc.txt")
    with open(acc_file, "w") as f:
        f.write(str(avg_acc))
        
    print(f"Final Eval Accuracy: {avg_acc:.4f}")

if __name__ == "__main__":
    main()
