import pandas as pd
import json
import os
from typing import List, Callable, Dict
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    output_path: str = "math12k_baseline_results.jsonl",
) -> float:


    print(f"Starting inference for {len(prompts)} examples...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    scores = []

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gold_answer = ground_truths[i]

        # Score the output
        eval_result = reward_fn(generated_text, gold_answer)
        score = eval_result.get("reward", 0.0)
        scores.append(score)

        results.append({
            "instruction": prompts[i],
            "generation": generated_text,
            "ground_truth": gold_answer,
            "raw_score_result": eval_result,
            "score": score,
        })

    avg_accuracy = sum(scores) / len(scores) if scores else 0
    print(f"Evaluation Complete. Accuracy: {avg_accuracy:.4f}")

    # Save results
    with open(output_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")
    print(f"Results saved to {output_path}")
    return avg_accuracy


if __name__ == "__main__":
    DATA_FILE = "../data/gsm8k/test.jsonl"
    MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
    PROMPT_TEMPLATE_PATH = "./prompts/r1_zero.prompt"

    df = pd.read_json(DATA_FILE, lines=True)

    # Load Prompt Template
    with open(PROMPT_TEMPLATE_PATH, "r") as f:
        r1_zero_template = f.read().strip()

    # Format Prompts and extract clean Ground Truths
    formatted_prompts = []
    ground_truths = []

    for _, row in df.iterrows():
        # 1. Format the prompt
        formatted_prompts.append(r1_zero_template.format(question=row['question']))
        # 2. Extract only the number after ####
        clean_answer = str(row['answer']).split("####")[-1].strip()
        ground_truths.append(clean_answer)

    # Initialize vLLM
    llm = LLM(model=MODEL_NAME, tensor_parallel_size=1)
    
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # Run Evaluation
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=formatted_prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_path="math12k_baseline_results.jsonl",
    )

