import os
import json
import torch
import wandb
import logging
import argparse
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from vllm import LLM, SamplingParams
from unittest.mock import patch
from typing import List, Dict, Any, Callable
from torch.utils.data import DataLoader, Dataset

from cs336_alignment.sft_helper import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.8):
    from vllm.model_executor import set_random_seed as vllm_set_random_seed
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def load_gsm8k(path: str):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_gsm8k_example(example: Dict[str, str]):
    question = example["question"]
    full_answer = example["answer"]
    if "####" in full_answer:
        cot, answer = full_answer.split("####")
        cot = cot.strip()
        answer = answer.strip()
    else:
        cot = ""
        answer = full_answer.strip()
    
    prompt = f"Question: {question}\nAnswer: <think>\n"
    response = f"{cot} </think> <answer> {answer} </answer>"
    return {"prompt": prompt, "response": response, "original_answer": answer}

class SFTDataset(Dataset):
    def __init__(self, examples: List[Dict[str, str]], tokenizer, max_length=1024):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch, tokenizer):
    prompts = [ex["prompt"] for ex in batch]
    responses = [ex["response"] for ex in batch]
    return tokenize_prompt_and_output(prompts, responses, tokenizer)

def evaluate_vllm_local(model, vllm_instance, eval_data, step):
    load_policy_into_vllm_instance(model, vllm_instance)
    
    prompts = [ex["prompt"] for ex in eval_data]
    ground_truths = [ex["original_answer"] for ex in eval_data]
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop=["</answer>"]
    )
    
    outputs = vllm_instance.generate(prompts, sampling_params)
    
    rewards = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        if "</answer>" not in generated_text:
            generated_text += "</answer>"
        
        # Note: vLLM generated text doesn't include the prompt.
        # Our prompt ends with <think>\n.
        # So generated_text should be CoT + </think> <answer> ans </answer>.
        reward_dict = r1_zero_reward_fn(generated_text, ground_truths[i])
        rewards.append(reward_dict.get("reward", 0.0))
        
    avg_acc = sum(rewards) / len(rewards) if rewards else 0.0
    wandb.log({"eval/accuracy": avg_acc, "eval_step": step})
    logger.info(f"Step {step}: Eval Accuracy = {avg_acc:.4f}")
    return avg_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--filter_correct", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--eval_subset_size", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project_name", type=str, default="gsm8k-sft")
    parser.add_argument("--output_dir", type=str, default="outputs/sft")
    args = parser.parse_args()

    set_seed(args.seed)

    # Initialize WandB
    run_name = f"sft-{args.num_examples}-{args.lr}-{args.batch_size}"
    if args.filter_correct:
        run_name += "-filtered"
    wandb.init(project=args.project_name, name=run_name, config=vars(args))
    
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    device = "cuda:0"
    vllm_device = "cuda:1"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    model.config.use_cache = False

    # Load and format data
    train_raw = load_gsm8k(args.train_path)
    test_raw = load_gsm8k(args.test_path)

    formatted_train = [format_gsm8k_example(ex) for ex in train_raw]
    formatted_test = [format_gsm8k_example(ex) for ex in test_raw]

    if args.filter_correct:
        logger.info("Filtering dataset for correct examples...")
        filtered_train = []
        for ex in tqdm(formatted_train):
            reward_dict = r1_zero_reward_fn(ex["response"], ex["original_answer"])
            if reward_dict.get("reward", 0.0) > 0.5:
                filtered_train.append(ex)
        formatted_train = filtered_train
        logger.info(f"Filtered dataset size: {len(formatted_train)}")
        wandb.config.update({"filtered_dataset_size": len(formatted_train)}, allow_val_change=True)

    if args.num_examples > 0 and args.num_examples < len(formatted_train):
        formatted_train = random.sample(formatted_train, args.num_examples)
        logger.info(f"Using {len(formatted_train)} unique examples for SFT")

    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size")

    train_dataset = SFTDataset(formatted_train, tokenizer)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.micro_batch_size, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    num_training_steps = (len(train_dataloader) * args.epochs) // (args.batch_size // args.micro_batch_size)
    if args.max_steps > 0:
        num_training_steps = min(num_training_steps, args.max_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=num_training_steps
    )

    # Initialize vLLM for evaluation
    vllm_instance = init_vllm(args.model_id, vllm_device, args.seed)

    # Training loop
    model.train()
    global_step = 0
    train_step_count = 0
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    
    logger.info(f"Starting training for {args.epochs} epochs, total steps: {num_training_steps}")
    
    for epoch in range(args.epochs):
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)
            
            # Forward pass and compute log-probs
            out = get_response_log_probs(model, input_ids, labels)
            policy_log_probs = out["log_probs"]
            
            # Normalize by average response tokens per example in the global batch.
            normalize_constant = response_mask.sum().item() / response_mask.shape[0]
            if normalize_constant == 0:
                continue
                
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs,
                response_mask,
                gradient_accumulation_steps,
                normalize_constant=normalize_constant
            )
            
            train_step_count += 1
            
            if train_step_count % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train_step": global_step
                })
                
                if global_step % args.eval_every == 0:
                    logger.info(f"Evaluating at step {global_step}...")
                    model.eval()
                    # Evaluate on a subset of test set for speed during training
                    eval_subset = formatted_test[: args.eval_subset_size]
                    evaluate_vllm_local(model, vllm_instance, eval_subset, global_step)
                    model.train()
                
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # Final evaluation
    logger.info("Final evaluation...")
    model.eval()
    evaluate_vllm_local(model, vllm_instance, formatted_test, global_step)
    
    size_tag = "full" if args.num_examples <= 0 or args.num_examples >= len(train_raw) else str(args.num_examples)
    output_dir = os.path.join(args.output_dir, size_tag)
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    wandb.finish()

if __name__ == "__main__":
    main()
