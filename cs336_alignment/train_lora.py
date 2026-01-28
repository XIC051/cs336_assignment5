import os
import json
import torch
import wandb
import logging
import argparse
import random
from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from cs336_alignment.sft_helper import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    def __init__(self, examples: List[Dict[str, str]], tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, tokenizer):
    prompts = [ex["prompt"] for ex in batch]
    responses = [ex["response"] for ex in batch]
    return tokenize_prompt_and_output(prompts, responses, tokenizer)


def evaluate_local(model, tokenizer, eval_data, step, device, max_new_tokens=512):
    prompts = [ex["prompt"] for ex in eval_data]
    ground_truths = [ex["original_answer"] for ex in eval_data]

    rewards = []
    model.eval()
    for prompt, gt in zip(prompts, ground_truths):
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[:, prompt_len:]
        response_str = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        if "</answer>" not in response_str:
            response_str += "</answer>"

        reward_dict = r1_zero_reward_fn(response_str, gt)
        rewards.append(reward_dict.get("reward", 0.0))

    avg_acc = sum(rewards) / len(rewards) if rewards else 0.0
    wandb.log({"eval/accuracy": avg_acc, "eval_step": step})
    logger.info(f"Step {step}: Eval Accuracy = {avg_acc:.4f}")
    model.train()
    return avg_acc


def parse_sample_sizes(sample_sizes: str, num_examples: int, dataset_len: int) -> List[int]:
    if num_examples > 0:
        return [num_examples]

    sizes = []
    for part in sample_sizes.split(","):
        token = part.strip().lower()
        if token in {"full", "all", "-1"}:
            sizes.append(-1)
        else:
            sizes.append(int(token))
    return sizes


def size_tag_from_examples(num_examples: int, dataset_len: int) -> str:
    if num_examples <= 0 or num_examples >= dataset_len:
        return "full"
    return str(num_examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--sample_sizes", type=str, default="128,256,512,1024,full")
    parser.add_argument("--filter_correct", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--eval_subset_size", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project_name", type=str, default="gsm8k-lora")
    parser.add_argument("--output_dir", type=str, default="outputs/lora_sft")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    args = parser.parse_args()

    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size")

    set_seed(args.seed)
    device = torch.device("cuda:0")

    # Load and format data once.
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

    sample_sizes = parse_sample_sizes(args.sample_sizes, args.num_examples, len(formatted_train))
    rng = random.Random(args.seed)

    for size in sample_sizes:
        size_tag = size_tag_from_examples(size, len(formatted_train))
        run_name = f"lora-{size_tag}-{args.lr}-{args.batch_size}"
        if args.filter_correct:
            run_name += "-filtered"
        wandb.init(project=args.project_name, name=run_name, config=vars(args))
        if args.filter_correct:
            wandb.config.update(
                {"filtered_dataset_size": len(formatted_train)}, allow_val_change=True
            )

        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

        if size > 0 and size < len(formatted_train):
            train_subset = rng.sample(formatted_train, size)
            logger.info(f"Using {len(train_subset)} unique examples for LoRA SFT")
        else:
            train_subset = formatted_train
            logger.info("Using full dataset for LoRA SFT")

        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        ).to(device)
        model.config.use_cache = False
        if not args.no_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        train_dataset = SFTDataset(train_subset, tokenizer)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.micro_batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, tokenizer),
        )

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr
        )

        num_training_steps = (len(train_dataloader) * args.epochs) // (
            args.batch_size // args.micro_batch_size
        )
        if args.max_steps > 0:
            num_training_steps = min(num_training_steps, args.max_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps,
        )

        model.train()
        global_step = 0
        train_step_count = 0
        gradient_accumulation_steps = args.batch_size // args.micro_batch_size

        logger.info(
            f"Starting LoRA training for {args.epochs} epochs, total steps: {num_training_steps}"
        )

        for epoch in range(args.epochs):
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                response_mask = batch["response_mask"].to(device)

                out = get_response_log_probs(model, input_ids, labels)
                policy_log_probs = out["log_probs"]

                normalize_constant = response_mask.sum().item() / response_mask.shape[0]
                if normalize_constant == 0:
                    continue

                loss, _ = sft_microbatch_train_step(
                    policy_log_probs,
                    response_mask,
                    gradient_accumulation_steps,
                    normalize_constant=normalize_constant,
                )

                train_step_count += 1
                if train_step_count % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/lr": scheduler.get_last_lr()[0],
                            "train_step": global_step,
                        }
                    )

                    if global_step % args.eval_every == 0:
                        logger.info(f"Evaluating at step {global_step}...")
                        eval_subset = formatted_test[: args.eval_subset_size]
                        evaluate_local(model, tokenizer, eval_subset, global_step, device)

                    if args.max_steps > 0 and global_step >= args.max_steps:
                        break
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        logger.info("Final evaluation...")
        evaluate_local(model, tokenizer, formatted_test, global_step, device)

        output_dir = os.path.join(args.output_dir, size_tag)
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        wandb.finish()
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
