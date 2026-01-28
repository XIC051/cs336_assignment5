import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch.nn.functional as F
from typing import Callable, Any
import logging

def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer):
    ## 
    all_full_ids = []
    all_response_masks = []
    
    for p, o in zip(prompt_strs, output_strs):
        p_ids = tokenizer(p, add_special_tokens=True).input_ids
        o_ids = tokenizer(o, add_special_tokens=False).input_ids
        
        full_ids = p_ids + o_ids
        # mask is 0 for prompt, 1 for response
        mask = [0] * len(p_ids) + [1] * len(o_ids)
        
        all_full_ids.append(full_ids)
        all_response_masks.append(mask)
        
    max_len = max(len(ids) for ids in all_full_ids)
    
    batch_input_ids = []
    batch_labels = []
    batch_response_mask = []
    
    # Use pad_token_id if available, otherwise fallback to eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    for ids, mask in zip(all_full_ids, all_response_masks):
        num_padding = max_len - len(ids)
        # Pad right
        padded_ids = ids + [pad_token_id] * num_padding
        padded_mask = mask + [0] * num_padding # Mask padding as 0
        input_ids = padded_ids[:-1]
        labels = padded_ids[1:]
        # response_mask: a mask on the response tokens in the labels.
        # labels[i] corresponds to padded_ids[i+1], so we use padded_mask[i+1].
        resp_mask = padded_mask[1:]
        
        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
        batch_response_mask.append(resp_mask)
        
    return {
        "input_ids": torch.tensor(batch_input_ids), ## (batch_size, sequence_length)
        "labels": torch.tensor(batch_labels), ## (batch_size, sequence_length)
        "response_mask": torch.tensor(batch_response_mask) ## (batch_size, sequence_length)
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token prediction.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Args:
        model: PreTrainedModel HuggingFace model used for scoring.
        input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt + response tokens.
        labels: torch.Tensor shape (batch_size, sequence_length), labels.
        return_token_entropy: bool If True, also return per-token entropy.
    Returns:
        dict[str, torch.Tensor].
        "log_probs" shape (batch_size, sequence_length), conditional log-probabilities log pθ (xt |x<t).
        "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy.
    """
    outputs = model(input_ids)
    logits = outputs.logits  # (batch_size, sequence_length, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1) ## shape: (batch_size, sequence_length, vocab_size)

    # Gather the log-probabilities for the label tokens
    # labels shape is (batch_size, sequence_length)
    token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    results = {"log_probs": token_log_probs}

    if return_token_entropy:
        results["token_entropy"] = compute_entropy(logits)

    return results


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those elements where mask == 1.
    """
    masked_tensor = tensor * mask
    if dim is None:
        return torch.sum(masked_tensor) / normalize_constant
    else:
        return torch.sum(masked_tensor, dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    """
    # SFT loss is the negative log-likelihood of the response tokens.
    # We normalize by: normalize_constant * batch_size * gradient_accumulation_steps
    # to keep gradients consistent across microbatches.
    mask = response_mask.to(dtype=policy_log_probs.dtype)
    masked_sum = torch.sum(policy_log_probs * mask)
    batch_size = policy_log_probs.shape[0]
    loss = -masked_sum / (normalize_constant * batch_size * gradient_accumulation_steps)

    loss.backward()

    metadata = {
        "loss": loss.detach(),
    }

    return loss.detach(), metadata


def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    device: torch.device | str,
    **generation_kwargs,
):
    """
    Log generations from the model for a set of prompts.
    """
    model.eval()
    results = []
    logger = logging.getLogger(__name__)

    for prompt, gt in zip(prompts, ground_truths):
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)

        with torch.no_grad():
            # Generate response
            output_ids = model.generate(**inputs, **generation_kwargs)
            # Get only the generated part
            prompt_len = inputs.input_ids.shape[1]
            generated_ids = output_ids[:, prompt_len:]
            response_str = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # Get reward info
            reward_info = reward_fn(response_str, gt)

            # Calculate entropy and length using the helpers
            tokenized = tokenize_prompt_and_output([prompt], [response_str], tokenizer)
            for k, v in tokenized.items():
                tokenized[k] = v.to(device)

            # get_response_log_probs will return log_probs and token_entropy
            # on the labels (which are the response tokens shifted)
            out = get_response_log_probs(
                model,
                tokenized["input_ids"],
                tokenized["labels"],
                return_token_entropy=True
            )

            resp_mask = tokenized["response_mask"]  # (1, L)
            token_entropy = out["token_entropy"]  # (1, L)

            # Average entropy over response tokens only
            avg_entropy = (token_entropy * resp_mask).sum() / resp_mask.sum()
            response_len = resp_mask.sum().item()

            results.append({
                "prompt": prompt,
                "response": response_str,
                "ground_truth": gt,
                "reward_info": reward_info,
                "avg_entropy": avg_entropy.item(),
                "length": response_len,
                "is_correct": reward_info.get("reward", 0.0) > 0.5
            })

    # Stats calculation
    total_len = sum(r["length"] for r in results)
    avg_len = total_len / len(results) if results else 0.0

    correct_results = [r for r in results if r["is_correct"]]
    incorrect_results = [r for r in results if not r["is_correct"]]

    avg_len_correct = sum(r["length"] for r in correct_results) / len(correct_results) if correct_results else 0.0
    avg_len_incorrect = (
        sum(r["length"] for r in incorrect_results) / len(incorrect_results)
        if incorrect_results
        else 0.0
    )

    # Logging
    for i, res in enumerate(results):
        logger.info(f"--- Generation {i+1} ---")
        logger.info(f"Prompt: {res['prompt']}")
        logger.info(f"Response: {res['response']}")
        logger.info(f"Ground Truth: {res['ground_truth']}")
        logger.info(f"Reward Info: {res['reward_info']}")
        logger.info(f"Avg Token Entropy: {res['avg_entropy']:.4f}")
        logger.info(f"Response Length: {res['length']}")

    logger.info(f"--- Summary Statistics ---")
    logger.info(f"Average Length: {avg_len:.2f}")
    logger.info(f"Average Length (Correct): {avg_len_correct:.2f}")
    logger.info(f"Average Length (Incorrect): {avg_len_incorrect:.2f}")

    return {
        "results": results,
        "avg_len": avg_len,
        "avg_len_correct": avg_len_correct,
        "avg_len_incorrect": avg_len_incorrect
    }

   