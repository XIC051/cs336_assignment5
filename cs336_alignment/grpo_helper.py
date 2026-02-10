from typing import Callable

import torch


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by the group size.
    """
    rollout_batch_size = len(rollout_responses)
    if rollout_batch_size != len(repeated_ground_truths):
        raise ValueError(
            "rollout_responses and repeated_ground_truths must have the same length."
        )
    if group_size <= 0:
        raise ValueError("group_size must be a positive integer.")
    if rollout_batch_size % group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size.")

    rewards = []
    format_rewards = []
    answer_rewards = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        score = reward_fn(response, ground_truth)
        rewards.append(score["reward"])
        format_rewards.append(score.get("format_reward", score["reward"]))
        answer_rewards.append(score.get("answer_reward", score["reward"]))

    raw_rewards = torch.tensor(rewards, dtype=torch.float32)
    num_groups = rollout_batch_size // group_size
    rewards_grouped = raw_rewards.reshape(num_groups, group_size)

    group_means = rewards_grouped.mean(dim=1, keepdim=True)
    advantages = rewards_grouped - group_means
    if normalize_by_std:
        group_stds = rewards_grouped.std(dim=1, keepdim=True, unbiased=False)
        advantages = advantages / (group_stds + advantage_eps)

    advantages = advantages.reshape(rollout_batch_size)

    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std(unbiased=False).item(),
        "reward_min": raw_rewards.min().item(),
        "reward_max": raw_rewards.max().item(),
        "format_reward_mean": torch.tensor(format_rewards, dtype=torch.float32)
        .mean()
        .item(),
        "answer_reward_mean": torch.tensor(answer_rewards, dtype=torch.float32)
        .mean()
        .item(),
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std(unbiased=False).item(),
    }

    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the per-token policy gradient loss using raw rewards or advantages.
    """
    if raw_rewards_or_advantages.ndim == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(1)
    if raw_rewards_or_advantages.ndim != 2 or raw_rewards_or_advantages.shape[1] != 1:
        raise ValueError(
            "raw_rewards_or_advantages must have shape (batch_size,) or (batch_size, 1)."
        )

    rewards = raw_rewards_or_advantages.to(dtype=policy_log_probs.dtype)
    return -rewards * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the per-token GRPO-Clip loss.
    """
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(1)
    if advantages.ndim != 2 or advantages.shape[1] != 1:
        raise ValueError("advantages must have shape (batch_size,) or (batch_size, 1).")

    advantages = advantages.to(dtype=policy_log_probs.dtype)
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    unclipped_loss = -advantages * ratio
    clipped_loss = -advantages * clipped_ratio
    loss = torch.minimum(unclipped_loss, clipped_loss)

    clip_mask = clipped_loss < unclipped_loss
    metadata = {"clip_mask": clip_mask}
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function.
    """
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards is required for loss_type='no_baseline'.")
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        )
        return loss, {}

    if loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError(
                "advantages is required for loss_type='reinforce_with_baseline'."
            )
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )
        return loss, {}

    if loss_type == "grpo_clip":
        if advantages is None:
            raise ValueError("advantages is required for loss_type='grpo_clip'.")
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for loss_type='grpo_clip'.")
        if cliprange is None:
            raise ValueError("cliprange is required for loss_type='grpo_clip'.")
        return compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )

    raise ValueError(
        "loss_type must be one of 'no_baseline', 'reinforce_with_baseline', or 'grpo_clip'."
    )


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a dimension, considering only mask == 1.
    """
    mask = mask.to(dtype=tensor.dtype)
    masked_tensor = tensor * mask
    if dim is None:
        total = masked_tensor.sum()
        count = mask.sum()
    else:
        total = masked_tensor.sum(dim=dim)
        count = mask.sum(dim=dim)

    safe_count = torch.clamp(count, min=1)
    return total / safe_count


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be a positive integer.")

    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    mask = response_mask.to(dtype=per_token_loss.dtype)
    loss = masked_mean(per_token_loss, mask=mask, dim=None)
    loss = loss / gradient_accumulation_steps
    loss.backward()

    return loss.detach(), metadata
