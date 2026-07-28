"""
Training loop improvements — requirements.md Section 10.

Design note on Section 10's "design (not implement) with future multi-GPU
(DistributedDataParallel) in mind": every piece here (scheduler, EMA, checkpoint manager, logger)
is a plain class taking the model/optimizer it operates on as an explicit argument — no module-
level singletons or global state — so wrapping the underlying model in `DistributedDataParallel`
later doesn't require touching any of this; `EMA` in particular tracks shadow params by name,
which works whether the source model is wrapped in DDP or not (DDP's `.module` attribute is the
caller's problem, not this module's).
"""
import copy
import math
from pathlib import Path

import torch


def build_lr_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    """
    Warmup + cosine decay (Section 10), replacing DFL's flat LR. Linear warmup from 0 to the
    optimizer's base LR over `warmup_steps`, then cosine decay from base LR down to
    `min_lr_ratio * base_lr` over the remaining steps.
    """
    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be < total_steps")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """
    Exponential moving average of a model's parameters (Section 10) — shadow weights for
    inference/preview, reducing flicker from raw training-weight oscillation. Tracked by
    parameter name (via state_dict), independent of whatever wraps the source model.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        for name, shadow_param in self.shadow.items():
            model_param = model_state[name]
            if shadow_param.dtype.is_floating_point:
                shadow_param.mul_(self.decay).add_(model_param, alpha=1.0 - self.decay)
            else:
                shadow_param.copy_(model_param)

    def copy_to(self, model):
        """Loads the EMA shadow weights into `model` (e.g. for a preview/inference pass)."""
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


class GradientAccumulator:
    """
    Simulates a larger effective batch size than VRAM allows (Section 10) by scaling the loss
    and only stepping the optimizer every `accumulation_steps` calls to `step()`.
    """

    def __init__(self, accumulation_steps=1):
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be >= 1")
        self.accumulation_steps = accumulation_steps
        self._count = 0

    def scale_loss(self, loss):
        """Call before .backward() on the raw (unscaled) loss."""
        return loss / self.accumulation_steps

    def step(self, optimizer):
        """Call after .backward() every micro-batch. Returns True if the optimizer actually
        stepped this call (and its gradients were zeroed), False if still accumulating."""
        self._count += 1
        if self._count >= self.accumulation_steps:
            optimizer.step()
            optimizer.zero_grad()
            self._count = 0
            return True
        return False


def train_val_split(dataset_len, val_fraction=0.05, seed=42):
    """
    Reserves a small held-out slice (Section 10's validation split — never trained on, used to
    measure generalization rather than judging quality only on memorized frames). Returns
    (train_indices, val_indices), both sorted lists.
    """
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be in [0, 1)")
    n_val = max(1, int(dataset_len * val_fraction)) if dataset_len > 0 else 0
    n_val = min(n_val, dataset_len)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(dataset_len, generator=generator).tolist()
    val_indices = sorted(perm[:n_val])
    train_indices = sorted(perm[n_val:])
    return train_indices, val_indices


class CheckpointManager:
    """
    Retains the best-by-validation-metric checkpoint (Section 10), not just latest-by-time, so a
    later degraded model doesn't overwrite a good one. Assumes lower metric is better (e.g.
    LPIPS); pass `higher_is_better=True` for metrics like SSIM.
    """

    def __init__(self, checkpoint_dir, higher_is_better=False):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.higher_is_better = higher_is_better
        self.best_metric = None

    def _is_better(self, metric):
        if self.best_metric is None:
            return True
        return metric > self.best_metric if self.higher_is_better else metric < self.best_metric

    def maybe_save(self, metric, state_dict, filename="best.pt"):
        """Saves `state_dict` (a plain dict — caller decides what goes in it: model/optimizer/EMA/
        step) if `metric` improves on the best seen so far. Returns True if it saved."""
        if not self._is_better(metric):
            return False
        self.best_metric = metric
        torch.save(state_dict, self.checkpoint_dir / filename)
        return True

    def save_latest(self, state_dict, filename="latest.pt"):
        torch.save(state_dict, self.checkpoint_dir / filename)

    def load(self, filename="best.pt", map_location="cpu"):
        return torch.load(self.checkpoint_dir / filename, map_location=map_location, weights_only=True)


class TrainingLogger:
    """Thin wrapper around torch.utils.tensorboard.SummaryWriter (Section 10: track LPIPS,
    identity similarity, and loss curves rather than DFL's basic preview window)."""

    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag_value_dict, step):
        for tag, value in tag_value_dict.items():
            self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()
