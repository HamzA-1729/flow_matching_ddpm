"""
Deep Learning Utilities:
1. Deterministic seed management.
2. Exponential Moving Average (EMA) parameter tracking.
3. Cosine learning rate scheduler with linear warmup.
4. Image grid un-normalization and visualization saving.
"""

import os
import math
import random
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image


def set_seed(seed: int = 42) -> None:
    """
    Fix random seeds across Python, NumPy, and PyTorch for 100% deterministic reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EMA:
    """
    Exponential Moving Average (EMA) of model parameters.

    Formula:
        theta_EMA = beta * theta_EMA + (1 - beta) * theta_train

    Why EMA matters:
        Diffusion and flow training gradients fluctuate significantly across batches.
        Maintaining an EMA shadow copy of weights produces significantly sharper,
        lower-variance generations during sampling.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        # Create shadow parameters detached from autograd graph
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update the shadow parameters with the current model weights.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        """
        Copy shadow weights into the target model for evaluation/sampling.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name].data)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 1e-3,
):
    """
    Creates a learning rate scheduler with linear warmup followed by cosine annealing.

    Phase 1 (step < warmup_steps):
        lr = lr_max * (step / warmup_steps)
    Phase 2 (step >= warmup_steps):
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_image_grid(
    tensor: torch.Tensor,
    path: str,
    nrow: int = 8,
) -> None:
    """
    Un-normalizes [-1, 1] generated image tensors back to [0, 1] and saves a PNG grid.

    Args:
        tensor: (B, C, H, W) tensor with values in [-1.0, 1.0].
        path: Destination filepath for the PNG file.
        nrow: Number of images displayed per row.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Un-normalize from [-1, 1] to [0, 1]: x_01 = (x_11 + 1.0) / 2.0
    img = (tensor.detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0
    save_image(img, path, nrow=nrow)
