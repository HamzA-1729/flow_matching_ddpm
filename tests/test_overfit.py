"""
1-Batch Overfit Test (MIT 6.S191 Ground Rule):
Asserts that both Flow Matching and DDPM can memorize and overfit a single fixed batch.
To properly test regression convergence, the latent noise and time step are fixed
so the model learns to map specific noisy inputs to their deterministic targets.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from flow_matching_ddpm.model import DiT


def test_overfit_flow_matching_one_batch():
    """
    Test that DiT drops loss by >85% when fitting fixed flow velocity targets.
    """
    torch.manual_seed(42)

    # 1. Tiny DiT architecture
    model = DiT(
        img_size=28,
        patch_size=4,
        in_channels=1,
        hidden_dim=32,
        depth=2,
        num_heads=2,
        num_classes=10,
    )
    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)

    # 2. Fixed batch of data, noise, time, and class labels
    x_data = torch.randn(4, 1, 28, 28)
    x_noise = torch.randn(4, 1, 28, 28)
    t = torch.tensor([0.2, 0.4, 0.6, 0.8])
    y = torch.tensor([0, 1, 2, 3])

    # Target velocity along the straight path
    t_broadcast = t.view(4, 1, 1, 1)
    x_t = (1.0 - t_broadcast) * x_noise + t_broadcast * x_data
    target_v = x_data - x_noise

    # 3. Track initial loss
    with torch.no_grad():
        initial_pred = model(x_t, t, y)
        initial_loss = F.mse_loss(initial_pred, target_v).item()

    # 4. Train loop for 80 steps
    for _ in range(80):
        optimizer.zero_grad()
        pred_v = model(x_t, t, y)
        loss = F.mse_loss(pred_v, target_v)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    loss_reduction = (initial_loss - final_loss) / initial_loss
    print(f"Flow Matching Fixed Batch: Initial={initial_loss:.4f}, Final={final_loss:.4f}, Reduction={loss_reduction*100:.1f}%")

    assert loss_reduction > 0.85, f"Flow Matching failed to overfit 1 batch! Reduction was {loss_reduction*100:.1f}%"


def test_overfit_ddpm_one_batch():
    """
    Test that DiT drops loss by >85% when fitting fixed DDPM noise targets.
    """
    torch.manual_seed(42)

    model = DiT(
        img_size=28,
        patch_size=4,
        in_channels=1,
        hidden_dim=32,
        depth=2,
        num_heads=2,
        num_classes=10,
    )
    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)

    x_data = torch.randn(4, 1, 28, 28)
    target_eps = torch.randn(4, 1, 28, 28)
    t = torch.tensor([100, 300, 500, 700])
    y = torch.tensor([0, 1, 2, 3])

    # Noisy input latent
    alpha_bar = torch.tensor([0.8, 0.5, 0.3, 0.1]).view(4, 1, 1, 1)
    x_t = torch.sqrt(alpha_bar) * x_data + torch.sqrt(1.0 - alpha_bar) * target_eps

    with torch.no_grad():
        initial_pred = model(x_t, t, y)
        initial_loss = F.mse_loss(initial_pred, target_eps).item()

    for _ in range(80):
        optimizer.zero_grad()
        pred_eps = model(x_t, t, y)
        loss = F.mse_loss(pred_eps, target_eps)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    loss_reduction = (initial_loss - final_loss) / initial_loss
    print(f"DDPM Fixed Batch: Initial={initial_loss:.4f}, Final={final_loss:.4f}, Reduction={loss_reduction*100:.1f}%")

    assert loss_reduction > 0.85, f"DDPM failed to overfit 1 batch! Reduction was {loss_reduction*100:.1f}%"
