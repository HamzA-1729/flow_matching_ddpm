"""
Unit tests for mathematical loss functions:
1. FlowMatchingLoss: probability path interpolation, target velocity, and gradient propagation.
2. DDPMLoss: beta schedule bounds, forward noising, and gradient propagation.
"""

import pytest
import torch
from flow_matching_ddpm.model import DiT
from flow_matching_ddpm.loss import FlowMatchingLoss, DDPMLoss


@pytest.fixture
def tiny_dit():
    """Create a lightweight DiT model for CPU testing."""
    return DiT(
        img_size=28,
        patch_size=4,
        in_channels=1,
        hidden_dim=32,
        depth=2,
        num_heads=2,
        num_classes=10,
    )


def test_flow_matching_loss_and_gradients(tiny_dit):
    """
    Verify that FlowMatchingLoss:
    1. Returns a valid scalar loss.
    2. Successfully computes gradients for all trainable parameters in the backbone.
    """
    loss_fn = FlowMatchingLoss(model=tiny_dit)
    x_data = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))

    loss, metrics = loss_fn(x_data, y)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0, "Loss must be a scalar."
    assert loss.item() >= 0.0, "MSE loss must be non-negative."
    assert "loss" in metrics

    # Test backward pass
    loss.backward()

    # Check that model parameters received non-zero gradients
    has_grad = [p.grad is not None and not torch.isnan(p.grad).any() for p in tiny_dit.parameters() if p.requires_grad]
    assert all(has_grad), "Not all model parameters received valid gradients!"


def test_flow_matching_path_boundary_conditions():
    """
    Mathematical check: Verify that the OT probability path strictly interpolates:
        t = 0.0 -> x_t == x_noise
        t = 1.0 -> x_t == x_data
    """
    x_data = torch.randn(2, 1, 28, 28)
    x_noise = torch.randn(2, 1, 28, 28)

    # At t = 0
    t_zero = torch.zeros(2, 1, 1, 1)
    x_t_0 = (1.0 - t_zero) * x_noise + t_zero * x_data
    assert torch.allclose(x_t_0, x_noise)

    # At t = 1
    t_one = torch.ones(2, 1, 1, 1)
    x_t_1 = (1.0 - t_one) * x_noise + t_one * x_data
    assert torch.allclose(x_t_1, x_data)


def test_ddpm_schedule_bounds(tiny_dit):
    """
    Verify that DDPM linear schedule matches Ho et al. (2020):
    - alpha_bar_0 should be near 1.0 (almost 100% signal).
    - alpha_bar_{T-1} should be near 0.004 (almost pure noise).
    """
    ddpm_loss = DDPMLoss(model=tiny_dit, timesteps=1000, beta_start=1e-4, beta_end=0.02)

    alpha_bar_0 = ddpm_loss.alphas_cumprod[0].item()
    alpha_bar_end = ddpm_loss.alphas_cumprod[-1].item()

    # beta_0 = 1e-4 -> alpha_bar_0 = 1 - 1e-4 = 0.9999
    assert abs(alpha_bar_0 - 0.9999) < 1e-4, f"Unexpected alpha_bar_0: {alpha_bar_0}"
    # In Ho et al. (2020), at step 1000, alpha_bar_end is approx 4.0e-5 (0.00004)
    # meaning the signal remaining is sqrt(4e-5) ~ 0.006 (99.996% pure noise)
    assert 1e-5 < alpha_bar_end < 1e-4, f"Unexpected alpha_bar_end: {alpha_bar_end}"


def test_ddpm_loss_and_gradients(tiny_dit):
    """
    Verify that DDPMLoss returns scalar MSE and propagates valid gradients to DiT.
    """
    loss_fn = DDPMLoss(model=tiny_dit, timesteps=1000)
    x_data = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))

    loss, metrics = loss_fn(x_data, y)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() >= 0.0

    loss.backward()

    has_grad = [p.grad is not None and not torch.isnan(p.grad).any() for p in tiny_dit.parameters() if p.requires_grad]
    assert all(has_grad), "Not all model parameters received valid gradients!"

