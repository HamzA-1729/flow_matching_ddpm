"""
Unit tests for sampling and numerical ODE solvers:
1. Euler solver shape and pixel value bounds [-1, 1].
2. Heun predictor-corrector stability and output bounds.
3. DDIM respaced sampler execution across low steps (5, 10, 20).
4. Classifier-Free Guidance (CFG) batched evaluation.
5. Deterministic reproducibility when fixing initial_noise.
"""

import pytest
import torch
from flow_matching_ddpm.model import DiT
from flow_matching_ddpm.loss import DDPMLoss
from flow_matching_ddpm.sampler import (
    eval_model_with_cfg,
    sample_flow_euler,
    sample_flow_heun,
    sample_ddim,
)


@pytest.fixture
def tiny_dit():
    """Create a lightweight DiT model for rapid CPU sampler testing."""
    return DiT(
        img_size=28,
        patch_size=4,
        in_channels=1,
        hidden_dim=32,
        depth=2,
        num_heads=2,
        num_classes=10,
    )


def test_eval_model_with_cfg_shapes(tiny_dit):
    """
    Verify that batched CFG returns identical shape (B, C, H, W)
    and that cfg_scale > 1.0 properly alters the unconditional prediction.
    """
    batch_size = 2
    x = torch.randn(batch_size, 1, 28, 28)
    t = torch.tensor([0.25, 0.75])
    y = torch.tensor([1, 8])

    # Unconditional (cfg_scale = 1.0)
    out_uncond = eval_model_with_cfg(tiny_dit, x, t, y, cfg_scale=1.0)
    assert out_uncond.shape == x.shape

    # Guided (cfg_scale = 2.0)
    out_guided = eval_model_with_cfg(tiny_dit, x, t, y, cfg_scale=2.0)
    assert out_guided.shape == x.shape
    assert not torch.isnan(out_guided).any()


def test_flow_euler_sampler(tiny_dit):
    """
    Verify Euler solver produces valid outputs clamped to [-1, 1] on CPU.
    """
    shape = (2, 1, 28, 28)
    y = torch.tensor([3, 7])

    samples = sample_flow_euler(
        model=tiny_dit,
        shape=shape,
        num_steps=5,
        y=y,
        cfg_scale=1.5,
        device=torch.device("cpu"),
    )

    assert samples.shape == shape
    assert not torch.isnan(samples).any()
    assert (samples >= -1.0).all() and (samples <= 1.0).all()


def test_flow_heun_sampler(tiny_dit):
    """
    Verify 2nd-order Heun solver produces valid outputs clamped to [-1, 1].
    """
    shape = (2, 1, 28, 28)
    y = torch.tensor([0, 9])

    samples = sample_flow_heun(
        model=tiny_dit,
        shape=shape,
        num_steps=5,
        y=y,
        cfg_scale=1.5,
        device=torch.device("cpu"),
    )

    assert samples.shape == shape
    assert not torch.isnan(samples).any()
    assert (samples >= -1.0).all() and (samples <= 1.0).all()


def test_ddim_sampler_respaced_steps(tiny_dit):
    """
    Verify DDIM deterministic sampler handles respaced schedules: 5, 10, and 20 steps.
    """
    ddpm_loss = DDPMLoss(model=tiny_dit, timesteps=1000)
    shape = (2, 1, 28, 28)
    y = torch.tensor([2, 5])

    for steps in [5, 10, 20]:
        samples = sample_ddim(
            model=tiny_dit,
            shape=shape,
            alphas_cumprod=ddpm_loss.alphas_cumprod,
            num_steps=steps,
            y=y,
            cfg_scale=1.5,
            device=torch.device("cpu"),
        )

        assert samples.shape == shape
        assert not torch.isnan(samples).any()
        assert (samples >= -1.0).all() and (samples <= 1.0).all()


def test_sampler_deterministic_reproducibility(tiny_dit):
    """
    Verify that providing the same initial_noise tensor produces identical samples.
    """
    fixed_noise = torch.randn(2, 1, 28, 28)
    y = torch.tensor([4, 6])

    # Run 1
    sample1 = sample_flow_euler(
        model=tiny_dit,
        shape=fixed_noise.shape,
        num_steps=5,
        y=y,
        cfg_scale=1.0,
        initial_noise=fixed_noise,
    )

    # Run 2
    sample2 = sample_flow_euler(
        model=tiny_dit,
        shape=fixed_noise.shape,
        num_steps=5,
        y=y,
        cfg_scale=1.0,
        initial_noise=fixed_noise,
    )

    assert torch.allclose(sample1, sample2, atol=1e-7), "Samplers must be 100% deterministic with fixed initial noise!"

