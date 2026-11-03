"""
Sampling & Numerical ODE Solvers:
1. Batched Classifier-Free Guidance (CFG).
2. Euler Integrator (1st order ODE for Flow Matching).
3. Heun Integrator (2nd order predictor-corrector ODE for Flow Matching).
4. DDIM Sampler (Deterministic reverse diffusion with uniform schedule respacing).
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn


def eval_model_with_cfg(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    cfg_scale: float = 1.0,
    null_class_idx: int = 10,
) -> torch.Tensor:
    """
    Evaluate the model with Classifier-Free Guidance (CFG) in a single batched pass.

    Formula:
        v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)

    Args:
        model: Shared DiT backbone.
        x: Image latent tensor of shape (B, C, H, W).
        t: Time conditioning tensor of shape (B,).
        y: Target class labels of shape (B,). If None, runs unconditionally.
        cfg_scale: Guidance weight w (w=1.0 means no guidance).
        null_class_idx: Index reserved for unconditional null token (10 for MNIST).

    Returns:
        Guided prediction (velocity or noise) of shape (B, C, H, W).
    """
    # If no guidance or no class labels provided, run standard single-pass forward
    if cfg_scale <= 1.0 or y is None:
        return model(x, t, y)

    batch_size = x.shape[0]

    # Double batch along batch dimension: [conditional, unconditional]
    x_in = torch.cat([x, x], dim=0)
    t_in = torch.cat([t, t], dim=0)

    # Class labels: first half gets target y, second half gets null token
    null_y = torch.full((batch_size,), null_class_idx, dtype=torch.long, device=x.device)
    y_in = torch.cat([y, null_y], dim=0)

    # Single batched forward pass through DiT
    pred_all = model(x_in, t_in, y_in)

    # Split into conditional and unconditional halves
    pred_cond, pred_uncond = pred_all.chunk(2, dim=0)

    # Apply CFG linear combination
    pred_guided = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
    return pred_guided


@torch.no_grad()
def sample_flow_euler(
    model: nn.Module,
    shape: Tuple[int, ...],
    num_steps: int = 20,
    y: Optional[torch.Tensor] = None,
    cfg_scale: float = 1.0,
    device: torch.device = torch.device("cpu"),
    initial_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generate samples from Flow Matching using a 1st-Order Euler ODE solver.

    Integration:
        dx/dt = v_theta(x, t)  from t = 0.0 to t = 1.0
        x_{t + dt} = x_t + dt * v_theta(x_t, t)

    Number of Function Evaluations (NFE):
        num_steps * (2 if cfg_scale > 1.0 else 1)
    """
    model.eval()

    # Initial Gaussian noise latent at t = 0
    if initial_noise is not None:
        x = initial_noise.clone().to(device)
    else:
        x = torch.randn(shape, device=device)

    dt = 1.0 / num_steps

    for step in range(num_steps):
        t_scalar = step * dt
        t = torch.full((x.shape[0],), t_scalar, device=device, dtype=torch.float32)

        # Predict velocity vector field: v = dx/dt
        v = eval_model_with_cfg(model, x, t, y, cfg_scale=cfg_scale)

        # Euler forward step
        x = x + dt * v

    # Final generated image at t = 1.0, clamped to valid pixel range [-1, 1]
    return torch.clamp(x, -1.0, 1.0)


@torch.no_grad()
def sample_flow_heun(
    model: nn.Module,
    shape: Tuple[int, ...],
    num_steps: int = 10,
    y: Optional[torch.Tensor] = None,
    cfg_scale: float = 1.0,
    device: torch.device = torch.device("cpu"),
    initial_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generate samples from Flow Matching using a 2nd-Order Heun Predictor-Corrector ODE solver.

    Algorithm:
        1. Predictor:  x_pred = x_t + dt * v(x_t, t)
        2. Corrector:  v_next = v(x_pred, t + dt)
        3. Step:       x_{t + dt} = x_t + (dt / 2) * (v + v_next)

    Number of Function Evaluations (NFE):
        2 * num_steps * (2 if cfg_scale > 1.0 else 1)
    """
    model.eval()

    if initial_noise is not None:
        x = initial_noise.clone().to(device)
    else:
        x = torch.randn(shape, device=device)

    dt = 1.0 / num_steps

    for step in range(num_steps):
        t_curr = step * dt
        t_next = min(1.0, (step + 1) * dt)

        t_curr_tensor = torch.full((x.shape[0],), t_curr, device=device, dtype=torch.float32)
        t_next_tensor = torch.full((x.shape[0],), t_next, device=device, dtype=torch.float32)

        # 1. Slope at current state (Predictor slope)
        v_curr = eval_model_with_cfg(model, x, t_curr_tensor, y, cfg_scale=cfg_scale)

        # 2. Predicted state at next time step
        x_pred = x + dt * v_curr

        # 3. Slope at predicted state (Corrector slope)
        v_next = eval_model_with_cfg(model, x_pred, t_next_tensor, y, cfg_scale=cfg_scale)

        # 4. Average slopes and advance state
        x = x + (dt / 2.0) * (v_curr + v_next)

    return torch.clamp(x, -1.0, 1.0)


@torch.no_grad()
def sample_ddim(
    model: nn.Module,
    shape: Tuple[int, ...],
    alphas_cumprod: torch.Tensor,
    num_steps: int = 20,
    y: Optional[torch.Tensor] = None,
    cfg_scale: float = 1.0,
    device: torch.device = torch.device("cpu"),
    initial_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generate samples using Denoising Diffusion Implicit Models (DDIM, Song et al. 2020).

    Uses uniform schedule respacing (Nichol & Dhariwal, Improved DDPM) to extract
    a deterministic ODE reverse trajectory from 1,000-step trained weights.

    Formula:
        x̂_0 = (x_t - sqrt(1 - ᾱ_t) * eps_pred) / sqrt(ᾱ_t)
        x_{t-1} = sqrt(ᾱ_{t-1}) * x̂_0 + sqrt(1 - ᾱ_{t-1}) * eps_pred
    """
    model.eval()
    total_timesteps = len(alphas_cumprod)

    # 1. Uniformly respace 1,000 steps into num_steps indices (e.g. 0, 50, 100, ..., 950)
    step_indices = torch.linspace(0, total_timesteps - 1, num_steps, dtype=torch.long, device=device)

    if initial_noise is not None:
        x = initial_noise.clone().to(device)
    else:
        x = torch.randn(shape, device=device)

    # Reverse diffusion loop: from last step index down to 0
    for i in reversed(range(num_steps)):
        curr_t_idx = step_indices[i]
        # Previous timestep index (or 0 if on the last step)
        prev_t_idx = step_indices[i - 1] if i > 0 else None

        alpha_bar_curr = alphas_cumprod[curr_t_idx]
        alpha_bar_prev = alphas_cumprod[prev_t_idx] if prev_t_idx is not None else torch.tensor(1.0, device=device)

        # Timestep passed to model: 1-indexed [1, 1000]
        t_model = torch.full((x.shape[0],), curr_t_idx.item() + 1, device=device, dtype=torch.long)

        # Predict noise eps_theta
        eps_pred = eval_model_with_cfg(model, x, t_model, y, cfg_scale=cfg_scale)

        # 2. Predict clean image x̂_0
        pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_curr) * eps_pred) / torch.sqrt(alpha_bar_curr)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # 3. Deterministic DDIM step (eta = 0.0)
        direction_to_xt = torch.sqrt(1.0 - alpha_bar_prev) * eps_pred
        x = torch.sqrt(alpha_bar_prev) * pred_x0 + direction_to_xt

    return torch.clamp(x, -1.0, 1.0)
