"""
Mathematical Loss Functions for Generative Modeling:
1. FlowMatchingLoss: Continuous-time straight probability paths predicting velocity field.
2. DDPMLoss: Discrete-time linear schedule forward diffusion predicting injected noise.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatchingLoss(nn.Module):
    """
    Continuous-Time Optimal Transport Flow Matching Loss.

    Mathematical Formulation:
        - Draw continuous time: t ~ U[0, 1]
        - Sample initial Gaussian noise: x_noise ~ N(0, I)
        - Straight-line probability path:
            x_t = (1 - t) * x_noise + t * x_data
        - Target velocity vector field (time derivative of path):
            u_t = d(x_t)/dt = x_data - x_noise
        - Objective:
            L_FM = E_{x_data, x_noise, t, y} [ || v_theta(x_t, t, y) - (x_data - x_noise) ||^2 ]
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        x_data: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the flow matching velocity regression loss on a batch of data.

        Args:
            x_data: Clean image tensor of shape (B, C, H, W)
            y: Optional class labels (B,)

        Returns:
            loss: Scalar tensor representing MSE between predicted and target velocity.
            metrics: Dictionary containing logged diagnostic values.
        """
        batch_size = x_data.shape[0]
        device = x_data.device

        # 1. Sample continuous time uniformly: t ~ U[0, 1]
        t = torch.rand(batch_size, device=device)

        # 2. Sample standard Gaussian noise: x_noise ~ N(0, I)
        x_noise = torch.randn_like(x_data)

        # 3. Interpolate along the straight-line OT path:
        # Reshape t to (B, 1, 1, 1) for clean spatial broadcasting over (B, C, H, W)
        t_broadcast = t.view(batch_size, 1, 1, 1)
        x_t = (1.0 - t_broadcast) * x_noise + t_broadcast * x_data

        # 4. Target conditional velocity field: u_t = x_data - x_noise
        target_velocity = x_data - x_noise

        # 5. Predict velocity field using the shared DiT backbone
        pred_velocity = self.model(x_t, t, y)

        # 6. Mean Squared Error (MSE) loss
        loss = F.mse_loss(pred_velocity, target_velocity)

        metrics = {
            "loss": loss.item(),
            "target_norm": target_velocity.norm(dim=1).mean().item(),
            "pred_norm": pred_velocity.norm(dim=1).mean().item(),
        }

        return loss, metrics


class DDPMLoss(nn.Module):
    """
    Denoising Diffusion Probabilistic Models (DDPM) Noise Prediction Loss.

    Mathematical Formulation:
        - Linear variance schedule: beta_1, ..., beta_T
        - Signal retention schedule:
            alpha_t = 1 - beta_t
            alpha_bar_t = prod_{s=1}^t alpha_s
        - Sample discrete step: t ~ U{0, ..., T-1}
        - Sample injected Gaussian noise: eps ~ N(0, I)
        - Forward noising process:
            x_t = sqrt(alpha_bar_t) * x_data + sqrt(1 - alpha_bar_t) * eps
        - Objective:
            L_DDPM = E_{x_data, eps, t, y} [ || eps_theta(x_t, t, y) - eps ||^2 ]
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # Precompute the linear variance schedule according to Ho et al. (2020)
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Precompute square root coefficients for forward diffusion
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Register all schedule buffers (persistent, moved to device, non-trainable)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)

    def forward(
        self,
        x_data: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the DDPM noise prediction loss on a batch of data.

        Args:
            x_data: Clean image tensor of shape (B, C, H, W)
            y: Optional class labels (B,)

        Returns:
            loss: Scalar tensor representing MSE between predicted and actual injected noise.
            metrics: Dictionary containing logged diagnostic values.
        """
        batch_size = x_data.shape[0]
        device = x_data.device

        # 1. Sample discrete timestep: t ~ U{0, ..., T-1}
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

        # 2. Sample standard Gaussian noise: eps ~ N(0, I)
        eps = torch.randn_like(x_data)

        # 3. Extract schedule coefficients for sampled timesteps and reshape to (B, 1, 1, 1)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(batch_size, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1, 1, 1)

        # 4. Generate noisy latent x_t
        x_t = sqrt_alpha_bar * x_data + sqrt_one_minus_alpha_bar * eps

        # 5. Predict noise using the shared DiT backbone
        # We pass discrete (t + 1) in range [1, 1000] matching DDPM convention
        pred_eps = self.model(x_t, t + 1, y)

        # 6. Mean Squared Error (MSE) loss
        loss = F.mse_loss(pred_eps, eps)

        metrics = {
            "loss": loss.item(),
            "pred_norm": pred_eps.norm(dim=1).mean().item(),
        }

        return loss, metrics
