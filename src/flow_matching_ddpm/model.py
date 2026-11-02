"""
DiT (Diffusion Transformer) Backbone with AdaLN-Zero Conditioning.

Shared neural network engine for both DDPM and Flow Matching.
Architecture features:
1. PatchEmbed: 2D image -> sequence of patch tokens.
2. SinusoidalTimeEmbedding: continuous/discrete harmonic frequencies.
3. 2D Sinusoidal Positional Encoding: spatial grid coordinates.
4. AdaLNZeroBlock: Multi-Head Self-Attention + MLP modulated by (gamma, beta, alpha).
5. Zero-initialization: modulation gates start at exact zero (identity mapping at step 0).
6. FinalLayer + Unpatchify: projects tokens back to image dimensions (B, C, H, W).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """
    Generate 2D sinusoidal position embeddings for a square grid of patches.

    Args:
        embed_dim: Token embedding dimension (must be even, and divisible by 4 for 2D).
        grid_size: Number of patches along height/width (e.g. 7 for 28x28 with patch 4).

    Returns:
        pos_embed: (grid_size * grid_size, embed_dim) fixed sinusoidal positional encoding.
    """
    assert embed_dim % 4 == 0, f"embed_dim ({embed_dim}) must be divisible by 4 for 2D sin-cos."

    # Grid coordinates: shape (grid_size, grid_size)
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    mesh_w, mesh_h = torch.meshgrid(grid_w, grid_h, indexing="xy")

    # Half the channels for height, half for width
    dim_per_axis = embed_dim // 2

    # Compute frequencies: omega_i = 1 / (10000^(2i / dim))
    omega = torch.arange(dim_per_axis // 2, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / (dim_per_axis // 2)))

    # Compute outer products: (grid_size, grid_size) x (dim_per_axis // 2)
    out_h = mesh_h.flatten().unsqueeze(1) * omega.unsqueeze(0)  # (N, dim//4)
    out_w = mesh_w.flatten().unsqueeze(1) * omega.unsqueeze(0)  # (N, dim//4)

    # Cat sin and cos for height and width independently
    emb_h = torch.cat([torch.sin(out_h), torch.cos(out_h)], dim=1)  # (N, dim//2)
    emb_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)  # (N, dim//2)

    # Concatenate H and W embeddings -> (N, embed_dim)
    pos_embed = torch.cat([emb_h, emb_w], dim=1)
    return pos_embed


class SinusoidalTimeEmbedding(nn.Module):
    """
    Projects scalar time t into a continuous harmonic frequency vector.

    Formula:
        PE(t, 2i)   = sin(t * omega_i)
        PE(t, 2i+1) = cos(t * omega_i)
        where omega_i = 10000^(-2i / dim)
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "Time embedding dimension must be even."
        self.dim = dim
        half_dim = dim // 2
        # Precompute frequency exponents
        frequencies = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) float or integer tensor of timesteps.
        Returns:
            embedding: (B, dim) continuous sinusoidal representation.
        """
        # Ensure 1D float tensor
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.float().unsqueeze(1)  # (B, 1)

        # Outer product: (B, 1) * (1, half_dim) -> (B, half_dim)
        args = t * self.frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding using a non-overlapping Conv2d.

    Takes an image (B, C, H, W) and splits it into (H/p)*(W/p) patches of size (p, p),
    projecting each patch to a vector of size hidden_dim.
    """

    def __init__(self, img_size: int, patch_size: int, in_channels: int, hidden_dim: int):
        super().__init__()
        assert img_size % patch_size == 0, f"img_size {img_size} must be divisible by patch_size {patch_size}"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        # Non-overlapping convolution is mathematically equivalent to unfold + linear
        self.proj = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            tokens: (B, N, hidden_dim) where N = num_patches
        """
        # (B, C, H, W) -> (B, hidden_dim, grid_size, grid_size)
        x = self.proj(x)
        # Flatten spatial dims and transpose -> (B, N, hidden_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class AdaLNZeroBlock(nn.Module):
    """
    Adaptive LayerNorm (AdaLN) Transformer Block with Zero Initialization.

    In standard LayerNorm, scale (gamma) and shift (beta) are static learned vectors.
    In AdaLN, gamma, beta, and a gating scalar alpha are dynamically generated
    by projecting the conditioning vector c = Embed(t) + Embed(y).

    Crucially, all modulation projections and gates are initialized to EXACT ZERO,
    guaranteeing that at initialization:
        block(x, c) == x  (the block is an exact mathematical identity)
    This prevents gradient explosions and variance collapse at the start of training.
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads

        # Normalization with elementwise_affine=False because AdaLN supplies gamma & beta
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )

        # Modulation projection: takes conditioning vector c and outputs 6 scalars per channel:
        # (gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )

        # Zero-initialize the modulation linear layer
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token tensor of shape (B, N, hidden_dim)
            c: Conditioning vector of shape (B, hidden_dim)

        Returns:
            x: Modulated token tensor of shape (B, N, hidden_dim)
        """
        # Compute the 6 modulation parameters: (B, 6 * hidden_dim)
        mod = self.adaLN_modulation(c)
        gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2 = mod.chunk(6, dim=-1)

        # Self-Attention sub-layer with adaptive scale, shift, and gate
        # Unsqueeze(1) broadcasts modulation over all tokens N
        norm_x1 = self.norm1(x) * (1.0 + gamma_1.unsqueeze(1)) + beta_1.unsqueeze(1)
        attn_out, _ = self.attn(norm_x1, norm_x1, norm_x1, need_weights=False)
        x = x + alpha_1.unsqueeze(1) * attn_out

        # MLP sub-layer with adaptive scale, shift, and gate
        norm_x2 = self.norm2(x) * (1.0 + gamma_2.unsqueeze(1)) + beta_2.unsqueeze(1)
        mlp_out = self.mlp(norm_x2)
        x = x + alpha_2.unsqueeze(1) * mlp_out

        return x


class FinalLayer(nn.Module):
    """
    Final AdaLN-modulated projection layer.
    Modulates tokens using conditioning vector c, applies LayerNorm,
    and linearly projects each token back to patch_size * patch_size * out_channels pixels.
    """

    def __init__(self, hidden_dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )

        # Zero-initialize the final layers so output is zero at initialization
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, hidden_dim)
            c: (B, hidden_dim)
        Returns:
            out: (B, N, patch_size * patch_size * out_channels)
        """
        mod = self.adaLN_modulation(c)
        gamma, beta = mod.chunk(2, dim=-1)
        x = self.norm_final(x) * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) Backbone.

    Equipped with:
    - PatchEmbed for image tokenization.
    - 2D Sinusoidal positional embeddings.
    - Conditioning embedding for continuous time t and discrete class labels y.
    - Stack of AdaLNZeroBlocks.
    - FinalLayer + unpatchify projection back to image shape (B, C, H, W).
    """

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        hidden_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
        time_scale: float = 1000.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.time_scale = time_scale

        # 1. Patch projection
        self.x_embedder = PatchEmbed(img_size, patch_size, in_channels, hidden_dim)
        grid_size = img_size // patch_size
        self.num_patches = self.x_embedder.num_patches

        # 2. Fixed 2D sinusoidal position encodings: (1, num_patches, hidden_dim)
        pos_embed = get_2d_sincos_pos_embed(hidden_dim, grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))

        # 3. Conditioning Embeddings
        # Time MLP: Sinusoidal projection followed by 2-layer MLP
        self.time_proj = SinusoidalTimeEmbedding(hidden_dim)
        self.t_embedder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Class embedding: includes (num_classes + 1) rows.
        # The extra row index `num_classes` is reserved for the unconditional null token (CFG).
        self.y_embedder = nn.Embedding(num_classes + 1, hidden_dim)

        # 4. AdaLN-Zero Transformer Blocks
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])

        # 5. Final output layer
        self.final_layer = FinalLayer(hidden_dim, patch_size, self.out_channels)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reassembles flattened patch tokens back into a 2D image tensor.

        Args:
            x: (B, N, patch_size * patch_size * out_channels)
        Returns:
            imgs: (B, out_channels, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = self.img_size // p
        assert h * w == x.shape[1], f"Grid {h}x{w} does not match token count {x.shape[1]}"

        # (B, h*w, p*p*c) -> (B, h, w, p, p, c)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        # Permute to (B, c, h, p, w, p)
        x = torch.einsum("nhwpqc->nchpwq", x)
        # Reshape to final image dimensions: (B, c, h*p, w*p)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for either DDPM (predicting noise) or Flow Matching (predicting velocity).

        Args:
            x: Image tensor (B, C, H, W)
            t: Continuous or discrete time tensor (B,)
            y: Optional class labels (B,). If None, null token index is used.

        Returns:
            output: Predicted noise or velocity with exact shape (B, C, H, W)
        """
        # 1. Scale time if continuous (in [0, 1]) so it spans identical frequency bands as DDPM
        # If max(t) <= 1.0, scale up by time_scale (1000.0)
        t_scaled = t.float()
        if t_scaled.max() <= 1.0:
            t_scaled = t_scaled * self.time_scale

        t_emb = self.t_embedder(self.time_proj(t_scaled))  # (B, hidden_dim)

        # 2. Class label embedding
        if y is None:
            # Default to null token (index num_classes) for unconditional evaluation
            y = torch.full((x.shape[0],), self.num_classes, dtype=torch.long, device=x.device)
        y_emb = self.y_embedder(y)  # (B, hidden_dim)

        # Combined conditioning vector
        c = t_emb + y_emb  # (B, hidden_dim)

        # 3. Patchify and add fixed 2D positional encoding
        tokens = self.x_embedder(x) + self.pos_embed  # (B, N, hidden_dim)

        # 4. Pass through AdaLN-Zero Transformer Blocks
        for block in self.blocks:
            tokens = block(tokens, c)

        # 5. Modulated projection and unpatchify back to image
        out_tokens = self.final_layer(tokens, c)  # (B, N, p*p*C)
        out_img = self.unpatchify(out_tokens)    # (B, C, H, W)

        return out_img

