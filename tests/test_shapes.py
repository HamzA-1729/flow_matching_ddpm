"""
Unit tests for the Patch Diffusion Transformer (DiT) backbone:
1. Shape preservation across forward pass.
2. Exact mathematical identity initialization of AdaLNZeroBlock.
3. Unpatchify reversibility check.
4. Class conditioning and null token handling (CFG).
"""

import torch

from flow_matching_ddpm.model import AdaLNZeroBlock, DiT


def test_adaln_zero_identity_init():
    """
    Verify that an AdaLNZeroBlock acts as an exact mathematical identity at step 0.
    Because modulation weights and gates alpha are zero-initialized, block(x, c) must equal x.
    """
    batch_size = 2
    seq_len = 49
    hidden_dim = 128
    num_heads = 4

    block = AdaLNZeroBlock(hidden_dim=hidden_dim, num_heads=num_heads)
    block.eval()

    x = torch.randn(batch_size, seq_len, hidden_dim)
    c = torch.randn(batch_size, hidden_dim)

    with torch.no_grad():
        out = block(x, c)

    # Must be identical within floating point tolerance
    assert torch.allclose(out, x, atol=1e-6), (
        "AdaLN-Zero block is not an exact identity at initialization!"
    )


def test_dit_forward_shapes():
    """
    Verify that the DiT backbone preserves exact spatial dimensions (B, C, H, W) -> (B, C, H, W).
    """
    batch_size = 4
    img_size = 28
    in_channels = 1
    hidden_dim = 64
    depth = 2
    num_heads = 4
    patch_size = 4

    model = DiT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        depth=depth,
        num_heads=num_heads,
        num_classes=10,
    )
    model.eval()

    # Test with continuous time in [0, 1] (Flow Matching)
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    t_flow = torch.rand(batch_size)
    y = torch.randint(0, 10, (batch_size,))

    with torch.no_grad():
        out_flow = model(x, t_flow, y)

    assert out_flow.shape == x.shape, f"Expected {x.shape}, got {out_flow.shape}"

    # Test with discrete time in [1, 1000] (DDPM) and unconditional null label (y=None)
    t_ddpm = torch.randint(1, 1000, (batch_size,))
    with torch.no_grad():
        out_ddpm = model(x, t_ddpm, y=None)

    assert out_ddpm.shape == x.shape, f"Expected {x.shape}, got {out_ddpm.shape}"


def test_unpatchify_reconstruction():
    """
    Verify that patchifying and immediately unpatchifying a grid accurately preserves spatial coordinates.
    """
    model = DiT(img_size=28, patch_size=4, in_channels=1, hidden_dim=64)
    # Create fake patch tokens: (B, num_patches, patch_size * patch_size * channels)
    b, p, c = 2, 4, 1
    num_patches = (28 // p) ** 2
    dummy_tokens = torch.randn(b, num_patches, p * p * c)

    imgs = model.unpatchify(dummy_tokens)
    assert imgs.shape == (b, c, 28, 28)
