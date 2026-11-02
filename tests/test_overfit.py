"""
Sanity check: a small DiT must be able to memorise a single batch.

Uses the Flow Matching velocity objective inline (loss.py is not implemented yet).
A model that cannot overfit one fixed batch has a bug in its forward/backward path.
"""

import torch
import torch.nn.functional as F

from flow_matching_ddpm.model import DiT


def test_single_batch_overfit():
    torch.manual_seed(0)
    model = DiT(img_size=28, patch_size=4, in_channels=1, hidden_dim=64, depth=2, num_heads=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x_data = torch.randn(8, 1, 28, 28)
    x_noise = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    t = torch.rand(8)
    x_t = (1 - t.view(-1, 1, 1, 1)) * x_noise + t.view(-1, 1, 1, 1) * x_data
    target = x_data - x_noise

    first = None
    for _ in range(100):
        loss = F.mse_loss(model(x_t, t, y), target)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert loss.item() < 0.5 * first, f"loss only went {first:.3f} -> {loss.item():.3f}"
