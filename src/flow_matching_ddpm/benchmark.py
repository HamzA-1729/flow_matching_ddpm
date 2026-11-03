"""
Comparative Benchmark & Step-Sweep Grid Generator:
Evaluates Flow Matching (Euler, Heun) vs DDPM (DDIM) under identical initial noise
across matching Number of Function Evaluations (NFE) in [5, 10, 20, 50, 100].

Usage:
    uv run python -m flow_matching_ddpm.benchmark --config configs/mnist_dit.yaml
"""

import os
import argparse
import yaml
import torch
from torchvision.utils import make_grid, save_image

from flow_matching_ddpm.model import DiT
from flow_matching_ddpm.loss import DDPMLoss
from flow_matching_ddpm.sampler import sample_flow_euler, sample_flow_heun, sample_ddim
from flow_matching_ddpm.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate NFE comparison grid")
    parser.add_argument("--config", type=str, default="configs/mnist_dit.yaml", help="Path to YAML config")
    parser.add_argument("--flow-ckpt", type=str, default="checkpoints/flow_best.pt", help="Flow matching checkpoint")
    parser.add_argument("--ddpm-ckpt", type=str, default="checkpoints/ddpm_best.pt", help="DDPM checkpoint")
    parser.add_argument("--output", type=str, default="outputs/nfe_comparison_grid.png", help="Output PNG path")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    return parser.parse_args()


def load_model_from_checkpoint(ckpt_path: str, cfg: dict, device: torch.device) -> DiT:
    """Instantiate DiT backbone and load EMA shadow weights from checkpoint."""
    ds_cfg = cfg["dataset"]
    m_cfg = cfg["model"]

    model = DiT(
        img_size=ds_cfg["img_size"],
        patch_size=m_cfg["patch_size"],
        in_channels=ds_cfg["in_channels"],
        hidden_dim=m_cfg["hidden_dim"],
        depth=m_cfg["depth"],
        num_heads=m_cfg["num_heads"],
        mlp_ratio=m_cfg.get("mlp_ratio", 4.0),
        num_classes=ds_cfg["num_classes"],
        time_scale=m_cfg.get("time_scale", 1000.0),
    ).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}. Please train the model first!")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Prefer EMA state if available for smoother generation (strict=False to allow persistent buffers)
    if "ema_state" in ckpt:
        model.load_state_dict(ckpt["ema_state"], strict=False)
        print(f"📦 Loaded EMA weights from {ckpt_path}")
    else:
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"📦 Loaded raw weights from {ckpt_path}")

    model.eval()
    return model


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🔬 Starting NFE Benchmark on device: {device}")

    # 1. Load trained models
    flow_model = load_model_from_checkpoint(args.flow_ckpt, cfg, device)
    ddpm_model = load_model_from_checkpoint(args.ddpm_ckpt, cfg, device)

    # Precompute DDPM schedule for DDIM sampler
    ddpm_loss_helper = DDPMLoss(
        model=ddpm_model,
        timesteps=cfg.get("ddpm", {}).get("timesteps", 1000),
        beta_start=float(cfg.get("ddpm", {}).get("beta_start", 1e-4)),
        beta_end=float(cfg.get("ddpm", {}).get("beta_end", 0.02)),
    ).to(device)

    # 2. Fixed initial Gaussian noise: 1 sample per class (digits 0-9)
    ds_cfg = cfg["dataset"]
    num_classes = 10
    shape = (num_classes, ds_cfg["in_channels"], ds_cfg["img_size"], ds_cfg["img_size"])
    fixed_initial_noise = torch.randn(shape, device=device)
    target_digits = torch.arange(num_classes, device=device)

    cfg_scale = float(cfg.get("sampling", {}).get("cfg_scale", 2.0))
    # With batched CFG (cfg_scale > 1.0), each model call evaluates [cond, uncond], doubling NFE:
    cfg_factor = 2 if cfg_scale > 1.0 else 1

    # Desired evaluation NFEs (Number of Function Evaluations)
    nfe_targets = cfg.get("sampling", {}).get("nfe_list", [5, 10, 20, 50, 100])
    print(f"📊 Target NFEs: {nfe_targets} (CFG Scale: {cfg_scale})")

    # Grid storage: rows = (method x NFE), cols = digits 0-9
    all_rows = []
    labels_row = []

    for nfe in nfe_targets:
        print(f"\n--- Benchmarking NFE = {nfe} ---")

        # -------------------------------------------------------------
        # 1. Flow Matching (Euler Solver)
        # Euler takes 1 forward pass per step -> steps = nfe // cfg_factor
        # -------------------------------------------------------------
        euler_steps = max(1, nfe // cfg_factor)
        print(f"  • Flow Matching (Euler)  -> {euler_steps} steps (NFE: {euler_steps * cfg_factor})")
        samples_euler = sample_flow_euler(
            model=flow_model,
            shape=shape,
            num_steps=euler_steps,
            y=target_digits,
            cfg_scale=cfg_scale,
            device=device,
            initial_noise=fixed_initial_noise,
        )
        all_rows.append(samples_euler)
        labels_row.append(f"Flow (Euler) - {nfe} NFE")

        # -------------------------------------------------------------
        # 2. Flow Matching (Heun 2nd-Order Solver)
        # Heun takes 2 forward passes per step -> steps = nfe // (2 * cfg_factor)
        # -------------------------------------------------------------
        heun_steps = max(1, nfe // (2 * cfg_factor))
        print(f"  • Flow Matching (Heun)   -> {heun_steps} steps (NFE: {heun_steps * 2 * cfg_factor})")
        samples_heun = sample_flow_heun(
            model=flow_model,
            shape=shape,
            num_steps=heun_steps,
            y=target_digits,
            cfg_scale=cfg_scale,
            device=device,
            initial_noise=fixed_initial_noise,
        )
        all_rows.append(samples_heun)
        labels_row.append(f"Flow (Heun) - {nfe} NFE")

        # -------------------------------------------------------------
        # 3. DDPM (DDIM Deterministic Solver with respaced schedule)
        # DDIM takes 1 forward pass per step -> steps = nfe // cfg_factor
        # -------------------------------------------------------------
        ddim_steps = max(1, nfe // cfg_factor)
        print(f"  • DDPM (DDIM)            -> {ddim_steps} steps (NFE: {ddim_steps * cfg_factor})")
        samples_ddim = sample_ddim(
            model=ddpm_model,
            shape=shape,
            alphas_cumprod=ddpm_loss_helper.alphas_cumprod,
            num_steps=ddim_steps,
            y=target_digits,
            cfg_scale=cfg_scale,
            device=device,
            initial_noise=fixed_initial_noise,
        )
        all_rows.append(samples_ddim)
        labels_row.append(f"DDPM (DDIM) - {nfe} NFE")

    # Combine all images into a single comparative grid
    # Stack along batch: (Total Rows * 10 digits, C, H, W)
    combined_tensor = torch.cat(all_rows, dim=0)

    # Un-normalize from [-1, 1] to [0, 1]
    combined_tensor = (combined_tensor.detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0

    # Save grid (10 digits per row)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_image(combined_tensor, args.output, nrow=num_classes)
    print(f"\n🎉 Comparative NFE Grid successfully saved to: {args.output}")


if __name__ == "__main__":
    main()

