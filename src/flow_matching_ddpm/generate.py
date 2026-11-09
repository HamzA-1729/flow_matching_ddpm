"""
Standalone Production Inference Engine:
Generates custom digits (0-9) on demand using trained Flow Matching (Euler/Heun) or DDPM (DDIM).
Supports CPU and GPU inference, Classifier-Free Guidance (CFG), and latency reporting.

Usage:
    # 1. Generate digit 7 using Flow Matching (20 Euler steps):
    uv run python -m flow_matching_ddpm.generate --digit 7 --steps 20 --output digit_7.png

    # 2. Generate all digits (0-9) side-by-side using Heun solver:
    uv run python -m flow_matching_ddpm.generate --digit all --solver heun --steps 15 --output all_digits.png

    # 3. Generate using DDPM (DDIM sampler):
    uv run python -m flow_matching_ddpm.generate --mode ddpm --digit 3 --steps 20 --output ddpm_3.png
"""

import os
import time
import argparse
import yaml
import torch
from torchvision.utils import save_image

from flow_matching_ddpm.model import DiT
from flow_matching_ddpm.loss import DDPMLoss
from flow_matching_ddpm.sampler import sample_flow_euler, sample_flow_heun, sample_ddim
from flow_matching_ddpm.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Inference CLI for Flow Matching & DDPM")
    parser.add_argument("--config", type=str, default="configs/mnist_dit.yaml", help="Path to config YAML")
    parser.add_argument("--mode", type=str, choices=["flow", "ddpm"], default="flow", help="Model paradigm")
    parser.add_argument("--solver", type=str, choices=["euler", "heun", "ddim"], default=None, help="ODE solver (defaults to euler for flow, ddim for ddpm)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path (optional)")
    parser.add_argument("--digit", type=str, default="7", help="Target digit (0-9, or 'all')")
    parser.add_argument("--steps", type=int, default=20, help="Number of sampling steps")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="Classifier-Free Guidance weight w (1.0 = none)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    parser.add_argument("--output", type=str, default="outputs/generated_sample.png", help="Destination path for output PNG")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu or cuda)")
    return parser.parse_args()


def load_model(ckpt_path: str, cfg: dict, device: torch.device) -> DiT:
    """Instantiate DiT architecture and load trained EMA weights."""
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
        raise FileNotFoundError(f"Checkpoint not found at '{ckpt_path}'. Run training first!")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "ema_state" in ckpt:
        model.load_state_dict(ckpt["ema_state"], strict=False)
    else:
        model.load_state_dict(ckpt["model_state"], strict=False)

    model.eval()
    return model


def main():
    args = parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Set seed if specified
    if args.seed is not None:
        set_seed(args.seed)

    # Device detection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine checkpoint path
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = f"checkpoints/{args.mode}_best.pt"

    # Default solver
    if args.solver is None:
        solver = "euler" if args.mode == "flow" else "ddim"
    else:
        solver = args.solver

    # Determine target digit(s)
    if args.digit.lower() == "all":
        targets = torch.arange(10, device=device)
        batch_size = 10
        nrow = 5
    else:
        digit_val = int(args.digit)
        assert 0 <= digit_val <= 9, f"Digit must be between 0 and 9, got {digit_val}"
        targets = torch.tensor([digit_val], device=device)
        batch_size = 1
        nrow = 1

    ds_cfg = cfg["dataset"]
    shape = (batch_size, ds_cfg["in_channels"], ds_cfg["img_size"], ds_cfg["img_size"])

    print(f"\n⚡ Antigravity Generative Inference Engine")
    print(f"  • Hardware:   {device}")
    print(f"  • Model:      {args.mode.upper()} ({solver.upper()} solver, {args.steps} steps)")
    print(f"  • Checkpoint: {ckpt_path}")
    print(f"  • Target:     Digit {args.digit} (CFG scale: {args.cfg_scale})")

    # Load model
    model = load_model(ckpt_path, cfg, device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  • Parameters: {param_count:,} ({param_count * 4 / (1024*1024):.2f} MB in FP32)")

    # Timed inference run
    start_time = time.perf_counter()

    if args.mode == "flow":
        if solver == "heun":
            samples = sample_flow_heun(
                model=model,
                shape=shape,
                num_steps=args.steps,
                y=targets,
                cfg_scale=args.cfg_scale,
                device=device,
            )
        else:
            samples = sample_flow_euler(
                model=model,
                shape=shape,
                num_steps=args.steps,
                y=targets,
                cfg_scale=args.cfg_scale,
                device=device,
            )
    else:
        # DDPM mode with DDIM sampler
        ddpm_loss_helper = DDPMLoss(
            model=model,
            timesteps=cfg.get("ddpm", {}).get("timesteps", 1000),
            beta_start=float(cfg.get("ddpm", {}).get("beta_start", 1e-4)),
            beta_end=float(cfg.get("ddpm", {}).get("beta_end", 0.02)),
        ).to(device)

        samples = sample_ddim(
            model=model,
            shape=shape,
            alphas_cumprod=ddpm_loss_helper.alphas_cumprod,
            num_steps=args.steps,
            y=targets,
            cfg_scale=args.cfg_scale,
            device=device,
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Un-normalize [-1, 1] -> [0, 1]
    samples_01 = (samples.detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_image(samples_01, args.output, nrow=nrow)

    print(f"\n✅ Generation complete in {elapsed_ms:.1f} ms!")
    print(f"🖼️ Saved output to: {args.output}\n")


if __name__ == "__main__":
    main()

