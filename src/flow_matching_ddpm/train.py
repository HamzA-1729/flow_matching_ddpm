"""
Unified Training Pipeline:
Trains either Flow Matching or DDPM using the shared DiT backbone and configuration.

Usage:
    uv run python -m flow_matching_ddpm.train --mode flow --config configs/mnist_dit.yaml
    uv run python -m flow_matching_ddpm.train --mode ddpm --config configs/mnist_dit.yaml
"""

import os
import argparse
import yaml
from tqdm import tqdm
import torch
from torch.optim import AdamW

from flow_matching_ddpm.model import DiT
from flow_matching_ddpm.loss import FlowMatchingLoss, DDPMLoss
from flow_matching_ddpm.sampler import sample_flow_euler, sample_ddim
from flow_matching_ddpm.dataset import get_dataloader
from flow_matching_ddpm.utils import set_seed, EMA, get_cosine_schedule_with_warmup, save_image_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Train DiT on Flow Matching or DDPM")
    parser.add_argument("--config", type=str, default="configs/mnist_dit.yaml", help="Path to YAML configuration")
    parser.add_argument("--mode", type=str, choices=["flow", "ddpm"], default="flow", help="Training paradigm")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration file
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Command line overrides
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size

    # Set deterministic random seed
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # Device auto-detection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training {args.mode.upper()} on device: {device}")

    # Build dataset loader
    ds_cfg = cfg["dataset"]
    train_loader = get_dataloader(
        name=ds_cfg["name"],
        img_size=ds_cfg["img_size"],
        in_channels=ds_cfg["in_channels"],
        batch_size=cfg["training"]["batch_size"],
        train=True,
    )

    # Build DiT Backbone
    model_cfg = cfg["model"]
    model = DiT(
        img_size=ds_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        in_channels=ds_cfg["in_channels"],
        hidden_dim=model_cfg["hidden_dim"],
        depth=model_cfg["depth"],
        num_heads=model_cfg["num_heads"],
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        num_classes=ds_cfg["num_classes"],
        time_scale=model_cfg.get("time_scale", 1000.0),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦 DiT Backbone Parameters: {num_params:,}")

    # Build Paradigm-Specific Loss Function
    if args.mode == "flow":
        loss_fn = FlowMatchingLoss(model=model).to(device)
    else:
        ddpm_cfg = cfg.get("ddpm", {})
        loss_fn = DDPMLoss(
            model=model,
            timesteps=ddpm_cfg.get("timesteps", 1000),
            beta_start=float(ddpm_cfg.get("beta_start", 1e-4)),
            beta_end=float(ddpm_cfg.get("beta_end", 0.02)),
        ).to(device)

    # Optimizer and Learning Rate Scheduler
    train_cfg = cfg["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    total_steps = len(train_loader) * train_cfg["epochs"]
    warmup_steps = train_cfg.get("warmup_steps", 250)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=float(train_cfg.get("min_lr", 1e-6)) / float(train_cfg["lr"]),
    )

    # Exponential Moving Average (EMA) tracker
    ema = EMA(model, decay=float(train_cfg.get("ema_decay", 0.999)))

    # Output directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    label_dropout = float(train_cfg.get("label_dropout", 0.10))
    null_token_idx = ds_cfg["num_classes"]

    # Training Loop
    global_step = 0
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg['epochs']}")
        for x_data, y in pbar:
            x_data = x_data.to(device)
            y = y.to(device)

            # Apply Classifier-Free Guidance (CFG) label dropout
            # Replace class label y with null_token_idx with probability label_dropout
            if label_dropout > 0.0:
                drop_mask = torch.rand(y.shape, device=device) < label_dropout
                y = torch.where(drop_mask, torch.full_like(y, null_token_idx), y)

            optimizer.zero_grad()
            loss, metrics = loss_fn(x_data, y)
            loss.backward()

            # Gradient Clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg["grad_clip"]))

            optimizer.step()
            scheduler.step()
            ema.update(model)

            epoch_loss += loss.item()
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "gnorm": f"{grad_norm.item():.2f}",
            })

        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ Epoch {epoch} complete. Average Loss: {avg_loss:.4f}")

        # Generate sample validation grid at end of epoch using EMA weights
        if epoch % 2 == 0 or epoch == train_cfg["epochs"]:
            print("🎨 Generating validation samples with EMA weights...")
            # Create temporary evaluation model and load EMA shadow weights
            eval_model = DiT(
                img_size=ds_cfg["img_size"],
                patch_size=model_cfg["patch_size"],
                in_channels=ds_cfg["in_channels"],
                hidden_dim=model_cfg["hidden_dim"],
                depth=model_cfg["depth"],
                num_heads=model_cfg["num_heads"],
                mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
                num_classes=ds_cfg["num_classes"],
                time_scale=model_cfg.get("time_scale", 1000.0),
            ).to(device)
            ema.copy_to(eval_model)
            eval_model.eval()

            # Generate digits 0 through 9
            eval_classes = torch.arange(10, device=device)
            sample_shape = (10, ds_cfg["in_channels"], ds_cfg["img_size"], ds_cfg["img_size"])

            if args.mode == "flow":
                samples = sample_flow_euler(
                    model=eval_model,
                    shape=sample_shape,
                    num_steps=20,
                    y=eval_classes,
                    cfg_scale=float(cfg.get("sampling", {}).get("cfg_scale", 2.0)),
                    device=device,
                )
            else:
                samples = sample_ddim(
                    model=eval_model,
                    shape=sample_shape,
                    alphas_cumprod=loss_fn.alphas_cumprod,
                    num_steps=20,
                    y=eval_classes,
                    cfg_scale=float(cfg.get("sampling", {}).get("cfg_scale", 2.0)),
                    device=device,
                )

            grid_path = f"outputs/{args.mode}_epoch_{epoch:02d}.png"
            save_image_grid(samples, path=grid_path, nrow=5)
            print(f"🖼️ Saved samples to {grid_path}")

            # Save checkpoint
            ckpt_path = f"checkpoints/{args.mode}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "ema_state": ema.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"💾 Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
