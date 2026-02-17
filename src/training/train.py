"""
Training loop for TAVSE speech enhancement models.

Features:
    - Mixed precision (AMP) training
    - Warmup + cosine LR schedule
    - Early stopping on validation SI-SNR
    - Top-K checkpoint management
    - TensorBoard logging
    - Gradient accumulation support

Usage:
    python -m src.training.train --config configs/audio_only.yaml
    python -m src.training.train --config configs/audio_rgb_thermal.yaml --resume
"""

import os
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Dict

from src.utils.config import load_config, load_config_with_overrides, TAVSEConfig
from src.models.tavse_model import TAVSEModel
from src.training.losses import TAVSELoss
from src.data.dataset import TAVSEDataset, collate_fn
from src.data.noise_mixer import NoiseMixer


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cosine_schedule_with_warmup(
    optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6
):
    """Cosine decay LR schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr / optimizer.defaults["lr"], cosine_decay)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class CheckpointManager:
    """Manages top-K checkpoints by validation SI-SNR."""

    def __init__(self, save_dir: str, keep_top_k: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.checkpoints = []  # list of (metric, path)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler,
        epoch: int,
        step: int,
        metric: float,
        cfg: TAVSEConfig,
    ):
        """Save checkpoint and manage top-K."""
        ckpt_path = self.save_dir / f"ckpt_epoch{epoch:03d}_sisnr{metric:.2f}.pt"
        latest_path = self.save_dir / "latest.pt"

        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "epoch": epoch,
            "step": step,
            "metric": metric,
            "config": {
                "experiment_name": cfg.experiment_name,
                "active_modalities": cfg.model.active_modalities,
            },
        }

        # Save latest
        torch.save(state, latest_path)

        # Save ranked checkpoint
        torch.save(state, ckpt_path)
        self.checkpoints.append((metric, str(ckpt_path)))

        # Keep only top-K by metric (higher is better for SI-SNR)
        self.checkpoints.sort(key=lambda x: x[0], reverse=True)
        while len(self.checkpoints) > self.keep_top_k:
            _, old_path = self.checkpoints.pop()
            if os.path.exists(old_path):
                os.remove(old_path)

    def load_latest(self, model, optimizer=None, scheduler=None, scaler=None):
        """Load the latest checkpoint. Returns (epoch, step, metric) or None."""
        latest = self.save_dir / "latest.pt"
        if not latest.exists():
            return None

        ckpt = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        return ckpt["epoch"], ckpt["step"], ckpt["metric"]


def train_epoch(
    model: TAVSEModel,
    loader: DataLoader,
    criterion: TAVSELoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    cfg: TAVSEConfig,
    epoch: int,
    global_step: int,
    writer: SummaryWriter,
) -> tuple:
    """Run one training epoch. Returns (avg_loss, global_step)."""
    model.train()
    total_loss = 0.0
    total_sisnr = 0.0
    total_mag = 0.0
    n_batches = 0

    use_rgb = "rgb" in cfg.model.active_modalities
    use_thermal = "thermal" in cfg.model.active_modalities
    device = next(model.parameters()).device

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        noisy_mag = batch["noisy_mag"].to(device)
        noisy_phase = batch["noisy_phase"].to(device)
        clean_mag = batch["clean_mag"].to(device)
        clean_wav = batch["clean_wav"].to(device)

        rgb_frames = batch["rgb_frames"].to(device) if use_rgb and batch["rgb_frames"] is not None else None
        thr_frames = batch["thr_frames"].to(device) if use_thermal and batch["thr_frames"] is not None else None

        # Forward pass with mixed precision
        with torch.amp.autocast("cuda", enabled=cfg.training.use_amp):
            outputs = model(noisy_mag, noisy_phase, rgb_frames, thr_frames)

        # Loss computation in FP32
        with torch.amp.autocast("cuda", enabled=False):
            losses = criterion(
                outputs["enhanced_wav"].float(),
                clean_wav.float(),
                outputs["enhanced_mag"].float(),
                clean_mag.float(),
            )
            loss = losses["total"] / cfg.training.grad_accum_steps

        # Backward
        scaler.scale(loss).backward()

        # Optimizer step (with gradient accumulation)
        if (batch_idx + 1) % cfg.training.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

        total_loss += losses["total"].item()
        total_sisnr += losses["si_snr"].item()
        total_mag += losses["magnitude"].item()
        n_batches += 1

        # Logging
        if global_step % cfg.training.log_interval == 0 and (batch_idx + 1) % cfg.training.grad_accum_steps == 0:
            writer.add_scalar("train/loss_total", losses["total"].item(), global_step)
            writer.add_scalar("train/loss_sisnr", losses["si_snr"].item(), global_step)
            writer.add_scalar("train/loss_mag", losses["magnitude"].item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, global_step


@torch.no_grad()
def validate(
    model: TAVSEModel,
    loader: DataLoader,
    criterion: TAVSELoss,
    cfg: TAVSEConfig,
    epoch: int,
    writer: SummaryWriter,
) -> float:
    """Run validation. Returns average SI-SNR (higher is better)."""
    model.eval()
    total_sisnr = 0.0
    total_loss = 0.0
    n_batches = 0

    use_rgb = "rgb" in cfg.model.active_modalities
    use_thermal = "thermal" in cfg.model.active_modalities
    device = next(model.parameters()).device

    for batch in loader:
        noisy_mag = batch["noisy_mag"].to(device)
        noisy_phase = batch["noisy_phase"].to(device)
        clean_mag = batch["clean_mag"].to(device)
        clean_wav = batch["clean_wav"].to(device)

        rgb_frames = batch["rgb_frames"].to(device) if use_rgb and batch["rgb_frames"] is not None else None
        thr_frames = batch["thr_frames"].to(device) if use_thermal and batch["thr_frames"] is not None else None

        outputs = model(noisy_mag, noisy_phase, rgb_frames, thr_frames)

        losses = criterion(
            outputs["enhanced_wav"],
            clean_wav,
            outputs["enhanced_mag"],
            clean_mag,
        )

        # Compute SI-SNR (positive, higher is better)
        val_sisnr = -losses["si_snr"].item()
        total_sisnr += val_sisnr
        total_loss += losses["total"].item()
        n_batches += 1

    avg_sisnr = total_sisnr / max(n_batches, 1)
    avg_loss = total_loss / max(n_batches, 1)

    writer.add_scalar("val/loss_total", avg_loss, epoch)
    writer.add_scalar("val/si_snr", avg_sisnr, epoch)

    return avg_sisnr


def main():
    parser = argparse.ArgumentParser(description="TAVSE Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--override", type=str, default=None, help="JSON string of config overrides")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────
    overrides = json.loads(args.override) if args.override else None
    cfg = load_config_with_overrides(args.config, overrides)
    set_seed(cfg.training.seed)

    print(f"\n{'='*60}")
    print(f"TAVSE Training: {cfg.experiment_name}")
    print(f"Active modalities: {cfg.model.active_modalities}")
    print(f"{'='*60}\n")

    # ── Device ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ── Model ─────────────────────────────────────────────────────
    model = TAVSEModel.from_config(cfg).to(device)
    param_counts = model.get_num_params()
    print(f"\nModel parameters:")
    for k, v in param_counts.items():
        print(f"  {k}: {v:,}")

    # ── Data ──────────────────────────────────────────────────────
    train_dataset = TAVSEDataset(cfg, split="train")
    val_dataset = TAVSEDataset(cfg, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        persistent_workers=cfg.training.persistent_workers,
        prefetch_factor=cfg.training.prefetch_factor,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        persistent_workers=cfg.training.persistent_workers,
        collate_fn=collate_fn,
    )

    print(f"\nDataset: {len(train_dataset)} train, {len(val_dataset)} val")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────
    criterion = TAVSELoss(
        si_snr_weight=cfg.training.loss_si_snr_weight,
        mag_weight=cfg.training.loss_mag_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=tuple(cfg.training.betas),
    )

    total_steps = len(train_loader) * cfg.training.max_epochs // cfg.training.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, cfg.training.warmup_steps, total_steps
    )

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.use_amp)

    # ── Checkpoint & Logging ──────────────────────────────────────
    ckpt_mgr = CheckpointManager(cfg.checkpoint_dir, cfg.training.keep_top_k)
    writer = SummaryWriter(log_dir=cfg.log_dir)

    # ── Resume ────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_sisnr = float("-inf")

    if args.resume:
        result = ckpt_mgr.load_latest(model, optimizer, scheduler, scaler)
        if result is not None:
            start_epoch, global_step, best_sisnr = result
            start_epoch += 1  # start from next epoch
            print(f"Resumed from epoch {start_epoch - 1}, step {global_step}, SI-SNR {best_sisnr:.2f}")
        else:
            print("No checkpoint found, starting from scratch")

    # ── Training loop ─────────────────────────────────────────────
    patience_counter = 0

    for epoch in range(start_epoch, cfg.training.max_epochs):
        epoch_start = time.time()

        # Train
        avg_loss, global_step = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, cfg, epoch, global_step, writer,
        )

        # Validate
        if (epoch + 1) % cfg.training.val_interval == 0:
            val_sisnr = validate(model, val_loader, criterion, cfg, epoch, writer)

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch:3d}/{cfg.training.max_epochs} | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Val SI-SNR: {val_sisnr:.2f} dB | "
                f"Best: {best_sisnr:.2f} dB | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Checkpoint
            ckpt_mgr.save(
                model, optimizer, scheduler, scaler,
                epoch, global_step, val_sisnr, cfg,
            )

            # Early stopping
            if val_sisnr > best_sisnr:
                best_sisnr = val_sisnr
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.training.patience:
                    print(f"\nEarly stopping at epoch {epoch} (patience={cfg.training.patience})")
                    break

    writer.close()
    print(f"\nTraining complete. Best SI-SNR: {best_sisnr:.2f} dB")
    print(f"Checkpoints: {cfg.checkpoint_dir}")
    print(f"Logs: {cfg.log_dir}")


if __name__ == "__main__":
    main()
