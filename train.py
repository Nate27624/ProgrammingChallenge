"""
train.py
========
Training and validation pipeline for the speech emotion recognition challenge.

Trains the baseline model, saves the best checkpoint, and plots loss curves.
Integrates Weights & Biases (wandb) for experiment tracking.

Usage:
    python train.py --data_dir dataset --results_dir results --team_name "Team Awesome"

    # Optional: give the run a custom name
    python train.py --data_dir dataset --results_dir results --team_name "Team Awesome" --run_name baseline_experiment

Outputs saved to <results_dir>/<team_name>/:
    best_model.pt     <- model weights with the lowest validation loss
    checkpoint.pt     <- latest checkpoint for resuming interrupted runs
    norm_stats.pt     <- training set normalisation statistics (needed by test.py)
    loss_curve.png    <- training and validation loss/accuracy curves
"""

import argparse
import json
import math
import os
import random
import sys
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from dataloader import get_dataloaders, N_MELS, N_MFCC
from baseline import BaselineLSTM
from model import BidirectionalMambaSER, CNNBiLSTMAttentionSER, TemporalFrequencyMambaSER
from wandb_compat import wandb


# ── Default hyperparameters ────────────────────────────────────────────────────
CONFIG = {
    "model_name": "mamba",
    "feature_type": "mfcc",
    "n_features": N_MFCC,
    "hidden_size":   128,
    "num_layers":    2,
    "dropout":       0.35,
    "mamba_d_model": 96,
    "mamba_d_state": 32,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
    "cnn_channels": 48,
    "frontend_type": "basic_cnn",
    "fusion_type": "concat",
    "pooling_type": "meanmax",
    "batch_size":    64,
    "learning_rate": 3e-4,
    "weight_decay":  1e-4,
    "optimizer_type": "adamw",
    "sam_rho": 0.05,
    "max_grad_norm": 1.0,
    "mixup_alpha": 0.0,
    "mixup_prob": 0.0,
    "loss_type": "ce",
    "focal_gamma": 2.0,
    "class_weighting": False,
    "label_smoothing": 0.05,
    "scheduler_type": "plateau",
    "warmup_epochs": 5,
    "min_lr_ratio": 0.1,
    "num_epochs":    100,
    "use_swa": False,
    "swa_start_epoch": 60,
    "swa_lr": 1e-4,
    "patience":      10,      # early stopping patience (epochs)
    "patience_lr":   5,      # ReduceLROnPlateau patience (epochs)
    "val_split":     0.15,
    "num_folds": 0,
    "fold_index": -1,
    "include_deltas": True,
    "specaugment": True,
    "num_time_masks": 2,
    "time_mask_param": 24,
    "num_freq_masks": 2,
    "freq_mask_param": 8,
    "speed_perturb_prob": 0.0,
    "speed_perturb_min": 0.9,
    "speed_perturb_max": 1.1,
    "num_workers": 0,
    "specaugment_on_gpu": True,
    "feature_cache_dir": "",
    "seed": 42,
}


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional class weighting and label smoothing."""

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        nll = -log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        if self.label_smoothing > 0.0:
            smooth_loss = -log_probs.mean(dim=1)
            ce = (1.0 - self.label_smoothing) * nll + self.label_smoothing * smooth_loss
        else:
            ce = nll

        if self.class_weights is not None:
            alpha_t = self.class_weights.gather(dim=0, index=targets)
        else:
            alpha_t = torch.ones_like(ce)

        focal_factor = (1.0 - pt).pow(self.gamma)
        loss = alpha_t * focal_factor * ce
        return loss.mean()


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization wrapper for a base optimizer."""

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho value: {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = ((torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("SAM requires closure; use first_step/second_step.")
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                scale = torch.abs(p) if group["adaptive"] else 1.0
                norms.append((scale * p.grad).norm(p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)


def _extract_train_labels(loader) -> list[int]:
    dataset = loader.dataset
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        base = dataset.dataset
        return [int(base.samples[idx][1]) for idx in dataset.indices]
    if hasattr(dataset, "samples"):
        return [int(label) for _, label in dataset.samples]
    raise ValueError("Unable to extract labels from train loader dataset.")


def build_class_weights(train_labels: list[int], device: torch.device) -> torch.Tensor:
    labels = torch.tensor(train_labels, dtype=torch.long)
    num_classes = int(labels.max().item()) + 1
    counts = torch.bincount(labels, minlength=num_classes).float().clamp_min(1.0)
    weights = counts.sum() / (counts * num_classes)
    return weights.to(device)


def build_lr_lambda(
    schedule_type: str,
    num_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
):
    warmup_epochs = max(1, warmup_epochs)

    if schedule_type == "warmup_invsqrt":
        def lr_lambda(epoch: int) -> float:
            step = epoch + 1
            if step <= warmup_epochs:
                return step / warmup_epochs
            return (warmup_epochs ** 0.5) / (step ** 0.5)
        return lr_lambda

    if schedule_type == "warmup_cosine":
        decay_epochs = max(1, num_epochs - warmup_epochs)

        def lr_lambda(epoch: int) -> float:
            step = epoch + 1
            if step <= warmup_epochs:
                return step / warmup_epochs
            progress = min(1.0, (step - warmup_epochs) / decay_epochs)
            cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        return lr_lambda

    raise ValueError(f"Unsupported scheduler_type: {schedule_type}")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_specaugment_batch(
    specs: torch.Tensor,
    num_time_masks: int,
    time_mask_param: int,
    num_freq_masks: int,
    freq_mask_param: int,
) -> torch.Tensor:
    """In-place style SpecAugment on GPU batch tensors (B, C, F, T)."""
    if specs.ndim != 4:
        return specs
    bsz, _, n_freq, n_time = specs.shape
    out = specs.clone()
    max_f = max(0, min(freq_mask_param, n_freq))
    max_t = max(0, min(time_mask_param, n_time))

    if max_f > 0 and num_freq_masks > 0:
        for _ in range(num_freq_masks):
            widths = torch.randint(0, max_f + 1, (bsz,), device=out.device)
            starts = torch.floor(
                torch.rand(bsz, device=out.device) * (n_freq - widths + 1).float()
            ).long()
            for i in range(bsz):
                width = int(widths[i].item())
                if width > 0:
                    start = int(starts[i].item())
                    out[i, :, start:start + width, :] = 0.0

    if max_t > 0 and num_time_masks > 0:
        for _ in range(num_time_masks):
            widths = torch.randint(0, max_t + 1, (bsz,), device=out.device)
            starts = torch.floor(
                torch.rand(bsz, device=out.device) * (n_time - widths + 1).float()
            ).long()
            for i in range(bsz):
                width = int(widths[i].item())
                if width > 0:
                    start = int(starts[i].item())
                    out[i, :, :, start:start + width] = 0.0
    return out


# ── One training epoch ─────────────────────────────────────────────────────────
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    max_grad_norm=None,
    mixup_alpha: float = 0.0,
    mixup_prob: float = 0.0,
    apply_gpu_specaugment: bool = False,
    num_time_masks: int = 0,
    time_mask_param: int = 0,
    num_freq_masks: int = 0,
    freq_mask_param: int = 0,
    use_sam: bool = False,
):
    """Run one training epoch. Logs step-level loss to wandb."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    skipped_oom = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    non_blocking = device.type == "cuda"
    for specs, labels in pbar:
        try:
            specs = specs.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            if apply_gpu_specaugment:
                specs = apply_specaugment_batch(
                    specs,
                    num_time_masks=num_time_masks,
                    time_mask_param=time_mask_param,
                    num_freq_masks=num_freq_masks,
                    freq_mask_param=freq_mask_param,
                )

            optimizer.zero_grad()
            use_mixup = (
                mixup_alpha > 0.0
                and mixup_prob > 0.0
                and torch.rand(1).item() < mixup_prob
                and specs.size(0) > 1
            )
            lam = None
            perm = None
            mixed_specs = specs
            if use_mixup:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(specs.size(0), device=device)
                mixed_specs = lam * specs + (1.0 - lam) * specs[perm]
            mixed_labels = labels

            if use_sam:
                logits = model(mixed_specs)
                if use_mixup:
                    loss = lam * criterion(logits, labels) + (1.0 - lam) * criterion(logits, labels[perm])
                else:
                    loss = criterion(logits, labels)
                loss.backward()
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, foreach=False)
                optimizer.first_step(zero_grad=True)

                logits = model(mixed_specs)
                if use_mixup:
                    loss_second = lam * criterion(logits, labels) + (1.0 - lam) * criterion(logits, labels[perm])
                else:
                    loss_second = criterion(logits, labels)
                loss_second.backward()
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, foreach=False)
                optimizer.second_step(zero_grad=True)
                loss_value = loss_second.item()
            else:
                logits = model(mixed_specs)
                if use_mixup:
                    loss = lam * criterion(logits, labels) + (1.0 - lam) * criterion(logits, labels[perm])
                else:
                    loss = criterion(logits, labels)
                loss.backward()
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, foreach=False)
                optimizer.step()
                loss_value = loss.item()
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg and device.type == "cuda":
                skipped_oom += 1
                optimizer.zero_grad(set_to_none=True)
                if "specs" in locals():
                    del specs
                if "labels" in locals():
                    del labels
                torch.cuda.empty_cache()
                pbar.set_postfix({"oom_skips": skipped_oom})
                continue
            raise

        total_loss += loss_value * specs.size(0)
        correct    += (logits.argmax(dim=1) == mixed_labels).sum().item()
        total      += specs.size(0)

        # Step-level logging
        wandb.log({"train/step_loss": loss_value})

        pbar.set_postfix({"loss": f"{loss_value:.4f}"})

    if skipped_oom > 0:
        print(f"  [warn] skipped {skipped_oom} train batch(es) due to CUDA OOM.")
    if total == 0:
        return float("inf"), 0.0
    return total_loss / total, correct / total


# ── Validation epoch ───────────────────────────────────────────────────────────
def validate(model, loader, criterion, device):
    """Run one validation epoch."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    skipped_oom = 0

    pbar = tqdm(loader, desc="  Val  ", leave=False)
    non_blocking = device.type == "cuda"
    with torch.no_grad():
        for specs, labels in pbar:
            try:
                specs = specs.to(device, non_blocking=non_blocking)
                labels = labels.to(device, non_blocking=non_blocking)
                logits = model(specs)
                loss   = criterion(logits, labels)
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg and device.type == "cuda":
                    skipped_oom += 1
                    if "specs" in locals():
                        del specs
                    if "labels" in locals():
                        del labels
                    torch.cuda.empty_cache()
                    pbar.set_postfix({"oom_skips": skipped_oom})
                    continue
                raise

            total_loss += loss.item() * specs.size(0)
            preds = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += specs.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if skipped_oom > 0:
        print(f"  [warn] skipped {skipped_oom} val batch(es) due to CUDA OOM.")
    if total == 0:
        return float("inf"), 0.0, 0.0
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / total, correct / total, weighted_f1


# ── Loss curve plotting ────────────────────────────────────────────────────────
def plot_curves(train_losses, val_losses, train_accs, val_accs,
                save_path, stopped_epoch=None):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, train_losses, "b-o", markersize=4, label="Train")
    ax1.plot(epochs, val_losses,   "r-o", markersize=4, label="Validation")
    if stopped_epoch:
        ax1.axvline(x=stopped_epoch, color="gray", linestyle="--",
                    label=f"Early stop (epoch {stopped_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in train_accs], "b-o", markersize=4,
             label="Train")
    ax2.plot(epochs, [a * 100 for a in val_accs],   "r-o", markersize=4,
             label="Validation")
    if stopped_epoch:
        ax2.axvline(x=stopped_epoch, color="gray", linestyle="--",
                    label=f"Early stop (epoch {stopped_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss curves saved to: {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    config = CONFIG.copy()
    config["model_name"] = args.model_name
    config["feature_type"] = args.feature_type
    if args.n_features is None:
        config["n_features"] = N_MFCC if args.feature_type == "mfcc" else N_MELS
    else:
        config["n_features"] = args.n_features
    config["include_deltas"] = args.include_deltas
    config["specaugment"] = args.specaugment
    config["num_time_masks"] = args.num_time_masks
    config["time_mask_param"] = args.time_mask_param
    config["num_freq_masks"] = args.num_freq_masks
    config["freq_mask_param"] = args.freq_mask_param
    config["hidden_size"] = args.hidden_size
    config["num_layers"] = args.num_layers
    config["dropout"] = args.dropout
    config["mamba_d_model"] = args.mamba_d_model
    config["mamba_d_state"] = args.mamba_d_state
    config["mamba_d_conv"] = args.mamba_d_conv
    config["mamba_expand"] = args.mamba_expand
    config["cnn_channels"] = args.cnn_channels
    config["frontend_type"] = args.frontend_type
    config["fusion_type"] = args.fusion_type
    config["pooling_type"] = args.pooling_type
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.learning_rate
    config["weight_decay"] = args.weight_decay
    config["optimizer_type"] = args.optimizer_type
    config["sam_rho"] = args.sam_rho
    config["max_grad_norm"] = args.max_grad_norm
    config["mixup_alpha"] = args.mixup_alpha
    config["mixup_prob"] = args.mixup_prob
    config["loss_type"] = args.loss_type
    config["focal_gamma"] = args.focal_gamma
    config["class_weighting"] = args.class_weighting
    config["label_smoothing"] = args.label_smoothing
    config["scheduler_type"] = args.scheduler_type
    config["warmup_epochs"] = args.warmup_epochs
    config["min_lr_ratio"] = args.min_lr_ratio
    config["num_epochs"] = args.num_epochs
    config["use_swa"] = args.use_swa
    config["swa_start_epoch"] = args.swa_start_epoch
    config["swa_lr"] = args.swa_lr
    config["patience"] = args.patience
    config["patience_lr"] = args.patience_lr
    config["val_split"] = args.val_split
    config["num_folds"] = args.num_folds
    config["fold_index"] = args.fold_index
    config["speed_perturb_prob"] = args.speed_perturb_prob
    config["speed_perturb_min"] = args.speed_perturb_min
    config["speed_perturb_max"] = args.speed_perturb_max
    config["num_workers"] = args.num_workers
    config["specaugment_on_gpu"] = args.specaugment_on_gpu
    config["feature_cache_dir"] = args.feature_cache_dir
    config["seed"] = args.seed

    # Windows multiprocessing data workers are unstable under low pagefile
    # conditions (shared-file mapping errors 1455). Force single-process loading.
    if os.name == "nt" and config["num_workers"] > 0:
        print(
            f"Warning: forcing num_workers from {config['num_workers']} to 0 on Windows "
            "to avoid shared-memory/pagefile worker crashes."
        )
        config["num_workers"] = 0
    if config["num_folds"] > 1 and config["fold_index"] < 0:
        raise ValueError("When --num_folds > 1, you must set --fold_index >= 0.")

    output_dir = Path(args.results_dir) / args.team_name.replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "checkpoint.pt"
    best_model_path = output_dir / "best_model.pt"
    norm_stats_path = output_dir / "norm_stats.pt"

    set_global_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        config["specaugment_on_gpu"] = False

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_loader, val_loader, _, mean, std = get_dataloaders(
        data_dir   = args.data_dir,
        val_split  = config["val_split"],
        batch_size = config["batch_size"],
        num_workers = config["num_workers"],
        include_deltas = config["include_deltas"],
        apply_specaugment = config["specaugment"] and not config["specaugment_on_gpu"],
        num_time_masks = config["num_time_masks"],
        time_mask_param = config["time_mask_param"],
        num_freq_masks = config["num_freq_masks"],
        freq_mask_param = config["freq_mask_param"],
        speed_perturb_prob = config["speed_perturb_prob"],
        speed_perturb_min = config["speed_perturb_min"],
        speed_perturb_max = config["speed_perturb_max"],
        feature_type = config["feature_type"],
        n_features = config["n_features"],
        feature_cache_dir = config["feature_cache_dir"] or None,
        num_folds = (config["num_folds"] if config["num_folds"] and config["num_folds"] > 1 else None),
        fold_index = (config["fold_index"] if config["fold_index"] >= 0 else None),
    )
    if config["specaugment"] and config["specaugment_on_gpu"]:
        print(
            "SpecAugment mode: GPU batch augmentation "
            f"(time_masks={config['num_time_masks']}, freq_masks={config['num_freq_masks']})"
        )
    torch.save(
        {
            "mean": mean,
            "std": std,
            "include_deltas": config["include_deltas"],
            "feature_type": config["feature_type"],
            "n_features": config["n_features"],
            "model_name": config["model_name"],
            "model_config": {
                "hidden_size": config["hidden_size"],
                "num_layers": config["num_layers"],
                "dropout": config["dropout"],
                "mamba_d_model": config["mamba_d_model"],
                "mamba_d_state": config["mamba_d_state"],
                "mamba_d_conv": config["mamba_d_conv"],
                "mamba_expand": config["mamba_expand"],
                "cnn_channels": config["cnn_channels"],
                "frontend_type": config["frontend_type"],
                "fusion_type": config["fusion_type"],
                "pooling_type": config["pooling_type"],
                "weight_decay": config["weight_decay"],
                "optimizer_type": config["optimizer_type"],
                "sam_rho": config["sam_rho"],
                "max_grad_norm": config["max_grad_norm"],
                "mixup_alpha": config["mixup_alpha"],
                "mixup_prob": config["mixup_prob"],
                "label_smoothing": config["label_smoothing"],
                "loss_type": config["loss_type"],
                "focal_gamma": config["focal_gamma"],
                "class_weighting": config["class_weighting"],
                "scheduler_type": config["scheduler_type"],
                "warmup_epochs": config["warmup_epochs"],
                "min_lr_ratio": config["min_lr_ratio"],
                "use_swa": config["use_swa"],
                "swa_start_epoch": config["swa_start_epoch"],
                "swa_lr": config["swa_lr"],
                "speed_perturb_prob": config["speed_perturb_prob"],
                "speed_perturb_min": config["speed_perturb_min"],
                "speed_perturb_max": config["speed_perturb_max"],
                "seed": config["seed"],
                "num_folds": config["num_folds"],
                "fold_index": config["fold_index"],
            },
        },
        norm_stats_path
    )
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(output_dir / "train_command.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(sys.argv) + "\n")

    # ── Model, optimiser, scheduler ───────────────────────────────────────────
    in_channels = 3 if config["include_deltas"] else 1
    if config["model_name"] == "mamba":
        model = BidirectionalMambaSER(
            in_channels=in_channels,
            n_features=config["n_features"],
            cnn_channels=config["cnn_channels"],
            d_model=config["mamba_d_model"],
            d_state=config["mamba_d_state"],
            d_conv=config["mamba_d_conv"],
            expand=config["mamba_expand"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            frontend_type=config["frontend_type"],
            fusion_type=config["fusion_type"],
            pooling_type=config["pooling_type"],
        ).to(device)
    elif config["model_name"] == "tf_mamba":
        model = TemporalFrequencyMambaSER(
            in_channels=in_channels,
            n_features=config["n_features"],
            d_model=config["mamba_d_model"],
            d_state=config["mamba_d_state"],
            d_conv=config["mamba_d_conv"],
            expand=config["mamba_expand"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            pooling_type=config["pooling_type"],
        ).to(device)
    elif config["model_name"] == "bilstm_attention":
        model = CNNBiLSTMAttentionSER(
            in_channels=in_channels,
            n_features=config["n_features"],
            cnn_channels=config["cnn_channels"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        ).to(device)
    elif config["model_name"] == "baseline":
        input_size = config["n_features"] * in_channels
        model = BaselineLSTM(
            input_size=input_size,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_name: {config['model_name']}")

    print(f"\nModel: {model.__class__.__name__}")
    print(f"Trainable parameters: {model.count_parameters():,}")

    class_weights = None
    if config["class_weighting"]:
        train_labels = _extract_train_labels(train_loader)
        class_weights = build_class_weights(train_labels, device)
        print(f"Using class-weighted loss with weights: {class_weights.tolist()}")

    if config["loss_type"] == "ce":
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config["label_smoothing"],
        )
    elif config["loss_type"] == "focal":
        criterion = FocalLoss(
            gamma=config["focal_gamma"],
            class_weights=class_weights,
            label_smoothing=config["label_smoothing"],
        )
    else:
        raise ValueError(f"Unsupported loss_type: {config['loss_type']}")

    if config["optimizer_type"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    elif config["optimizer_type"] == "sam":
        optimizer = SAM(
            model.parameters(),
            base_optimizer=torch.optim.AdamW,
            rho=config["sam_rho"],
            adaptive=False,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    else:
        raise ValueError(f"Unsupported optimizer_type: {config['optimizer_type']}")

    swa_model = None
    swa_scheduler = None
    swa_updates = 0
    if config["use_swa"]:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=config["swa_lr"])

    if config["scheduler_type"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=config["patience_lr"]
        )
    else:
        lr_lambda = build_lr_lambda(
            schedule_type=config["scheduler_type"],
            num_epochs=config["num_epochs"],
            warmup_epochs=config["warmup_epochs"],
            min_lr_ratio=config["min_lr_ratio"],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ── Resume logic ──────────────────────────────────────────────────────────
    start_epoch      = 0
    best_val_loss    = float("inf")
    best_val_f1      = float("-inf")
    patience_counter = 0
    stopped_epoch    = None
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    if hasattr(wandb, "util") and hasattr(wandb.util, "generate_id"):
        wandb_id = wandb.util.generate_id()
    else:
        wandb_id = uuid.uuid4().hex[:8]

    if checkpoint_path.exists():
        print(f"\nFound checkpoint at {checkpoint_path}. Resuming training...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as exc:
            print(f"Warning: could not restore scheduler state ({exc}). Continuing with fresh scheduler.")
        start_epoch      = ckpt["epoch"] + 1
        best_val_loss    = ckpt["best_val_loss"]
        best_val_f1      = ckpt.get("best_val_f1", float("-inf"))
        patience_counter = ckpt["patience_counter"]
        train_losses     = ckpt["train_losses"]
        val_losses       = ckpt["val_losses"]
        train_accs       = ckpt["train_accs"]
        val_accs         = ckpt["val_accs"]
        wandb_id         = ckpt["wandb_id"]
        swa_updates      = int(ckpt.get("swa_updates", 0))
        if swa_model is not None and "swa_model_state_dict" in ckpt and ckpt["swa_model_state_dict"] is not None:
            swa_model.load_state_dict(ckpt["swa_model_state_dict"])
        print(f"Resuming from epoch {start_epoch} | "
              f"best val loss so far: {best_val_loss:.4f}")
    else:
        print("No checkpoint found. Starting from scratch.")

    # ── Wandb initialisation ──────────────────────────────────────────────────
    # resume="allow" appends to the existing run when wandb_id matches
    run_name = args.run_name or (
        f"{config['model_name']}-{config['feature_type']}-lr{config['learning_rate']}"
    )
    wandb.init(
        project = "CSE 5526 - Programming Challenge",
        name    = run_name,
        config  = config,
        id      = wandb_id,
        resume  = "allow",
    )
    # Define epoch as the x-axis for all epoch-level metrics
    wandb.define_metric("epoch")
    wandb.define_metric("epoch/*", step_metric="epoch")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for up to {config['num_epochs']} epochs "
          f"(early stopping patience = {config['patience']})...\n")

    for epoch in range(start_epoch, config["num_epochs"]):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config["max_grad_norm"],
            mixup_alpha=config["mixup_alpha"],
            mixup_prob=config["mixup_prob"],
            apply_gpu_specaugment=config["specaugment"] and config["specaugment_on_gpu"],
            num_time_masks=config["num_time_masks"],
            time_mask_param=config["time_mask_param"],
            num_freq_masks=config["num_freq_masks"],
            freq_mask_param=config["freq_mask_param"],
            use_sam=config["optimizer_type"] == "sam",
        )
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, device
        )
        if config["use_swa"] and swa_model is not None and swa_scheduler is not None and epoch >= config["swa_start_epoch"]:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            swa_updates += 1
        elif config["scheduler_type"] == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        lr = optimizer.param_groups[0]["lr"]

        # Epoch-level wandb logging
        wandb.log({
            "epoch":             epoch,
            "epoch/train_loss":  train_loss,
            "epoch/val_loss":    val_loss,
            "epoch/train_acc":   train_acc * 100,
            "epoch/val_acc":     val_acc   * 100,
            "epoch/val_f1":      val_f1,
            "epoch/lr":          lr,
        })

        print(f"Epoch {epoch + 1:>3}/{config['num_epochs']}  "
              f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f}  "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}  "
              f"val_f1: {val_f1:.4f}  "
              f"lr: {lr:.2e}")

        # Save best model by weighted F1; use loss as tiebreaker
        is_better_f1 = val_f1 > best_val_f1 + 1e-6
        is_tie_better_loss = abs(val_f1 - best_val_f1) <= 1e-6 and val_loss < best_val_loss
        if is_better_f1 or is_tie_better_loss:
            best_val_f1      = val_f1
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(
                "  --> New best model saved "
                f"(val_f1: {best_val_f1:.4f}, val_loss: {best_val_loss:.4f})"
            )
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                stopped_epoch = epoch + 1
                print(f"\nEarly stopping triggered at epoch {stopped_epoch}.")
                break

        # Save latest checkpoint for resuming interrupted runs
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss":        best_val_loss,
            "best_val_f1":          best_val_f1,
            "patience_counter":     patience_counter,
            "train_losses":         train_losses,
            "val_losses":           val_losses,
            "train_accs":           train_accs,
            "val_accs":             val_accs,
            "wandb_id":             wandb_id,
            "swa_updates":          swa_updates,
            "swa_model_state_dict": swa_model.state_dict() if swa_model is not None else None,
            "config":               config,
        }, checkpoint_path)

    # ── Post-training ──────────────────────────────────────────────────────────
    if config["use_swa"] and swa_model is not None and swa_updates > 0:
        print(f"\nFinalizing SWA model from {swa_updates} update(s)...")
        update_bn(train_loader, swa_model, device=device)
        swa_val_loss, swa_val_acc, swa_val_f1 = validate(swa_model, val_loader, criterion, device)
        print(
            f"SWA val_loss: {swa_val_loss:.4f}  val_acc: {swa_val_acc:.4f}  val_f1: {swa_val_f1:.4f}"
        )
        wandb.log(
            {
                "swa/val_loss": swa_val_loss,
                "swa/val_acc": swa_val_acc * 100.0,
                "swa/val_f1": swa_val_f1,
                "swa/updates": swa_updates,
            }
        )
        is_better_swa = swa_val_f1 > best_val_f1 + 1e-6
        is_swa_tie_better_loss = abs(swa_val_f1 - best_val_f1) <= 1e-6 and swa_val_loss < best_val_loss
        if is_better_swa or is_swa_tie_better_loss:
            best_val_f1 = swa_val_f1
            best_val_loss = swa_val_loss
            torch.save(swa_model.state_dict(), best_model_path)
            print("  --> SWA replaced best_model.pt")
        torch.save(swa_model.state_dict(), output_dir / "swa_model.pt")

    plot_curves(
        train_losses, val_losses,
        train_accs,   val_accs,
        save_path     = output_dir / "loss_curve.png",
        stopped_epoch = stopped_epoch,
    )

    # Upload final loss curve image to wandb
    wandb.log({"loss_curve": wandb.Image(str(output_dir / "loss_curve.png"))})

    print(f"\nBest model saved to : {best_model_path}")
    print(f"Norm stats saved to : {norm_stats_path}")
    wandb.finish()


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the speech emotion recognition baseline."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset",
        help="Root dataset directory containing train/ and test/ "
             "(default: dataset)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Root directory where results are saved (default: results)",
    )
    parser.add_argument(
        "--team_name",
        type=str,
        default="baseline",
        help="Team name — output is saved to <results_dir>/<team_name>/ "
             "(default: baseline)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default='baseline_train',
        help="Optional wandb run name (default: auto-generated from config)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["baseline", "mamba", "tf_mamba", "bilstm_attention"],
        default="mamba",
        help="Model type to train (default: mamba).",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        choices=["mel", "mfcc"],
        default="mfcc",
        help="Input feature type (default: mfcc).",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=None,
        help="Number of feature bins (e.g., MFCC coeffs or mel bins).",
    )
    parser.add_argument(
        "--include_deltas",
        action="store_true",
        default=True,
        help="Use delta + delta-delta channels (default: enabled).",
    )
    parser.add_argument(
        "--no_include_deltas",
        action="store_false",
        dest="include_deltas",
        help="Disable delta + delta-delta channels.",
    )
    parser.add_argument(
        "--specaugment",
        action="store_true",
        default=True,
        help="Apply SpecAugment to train split only (default: enabled).",
    )
    parser.add_argument(
        "--no_specaugment",
        action="store_false",
        dest="specaugment",
        help="Disable SpecAugment.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Baseline LSTM hidden size (used when --model_name baseline).",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of recurrent/SSM layers (default: 2).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.35,
        help="Dropout probability (default: 0.35).",
    )
    parser.add_argument(
        "--mamba_d_model",
        type=int,
        default=96,
        help="Mamba hidden model width (default: 96).",
    )
    parser.add_argument(
        "--mamba_d_state",
        type=int,
        default=32,
        help="Mamba state dimension N (default: 32).",
    )
    parser.add_argument(
        "--mamba_d_conv",
        type=int,
        default=4,
        help="Mamba local convolution width (default: 4).",
    )
    parser.add_argument(
        "--mamba_expand",
        type=int,
        default=2,
        help="Mamba expansion factor E (default: 2).",
    )
    parser.add_argument(
        "--cnn_channels",
        type=int,
        default=48,
        help="CNN front-end output channels (default: 48).",
    )
    parser.add_argument(
        "--frontend_type",
        type=str,
        choices=["basic_cnn", "residual_cnn"],
        default="basic_cnn",
        help="CNN frontend variant for mamba model (default: basic_cnn).",
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        choices=["concat", "gated"],
        default="concat",
        help="Bidirectional fusion strategy for mamba model (default: concat).",
    )
    parser.add_argument(
        "--pooling_type",
        type=str,
        choices=["meanmax", "attention"],
        default="meanmax",
        help="Temporal pooling strategy for mamba model (default: meanmax).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Mini-batch size (default: 64).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="AdamW decoupled weight decay (default: 1e-4).",
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        choices=["adamw", "sam"],
        default="adamw",
        help="Optimizer type (default: adamw).",
    )
    parser.add_argument(
        "--sam_rho",
        type=float,
        default=0.05,
        help="SAM neighborhood radius rho (used when --optimizer_type sam).",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Global gradient clipping max-norm (default: 1.0).",
    )
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.0,
        help="Mixup Beta(alpha, alpha) parameter; 0 disables mixup (default: 0.0).",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=0.0,
        help="Per-batch probability of applying mixup (default: 0.0).",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["ce", "focal"],
        default="ce",
        help="Classification loss type (default: ce).",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma when --loss_type focal (default: 2.0).",
    )
    parser.add_argument(
        "--class_weighting",
        action="store_true",
        default=False,
        help="Enable inverse-frequency class weighting in CE/Focal.",
    )
    parser.add_argument(
        "--no_class_weighting",
        action="store_false",
        dest="class_weighting",
        help="Disable class weighting (default).",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.05,
        help="Cross-entropy label smoothing (default: 0.05).",
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        choices=["plateau", "warmup_invsqrt", "warmup_cosine"],
        default="plateau",
        help="LR scheduler (default: plateau).",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Warmup epochs for warmup schedulers (default: 5).",
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.1,
        help="Min LR / base LR ratio at end of warmup_cosine (default: 0.1).",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Maximum training epochs (default: 100).",
    )
    parser.add_argument(
        "--use_swa",
        action="store_true",
        default=False,
        help="Enable stochastic weight averaging during late training.",
    )
    parser.add_argument(
        "--swa_start_epoch",
        type=int,
        default=60,
        help="Epoch index to start SWA updates (default: 60).",
    )
    parser.add_argument(
        "--swa_lr",
        type=float,
        default=1e-4,
        help="SWA learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10).",
    )
    parser.add_argument(
        "--patience_lr",
        type=int,
        default=5,
        help="ReduceLROnPlateau patience (default: 5).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Validation split fraction (default: 0.15).",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=0,
        help="If >1, use stratified K-fold CV with this many folds.",
    )
    parser.add_argument(
        "--fold_index",
        type=int,
        default=-1,
        help="Fold index for K-fold mode (0-based). Used only when --num_folds > 1.",
    )
    parser.add_argument(
        "--speed_perturb_prob",
        type=float,
        default=0.0,
        help="Waveform speed perturbation probability on train split (default: 0.0).",
    )
    parser.add_argument(
        "--speed_perturb_min",
        type=float,
        default=0.9,
        help="Minimum speed perturbation factor (default: 0.9).",
    )
    parser.add_argument(
        "--speed_perturb_max",
        type=float,
        default=1.1,
        help="Maximum speed perturbation factor (default: 1.1).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker processes (default: 0).",
    )
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="",
        help="Optional directory to cache extracted features across runs.",
    )
    parser.add_argument(
        "--specaugment_on_gpu",
        action="store_true",
        default=True,
        help="Apply SpecAugment on GPU after batch transfer (default: enabled).",
    )
    parser.add_argument(
        "--specaugment_on_cpu",
        action="store_false",
        dest="specaugment_on_gpu",
        help="Apply SpecAugment inside CPU dataset pipeline instead.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--num_time_masks",
        type=int,
        default=2,
        help="Number of time masks for SpecAugment (default: 2).",
    )
    parser.add_argument(
        "--time_mask_param",
        type=int,
        default=24,
        help="Maximum width for each time mask (default: 24).",
    )
    parser.add_argument(
        "--num_freq_masks",
        type=int,
        default=2,
        help="Number of frequency masks for SpecAugment (default: 2).",
    )
    parser.add_argument(
        "--freq_mask_param",
        type=int,
        default=8,
        help="Maximum width for each frequency mask (default: 8).",
    )
    main(parser.parse_args())
