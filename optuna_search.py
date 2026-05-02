"""
optuna_search.py
================
Phase 3 hyperparameter search for SER with coupled capacity/regularization.

Objective: maximize validation weighted F1.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataloader import (
    SpeechEmotionDataset,
    SAMPLE_RATE,
    WIN_SIZE,
    HOP_SIZE,
    N_MELS,
    N_MFCC,
    MAX_FRAMES,
)
from model import BidirectionalMambaSER, CNNBiLSTMAttentionSER


try:
    import optuna
except Exception as exc:
    raise RuntimeError(
        "Optuna is not installed. Install with: pip install optuna"
    ) from exc


@dataclass
class DataBundle:
    train_dir: Path
    feature_transform: nn.Module
    apply_db: bool
    max_frames: int
    train_indices: list[int]
    val_indices: list[int]
    mean: torch.Tensor
    std: torch.Tensor


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


def apply_specaugment_batch(
    specs: torch.Tensor,
    num_time_masks: int,
    time_mask_param: int,
    num_freq_masks: int,
    freq_mask_param: int,
) -> torch.Tensor:
    """Batch SpecAugment on GPU tensors with shape (B, C, F, T)."""
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


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_mean_std(dataset: Subset) -> Tuple[torch.Tensor, torch.Tensor]:
    all_specs = [dataset[i][0] for i in range(len(dataset))]
    stacked = torch.stack(all_specs, dim=0)  # (N, C, F, T)
    mean = stacked.mean(dim=(0, 3), keepdim=True).squeeze(0)
    std = stacked.std(dim=(0, 3), keepdim=True).squeeze(0)
    return mean, std


def build_feature_transform(feature_type: str, n_features: int) -> Tuple[nn.Module, bool]:
    if feature_type == "mfcc":
        transform = T.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=n_features,
            melkwargs={
                "n_fft": WIN_SIZE,
                "hop_length": HOP_SIZE,
                "n_mels": N_MELS,
            },
        )
        return transform, False

    if feature_type == "mel":
        transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=WIN_SIZE,
            hop_length=HOP_SIZE,
            n_mels=n_features,
        )
        return transform, True

    raise ValueError(f"Unsupported feature_type: {feature_type}")


def prepare_data_bundle(
    data_dir: Path,
    feature_type: str,
    n_features: int,
    include_deltas: bool,
    val_split: float,
    random_seed: int,
    feature_cache_dir: Path | None = None,
) -> DataBundle:
    train_dir = data_dir / "train"
    feature_transform, apply_db = build_feature_transform(feature_type, n_features)

    full_train_raw = SpeechEmotionDataset(
        audio_dir=train_dir / "audio",
        labels_csv=train_dir / "train_labels.csv",
        transform=feature_transform,
        max_frames=MAX_FRAMES,
        include_deltas=include_deltas,
        apply_db=apply_db,
        feature_cache_dir=(feature_cache_dir / "train_raw") if feature_cache_dir else None,
        cache_tag=(
            f"{feature_type}_{n_features}_frames{MAX_FRAMES}_"
            f"{'deltas' if include_deltas else 'nodeltas'}_"
            f"{'db' if apply_db else 'lin'}"
        ),
    )

    all_indices = np.arange(len(full_train_raw))
    all_labels = np.array([label for _, label in full_train_raw.samples])
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=val_split,
        random_state=random_seed,
        shuffle=True,
        stratify=all_labels,
    )
    train_indices = train_idx.tolist()
    val_indices = val_idx.tolist()

    mean, std = compute_mean_std(Subset(full_train_raw, train_indices))

    return DataBundle(
        train_dir=train_dir,
        feature_transform=feature_transform,
        apply_db=apply_db,
        max_frames=MAX_FRAMES,
        train_indices=train_indices,
        val_indices=val_indices,
        mean=mean,
        std=std,
    )


def make_dataloaders_for_trial(
    bundle: DataBundle,
    batch_size: int,
    include_deltas: bool,
    num_workers: int,
    apply_cpu_specaugment: bool,
    num_time_masks: int,
    time_mask_param: int,
    num_freq_masks: int,
    freq_mask_param: int,
    speed_perturb_prob: float,
    speed_perturb_min: float,
    speed_perturb_max: float,
    feature_cache_dir: Path | None = None,
    cache_tag: str = "default",
) -> Tuple[DataLoader, DataLoader]:
    pin_memory = torch.cuda.is_available()
    train_dataset = SpeechEmotionDataset(
        audio_dir=bundle.train_dir / "audio",
        labels_csv=bundle.train_dir / "train_labels.csv",
        transform=bundle.feature_transform,
        max_frames=bundle.max_frames,
        mean=bundle.mean,
        std=bundle.std,
        include_deltas=include_deltas,
        apply_specaugment=apply_cpu_specaugment,
        num_time_masks=num_time_masks,
        time_mask_param=time_mask_param,
        num_freq_masks=num_freq_masks,
        freq_mask_param=freq_mask_param,
        apply_db=bundle.apply_db,
        speed_perturb_prob=speed_perturb_prob,
        speed_perturb_min=speed_perturb_min,
        speed_perturb_max=speed_perturb_max,
        feature_cache_dir=(feature_cache_dir / "train") if feature_cache_dir else None,
        cache_tag=cache_tag,
    )
    val_dataset = SpeechEmotionDataset(
        audio_dir=bundle.train_dir / "audio",
        labels_csv=bundle.train_dir / "train_labels.csv",
        transform=bundle.feature_transform,
        max_frames=bundle.max_frames,
        mean=bundle.mean,
        std=bundle.std,
        include_deltas=include_deltas,
        apply_db=bundle.apply_db,
        feature_cache_dir=(feature_cache_dir / "val") if feature_cache_dir else None,
        cache_tag=cache_tag,
    )

    train_subset = Subset(train_dataset, bundle.train_indices)
    val_subset = Subset(val_dataset, bundle.val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def build_class_weights_from_bundle(bundle: DataBundle, device: torch.device) -> torch.Tensor:
    labels = []
    raw = SpeechEmotionDataset(
        audio_dir=bundle.train_dir / "audio",
        labels_csv=bundle.train_dir / "train_labels.csv",
        transform=bundle.feature_transform,
        max_frames=bundle.max_frames,
        include_deltas=True,
        apply_db=bundle.apply_db,
    )
    for idx in bundle.train_indices:
        labels.append(int(raw.samples[idx][1]))
    label_tensor = torch.tensor(labels, dtype=torch.long)
    num_classes = int(label_tensor.max().item()) + 1
    counts = torch.bincount(label_tensor, minlength=num_classes).float().clamp_min(1.0)
    weights = counts.sum() / (counts * num_classes)
    return weights.to(device)


def build_lr_lambda(
    schedule_type: str,
    max_epochs: int,
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
        decay_epochs = max(1, max_epochs - warmup_epochs)

        def lr_lambda(epoch: int) -> float:
            step = epoch + 1
            if step <= warmup_epochs:
                return step / warmup_epochs
            progress = min(1.0, (step - warmup_epochs) / decay_epochs)
            cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        return lr_lambda

    raise ValueError(f"Unsupported scheduler_type: {schedule_type}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    max_grad_norm: float | None = None,
    mixup_alpha: float = 0.0,
    mixup_prob: float = 0.0,
    apply_gpu_specaugment: bool = False,
    num_time_masks: int = 0,
    time_mask_param: int = 0,
    num_freq_masks: int = 0,
    freq_mask_param: int = 0,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    non_blocking = device.type == "cuda"
    for specs, labels in loader:
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
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            if use_mixup:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(specs.size(0), device=device)
                mixed_specs = lam * specs + (1.0 - lam) * specs[perm]
                logits = model(mixed_specs)
                loss = lam * criterion(logits, labels) + (1.0 - lam) * criterion(logits, labels[perm])
            else:
                logits = model(specs)
                loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * specs.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += specs.size(0)

    return total_loss / total, correct / total


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        non_blocking = device.type == "cuda"
        for specs, labels in loader:
            specs = specs.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(specs)
                loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * specs.size(0)
            correct += (preds == labels).sum().item()
            total += specs.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    val_f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / total, correct / total, val_f1


ALL_FRONTEND_TYPES = ["basic_cnn", "residual_cnn"]
ALL_FUSION_TYPES = ["concat", "gated"]
ALL_POOLING_TYPES = ["meanmax", "attention"]


def all_architectures() -> List[Tuple[str, str, str]]:
    return [
        (frontend, fusion, pooling)
        for frontend in ALL_FRONTEND_TYPES
        for fusion in ALL_FUSION_TYPES
        for pooling in ALL_POOLING_TYPES
    ]


def parse_architectures(raw: str) -> List[Tuple[str, str, str]]:
    parsed: List[Tuple[str, str, str]] = []
    if not raw.strip():
        return parsed

    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid architecture token '{token}'. "
                "Expected format frontend:fusion:pooling"
            )
        frontend, fusion, pooling = parts
        if frontend not in ALL_FRONTEND_TYPES:
            raise ValueError(f"Unsupported frontend_type '{frontend}'")
        if fusion not in ALL_FUSION_TYPES:
            raise ValueError(f"Unsupported fusion_type '{fusion}'")
        if pooling not in ALL_POOLING_TYPES:
            raise ValueError(f"Unsupported pooling_type '{pooling}'")
        parsed.append((frontend, fusion, pooling))

    # Preserve order while removing duplicates
    uniq: List[Tuple[str, str, str]] = []
    for arch in parsed:
        if arch not in uniq:
            uniq.append(arch)
    return uniq


def resolve_architecture_from_params(params: Dict[str, object]) -> Tuple[str, str, str]:
    if "arch_triplet" in params:
        frontend, fusion, pooling = str(params["arch_triplet"]).split(":")
        return frontend, fusion, pooling
    return (
        str(params["frontend_type"]),
        str(params["fusion_type"]),
        str(params["pooling_type"]),
    )


def select_top_architectures_from_study(
    results_dir: Path,
    source_study_name: str,
    top_k: int,
) -> List[Tuple[str, str, str]]:
    source_dir = results_dir / "optuna" / source_study_name
    source_db = source_dir / f"{source_study_name}.db"
    if not source_db.exists():
        raise FileNotFoundError(f"Stage A study db not found: {source_db}")

    source_storage = f"sqlite:///{source_db.as_posix()}"
    source_study = optuna.load_study(study_name=source_study_name, storage=source_storage)
    complete_trials = [
        trial
        for trial in source_study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not complete_trials:
        raise RuntimeError(f"No completed trials found in source study '{source_study_name}'.")

    arch_best: Dict[Tuple[str, str, str], float] = {}
    for trial in complete_trials:
        try:
            arch = resolve_architecture_from_params(trial.params)
        except Exception:
            continue
        best_so_far = arch_best.get(arch, float("-inf"))
        arch_best[arch] = max(best_so_far, float(trial.value))

    ranked = sorted(arch_best.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        raise RuntimeError(
            f"Could not resolve architecture fields from source study '{source_study_name}'."
        )
    return [arch for arch, _ in ranked[:max(1, top_k)]]


def write_architecture_ranking(study: optuna.Study, out_path: Path) -> None:
    stats: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            continue
        try:
            arch = resolve_architecture_from_params(trial.params)
        except Exception:
            continue
        stats[arch].append(float(trial.value))

    rows = []
    for arch, values in stats.items():
        rows.append(
            {
                "frontend_type": arch[0],
                "fusion_type": arch[1],
                "pooling_type": arch[2],
                "num_trials": len(values),
                "mean_val_f1": float(np.mean(values)),
                "max_val_f1": float(np.max(values)),
            }
        )
    rows.sort(key=lambda row: row["max_val_f1"], reverse=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def sample_stage_a_config(
    trial: optuna.Trial,
    args: argparse.Namespace,
) -> Dict[str, float | int | str]:
    frontend_type = trial.suggest_categorical("frontend_type", ALL_FRONTEND_TYPES)
    fusion_type = trial.suggest_categorical("fusion_type", ALL_FUSION_TYPES)
    pooling_type = trial.suggest_categorical("pooling_type", ALL_POOLING_TYPES)
    repeat_id = trial.suggest_int("repeat_id", 0, max(args.stage_a_repeats - 1, 0))
    d_model = trial.suggest_categorical("mamba_d_model", [96])
    cnn_channels = trial.suggest_categorical("cnn_channels", [48])
    d_state = trial.suggest_categorical("mamba_d_state", [32])
    num_layers = trial.suggest_categorical("num_layers", [2])
    mamba_expand = trial.suggest_categorical("mamba_expand", [2])
    mamba_d_conv = trial.suggest_categorical("mamba_d_conv", [4])
    dropout = trial.suggest_categorical("dropout", [0.35])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-4])
    label_smoothing = trial.suggest_categorical("label_smoothing", [0.05])
    learning_rate = trial.suggest_categorical("learning_rate", [3e-4])
    loss_type = trial.suggest_categorical("loss_type", ["ce"])
    focal_gamma = trial.suggest_categorical("focal_gamma", [2.0])
    class_weighting = trial.suggest_categorical("class_weighting", [False])
    scheduler_type = trial.suggest_categorical("scheduler_type", ["plateau"])
    warmup_epochs = trial.suggest_categorical("warmup_epochs", [5])
    min_lr_ratio = trial.suggest_categorical("min_lr_ratio", [0.1])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [1.0])
    mixup_alpha = trial.suggest_categorical("mixup_alpha", [0.2])
    mixup_prob = trial.suggest_categorical("mixup_prob", [0.4])
    speed_perturb_prob = trial.suggest_categorical("speed_perturb_prob", [0.6])
    speed_perturb_min = trial.suggest_categorical("speed_perturb_min", [0.9])
    speed_perturb_max = trial.suggest_categorical("speed_perturb_max", [1.1])
    batch_size = trial.suggest_categorical("batch_size", [args.stage_a_batch_size])
    num_time_masks = trial.suggest_categorical("num_time_masks", [2])
    time_mask_param = trial.suggest_categorical("time_mask_param", [24])
    num_freq_masks = trial.suggest_categorical("num_freq_masks", [2])
    freq_mask_param = trial.suggest_categorical("freq_mask_param", [8])

    return {
        "mamba_d_model": d_model,
        "cnn_channels": cnn_channels,
        "mamba_d_state": d_state,
        "num_layers": num_layers,
        "mamba_expand": mamba_expand,
        "mamba_d_conv": mamba_d_conv,
        "frontend_type": frontend_type,
        "fusion_type": fusion_type,
        "pooling_type": pooling_type,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "learning_rate": learning_rate,
        "loss_type": loss_type,
        "focal_gamma": focal_gamma,
        "class_weighting": class_weighting,
        "scheduler_type": scheduler_type,
        "warmup_epochs": warmup_epochs,
        "min_lr_ratio": min_lr_ratio,
        "max_grad_norm": max_grad_norm,
        "mixup_alpha": mixup_alpha,
        "mixup_prob": mixup_prob,
        "speed_perturb_prob": speed_perturb_prob,
        "speed_perturb_min": speed_perturb_min,
        "speed_perturb_max": speed_perturb_max,
        "batch_size": batch_size,
        "num_time_masks": num_time_masks,
        "time_mask_param": time_mask_param,
        "num_freq_masks": num_freq_masks,
        "freq_mask_param": freq_mask_param,
        "repeat_id": repeat_id,
    }


def sample_stage_b_config(
    trial: optuna.Trial,
    allowed_architectures: List[Tuple[str, str, str]],
) -> Dict[str, float | int | str]:
    arch_choices = [f"{f}:{u}:{p}" for f, u, p in allowed_architectures]
    arch_triplet = trial.suggest_categorical("arch_triplet", arch_choices)
    frontend_type, fusion_type, pooling_type = arch_triplet.split(":")

    # Focused capacity search (removes weak low-width region)
    d_model = trial.suggest_categorical("mamba_d_model", [96, 128])
    cnn_channels = trial.suggest_categorical("cnn_channels", [48, 64])
    d_state = trial.suggest_categorical("mamba_d_state", [32, 64])
    num_layers = trial.suggest_int("num_layers", 2, 3)
    mamba_expand = trial.suggest_categorical("mamba_expand", [2, 4])
    mamba_d_conv = trial.suggest_categorical("mamba_d_conv", [3, 4, 5])

    capacity_score = 0
    if d_model >= 128:
        capacity_score += 1
    if cnn_channels >= 64:
        capacity_score += 1
    if d_state >= 64:
        capacity_score += 1
    if num_layers >= 3:
        capacity_score += 1

    # Coupled regularization (stronger for larger models)
    if capacity_score >= 3:
        dropout_min, dropout_max = 0.38, 0.60
        wd_min, wd_max = 5e-5, 3e-3
        ls_min, ls_max = 0.06, 0.16
        tm_lo, tm_hi, tm_step = 24, 56, 8
        fm_lo, fm_hi, fm_step = 8, 20, 4
        tmask_min, tmask_max = 2, 4
        fmask_min, fmask_max = 2, 4
    elif capacity_score >= 2:
        dropout_min, dropout_max = 0.33, 0.53
        wd_min, wd_max = 2e-5, 1.5e-3
        ls_min, ls_max = 0.04, 0.13
        tm_lo, tm_hi, tm_step = 20, 48, 4
        fm_lo, fm_hi, fm_step = 6, 16, 2
        tmask_min, tmask_max = 2, 3
        fmask_min, fmask_max = 2, 3
    else:
        dropout_min, dropout_max = 0.25, 0.45
        wd_min, wd_max = 1e-5, 5e-4
        ls_min, ls_max = 0.01, 0.10
        tm_lo, tm_hi, tm_step = 16, 40, 4
        fm_lo, fm_hi, fm_step = 4, 12, 2
        tmask_min, tmask_max = 1, 3
        fmask_min, fmask_max = 1, 3

    # Attention pooling tends to overfit more easily; raise regularization floor.
    if pooling_type == "attention":
        dropout_min = max(dropout_min, 0.35)
        ls_min = max(ls_min, 0.03)

    dropout = trial.suggest_float("dropout", dropout_min, dropout_max)
    weight_decay = trial.suggest_float("weight_decay", wd_min, wd_max, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", ls_min, ls_max)
    loss_type = trial.suggest_categorical("loss_type", ["ce", "focal"])
    if loss_type == "focal":
        focal_gamma = trial.suggest_float("focal_gamma", 1.5, 3.0)
    else:
        focal_gamma = 2.0
    class_weighting = trial.suggest_categorical("class_weighting", [False, True])
    scheduler_type = trial.suggest_categorical(
        "scheduler_type", ["plateau", "warmup_invsqrt", "warmup_cosine"]
    )
    warmup_epochs = trial.suggest_int("warmup_epochs", 2, 8)
    min_lr_ratio = trial.suggest_float("min_lr_ratio", 0.05, 0.30)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 2.0)
    mixup_alpha = trial.suggest_float("mixup_alpha", 0.1, 0.5)
    mixup_prob = trial.suggest_float("mixup_prob", 0.2, 0.5)
    speed_perturb_prob = trial.suggest_float("speed_perturb_prob", 0.0, 0.3)
    speed_perturb_min = trial.suggest_categorical("speed_perturb_min", [0.9])
    speed_perturb_max = trial.suggest_categorical("speed_perturb_max", [1.1])

    return {
        "mamba_d_model": d_model,
        "cnn_channels": cnn_channels,
        "mamba_d_state": d_state,
        "num_layers": num_layers,
        "mamba_expand": mamba_expand,
        "mamba_d_conv": mamba_d_conv,
        "frontend_type": frontend_type,
        "fusion_type": fusion_type,
        "pooling_type": pooling_type,
        "arch_triplet": arch_triplet,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "loss_type": loss_type,
        "focal_gamma": focal_gamma,
        "class_weighting": class_weighting,
        "scheduler_type": scheduler_type,
        "warmup_epochs": warmup_epochs,
        "min_lr_ratio": min_lr_ratio,
        "max_grad_norm": max_grad_norm,
        "mixup_alpha": mixup_alpha,
        "mixup_prob": mixup_prob,
        "speed_perturb_prob": speed_perturb_prob,
        "speed_perturb_min": speed_perturb_min,
        "speed_perturb_max": speed_perturb_max,
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "num_time_masks": trial.suggest_int("num_time_masks", tmask_min, tmask_max),
        "time_mask_param": trial.suggest_int("time_mask_param", tm_lo, tm_hi, step=tm_step),
        "num_freq_masks": trial.suggest_int("num_freq_masks", fmask_min, fmask_max),
        "freq_mask_param": trial.suggest_int("freq_mask_param", fm_lo, fm_hi, step=fm_step),
    }


def sample_bilstm_attention_config(
    trial: optuna.Trial,
    args: argparse.Namespace,
) -> Dict[str, float | int | str]:
    # Focused training-parameter search for CNN+BiLSTM+attention.
    cnn_channels = trial.suggest_categorical("cnn_channels", [48, 64, 80])
    hidden_size = trial.suggest_categorical("hidden_size", [192, 256, 320])
    num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])

    capacity_score = 0
    if cnn_channels >= 64:
        capacity_score += 1
    if hidden_size >= 256:
        capacity_score += 1
    if num_layers >= 2:
        capacity_score += 1

    if capacity_score >= 2:
        dropout_min, dropout_max = 0.30, 0.55
        wd_min, wd_max = 5e-5, 2e-3
        ls_min, ls_max = 0.04, 0.14
        tm_lo, tm_hi, tm_step = 20, 48, 4
        fm_lo, fm_hi, fm_step = 6, 16, 2
        tmask_min, tmask_max = 2, 4
        fmask_min, fmask_max = 2, 4
    else:
        dropout_min, dropout_max = 0.20, 0.45
        wd_min, wd_max = 1e-5, 8e-4
        ls_min, ls_max = 0.02, 0.10
        tm_lo, tm_hi, tm_step = 16, 40, 4
        fm_lo, fm_hi, fm_step = 4, 12, 2
        tmask_min, tmask_max = 1, 3
        fmask_min, fmask_max = 1, 3

    dropout = trial.suggest_float("dropout", dropout_min, dropout_max)
    weight_decay = trial.suggest_float("weight_decay", wd_min, wd_max, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", ls_min, ls_max)
    loss_type = trial.suggest_categorical("loss_type", ["ce", "focal"])
    if loss_type == "focal":
        focal_gamma = trial.suggest_float("focal_gamma", 1.5, 3.0)
    else:
        focal_gamma = 2.0
    class_weighting = trial.suggest_categorical("class_weighting", [False, True])
    scheduler_type = trial.suggest_categorical(
        "scheduler_type", ["plateau", "warmup_invsqrt", "warmup_cosine"]
    )
    warmup_epochs = trial.suggest_int("warmup_epochs", 2, 8)
    min_lr_ratio = trial.suggest_float("min_lr_ratio", 0.05, 0.30)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 2.0)
    mixup_alpha = trial.suggest_float("mixup_alpha", 0.1, 0.5)
    mixup_prob = trial.suggest_float("mixup_prob", 0.1, 0.5)
    speed_perturb_prob = trial.suggest_float("speed_perturb_prob", 0.0, 0.3)
    speed_perturb_min = trial.suggest_categorical("speed_perturb_min", [0.9])
    speed_perturb_max = trial.suggest_categorical("speed_perturb_max", [1.1])

    return {
        "model_name": "bilstm_attention",
        "cnn_channels": cnn_channels,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "loss_type": loss_type,
        "focal_gamma": focal_gamma,
        "class_weighting": class_weighting,
        "scheduler_type": scheduler_type,
        "warmup_epochs": warmup_epochs,
        "min_lr_ratio": min_lr_ratio,
        "max_grad_norm": max_grad_norm,
        "mixup_alpha": mixup_alpha,
        "mixup_prob": mixup_prob,
        "speed_perturb_prob": speed_perturb_prob,
        "speed_perturb_min": speed_perturb_min,
        "speed_perturb_max": speed_perturb_max,
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "num_time_masks": trial.suggest_int("num_time_masks", tmask_min, tmask_max),
        "time_mask_param": trial.suggest_int("time_mask_param", tm_lo, tm_hi, step=tm_step),
        "num_freq_masks": trial.suggest_int("num_freq_masks", fmask_min, fmask_max),
        "freq_mask_param": trial.suggest_int("freq_mask_param", fm_lo, fm_hi, step=fm_step),
    }


def build_objective(
    bundle: DataBundle,
    device: torch.device,
    args: argparse.Namespace,
    allowed_architectures: List[Tuple[str, str, str]],
):
    def objective(trial: optuna.Trial) -> float:
        if args.model_name == "bilstm_attention":
            if args.mode != "stage_b":
                raise ValueError("bilstm_attention search currently supports --mode stage_b only.")
            config = sample_bilstm_attention_config(trial, args)
        elif args.mode == "stage_a":
            config = sample_stage_a_config(trial, args)
        elif args.mode == "stage_b":
            config = sample_stage_b_config(trial, allowed_architectures)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")

        if args.fast_debug:
            # Force lightweight augmentation to verify trial progress quickly.
            config["mixup_alpha"] = 0.1
            config["mixup_prob"] = 0.2
            config["speed_perturb_prob"] = 0.0
            config["num_time_masks"] = 1
            config["time_mask_param"] = min(16, int(config["time_mask_param"]))
            config["num_freq_masks"] = 1
            config["freq_mask_param"] = min(6, int(config["freq_mask_param"]))

        repeat_offset = int(config.get("repeat_id", 0)) * 1000
        stage_b_repeats = max(1, int(args.stage_b_repeats)) if args.mode == "stage_b" else 1
        repeat_stride = max(1, int(args.stage_b_repeat_seed_stride))
        repeat_best_f1s: list[float] = []
        repeat_best_losses: list[float] = []

        for repeat_idx in range(stage_b_repeats):
            run_seed = args.seed + trial.number + repeat_offset + (repeat_idx * repeat_stride)
            set_global_seed(run_seed)
            model_tag = str(config.get("model_name", "mamba"))
            print(
                f"[trial {trial.number:03d}|repeat {repeat_idx+1}/{stage_b_repeats}] setup "
                f"model={model_tag} "
                f"bs={config['batch_size']} speed_p={float(config['speed_perturb_prob']):.2f} "
                f"mixup_p={float(config['mixup_prob']):.2f} seed={run_seed}"
            )
            setup_t0 = time.perf_counter()

            train_loader, val_loader = make_dataloaders_for_trial(
                bundle=bundle,
                batch_size=int(config["batch_size"]),
                include_deltas=True,
                num_workers=args.num_workers,
                apply_cpu_specaugment=bool(
                    (not args.specaugment_on_gpu) or device.type != "cuda"
                ),
                num_time_masks=int(config["num_time_masks"]),
                time_mask_param=int(config["time_mask_param"]),
                num_freq_masks=int(config["num_freq_masks"]),
                freq_mask_param=int(config["freq_mask_param"]),
                speed_perturb_prob=float(config["speed_perturb_prob"]),
                speed_perturb_min=float(config["speed_perturb_min"]),
                speed_perturb_max=float(config["speed_perturb_max"]),
                feature_cache_dir=(Path(args.feature_cache_dir) if args.feature_cache_dir else None),
                cache_tag=(
                    f"{args.feature_type}_{args.n_features}_frames{MAX_FRAMES}_"
                    f"deltas_{'db' if bundle.apply_db else 'lin'}"
                ),
            )
            setup_t1 = time.perf_counter()
            print(
                f"[trial {trial.number:03d}|repeat {repeat_idx+1}/{stage_b_repeats}] dataloaders ready "
                f"(train_batches={len(train_loader)}, val_batches={len(val_loader)}, "
                f"setup_sec={setup_t1 - setup_t0:.1f})"
            )

            if args.model_name == "bilstm_attention":
                model = CNNBiLSTMAttentionSER(
                    in_channels=3,
                    n_features=args.n_features,
                    cnn_channels=int(config["cnn_channels"]),
                    hidden_size=int(config["hidden_size"]),
                    num_layers=int(config["num_layers"]),
                    dropout=float(config["dropout"]),
                ).to(device)
            else:
                model = BidirectionalMambaSER(
                    in_channels=3,
                    n_features=args.n_features,
                    cnn_channels=int(config["cnn_channels"]),
                    d_model=int(config["mamba_d_model"]),
                    d_state=int(config["mamba_d_state"]),
                    d_conv=int(config["mamba_d_conv"]),
                    expand=int(config["mamba_expand"]),
                    num_layers=int(config["num_layers"]),
                    dropout=float(config["dropout"]),
                    frontend_type=str(config["frontend_type"]),
                    fusion_type=str(config["fusion_type"]),
                    pooling_type=str(config["pooling_type"]),
                ).to(device)

            class_weights = None
            if bool(config["class_weighting"]):
                class_weights = build_class_weights_from_bundle(bundle, device)

            if str(config["loss_type"]) == "ce":
                criterion = nn.CrossEntropyLoss(
                    weight=class_weights,
                    label_smoothing=float(config["label_smoothing"]),
                )
            elif str(config["loss_type"]) == "focal":
                criterion = FocalLoss(
                    gamma=float(config["focal_gamma"]),
                    class_weights=class_weights,
                    label_smoothing=float(config["label_smoothing"]),
                )
            else:
                raise ValueError(f"Unsupported loss_type: {config['loss_type']}")

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(config["learning_rate"]),
                weight_decay=float(config["weight_decay"]),
            )
            if str(config["scheduler_type"]) == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=args.patience_lr
                )
            else:
                lr_lambda = build_lr_lambda(
                    schedule_type=str(config["scheduler_type"]),
                    max_epochs=args.max_epochs,
                    warmup_epochs=int(config["warmup_epochs"]),
                    min_lr_ratio=float(config["min_lr_ratio"]),
                )
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=lr_lambda
                )
            amp_enabled = bool(args.amp and device.type == "cuda")
            scaler = torch.amp.GradScaler(enabled=amp_enabled)

            best_val_f1 = float("-inf")
            best_val_loss = float("inf")
            patience_counter = 0

            try:
                for epoch in range(args.max_epochs):
                    epoch_t0 = time.perf_counter()
                    train_loss, train_acc = train_one_epoch(
                        model,
                        train_loader,
                        criterion,
                        optimizer,
                        device,
                        scaler,
                        amp_enabled,
                        max_grad_norm=float(config["max_grad_norm"]),
                        mixup_alpha=float(config["mixup_alpha"]),
                        mixup_prob=float(config["mixup_prob"]),
                        apply_gpu_specaugment=bool(
                            args.specaugment_on_gpu and device.type == "cuda"
                        ),
                        num_time_masks=int(config["num_time_masks"]),
                        time_mask_param=int(config["time_mask_param"]),
                        num_freq_masks=int(config["num_freq_masks"]),
                        freq_mask_param=int(config["freq_mask_param"]),
                    )
                    val_loss, val_acc, val_f1 = validate(
                        model, val_loader, criterion, device, amp_enabled
                    )
                    if str(config["scheduler_type"]) == "plateau":
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                    report_step = repeat_idx * args.max_epochs + epoch
                    trial.report(val_f1, step=report_step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                    is_better_f1 = val_f1 > best_val_f1 + 1e-6
                    is_tie_better_loss = abs(val_f1 - best_val_f1) <= 1e-6 and val_loss < best_val_loss
                    if is_better_f1 or is_tie_better_loss:
                        best_val_f1 = val_f1
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= args.early_stop_patience:
                            break

                    print(
                        f"[trial {trial.number:03d}|repeat {repeat_idx+1}/{stage_b_repeats}] "
                        f"epoch {epoch+1:02d}/{args.max_epochs} "
                        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
                        f"epoch_sec={time.perf_counter() - epoch_t0:.1f}"
                    )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"[trial {trial.number:03d}|repeat {repeat_idx+1}/{stage_b_repeats}] OOM encountered; pruning trial.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise optuna.TrialPruned() from exc
                raise
            finally:
                del model, optimizer, scheduler, scaler, train_loader, val_loader
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            repeat_best_f1s.append(float(best_val_f1))
            repeat_best_losses.append(float(best_val_loss))
            print(
                f"[trial {trial.number:03d}|repeat {repeat_idx+1}/{stage_b_repeats}] "
                f"best_val_f1={best_val_f1:.4f} best_val_loss={best_val_loss:.4f}"
            )

        mean_f1 = float(np.mean(repeat_best_f1s))
        std_f1 = float(np.std(repeat_best_f1s)) if len(repeat_best_f1s) > 1 else 0.0
        robust_score = mean_f1 - float(args.stage_b_repeat_lambda) * std_f1
        trial.set_user_attr("repeat_best_f1s", repeat_best_f1s)
        trial.set_user_attr("repeat_best_losses", repeat_best_losses)
        trial.set_user_attr("repeat_mean_f1", mean_f1)
        trial.set_user_attr("repeat_std_f1", std_f1)
        trial.set_user_attr("repeat_robust_score", robust_score)
        print(
            f"[trial {trial.number:03d}] summary "
            f"mean_f1={mean_f1:.4f} std_f1={std_f1:.4f} "
            f"robust_score={robust_score:.4f} repeats={stage_b_repeats}"
        )

        return robust_score if args.mode == "stage_b" else mean_f1

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna Phase 3 search for Mamba SER.")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--study_name", type=str, default="mamba_phase3")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["mamba", "bilstm_attention"],
        default="mamba",
        help="Model family to optimize (default: mamba).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["stage_a", "stage_b"],
        default="stage_b",
        help="stage_a: architecture shortlist with fixed training config; "
        "stage_b: focused coupled capacity/regularization search.",
    )
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URI, e.g. sqlite:///results/optuna/mamba_phase3.db")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout in seconds for the study.")
    parser.add_argument("--max_epochs", type=int, default=35)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--patience_lr", type=int, default=4)
    parser.add_argument("--feature_type", type=str, choices=["mfcc", "mel"], default="mfcc")
    parser.add_argument("--n_features", type=int, default=N_MFCC)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="",
        help="Optional directory to cache extracted features across trials.",
    )
    parser.add_argument(
        "--specaugment_on_gpu",
        action="store_true",
        default=True,
        help="Apply SpecAugment on GPU batch tensors (default: enabled).",
    )
    parser.add_argument(
        "--specaugment_on_cpu",
        action="store_false",
        dest="specaugment_on_gpu",
        help="Apply SpecAugment in CPU dataset transform path instead.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Enable mixed precision on CUDA (default: enabled).",
    )
    parser.add_argument(
        "--no_amp",
        action="store_false",
        dest="amp",
        help="Disable mixed precision.",
    )
    parser.add_argument(
        "--stage_a_repeats",
        type=int,
        default=2,
        help="How many random-seed repeats per architecture in stage_a.",
    )
    parser.add_argument(
        "--stage_a_batch_size",
        type=int,
        default=32,
        help="Fixed batch size for stage_a architecture shortlist (default: 32).",
    )
    parser.add_argument(
        "--stage_b_architectures",
        type=str,
        default="",
        help="Comma list of architecture triplets for stage_b, e.g. "
        "'basic_cnn:concat:meanmax,residual_cnn:gated:attention'.",
    )
    parser.add_argument(
        "--stage_b_from_study",
        type=str,
        default="",
        help="Use top architectures from an existing stage_a study name.",
    )
    parser.add_argument(
        "--stage_b_top_k",
        type=int,
        default=2,
        help="Number of top architectures to import from --stage_b_from_study.",
    )
    parser.add_argument(
        "--fast_debug",
        action="store_true",
        help="Use lightweight augmentation settings to quickly validate trial progress/logging.",
    )
    parser.add_argument(
        "--stage_b_repeats",
        type=int,
        default=1,
        help="Number of seed repeats per Stage B trial (default: 1).",
    )
    parser.add_argument(
        "--stage_b_repeat_lambda",
        type=float,
        default=0.5,
        help="Robust score lambda for Stage B repeats: mean_f1 - lambda * std_f1 (default: 0.5).",
    )
    parser.add_argument(
        "--stage_b_repeat_seed_stride",
        type=int,
        default=1000,
        help="Seed offset between Stage B repeats (default: 1000).",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    out_dir = results_dir / "optuna" / args.study_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.storage is None:
        storage = f"sqlite:///{(out_dir / (args.study_name + '.db')).as_posix()}"
    else:
        storage = args.storage

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Preparing base dataset/cache for trial splits and normalization...")
    bundle = prepare_data_bundle(
        data_dir=data_dir,
        feature_type=args.feature_type,
        n_features=args.n_features,
        include_deltas=True,
        val_split=args.val_split,
        random_seed=args.seed,
        feature_cache_dir=Path(args.feature_cache_dir) if args.feature_cache_dir else None,
    )

    if args.model_name == "bilstm_attention" and args.mode == "stage_a":
        raise ValueError("bilstm_attention is not supported in --mode stage_a. Use --mode stage_b.")

    allowed_architectures = all_architectures()
    if args.model_name == "mamba" and args.mode == "stage_b":
        if args.stage_b_architectures.strip():
            allowed_architectures = parse_architectures(args.stage_b_architectures)
        elif args.stage_b_from_study.strip():
            allowed_architectures = select_top_architectures_from_study(
                results_dir=results_dir,
                source_study_name=args.stage_b_from_study.strip(),
                top_k=args.stage_b_top_k,
            )
        print(f"Stage B architecture pool: {allowed_architectures}")
    elif args.model_name == "bilstm_attention":
        print("Stage B model pool: bilstm_attention (single architecture family).")

    if args.mode == "stage_a":
        grid_space = {
            "frontend_type": ALL_FRONTEND_TYPES,
            "fusion_type": ALL_FUSION_TYPES,
            "pooling_type": ALL_POOLING_TYPES,
            "repeat_id": list(range(max(args.stage_a_repeats, 1))),
        }
        sampler = optuna.samplers.GridSampler(grid_space)
        pruner = optuna.pruners.NopPruner()
    else:
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=6,
            interval_steps=1,
        )

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    if args.mode == "stage_a":
        total_grid = 2 * 2 * 2 * max(args.stage_a_repeats, 1)
        effective_trials = min(args.n_trials, total_grid)
        print(
            f"Starting Stage A study '{args.study_name}' "
            f"with total_grid={total_grid}, n_trials={effective_trials}, "
            f"max_epochs={args.max_epochs}, early_stop_patience={args.early_stop_patience}"
        )
    else:
        effective_trials = args.n_trials
        print(
            f"Starting Stage B study '{args.study_name}' with n_trials={effective_trials}, "
            f"max_epochs={args.max_epochs}, early_stop_patience={args.early_stop_patience}, "
            f"repeats={max(1, args.stage_b_repeats)}, repeat_lambda={args.stage_b_repeat_lambda}"
        )
    objective = build_objective(
        bundle=bundle,
        device=device,
        args=args,
        allowed_architectures=allowed_architectures,
    )
    study.optimize(objective, n_trials=effective_trials, timeout=args.timeout, gc_after_trial=True)

    if args.model_name == "mamba":
        best_arch = resolve_architecture_from_params(study.best_trial.params)
        best_arch_dict: dict[str, str] | None = {
            "frontend_type": best_arch[0],
            "fusion_type": best_arch[1],
            "pooling_type": best_arch[2],
        }
    else:
        best_arch = None
        best_arch_dict = None
    best_repeat_mean = study.best_trial.user_attrs.get("repeat_mean_f1")
    best_repeat_std = study.best_trial.user_attrs.get("repeat_std_f1")
    best_repeat_score = study.best_trial.user_attrs.get("repeat_robust_score")
    best = {
        "model_name": args.model_name,
        "best_trial_number": study.best_trial.number,
        "best_objective_value": study.best_value,
        "best_repeat_mean_f1": best_repeat_mean,
        "best_repeat_std_f1": best_repeat_std,
        "best_repeat_robust_score": best_repeat_score,
        "best_params": study.best_trial.params,
        "best_architecture": best_arch_dict,
    }
    with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(out_dir / "trials.csv", index=False)
    if args.model_name == "mamba":
        write_architecture_ranking(study, out_dir / "architecture_ranking.json")

    retrain_cmd = [
        "python train.py",
        f"--data_dir {args.data_dir}",
        f"--results_dir {args.results_dir}",
        f"--team_name {args.study_name}_best",
        f"--run_name {args.study_name}_best",
        f"--model_name {args.model_name}",
        f"--feature_type {args.feature_type}",
        "--include_deltas",
        "--specaugment",
        "--specaugment_on_gpu",
        f"--cnn_channels {study.best_trial.params['cnn_channels']}",
        f"--num_layers {study.best_trial.params['num_layers']}",
        f"--dropout {study.best_trial.params['dropout']}",
        f"--learning_rate {study.best_trial.params['learning_rate']}",
        f"--weight_decay {study.best_trial.params['weight_decay']}",
        f"--max_grad_norm {study.best_trial.params['max_grad_norm']}",
        f"--mixup_alpha {study.best_trial.params['mixup_alpha']}",
        f"--mixup_prob {study.best_trial.params['mixup_prob']}",
        f"--loss_type {study.best_trial.params['loss_type']}",
        f"--focal_gamma {study.best_trial.params.get('focal_gamma', 2.0)}",
        "--class_weighting" if study.best_trial.params["class_weighting"] else "--no_class_weighting",
        f"--label_smoothing {study.best_trial.params['label_smoothing']}",
        f"--scheduler_type {study.best_trial.params['scheduler_type']}",
        f"--warmup_epochs {study.best_trial.params['warmup_epochs']}",
        f"--min_lr_ratio {study.best_trial.params['min_lr_ratio']}",
        f"--batch_size {study.best_trial.params['batch_size']}",
        f"--num_time_masks {study.best_trial.params['num_time_masks']}",
        f"--time_mask_param {study.best_trial.params['time_mask_param']}",
        f"--num_freq_masks {study.best_trial.params['num_freq_masks']}",
        f"--freq_mask_param {study.best_trial.params['freq_mask_param']}",
        f"--speed_perturb_prob {study.best_trial.params['speed_perturb_prob']}",
        f"--speed_perturb_min {study.best_trial.params['speed_perturb_min']}",
        f"--speed_perturb_max {study.best_trial.params['speed_perturb_max']}",
        f"--num_workers {args.num_workers}",
        f"--seed {args.seed}",
    ]
    if args.feature_cache_dir:
        retrain_cmd.append(f"--feature_cache_dir {args.feature_cache_dir}")
    if args.model_name == "mamba":
        retrain_cmd.extend(
            [
                f"--mamba_d_model {study.best_trial.params['mamba_d_model']}",
                f"--mamba_d_state {study.best_trial.params['mamba_d_state']}",
                f"--mamba_expand {study.best_trial.params['mamba_expand']}",
                f"--mamba_d_conv {study.best_trial.params['mamba_d_conv']}",
                f"--frontend_type {best_arch[0]}",
                f"--fusion_type {best_arch[1]}",
                f"--pooling_type {best_arch[2]}",
            ]
        )
    else:
        retrain_cmd.append(f"--hidden_size {study.best_trial.params['hidden_size']}")
    with open(out_dir / "retrain_command.txt", "w", encoding="utf-8") as f:
        f.write(" \\\n  ".join(retrain_cmd) + "\n")

    print("\nStudy complete.")
    print(f"Best trial: {study.best_trial.number}")
    if args.mode == "stage_b" and max(1, args.stage_b_repeats) > 1:
        print(
            f"Best robust score: {study.best_value:.4f} "
            f"(mean_f1={best_repeat_mean:.4f}, std_f1={best_repeat_std:.4f})"
        )
    else:
        print(f"Best val_f1: {study.best_value:.4f}")
    if args.model_name == "mamba":
        print(f"Best architecture: frontend={best_arch[0]}, fusion={best_arch[1]}, pooling={best_arch[2]}")
    else:
        print("Best model family: bilstm_attention")
    print(f"Artifacts written to: {out_dir}")
    print(f"Retrain command saved to: {out_dir / 'retrain_command.txt'}")


if __name__ == "__main__":
    main()
