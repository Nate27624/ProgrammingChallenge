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
from model import BidirectionalMambaSER


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
    num_time_masks: int,
    time_mask_param: int,
    num_freq_masks: int,
    freq_mask_param: int,
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
        apply_specaugment=True,
        num_time_masks=num_time_masks,
        time_mask_param=time_mask_param,
        num_freq_masks=num_freq_masks,
        freq_mask_param=freq_mask_param,
        apply_db=bundle.apply_db,
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
    )

    train_subset = Subset(train_dataset, bundle.train_indices)
    val_subset = Subset(val_dataset, bundle.val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
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
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(specs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
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
        for specs, labels in loader:
            specs, labels = specs.to(device), labels.to(device)
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
        if args.mode == "stage_a":
            config = sample_stage_a_config(trial, args)
        elif args.mode == "stage_b":
            config = sample_stage_b_config(trial, allowed_architectures)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")

        repeat_offset = int(config.get("repeat_id", 0)) * 1000
        set_global_seed(args.seed + trial.number + repeat_offset)

        train_loader, val_loader = make_dataloaders_for_trial(
            bundle=bundle,
            batch_size=int(config["batch_size"]),
            include_deltas=True,
            num_workers=args.num_workers,
            num_time_masks=int(config["num_time_masks"]),
            time_mask_param=int(config["time_mask_param"]),
            num_freq_masks=int(config["num_freq_masks"]),
            freq_mask_param=int(config["freq_mask_param"]),
        )

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

        optimizer = torch.optim.Adam(
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
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer, device, scaler, amp_enabled
                )
                val_loss, val_acc, val_f1 = validate(
                    model, val_loader, criterion, device, amp_enabled
                )
                if str(config["scheduler_type"]) == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

                trial.report(val_f1, step=epoch)
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
                    f"[trial {trial.number:03d}] "
                    f"epoch {epoch+1:02d}/{args.max_epochs} "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
                )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"[trial {trial.number:03d}] OOM encountered; pruning trial.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned() from exc
            raise
        finally:
            del model, optimizer, scheduler, scaler, train_loader, val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return best_val_f1

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna Phase 3 search for Mamba SER.")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--study_name", type=str, default="mamba_phase3")
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
    )

    allowed_architectures = all_architectures()
    if args.mode == "stage_b":
        if args.stage_b_architectures.strip():
            allowed_architectures = parse_architectures(args.stage_b_architectures)
        elif args.stage_b_from_study.strip():
            allowed_architectures = select_top_architectures_from_study(
                results_dir=results_dir,
                source_study_name=args.stage_b_from_study.strip(),
                top_k=args.stage_b_top_k,
            )
        print(f"Stage B architecture pool: {allowed_architectures}")

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
            f"max_epochs={args.max_epochs}, early_stop_patience={args.early_stop_patience}"
        )
    objective = build_objective(
        bundle=bundle,
        device=device,
        args=args,
        allowed_architectures=allowed_architectures,
    )
    study.optimize(objective, n_trials=effective_trials, timeout=args.timeout, gc_after_trial=True)

    best_arch = resolve_architecture_from_params(study.best_trial.params)
    best = {
        "best_trial_number": study.best_trial.number,
        "best_val_f1": study.best_value,
        "best_params": study.best_trial.params,
        "best_architecture": {
            "frontend_type": best_arch[0],
            "fusion_type": best_arch[1],
            "pooling_type": best_arch[2],
        },
    }
    with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(out_dir / "trials.csv", index=False)
    write_architecture_ranking(study, out_dir / "architecture_ranking.json")

    retrain_cmd = [
        "python train.py",
        f"--data_dir {args.data_dir}",
        f"--results_dir {args.results_dir}",
        f"--team_name {args.study_name}_best",
        f"--run_name {args.study_name}_best",
        "--model_name mamba",
        f"--feature_type {args.feature_type}",
        "--include_deltas",
        "--specaugment",
        f"--mamba_d_model {study.best_trial.params['mamba_d_model']}",
        f"--cnn_channels {study.best_trial.params['cnn_channels']}",
        f"--mamba_d_state {study.best_trial.params['mamba_d_state']}",
        f"--num_layers {study.best_trial.params['num_layers']}",
        f"--mamba_expand {study.best_trial.params['mamba_expand']}",
        f"--mamba_d_conv {study.best_trial.params['mamba_d_conv']}",
        f"--frontend_type {best_arch[0]}",
        f"--fusion_type {best_arch[1]}",
        f"--pooling_type {best_arch[2]}",
        f"--dropout {study.best_trial.params['dropout']}",
        f"--learning_rate {study.best_trial.params['learning_rate']}",
        f"--weight_decay {study.best_trial.params['weight_decay']}",
        f"--loss_type {study.best_trial.params['loss_type']}",
        f"--focal_gamma {study.best_trial.params['focal_gamma']}",
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
    ]
    with open(out_dir / "retrain_command.txt", "w", encoding="utf-8") as f:
        f.write(" \\\n  ".join(retrain_cmd) + "\n")

    print("\nStudy complete.")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best val_f1: {study.best_value:.4f}")
    print(f"Best architecture: frontend={best_arch[0]}, fusion={best_arch[1]}, pooling={best_arch[2]}")
    print(f"Artifacts written to: {out_dir}")
    print(f"Retrain command saved to: {out_dir / 'retrain_command.txt'}")


if __name__ == "__main__":
    main()
