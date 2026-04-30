"""
optuna_search.py
================
Phase 3 hyperparameter search for SER with coupled capacity/regularization.

Objective: maximize validation weighted F1.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
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
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * specs.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += specs.size(0)

    return total_loss / total, correct / total


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for specs, labels in loader:
            specs, labels = specs.to(device), labels.to(device)
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


def sample_coupled_config(trial: optuna.Trial) -> Dict[str, float | int | str]:
    # Capacity first
    d_model = trial.suggest_categorical("mamba_d_model", [64, 96, 128])
    cnn_channels = trial.suggest_categorical("cnn_channels", [32, 48, 64])
    d_state = trial.suggest_categorical("mamba_d_state", [16, 32, 64])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    mamba_expand = trial.suggest_categorical("mamba_expand", [2, 4])
    mamba_d_conv = trial.suggest_categorical("mamba_d_conv", [3, 4, 5])
    frontend_type = trial.suggest_categorical("frontend_type", ["basic_cnn", "residual_cnn"])
    fusion_type = trial.suggest_categorical("fusion_type", ["concat", "gated"])
    pooling_type = trial.suggest_categorical("pooling_type", ["meanmax", "attention"])

    capacity_score = 0
    if d_model >= 128:
        capacity_score += 1
    if cnn_channels >= 64:
        capacity_score += 1
    if d_state >= 64:
        capacity_score += 1
    if num_layers >= 3:
        capacity_score += 1

    # Coupled regularization ranges (stronger for higher capacity)
    if capacity_score >= 3:
        dropout = trial.suggest_float("dropout", 0.35, 0.55)
        weight_decay = trial.suggest_float("weight_decay", 3e-5, 3e-3, log=True)
        label_smoothing = trial.suggest_float("label_smoothing", 0.05, 0.15)
        num_time_masks = trial.suggest_int("num_time_masks", 2, 4)
        time_mask_param = trial.suggest_int("time_mask_param", 24, 56, step=8)
        num_freq_masks = trial.suggest_int("num_freq_masks", 2, 4)
        freq_mask_param = trial.suggest_int("freq_mask_param", 8, 20, step=4)
    elif capacity_score >= 2:
        dropout = trial.suggest_float("dropout", 0.30, 0.50)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        label_smoothing = trial.suggest_float("label_smoothing", 0.03, 0.12)
        num_time_masks = trial.suggest_int("num_time_masks", 2, 3)
        time_mask_param = trial.suggest_int("time_mask_param", 20, 48, step=4)
        num_freq_masks = trial.suggest_int("num_freq_masks", 2, 3)
        freq_mask_param = trial.suggest_int("freq_mask_param", 6, 16, step=2)
    else:
        dropout = trial.suggest_float("dropout", 0.20, 0.40)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.10)
        num_time_masks = trial.suggest_int("num_time_masks", 1, 3)
        time_mask_param = trial.suggest_int("time_mask_param", 16, 40, step=4)
        num_freq_masks = trial.suggest_int("num_freq_masks", 1, 3)
        freq_mask_param = trial.suggest_int("freq_mask_param", 4, 12, step=2)

    config = {
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
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 8e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "num_time_masks": num_time_masks,
        "time_mask_param": time_mask_param,
        "num_freq_masks": num_freq_masks,
        "freq_mask_param": freq_mask_param,
    }
    return config


def build_objective(
    bundle: DataBundle,
    device: torch.device,
    args: argparse.Namespace,
):
    def objective(trial: optuna.Trial) -> float:
        config = sample_coupled_config(trial)
        set_global_seed(args.seed + trial.number)

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

        criterion = nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=args.patience_lr
        )

        best_val_f1 = float("-inf")
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(args.max_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc, val_f1 = validate(
                model, val_loader, criterion, device
            )
            scheduler.step(val_loss)

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

        return best_val_f1

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna Phase 3 search for Mamba SER.")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--study_name", type=str, default="mamba_phase3")
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

    print(
        f"Starting Optuna study '{args.study_name}' with n_trials={args.n_trials}, "
        f"max_epochs={args.max_epochs}, early_stop_patience={args.early_stop_patience}"
    )
    objective = build_objective(bundle=bundle, device=device, args=args)
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, gc_after_trial=True)

    best = {
        "best_trial_number": study.best_trial.number,
        "best_val_f1": study.best_value,
        "best_params": study.best_trial.params,
    }
    with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(out_dir / "trials.csv", index=False)

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
        f"--frontend_type {study.best_trial.params['frontend_type']}",
        f"--fusion_type {study.best_trial.params['fusion_type']}",
        f"--pooling_type {study.best_trial.params['pooling_type']}",
        f"--dropout {study.best_trial.params['dropout']}",
        f"--learning_rate {study.best_trial.params['learning_rate']}",
        f"--weight_decay {study.best_trial.params['weight_decay']}",
        f"--label_smoothing {study.best_trial.params['label_smoothing']}",
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
    print(f"Artifacts written to: {out_dir}")
    print(f"Retrain command saved to: {out_dir / 'retrain_command.txt'}")


if __name__ == "__main__":
    main()
