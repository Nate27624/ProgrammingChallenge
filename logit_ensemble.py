"""
logit_ensemble.py
=================
Evaluate and/or export predictions from a logit-averaged ensemble of saved runs.

Example:
    python -u logit_ensemble.py \
      --results_dir results \
      --run_names m9_t11_seed123,m9_t11_seed2025,m9_t11_seed31415,m9_t11_seed27182,m9_t11_seed7 \
      --test_dir dataset/test \
      --ensemble_name m9_t11_top5_ensemble
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torchaudio.transforms as T
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from baseline import BaselineLSTM
from dataloader import (
    EMOTION_LABELS,
    IDX_TO_EMOTION,
    HOP_SIZE,
    MAX_FRAMES,
    N_MELS,
    N_MFCC,
    SAMPLE_RATE,
    WIN_SIZE,
    SpeechEmotionDataset,
)
from model import BidirectionalMambaSER, CNNBiLSTMAttentionSER


def parse_run_names(raw: str) -> list[str]:
    names = [token.strip() for token in raw.split(",") if token.strip()]
    if not names:
        raise ValueError("No run names provided.")
    return names


def build_model_from_stats(stats: dict[str, Any], device: torch.device) -> torch.nn.Module:
    include_deltas = bool(stats.get("include_deltas", False))
    feature_type = stats.get("feature_type", "mel")
    n_features = int(stats.get("n_features", N_MFCC if feature_type == "mfcc" else N_MELS))
    model_name = stats.get("model_name", "baseline")
    model_config = stats.get("model_config", {})
    in_channels = 3 if include_deltas else 1

    if model_name == "mamba":
        model = BidirectionalMambaSER(
            in_channels=in_channels,
            n_features=n_features,
            cnn_channels=int(model_config.get("cnn_channels", 64)),
            d_model=int(model_config.get("mamba_d_model", 128)),
            d_state=int(model_config.get("mamba_d_state", 32)),
            d_conv=int(model_config.get("mamba_d_conv", 4)),
            expand=int(model_config.get("mamba_expand", 2)),
            num_layers=int(model_config.get("num_layers", 2)),
            dropout=float(model_config.get("dropout", 0.2)),
            frontend_type=str(model_config.get("frontend_type", "basic_cnn")),
            fusion_type=str(model_config.get("fusion_type", "concat")),
            pooling_type=str(model_config.get("pooling_type", "meanmax")),
        )
    elif model_name == "bilstm_attention":
        model = CNNBiLSTMAttentionSER(
            in_channels=in_channels,
            n_features=n_features,
            cnn_channels=int(model_config.get("cnn_channels", 64)),
            hidden_size=int(model_config.get("hidden_size", 256)),
            num_layers=int(model_config.get("num_layers", 2)),
            dropout=float(model_config.get("dropout", 0.3)),
        )
    elif model_name == "baseline":
        input_size = n_features * in_channels
        model = BaselineLSTM(
            input_size=input_size,
            hidden_size=int(model_config.get("hidden_size", 128)),
            num_layers=int(model_config.get("num_layers", 2)),
            dropout=float(model_config.get("dropout", 0.0)),
        )
    else:
        raise ValueError(f"Unsupported model_name in stats: {model_name}")

    return model.to(device)


def build_test_loader(
    test_dir: Path,
    mean: torch.Tensor,
    std: torch.Tensor,
    include_deltas: bool,
    feature_type: str,
    n_features: int,
    batch_size: int,
) -> SpeechEmotionDataset:
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
        apply_db = False
    else:
        transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=WIN_SIZE,
            hop_length=HOP_SIZE,
            n_mels=n_features,
        )
        apply_db = True

    csv_files = list(test_dir.glob("*_labels.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_labels.csv found in {test_dir}")
    labels_csv = csv_files[0]

    dataset = SpeechEmotionDataset(
        audio_dir=test_dir / "audio",
        labels_csv=labels_csv,
        transform=transform,
        max_frames=MAX_FRAMES,
        mean=mean,
        std=std,
        include_deltas=include_deltas,
        apply_db=apply_db,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataset, loader


def save_submission(path: Path, clip_ids: list[str], preds: list[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "predicted_emotion"])
        for clip_id, pred in zip(clip_ids, preds):
            writer.writerow([clip_id, IDX_TO_EMOTION[pred]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Logit-averaged ensemble inference for SER.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--run_names", type=str, required=True, help="Comma-separated run folder names.")
    parser.add_argument("--test_dir", type=str, default="dataset/test")
    parser.add_argument("--ensemble_name", type=str, default="ensemble")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Optional comma-separated float weights aligned with run_names.",
    )
    args = parser.parse_args()

    run_names = parse_run_names(args.run_names)
    results_root = Path(args.results_dir)
    ensemble_dir = results_root / args.ensemble_name.replace(" ", "_")
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Using runs: {run_names}")

    run_dirs = [results_root / name for name in run_names]
    stats_paths = [d / "norm_stats.pt" for d in run_dirs]
    model_paths = [d / "best_model.pt" for d in run_dirs]

    for path in stats_paths + model_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    base_stats = torch.load(stats_paths[0], map_location="cpu")
    include_deltas = bool(base_stats.get("include_deltas", False))
    feature_type = base_stats.get("feature_type", "mel")
    n_features = int(base_stats.get("n_features", N_MFCC if feature_type == "mfcc" else N_MELS))
    mean = base_stats["mean"]
    std = base_stats["std"]

    for i, stats_path in enumerate(stats_paths[1:], start=1):
        stats_i = torch.load(stats_path, map_location="cpu")
        if bool(stats_i.get("include_deltas", False)) != include_deltas:
            raise ValueError(f"Run {run_names[i]} has different include_deltas setting.")
        if stats_i.get("feature_type", "mel") != feature_type:
            raise ValueError(f"Run {run_names[i]} has different feature_type setting.")
        if int(stats_i.get("n_features", n_features)) != n_features:
            raise ValueError(f"Run {run_names[i]} has different n_features setting.")

    dataset, loader = build_test_loader(
        test_dir=Path(args.test_dir),
        mean=mean,
        std=std,
        include_deltas=include_deltas,
        feature_type=feature_type,
        n_features=n_features,
        batch_size=args.batch_size,
    )

    if args.weights.strip():
        weights = [float(x.strip()) for x in args.weights.split(",") if x.strip()]
        if len(weights) != len(run_names):
            raise ValueError("weights length must match run_names length.")
    else:
        weights = [1.0] * len(run_names)

    models: list[torch.nn.Module] = []
    for run_name, stats_path, model_path in zip(run_names, stats_paths, model_paths):
        stats = torch.load(stats_path, map_location="cpu")
        model = build_model_from_stats(stats, device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
        print(f"Loaded model: {run_name}")

    all_preds: list[int] = []
    all_labels: list[int] = []
    clip_ids = [clip_id for clip_id, _ in dataset.samples]

    with torch.no_grad():
        for specs, labels in loader:
            specs = specs.to(device)
            logits_sum = None
            for weight, model in zip(weights, models):
                logits = model(specs) * float(weight)
                logits_sum = logits if logits_sum is None else logits_sum + logits
            preds = logits_sum.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"\nEnsemble weighted F1: {weighted_f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=[IDX_TO_EMOTION[i] for i in range(len(EMOTION_LABELS))]))

    submission_path = ensemble_dir / f"{args.ensemble_name.replace(' ', '_')}.csv"
    save_submission(submission_path, clip_ids, all_preds)
    print(f"Submission saved: {submission_path}")

    summary = {
        "ensemble_name": args.ensemble_name,
        "run_names": run_names,
        "weights": weights,
        "weighted_f1": float(weighted_f1),
        "feature_type": feature_type,
        "n_features": n_features,
        "include_deltas": include_deltas,
        "submission_csv": str(submission_path),
    }
    summary_path = ensemble_dir / "ensemble_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
