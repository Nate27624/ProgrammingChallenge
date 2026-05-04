"""
test_ensemble.py
================
Evaluate a logit-averaged ensemble over multiple trained runs.

Each run is expected at:
  <results_dir>/<run_name>/best_model.pt
  <results_dir>/<run_name>/norm_stats.pt

Usage example:
  python -u test_ensemble.py \
    --results_dir results \
    --test_dir dataset/test \
    --run_names run_a,run_b,run_c \
    --team_name ensemble_run_abc
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torchaudio.transforms as T
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from baseline import BaselineLSTM
from dataloader import (
    EMOTION_LABELS,
    HOP_SIZE,
    IDX_TO_EMOTION,
    MAX_FRAMES,
    N_MELS,
    N_MFCC,
    SAMPLE_RATE,
    WIN_SIZE,
    SpeechEmotionDataset,
)
from model import BidirectionalMambaSER, CNNBiLSTMAttentionSER, TemporalFrequencyMambaSER


def build_model(model_name: str, model_config: dict, in_channels: int, n_features: int, device: torch.device):
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
    elif model_name == "tf_mamba":
        model = TemporalFrequencyMambaSER(
            in_channels=in_channels,
            n_features=n_features,
            d_model=int(model_config.get("mamba_d_model", 128)),
            d_state=int(model_config.get("mamba_d_state", 32)),
            d_conv=int(model_config.get("mamba_d_conv", 4)),
            expand=int(model_config.get("mamba_expand", 2)),
            num_layers=int(model_config.get("num_layers", 2)),
            dropout=float(model_config.get("dropout", 0.2)),
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
    else:
        model = BaselineLSTM(
            input_size=n_features * in_channels,
            hidden_size=int(model_config.get("hidden_size", 128)),
            num_layers=int(model_config.get("num_layers", 2)),
            dropout=float(model_config.get("dropout", 0.0)),
        )
    return model.to(device)


def build_loader(test_dir: Path, stats: dict, batch_size: int):
    include_deltas = bool(stats.get("include_deltas", False))
    feature_type = stats.get("feature_type", "mel")
    n_features = int(stats.get("n_features", N_MFCC if feature_type == "mfcc" else N_MELS))
    mean, std = stats["mean"], stats["std"]

    if feature_type == "mfcc":
        transform = T.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=n_features,
            melkwargs={"n_fft": WIN_SIZE, "hop_length": HOP_SIZE, "n_mels": N_MELS},
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
        raise FileNotFoundError(f"No *_labels.csv in {test_dir}")
    labels_csv = csv_files[0]

    ds = SpeechEmotionDataset(
        audio_dir=test_dir / "audio",
        labels_csv=labels_csv,
        transform=transform,
        max_frames=MAX_FRAMES,
        mean=mean,
        std=std,
        include_deltas=include_deltas,
        apply_db=apply_db,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return ds, loader, include_deltas, n_features, stats.get("model_name", "baseline"), stats.get("model_config", {})


def evaluate_ensemble(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = Path(args.test_dir)
    results_dir = Path(args.results_dir)
    run_names = [x.strip() for x in args.run_names.split(",") if x.strip()]
    if not run_names:
        raise ValueError("No run_names provided")

    all_logits = []
    labels_ref = None
    clip_ids_ref = None

    for run_name in run_names:
        run_dir = results_dir / run_name
        model_path = run_dir / "best_model.pt"
        stats_path = run_dir / "norm_stats.pt"
        if not model_path.exists() or not stats_path.exists():
            raise FileNotFoundError(f"Missing best_model.pt or norm_stats.pt for run '{run_name}'")

        stats = torch.load(stats_path, map_location="cpu")
        ds, loader, include_deltas, n_features, model_name, model_config = build_loader(
            test_dir, stats, args.batch_size
        )
        model = build_model(model_name, model_config, 3 if include_deltas else 1, n_features, device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        logits_chunks = []
        labels = []
        with torch.no_grad():
            for specs, y in loader:
                specs = specs.to(device)
                logits = model(specs).cpu()
                logits_chunks.append(logits)
                labels.extend(y.tolist())
        run_logits = torch.cat(logits_chunks, dim=0)
        all_logits.append(run_logits)

        if labels_ref is None:
            labels_ref = labels
            clip_ids_ref = [c for c, _ in ds.samples]
        else:
            if labels != labels_ref:
                raise RuntimeError(f"Label order mismatch for run '{run_name}'")

        print(f"Loaded logits from {run_name}: {tuple(run_logits.shape)}")

    logits_avg = torch.stack(all_logits, dim=0).mean(dim=0)
    preds = logits_avg.argmax(dim=1).tolist()
    labels = labels_ref or []

    weighted_f1 = f1_score(labels, preds, average="weighted")
    emotion_names = [IDX_TO_EMOTION[i] for i in range(len(EMOTION_LABELS))]
    print("\n" + "=" * 50)
    print(f"Ensemble weighted F1: {weighted_f1:.4f}")
    print(classification_report(labels, preds, target_names=emotion_names))

    out_dir = results_dir / args.team_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.team_name}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "predicted_emotion"])
        for clip_id, p in zip(clip_ids_ref or [], preds):
            w.writerow([clip_id, IDX_TO_EMOTION[p]])

    print(f"Saved ensemble submission: {out_csv}")


def main():
    p = argparse.ArgumentParser(description="Logit-average ensemble test evaluator.")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--test_dir", type=str, default="dataset/test")
    p.add_argument("--run_names", type=str, required=True, help="Comma-separated run names under results/")
    p.add_argument("--team_name", type=str, required=True, help="Output team/run name for ensemble csv")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()
    evaluate_ensemble(args)


if __name__ == "__main__":
    main()
