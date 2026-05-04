"""
test.py
=======
Loads a saved model checkpoint and evaluates it on the test set.

Logs results to Weights & Biases — weighted F1, per-class F1, and a
confusion matrix. Always writes a leaderboard submission CSV named
after the team.

Usage:
    python test.py \
        --test_dir    dataset/test \
        --results_dir results \
        --team_name   "Team Awesome"

    # Optional: give the wandb run a custom name
    python test.py \
        --test_dir    dataset/test \
        --results_dir results \
        --team_name   "Team Awesome" \
        --run_name    baseline-test

Output:
    Weighted F1-score and per-class F1-score printed to stdout and
    logged to wandb under the project "CSE 5526 - Programming Challenge".
    Team_Awesome.csv written to <results_dir>/<team_name>/ —
    commit and push this file to update the leaderboard.
"""

import argparse
import csv
import torch
import torchaudio.transforms as T
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from dataloader import (SpeechEmotionDataset, EMOTION_LABELS, IDX_TO_EMOTION,
                        SAMPLE_RATE, WIN_SIZE, HOP_SIZE, N_MELS, N_MFCC, MAX_FRAMES)
from baseline import BaselineLSTM
from model import BidirectionalMambaSER, CNNBiLSTMAttentionSER, TemporalFrequencyMambaSER
from wandb_compat import wandb


# ── Inference ──────────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    """Run inference and return predictions and ground-truth labels."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for specs, labels in loader:
            specs = specs.to(device)
            preds = model(specs).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    return all_preds, all_labels


def evaluate_ensemble(models, loader, device):
    """Run inference with an ensemble by averaging member logits."""
    for model in models:
        model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for specs, labels in loader:
            specs = specs.to(device)
            logits_sum = None
            for model in models:
                logits = model(specs)
                logits_sum = logits if logits_sum is None else logits_sum + logits
            preds = logits_sum.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    return all_preds, all_labels


def build_model(model_name, model_config, in_channels, n_features, device):
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
        input_size = n_features * in_channels
        model = BaselineLSTM(
            input_size=input_size,
            hidden_size=int(model_config.get("hidden_size", 128)),
            num_layers=int(model_config.get("num_layers", 2)),
            dropout=float(model_config.get("dropout", 0.0)),
        )
    return model.to(device)


# ── Reporting ──────────────────────────────────────────────────────────────────
def report(preds, labels):
    """Print metrics to stdout and return them for wandb logging."""
    emotion_names = [IDX_TO_EMOTION[i] for i in range(len(EMOTION_LABELS))]
    weighted_f1   = f1_score(labels, preds, average="weighted")
    per_class_f1  = f1_score(labels, preds, average=None,
                              labels=list(range(len(EMOTION_LABELS))))

    print(f"\n{'='*50}")
    print(f"Weighted F1:  {weighted_f1:.4f}")
    print(f"\nPer-class F1:")
    for name, score in zip(emotion_names, per_class_f1):
        bar = "█" * int(score * 20)
        print(f"  {name:<10} {score:.4f}  {bar}")
    print(f"\nClassification report:")
    print(classification_report(labels, preds, target_names=emotion_names))

    return weighted_f1, per_class_f1, emotion_names


# ── Wandb logging ──────────────────────────────────────────────────────────────
def log_to_wandb(preds, labels, weighted_f1, per_class_f1, emotion_names):
    """Log all metrics to wandb in a single call to avoid table truncation."""

    # Build per-class F1 table — all 6 rows
    f1_table = wandb.Table(columns=["emotion", "f1_score"])
    for name, score in zip(emotion_names, per_class_f1):
        f1_table.add_data(name, round(float(score), 4))

    # Build confusion matrix table — all 36 cells (6x6)
    cm = confusion_matrix(labels, preds, labels=list(range(len(emotion_names))))
    cm_table = wandb.Table(columns=["Actual", "Predicted", "nPredictions"])
    for i, actual in enumerate(emotion_names):
        for j, predicted in enumerate(emotion_names):
            cm_table.add_data(actual, predicted, int(cm[i][j]))

    # Log everything in ONE call — prevents wandb from paginating tables
    # across multiple steps and showing only the last row in the summary panel
    wandb.log({
        "test/weighted_f1":            weighted_f1,
        "test/per_class_f1":           f1_table,
        "test/confusion_matrix_table": cm_table,
        "test/confusion_matrix":       wandb.plot.confusion_matrix(
            probs       = None,
            y_true      = labels,
            preds       = preds,
            class_names = emotion_names,
        ),
    })

    # Also store per-class F1 as individual scalars in the run summary
    # so they appear in the runs comparison table
    wandb.summary["test/weighted_f1"] = weighted_f1
    for name, score in zip(emotion_names, per_class_f1):
        wandb.summary[f"test/f1_{name}"] = round(float(score), 4)

    print("Results logged to wandb.")


# ── Leaderboard submission ─────────────────────────────────────────────────────
def save_submission(team_name, clip_ids, preds, results_dir):
    """Write <team_name>.csv for the leaderboard.

    Columns: clip_id, predicted_emotion
    The leaderboard script computes the score server-side from this CSV
    against the ground truth — no self-reported scores.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = results_dir / (team_name.replace(" ", "_") + ".csv")
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "predicted_emotion"])
        for clip_id, pred in zip(clip_ids, preds):
            writer.writerow([clip_id, IDX_TO_EMOTION[pred]])
    print(f"\nSubmission saved  : {filename}")
    print("Commit and push this file to update the leaderboard.")


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir) / args.team_name.replace(" ", "_")
    model_path  = results_dir / "best_model.pt"
    stats_path  = results_dir / "norm_stats.pt"
    # Submission-friendly fallback: allow root-level artifacts.
    if not model_path.exists():
        fallback_model = Path("best_model.pt")
        if fallback_model.exists():
            model_path = fallback_model
    if not stats_path.exists():
        fallback_stats = Path("norm_stats.pt")
        if fallback_stats.exists():
            stats_path = fallback_stats

    print(f"Device: {device}")

    # Load normalisation statistics saved by train.py
    stats     = torch.load(stats_path, map_location="cpu")
    mean, std = stats["mean"], stats["std"]
    include_deltas = bool(stats.get("include_deltas", False))
    feature_type = stats.get("feature_type", "mel")
    n_features = int(stats.get("n_features", N_MFCC if feature_type == "mfcc" else N_MELS))
    model_name = stats.get("model_name", "baseline")
    model_config = stats.get("model_config", {})

    # Locate labels CSV inside the test directory
    test_dir  = Path(args.test_dir)
    csv_files = list(test_dir.glob("*_labels.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_labels.csv found in {test_dir}")
    labels_csv = csv_files[0]
    print(f"\nTest directory : {test_dir}")
    print(f"Labels file    : {labels_csv.name}")

    # Build test DataLoader
    if feature_type == "mfcc":
        feature_transform = T.MFCC(
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
        feature_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=WIN_SIZE,
            hop_length=HOP_SIZE,
            n_mels=n_features,
        )
        apply_db = True

    test_dataset = SpeechEmotionDataset(
        audio_dir  = test_dir / "audio",
        labels_csv = labels_csv,
        transform  = feature_transform,
        max_frames = MAX_FRAMES,
        mean       = mean,
        std        = std,
        include_deltas = include_deltas,
        apply_db = apply_db,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=0
    )
    print(f"Test clips     : {len(test_dataset)}")

    # Load model
    in_channels = 3 if include_deltas else 1
    ckpt = torch.load(model_path, map_location="cpu")
    if (
        model_name == "ensemble"
        and isinstance(ckpt, dict)
        and ckpt.get("checkpoint_type") == "ensemble_v1"
    ):
        members = ckpt.get("members", [])
        ensemble_models = []
        for member in members:
            m_name = member.get("model_name", "baseline")
            m_cfg = member.get("model_config", {})
            state_dict = member.get("state_dict", None)
            if state_dict is None:
                continue
            model = build_model(m_name, m_cfg, in_channels, n_features, device)
            model.load_state_dict(state_dict)
            ensemble_models.append(model)
        if not ensemble_models:
            raise RuntimeError("Ensemble checkpoint has no valid members.")
        print(f"Ensemble loaded from : {model_path} (members={len(ensemble_models)})")
        preds, labels = evaluate_ensemble(ensemble_models, test_loader, device)
    else:
        model = build_model(model_name, model_config, in_channels, n_features, device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt and "checkpoint_type" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"Model loaded from : {model_path}")
        preds, labels = evaluate(model, test_loader, device)

    # Initialise wandb
    run_name = args.run_name or results_dir.name
    wandb.init(
        project = "CSE 5526 - Programming Challenge",
        name    = f"{run_name}",
        config  = {
            "results_dir": str(results_dir),
            "test_dir":    str(test_dir),
        },
    )

    # Evaluate and log
    weighted_f1, per_class_f1, emo_names = report(preds, labels)
    log_to_wandb(preds, labels, weighted_f1, per_class_f1, emo_names)
    wandb.finish()

    # Extract clip IDs in the same order as predictions (shuffle=False)
    clip_ids = [clip_id for clip_id, _ in test_dataset.samples]

    # Always write leaderboard submission CSV
    save_submission(args.team_name, clip_ids, preds, results_dir)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model on the test set."
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="dataset/test",
        help="Path to test directory containing audio/ and *_labels.csv "
             "(default: dataset/test)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Root directory where results are stored (default: results)",
    )
    parser.add_argument(
        "--team_name",
        type=str,
        default="baseline",
        help="Team name — used as the submission filename (e.g. 'Team Awesome' -> Team_Awesome.csv)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="baseline_test",
        help="Optional wandb run name (default: inferred from results_dir)",
    )
    main(parser.parse_args())
