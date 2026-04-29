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
import wandb
import torch
import torchaudio.transforms as T
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from dataloader import (SpeechEmotionDataset, EMOTION_LABELS, IDX_TO_EMOTION,
                        SAMPLE_RATE, WIN_SIZE, HOP_SIZE, N_MELS, MAX_FRAMES)
from baseline import BaselineLSTM


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

    print(f"Device: {device}")

    # Load normalisation statistics saved by train.py
    stats     = torch.load(stats_path, map_location="cpu")
    mean, std = stats["mean"], stats["std"]

    # Locate labels CSV inside the test directory
    test_dir  = Path(args.test_dir)
    csv_files = list(test_dir.glob("*_labels.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_labels.csv found in {test_dir}")
    labels_csv = csv_files[0]
    print(f"\nTest directory : {test_dir}")
    print(f"Labels file    : {labels_csv.name}")

    # Build test DataLoader
    mel_transform = T.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft       = WIN_SIZE,
        hop_length  = HOP_SIZE,
        n_mels      = N_MELS,
    )
    test_dataset = SpeechEmotionDataset(
        audio_dir  = test_dir / "audio",
        labels_csv = labels_csv,
        transform  = mel_transform,
        max_frames = MAX_FRAMES,
        mean       = mean,
        std        = std,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=0
    )
    print(f"Test clips     : {len(test_dataset)}")

    # Load model
    model = BaselineLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from : {model_path}")

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
    preds, labels                        = evaluate(model, test_loader, device)
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