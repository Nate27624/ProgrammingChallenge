"""
super_submission_vote.py
========================
Create a weighted-vote "super submission" from multiple prediction CSV files.

Supports sources as:
1) Local file path, e.g. results/run_a/run_a.csv
2) Git object spec, e.g. origin/Nate:results/run_b/run_b.csv

When labels CSV is provided, this can:
- score each source (weighted F1)
- use those F1 scores as vote weights
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.metrics import f1_score

EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
EMO2IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX2EMO = {i: e for i, e in enumerate(EMOTIONS)}


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def load_text(source: str) -> str:
    if ":" in source and not Path(source).exists():
        # Treat as git object spec: <ref>:<path>
        return subprocess.check_output(["git", "show", source], text=True, errors="replace")
    return Path(source).read_text(encoding="utf-8", errors="replace")


def read_submission(source: str) -> Dict[str, int]:
    txt = load_text(source)
    rows = {}
    rd = csv.DictReader(io.StringIO(txt))
    for row in rd:
        clip = row["clip_id"].strip()
        emo = row["predicted_emotion"].strip().lower()
        rows[clip] = EMO2IDX[emo]
    return rows


def read_labels(labels_csv: str) -> Dict[str, int]:
    rows = {}
    with open(labels_csv, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        col = "emotion" if "emotion" in rd.fieldnames else "predicted_emotion"
        for row in rd:
            rows[row["clip_id"].strip()] = EMO2IDX[row[col].strip().lower()]
    return rows


def score_submission(preds: Dict[str, int], labels: Dict[str, int]) -> float:
    y_true, y_pred = [], []
    for clip_id, y in labels.items():
        if clip_id in preds:
            y_true.append(y)
            y_pred.append(preds[clip_id])
    if not y_true:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="weighted"))


def vote_submissions(
    submissions: List[Tuple[str, Dict[str, int]]],
    weights: Dict[str, float],
) -> Dict[str, int]:
    clip_ids = None
    for _, preds in submissions:
        if clip_ids is None:
            clip_ids = set(preds.keys())
        else:
            clip_ids &= set(preds.keys())
    clip_ids = sorted(clip_ids or [])

    out = {}
    for clip in clip_ids:
        score = [0.0] * len(EMOTIONS)
        for name, preds in submissions:
            w = weights.get(name, 1.0)
            score[preds[clip]] += w
        best_idx = max(range(len(EMOTIONS)), key=lambda i: (score[i], -i))
        out[clip] = best_idx
    return out


def save_submission(out_csv: Path, preds: Dict[str, int]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "predicted_emotion"])
        for clip_id in sorted(preds.keys()):
            w.writerow([clip_id, IDX2EMO[preds[clip_id]]])


def main() -> None:
    p = argparse.ArgumentParser(description="Weighted vote combiner for submission CSV files.")
    p.add_argument("--sources", type=str, required=True, help="Comma-separated csv sources (path or gitref:path).")
    p.add_argument("--out_csv", type=str, required=True, help="Output submission CSV path.")
    p.add_argument("--labels_csv", type=str, default="", help="Optional labels csv for scoring/weighting.")
    p.add_argument("--weight_mode", type=str, default="uniform", choices=["uniform", "f1"])
    args = p.parse_args()

    sources = parse_csv_list(args.sources)
    if not sources:
        raise ValueError("No sources provided.")

    submissions: List[Tuple[str, Dict[str, int]]] = []
    for src in sources:
        preds = read_submission(src)
        submissions.append((src, preds))
        print(f"Loaded {src} ({len(preds)} rows)")

    labels = read_labels(args.labels_csv) if args.labels_csv else None
    weights: Dict[str, float] = {}

    if args.weight_mode == "f1":
        if labels is None:
            raise ValueError("--labels_csv is required for --weight_mode f1")
        for name, preds in submissions:
            f1 = score_submission(preds, labels)
            # Clamp minimum positive weight so weak models do not become zeroed.
            weights[name] = max(1e-4, f1)
            print(f"  score {name}: weighted_f1={f1:.4f}")
    else:
        for name, _ in submissions:
            weights[name] = 1.0

    merged = vote_submissions(submissions, weights)
    save_submission(Path(args.out_csv), merged)
    print(f"Saved super submission: {args.out_csv}")

    if labels is not None:
        merged_f1 = score_submission(merged, labels)
        print(f"Ensemble weighted_f1: {merged_f1:.4f}")


if __name__ == "__main__":
    main()
