"""
run_cv.py
=========
Lightweight K-fold driver for train.py.

Runs train.py for each fold (stratified K-fold mode) and reports aggregate
validation weighted F1 statistics from saved checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> int:
    """Run one command with optional env overrides and return exit code."""
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    proc = subprocess.run(cmd, env=run_env)
    return proc.returncode


def main() -> None:
    """Run train.py across all folds and persist per-fold + aggregate metrics."""
    parser = argparse.ArgumentParser(description="Run stratified K-fold CV using train.py.")
    parser.add_argument("--base_team_name", type=str, required=True)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="Split/random seed.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--train_args",
        type=str,
        required=True,
        help="Quoted train.py args excluding team_name/run_name/fold args.",
    )
    parser.add_argument("--wandb_mode", type=str, default="disabled", choices=["disabled", "offline", "online"])
    parser.add_argument(
        "--cuda_alloc_conf",
        type=str,
        default="max_split_size_mb:64",
        help="Value for PYTORCH_CUDA_ALLOC_CONF during fold training. Use empty string to disable.",
    )
    args = parser.parse_args()

    if args.num_folds < 2:
        raise ValueError("--num_folds must be >= 2")

    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    cv_rows: list[dict] = []
    env = {"WANDB_MODE": args.wandb_mode}
    if args.cuda_alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf

    print(
        f"Starting CV: base_team_name={args.base_team_name} "
        f"num_folds={args.num_folds} seed={args.seed}"
    )
    # train_args is passed as one quoted string; split it into argv tokens.
    base = shlex.split(args.train_args)
    for fold_idx in range(args.num_folds):
        team_name = f"{args.base_team_name}_fold{fold_idx}"
        run_name = team_name
        cmd = [
            "python",
            "-u",
            "train.py",
            *base,
            "--team_name",
            team_name,
            "--run_name",
            run_name,
            "--results_dir",
            args.results_dir,
            "--seed",
            str(args.seed),
            "--num_folds",
            str(args.num_folds),
            "--fold_index",
            str(fold_idx),
        ]
        print(f"\n[fold {fold_idx + 1}/{args.num_folds}]")
        print("train:", " ".join(shlex.quote(x) for x in cmd))
        t0 = time.time()
        rc = run_cmd(cmd, env=env)
        dur = time.time() - t0

        run_dir = results_root / team_name
        ckpt_path = run_dir / "checkpoint.pt"
        best_model_path = run_dir / "best_model.pt"
        best_val_f1 = None
        best_val_loss = None
        epochs_done = None

        # Read fold metrics from checkpoint so interrupted runs still count.
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            best_val_f1 = float(ckpt.get("best_val_f1", float("nan")))
            best_val_loss = float(ckpt.get("best_val_loss", float("nan")))
            epochs_done = int(ckpt.get("epoch", -1)) + 1

        if rc != 0 and not best_model_path.exists():
            print(f"fold failed rc={rc} and no best_model.pt")
        elif rc != 0 and best_model_path.exists():
            print(f"fold rc={rc}, but best_model.pt exists; keeping fold metrics.")

        cv_rows.append(
            {
                "fold_index": fold_idx,
                "team_name": team_name,
                "return_code": rc,
                "duration_sec": dur,
                "epochs_done": epochs_done,
                "best_val_f1": best_val_f1,
                "best_val_loss": best_val_loss,
                "best_model_exists": best_model_path.exists(),
            }
        )

    valid = [r for r in cv_rows if r["best_val_f1"] is not None and np.isfinite(r["best_val_f1"])]
    vals = [float(r["best_val_f1"]) for r in valid]
    mean_f1 = float(np.mean(vals)) if vals else None
    std_f1 = float(np.std(vals)) if vals else None

    out_dir = results_root / "cv_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.base_team_name}_cv.csv"
    out_json = out_dir / f"{args.base_team_name}_cv.json"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "fold_index",
                "team_name",
                "return_code",
                "duration_sec",
                "epochs_done",
                "best_val_f1",
                "best_val_loss",
                "best_model_exists",
            ],
        )
        w.writeheader()
        for row in cv_rows:
            w.writerow(row)

    summary = {
        "base_team_name": args.base_team_name,
        "num_folds": args.num_folds,
        "seed": args.seed,
        "mean_best_val_f1": mean_f1,
        "std_best_val_f1": std_f1,
        "fold_rows": cv_rows,
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nCV Summary")
    print(f"  mean best_val_f1: {mean_f1}")
    print(f"  std  best_val_f1: {std_f1}")
    print(f"  csv: {out_csv}")
    print(f"  json: {out_json}")


if __name__ == "__main__":
    main()
