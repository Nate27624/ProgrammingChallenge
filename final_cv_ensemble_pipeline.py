"""
final_cv_ensemble_pipeline.py
=============================
Pipeline:
1) Run CV training for multiple seeds via run_cv.py
2) Rank all fold models by best_val_f1
3) Build one final ensemble checkpoint from top-k fold models

Outputs:
  results/<ensemble_team_name>/best_model.pt
  results/<ensemble_team_name>/norm_stats.pt

This script is intentionally orchestration-only:
  - it does not define model/training logic itself
  - it composes existing scripts and standardizes the sequence
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


def parse_ints(raw: str) -> list[int]:
    """Parse comma-separated integer seeds."""
    out = []
    for x in raw.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    if not out:
        raise ValueError("No seed values provided.")
    return out


def run(cmd: list[str], env: dict[str, str] | None = None) -> int:
    """Run a subprocess command and return its exit code."""
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    print("run:", " ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.run(cmd, env=run_env)
    return proc.returncode


def main() -> None:
    """Execute end-to-end CV sweep, rank folds, and build final ensemble."""
    p = argparse.ArgumentParser(description="Run CV over seeds and build final ensemble checkpoint.")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--base_name", type=str, default="final_cv")
    p.add_argument("--seeds", type=str, required=True, help="Comma-separated split seeds")
    p.add_argument("--num_folds", type=int, default=3)
    p.add_argument("--train_args", type=str, required=True, help="Quoted args passed to train.py via run_cv.py")
    p.add_argument("--top_k_models", type=int, default=4)
    p.add_argument("--ensemble_team_name", type=str, default="final_ensemble_model")
    p.add_argument("--wandb_mode", type=str, default="offline", choices=["disabled", "offline", "online"])
    p.add_argument("--cuda_alloc_conf", type=str, default="max_split_size_mb:64")
    args = p.parse_args()

    # ------------------------------------------------------------
    # 1) Parse inputs and run per-seed CV jobs
    # ------------------------------------------------------------
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_ints(args.seeds)

    for seed in seeds:
        cv_name = f"{args.base_name}_s{seed}"
        cmd = [
            "python",
            "-u",
            "run_cv.py",
            "--base_team_name",
            cv_name,
            "--num_folds",
            str(args.num_folds),
            "--seed",
            str(seed),
            "--results_dir",
            args.results_dir,
            "--wandb_mode",
            args.wandb_mode,
            "--cuda_alloc_conf",
            args.cuda_alloc_conf,
            "--train_args",
            args.train_args,
        ]
        # Every seed gets its own CV namespace so summaries do not overwrite.
        rc = run(cmd)
        if rc != 0:
            print(f"Warning: CV run returned rc={rc} for seed={seed}")

    # ------------------------------------------------------------
    # 2) Collect and rank fold candidates from summary JSON files
    # ------------------------------------------------------------
    # Collect all fold rows from summaries
    summary_dir = results_dir / "cv_summaries"
    fold_rows = []
    for seed in seeds:
        cv_name = f"{args.base_name}_s{seed}"
        summary_path = summary_dir / f"{cv_name}_cv.json"
        if not summary_path.exists():
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in data.get("fold_rows", []):
            f1 = row.get("best_val_f1")
            run_name = row.get("team_name")
            if f1 is None or run_name is None:
                continue
            run_dir = results_dir / run_name
            # Keep only runs that produced both required artifacts.
            if not (run_dir / "best_model.pt").exists():
                continue
            if not (run_dir / "norm_stats.pt").exists():
                continue
            fold_rows.append((float(f1), run_name))

    if not fold_rows:
        raise RuntimeError("No valid fold models found to ensemble.")

    # Rank by best fold validation F1 and keep top-k as ensemble members.
    fold_rows.sort(reverse=True)
    top_runs = [r for _, r in fold_rows[: args.top_k_models]]
    print("Top runs selected for ensemble:")
    for i, (f1, run_name) in enumerate(fold_rows[: args.top_k_models], 1):
        print(f"  {i}. {run_name}  best_val_f1={f1:.4f}")

    # ------------------------------------------------------------
    # 3) Build a single-file ensemble bundle from top candidates
    # ------------------------------------------------------------
    cmd = [
        "python",
        "-u",
        "create_ensemble_checkpoint.py",
        "--results_dir",
        args.results_dir,
        "--run_names",
        ",".join(top_runs),
        "--team_name",
        args.ensemble_team_name,
    ]
    rc = run(cmd)
    if rc != 0:
        raise RuntimeError(f"Failed to build ensemble checkpoint (rc={rc}).")

    print(f"Final ensemble model written to: {results_dir / args.ensemble_team_name}")


if __name__ == "__main__":
    main()
