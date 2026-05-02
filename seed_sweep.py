"""
seed_sweep.py
=============
Run the same train.py command across multiple random seeds and summarize results.

Example:
    python seed_sweep.py \
      --train_cmd "python -u train.py --data_dir dataset --results_dir results --model_name mamba ..." \
      --base_name m9_t11 \
      --seeds 42,1337,7,17,123
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

import torch


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("No valid seeds provided.")
    return seeds


def read_checkpoint_metrics(run_dir: Path) -> dict[str, Any]:
    ckpt_path = run_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        return {
            "checkpoint_found": False,
            "best_val_f1": None,
            "best_val_loss": None,
            "last_epoch_idx": None,
        }

    ckpt = torch.load(ckpt_path, map_location="cpu")
    return {
        "checkpoint_found": True,
        "best_val_f1": ckpt.get("best_val_f1"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "last_epoch_idx": ckpt.get("epoch"),
    }


def build_run_command(train_cmd: str, seed: int, run_name: str, results_dir: Path) -> list[str]:
    cmd = shlex.split(train_cmd)
    cmd.extend(
        [
            "--seed",
            str(seed),
            "--team_name",
            run_name,
            "--run_name",
            run_name,
            "--results_dir",
            str(results_dir),
        ]
    )
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run train.py across multiple seeds and rank outcomes.")
    parser.add_argument("--train_cmd", type=str, required=True, help="Base train command (quoted string).")
    parser.add_argument("--base_name", type=str, required=True, help="Prefix for run/team names.")
    parser.add_argument("--seeds", type=str, required=True, help="Comma-separated seed list (e.g. 42,1337,7).")
    parser.add_argument("--results_dir", type=str, default="results", help="Root results directory (default: results).")
    parser.add_argument(
        "--summary_dir",
        type=str,
        default="results/seed_sweeps",
        help="Directory where summary files are written.",
    )
    parser.add_argument("--skip_existing", action="store_true", help="Skip runs that already have checkpoint.pt.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only, do not execute.")
    parser.add_argument("--sleep_sec", type=float, default=0.0, help="Optional pause between runs.")
    parser.add_argument("--top_k", type=int, default=5, help="How many top runs to print (default: 5).")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    results_dir = Path(args.results_dir)
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    print(f"Starting seed sweep: base_name={args.base_name}, seeds={seeds}")

    for idx, seed in enumerate(seeds, start=1):
        run_name = f"{args.base_name}_seed{seed}"
        run_dir = results_dir / run_name
        cmd = build_run_command(args.train_cmd, seed=seed, run_name=run_name, results_dir=results_dir)
        cmd_str = " ".join(shlex.quote(c) for c in cmd)

        print(f"\n[{idx}/{len(seeds)}] {run_name}")
        print(f"Command: {cmd_str}")

        skipped = False
        return_code: int | None = None
        duration_sec: float | None = None

        if args.skip_existing and (run_dir / "checkpoint.pt").exists():
            skipped = True
            print("  Skipping (checkpoint already exists).")
        elif args.dry_run:
            skipped = True
            print("  Dry run mode (not executed).")
        else:
            t0 = time.time()
            proc = subprocess.run(cmd)
            duration_sec = time.time() - t0
            return_code = proc.returncode
            print(f"  Return code: {return_code} | duration_sec={duration_sec:.1f}")

        metrics = read_checkpoint_metrics(run_dir)
        row = {
            "seed": seed,
            "run_name": run_name,
            "return_code": return_code,
            "skipped": skipped,
            "duration_sec": duration_sec,
            **metrics,
        }
        rows.append(row)

        if args.sleep_sec > 0 and idx < len(seeds):
            time.sleep(args.sleep_sec)

    def _sort_key(item: dict[str, Any]) -> float:
        f1 = item.get("best_val_f1")
        return float(f1) if f1 is not None else float("-inf")

    ranked = sorted(rows, key=_sort_key, reverse=True)

    csv_path = summary_dir / f"{args.base_name}_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "run_name",
                "return_code",
                "skipped",
                "duration_sec",
                "checkpoint_found",
                "best_val_f1",
                "best_val_loss",
                "last_epoch_idx",
            ],
        )
        writer.writeheader()
        for row in ranked:
            writer.writerow(row)

    json_path = summary_dir / f"{args.base_name}_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_name": args.base_name,
                "seeds": seeds,
                "results_dir": str(results_dir),
                "rows_ranked": ranked,
            },
            f,
            indent=2,
        )

    print("\nTop runs:")
    for i, row in enumerate(ranked[: max(1, args.top_k)], start=1):
        print(
            f"  {i}. {row['run_name']}: "
            f"best_val_f1={row['best_val_f1']} "
            f"loss={row['best_val_loss']} "
            f"epoch_idx={row['last_epoch_idx']}"
        )
    print(f"\nSummary written to:\n  {csv_path}\n  {json_path}")


if __name__ == "__main__":
    main()
