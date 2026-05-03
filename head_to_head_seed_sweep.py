"""
head_to_head_seed_sweep.py
==========================
Run two train.py configurations across identical seeds, evaluate each run via
test.py weighted F1, and rank outcomes.

Example:
    python -u head_to_head_seed_sweep.py \
      --name_a ours \
      --cmd_a "python -u train.py --data_dir dataset --model_name bilstm_attention ..." \
      --name_b ujwal \
      --cmd_b "python -u train.py --data_dir dataset --model_name bilstm_attention ..." \
      --seeds 42,2025,777,1337 \
      --results_dir results
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


WEIGHTED_F1_RE = re.compile(r"Weighted F1:\s*([0-9]*\.?[0-9]+)")


@dataclass
class Profile:
    name: str
    train_cmd: str


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


def build_train_command(train_cmd: str, seed: int, run_name: str, results_dir: Path) -> list[str]:
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


def build_test_command(test_cmd: str, run_name: str, results_dir: Path) -> list[str]:
    cmd = shlex.split(test_cmd)
    cmd.extend(
        [
            "--results_dir",
            str(results_dir),
            "--team_name",
            run_name,
            "--run_name",
            f"eval_{run_name}",
        ]
    )
    return cmd


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


def parse_weighted_f1(stdout: str) -> float | None:
    m = WEIGHTED_F1_RE.search(stdout)
    if not m:
        return None
    return float(m.group(1))


def run_command(cmd: list[str], capture: bool = False) -> tuple[int, str]:
    if capture:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    proc = subprocess.run(cmd)
    return proc.returncode, ""


def topk_mean(values: list[float], k: int) -> float | None:
    if not values:
        return None
    ranked = sorted(values, reverse=True)
    picked = ranked[: max(1, min(k, len(ranked)))]
    return float(sum(picked) / len(picked))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Head-to-head seed sweep for two train configs, ranked by test weighted F1."
    )
    parser.add_argument("--name_a", type=str, required=True)
    parser.add_argument("--cmd_a", type=str, required=True, help="Quoted base train command for profile A.")
    parser.add_argument("--name_b", type=str, required=True)
    parser.add_argument("--cmd_b", type=str, required=True, help="Quoted base train command for profile B.")
    parser.add_argument("--seeds", type=str, required=True, help="Comma-separated seeds.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--test_cmd", type=str, default="python -u test.py", help="Base test command.")
    parser.add_argument("--summary_dir", type=str, default="results/head_to_head_sweeps")
    parser.add_argument("--base_name", type=str, default="head_to_head")
    parser.add_argument("--skip_existing", action="store_true", help="Skip train when checkpoint exists.")
    parser.add_argument("--skip_test_if_exists", action="store_true", help="Reuse prior test score file if present.")
    parser.add_argument("--sleep_sec", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    profiles = [
        Profile(name=args.name_a.strip(), train_cmd=args.cmd_a.strip()),
        Profile(name=args.name_b.strip(), train_cmd=args.cmd_b.strip()),
    ]
    for p in profiles:
        if not p.name:
            raise ValueError("Profile names must be non-empty.")

    seeds = parse_seeds(args.seeds)
    results_dir = Path(args.results_dir)
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    test_cache: dict[str, float] = {}

    total_jobs = len(seeds) * len(profiles)
    job_idx = 0
    print(f"Starting head-to-head sweep: profiles={[p.name for p in profiles]}, seeds={seeds}")

    for seed in seeds:
        for profile in profiles:
            job_idx += 1
            run_name = f"{args.base_name}_{profile.name}_seed{seed}"
            run_dir = results_dir / run_name
            checkpoint_path = run_dir / "checkpoint.pt"
            test_f1_path = run_dir / "test_weighted_f1.txt"

            print(f"\n[{job_idx}/{total_jobs}] profile={profile.name} seed={seed} run={run_name}")

            train_cmd = build_train_command(profile.train_cmd, seed=seed, run_name=run_name, results_dir=results_dir)
            train_cmd_str = " ".join(shlex.quote(c) for c in train_cmd)
            train_skipped = False
            train_rc: int | None = None
            train_sec: float | None = None

            if args.skip_existing and checkpoint_path.exists():
                train_skipped = True
                print("  Train skipped (checkpoint exists).")
            else:
                print(f"  Train cmd: {train_cmd_str}")
                t0 = time.time()
                train_rc, _ = run_command(train_cmd, capture=False)
                train_sec = time.time() - t0
                print(f"  Train return code: {train_rc} | sec={train_sec:.1f}")

            test_rc: int | None = None
            test_sec: float | None = None
            weighted_f1: float | None = None
            test_stdout = ""

            can_reuse = args.skip_test_if_exists and test_f1_path.exists()
            if can_reuse:
                try:
                    weighted_f1 = float(test_f1_path.read_text(encoding="utf-8").strip())
                    print(f"  Reused cached test weighted F1: {weighted_f1:.4f}")
                except Exception:
                    weighted_f1 = None

            if weighted_f1 is None:
                test_cmd = build_test_command(args.test_cmd, run_name=run_name, results_dir=results_dir)
                test_cmd_str = " ".join(shlex.quote(c) for c in test_cmd)
                print(f"  Test cmd: {test_cmd_str}")
                t1 = time.time()
                test_rc, test_stdout = run_command(test_cmd, capture=True)
                test_sec = time.time() - t1
                weighted_f1 = parse_weighted_f1(test_stdout)
                print(
                    f"  Test return code: {test_rc} | sec={test_sec:.1f} | "
                    f"weighted_f1={weighted_f1 if weighted_f1 is not None else 'NA'}"
                )
                if weighted_f1 is not None:
                    test_f1_path.write_text(f"{weighted_f1:.8f}\n", encoding="utf-8")
                elif test_stdout:
                    tail = "\n".join(test_stdout.splitlines()[-20:])
                    print("  Could not parse Weighted F1 from test.py output tail:\n" + tail)

            ckpt_metrics = read_checkpoint_metrics(run_dir)
            row = {
                "profile": profile.name,
                "seed": seed,
                "run_name": run_name,
                "train_return_code": train_rc,
                "train_skipped": train_skipped,
                "train_duration_sec": train_sec,
                "test_return_code": test_rc,
                "test_duration_sec": test_sec,
                "test_weighted_f1": weighted_f1,
                **ckpt_metrics,
            }
            rows.append(row)
            if weighted_f1 is not None:
                test_cache[run_name] = weighted_f1

            if args.sleep_sec > 0 and job_idx < total_jobs:
                time.sleep(args.sleep_sec)

    ranked = sorted(
        rows,
        key=lambda r: float(r["test_weighted_f1"]) if r["test_weighted_f1"] is not None else float("-inf"),
        reverse=True,
    )

    by_profile: dict[str, list[float]] = {}
    for p in profiles:
        vals = [float(r["test_weighted_f1"]) for r in rows if r["profile"] == p.name and r["test_weighted_f1"] is not None]
        by_profile[p.name] = vals

    profile_summary: dict[str, Any] = {}
    for p in profiles:
        vals = by_profile[p.name]
        profile_summary[p.name] = {
            "num_runs": len(vals),
            "best_test_weighted_f1": max(vals) if vals else None,
            "mean_test_weighted_f1": (sum(vals) / len(vals)) if vals else None,
            "top3_mean_test_weighted_f1": topk_mean(vals, 3),
            "top5_mean_test_weighted_f1": topk_mean(vals, 5),
        }

    csv_path = summary_dir / f"{args.base_name}_head_to_head.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "profile",
                "seed",
                "run_name",
                "train_return_code",
                "train_skipped",
                "train_duration_sec",
                "test_return_code",
                "test_duration_sec",
                "test_weighted_f1",
                "checkpoint_found",
                "best_val_f1",
                "best_val_loss",
                "last_epoch_idx",
            ],
        )
        writer.writeheader()
        for row in ranked:
            writer.writerow(row)

    json_path = summary_dir / f"{args.base_name}_head_to_head.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_name": args.base_name,
                "profiles": [p.name for p in profiles],
                "seeds": seeds,
                "results_dir": str(results_dir),
                "profile_summary": profile_summary,
                "rows_ranked": ranked,
            },
            f,
            indent=2,
        )

    print("\nTop runs by test weighted F1:")
    for i, row in enumerate(ranked[: max(1, args.top_k)], start=1):
        print(
            f"  {i}. {row['run_name']}: "
            f"profile={row['profile']} "
            f"seed={row['seed']} "
            f"test_f1={row['test_weighted_f1']} "
            f"val_f1={row['best_val_f1']}"
        )

    print("\nProfile summary:")
    for name, stats in profile_summary.items():
        print(
            f"  {name}: "
            f"best={stats['best_test_weighted_f1']} "
            f"mean={stats['mean_test_weighted_f1']} "
            f"top3_mean={stats['top3_mean_test_weighted_f1']} "
            f"top5_mean={stats['top5_mean_test_weighted_f1']}"
        )

    print(f"\nSummary written to:\n  {csv_path}\n  {json_path}")


if __name__ == "__main__":
    main()

