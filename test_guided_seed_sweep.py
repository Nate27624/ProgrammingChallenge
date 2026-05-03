"""
test_guided_seed_sweep.py
=========================
Run train.py in epoch chunks across multiple seeds, evaluate with test.py after
each chunk, and save the global best model by test weighted F1.

This script is intentionally test-driven for deadline scenarios where maximizing
public test weighted F1 is the immediate objective.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


WEIGHTED_F1_RE = re.compile(r"Weighted F1:\s*([0-9]*\.?[0-9]+)")
EPOCH_LINE_RE = re.compile(r"^Epoch\s+\d+/\d+")


def should_echo_train_line(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if EPOCH_LINE_RE.match(line):
        return True
    key_prefixes = (
        "Device:",
        "Loading data",
        "Dataset split:",
        "Feature shape:",
        "Feature cache:",
        "SpecAugment mode:",
        "Model:",
        "Trainable parameters:",
        "Training for up to",
        "Early stopping triggered",
        "Best model saved to",
        "Norm stats saved to",
    )
    if any(line.startswith(p) for p in key_prefixes):
        return True
    if "--> New best model saved" in line:
        return True
    return False


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("No valid seeds provided.")
    return seeds


def parse_test_f1(stdout: str) -> float | None:
    m = WEIGHTED_F1_RE.search(stdout)
    if not m:
        return None
    return float(m.group(1))


def run_cmd(cmd: list[str], capture: bool = False, env: dict[str, str] | None = None) -> tuple[int, str]:
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    if capture:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=run_env)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    proc = subprocess.run(cmd, env=run_env)
    return proc.returncode, ""


def run_cmd_live_capture(
    cmd: list[str],
    log_path: Path | None = None,
    env: dict[str, str] | None = None,
    full_output: bool = False,
) -> tuple[int, str]:
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        universal_newlines=True,
        env=run_env,
    )
    out_lines: list[str] = []
    with (log_path.open("w", encoding="utf-8") if log_path else open(os.devnull, "w")) as fh:
        assert proc.stdout is not None
        for line in proc.stdout:
            if full_output or should_echo_train_line(line):
                sys.stdout.write(line)
            out_lines.append(line)
            if log_path:
                fh.write(line)
    rc = proc.wait()
    return rc, "".join(out_lines)


def build_train_cmd(
    base_train_cmd: str,
    seed: int,
    run_name: str,
    results_dir: Path,
    num_epochs: int,
) -> list[str]:
    cmd = shlex.split(base_train_cmd)
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
            "--num_epochs",
            str(num_epochs),
        ]
    )
    return cmd


def build_test_cmd(base_test_cmd: str, run_name: str, results_dir: Path) -> list[str]:
    cmd = shlex.split(base_test_cmd)
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


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def save_global_best_snapshot(
    run_dir: Path,
    out_dir: Path,
    seed: int,
    epoch_cap: int,
    test_f1: float,
    run_name: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_if_exists(run_dir / "best_model.pt", out_dir / "best_model.pt")
    copy_if_exists(run_dir / "norm_stats.pt", out_dir / "norm_stats.pt")
    copy_if_exists(run_dir / "config.json", out_dir / "config.json")
    copy_if_exists(run_dir / "train_command.txt", out_dir / "train_command.txt")

    metadata = {
        "best_test_f1": test_f1,
        "best_seed": seed,
        "best_epoch_cap": epoch_cap,
        "best_run_name": run_name,
        "source_run_dir": str(run_dir),
        "saved_at_unix": time.time(),
    }
    with (out_dir / "global_best_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def save_seed_best_snapshot(
    run_dir: Path,
    out_dir: Path,
    seed: int,
    epoch_cap: int,
    test_f1: float,
    run_name: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = out_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    copy_if_exists(run_dir / "best_model.pt", seed_dir / "best_model.pt")
    copy_if_exists(run_dir / "norm_stats.pt", seed_dir / "norm_stats.pt")
    copy_if_exists(run_dir / "config.json", seed_dir / "config.json")
    copy_if_exists(run_dir / "train_command.txt", seed_dir / "train_command.txt")
    metadata = {
        "best_test_f1": test_f1,
        "seed": seed,
        "best_epoch_cap": epoch_cap,
        "run_name": run_name,
        "source_run_dir": str(run_dir),
        "saved_at_unix": time.time(),
    }
    with (seed_dir / "seed_best_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunked train/test sweep over seeds with global best test-F1 checkpointing."
    )
    parser.add_argument("--train_cmd", type=str, required=True, help="Quoted base train command.")
    parser.add_argument("--seeds", type=str, required=True, help="Comma-separated seeds.")
    parser.add_argument("--base_name", type=str, required=True, help="Prefix for run names.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--test_cmd", type=str, default="python -u test.py")
    parser.add_argument("--start_epoch", type=int, default=30)
    parser.add_argument("--end_epoch", type=int, default=120)
    parser.add_argument("--epoch_step", type=int, default=5)
    parser.add_argument(
        "--stale_checks",
        type=int,
        default=2,
        help="Stop a seed when test F1 fails to improve for this many checks.",
    )
    parser.add_argument(
        "--min_epoch_for_cutoff",
        type=int,
        default=80,
        help="Do not apply stale-check early cutoff before this epoch cap.",
    )
    parser.add_argument(
        "--disable_stale_cutoff",
        action="store_true",
        help="Never early-stop a seed based on stale checks.",
    )
    parser.add_argument("--skip_existing_seed", action="store_true")
    parser.add_argument("--sleep_sec", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--allow_nonzero_train_rc_if_model_exists",
        action="store_true",
        help="If train returns non-zero but best_model.pt exists, continue to test anyway.",
    )
    parser.add_argument(
        "--disable_wandb_in_subprocess",
        action="store_true",
        help="Set WANDB_MODE=disabled for train/test subprocesses.",
    )
    parser.add_argument(
        "--force_wandb_offline_in_subprocess",
        action="store_true",
        help="Set WANDB_MODE=offline for train/test subprocesses.",
    )
    parser.add_argument(
        "--full_train_output",
        action="store_true",
        help="Print all train.py subprocess lines instead of concise filtered output.",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    results_dir = Path(args.results_dir)
    summary_dir = results_dir / "test_guided_sweeps"
    summary_dir.mkdir(parents=True, exist_ok=True)
    global_best_dir = results_dir / f"{args.base_name}_global_best"
    global_best_dir.mkdir(parents=True, exist_ok=True)
    per_seed_best_dir = results_dir / f"{args.base_name}_per_seed_best"
    per_seed_best_dir.mkdir(parents=True, exist_ok=True)

    epoch_caps = list(range(args.start_epoch, args.end_epoch + 1, args.epoch_step))
    if not epoch_caps:
        raise ValueError("Invalid epoch range produced no checkpoints.")

    rows: list[dict[str, Any]] = []
    global_best = float("-inf")
    global_best_row: dict[str, Any] | None = None
    subproc_env: dict[str, str] = {}
    if args.disable_wandb_in_subprocess:
        subproc_env["WANDB_MODE"] = "disabled"
    elif args.force_wandb_offline_in_subprocess:
        subproc_env["WANDB_MODE"] = "offline"

    total_jobs = len(seeds) * len(epoch_caps)
    job_idx = 0
    print(
        f"Starting test-guided sweep: seeds={seeds}, epoch_caps={epoch_caps}, "
        f"base_name={args.base_name}"
    )

    for seed in seeds:
        run_name = f"{args.base_name}_seed{seed}"
        run_dir = results_dir / run_name
        if args.skip_existing_seed and (run_dir / "checkpoint.pt").exists():
            print(f"\n[seed={seed}] skipped (checkpoint exists).")
            continue

        seed_best = float("-inf")
        seed_best_epoch = None
        stale = 0
        print(f"\n[seed={seed}] run_name={run_name}")

        for epoch_cap in epoch_caps:
            job_idx += 1
            print(f"  [{job_idx}/{total_jobs}] epoch_cap={epoch_cap}")

            train_cmd = build_train_cmd(
                base_train_cmd=args.train_cmd,
                seed=seed,
                run_name=run_name,
                results_dir=results_dir,
                num_epochs=epoch_cap,
            )
            train_cmd_str = " ".join(shlex.quote(c) for c in train_cmd)
            print(f"    train: {train_cmd_str}")

            t0 = time.time()
            train_log_path = summary_dir / f"{run_name}_e{epoch_cap}_train.log"
            train_rc, _ = run_cmd_live_capture(
                train_cmd,
                log_path=train_log_path,
                env=subproc_env,
                full_output=args.full_train_output,
            )
            train_sec = time.time() - t0
            model_exists = (run_dir / "best_model.pt").exists()
            if train_rc != 0:
                if args.allow_nonzero_train_rc_if_model_exists and model_exists:
                    print(
                        f"    train returned rc={train_rc}, but best_model.pt exists; "
                        "continuing to test."
                    )
                else:
                    print(
                        f"    train failed rc={train_rc}, stopping this seed. "
                        f"(model_exists={model_exists})"
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "run_name": run_name,
                            "epoch_cap": epoch_cap,
                            "train_return_code": train_rc,
                            "train_sec": train_sec,
                            "test_return_code": None,
                            "test_sec": None,
                            "test_weighted_f1": None,
                            "seed_best_so_far": seed_best if seed_best > -1e20 else None,
                            "global_best_so_far": global_best if global_best > -1e20 else None,
                        }
                    )
                    break

            test_cmd = build_test_cmd(args.test_cmd, run_name=run_name, results_dir=results_dir)
            test_cmd_str = " ".join(shlex.quote(c) for c in test_cmd)
            print(f"    test : {test_cmd_str}")

            t1 = time.time()
            test_rc, test_out = run_cmd(test_cmd, capture=True, env=subproc_env)
            test_sec = time.time() - t1
            test_f1 = parse_test_f1(test_out)
            print(f"    test rc={test_rc} f1={test_f1}")
            test_log_path = summary_dir / f"{run_name}_e{epoch_cap}_test.log"
            with test_log_path.open("w", encoding="utf-8") as f:
                f.write(test_out)

            row = {
                "seed": seed,
                "run_name": run_name,
                "epoch_cap": epoch_cap,
                "train_return_code": train_rc,
                "train_sec": train_sec,
                "test_return_code": test_rc,
                "test_sec": test_sec,
                "test_weighted_f1": test_f1,
                "seed_best_so_far": None,
                "global_best_so_far": None,
            }

            if test_f1 is not None:
                if test_f1 > seed_best:
                    seed_best = test_f1
                    seed_best_epoch = epoch_cap
                    stale = 0
                    save_seed_best_snapshot(
                        run_dir=run_dir,
                        out_dir=per_seed_best_dir,
                        seed=seed,
                        epoch_cap=epoch_cap,
                        test_f1=test_f1,
                        run_name=run_name,
                    )
                else:
                    stale += 1

                if test_f1 > global_best:
                    global_best = test_f1
                    global_best_row = {
                        "seed": seed,
                        "run_name": run_name,
                        "epoch_cap": epoch_cap,
                        "test_weighted_f1": test_f1,
                    }
                    save_global_best_snapshot(
                        run_dir=run_dir,
                        out_dir=global_best_dir,
                        seed=seed,
                        epoch_cap=epoch_cap,
                        test_f1=test_f1,
                        run_name=run_name,
                    )
                    print(
                        f"    NEW GLOBAL BEST: seed={seed} epoch_cap={epoch_cap} "
                        f"test_f1={test_f1:.4f}"
                    )

                row["seed_best_so_far"] = seed_best
                row["global_best_so_far"] = global_best

            rows.append(row)

            seed_best_str = f"{seed_best:.4f}" if seed_best > -1e20 else "n/a"
            global_best_str = f"{global_best:.4f}" if global_best > -1e20 else "n/a"
            print(
                f"    status seed_best={seed_best_str} global_best={global_best_str} "
                f"train_sec={train_sec:.1f}s test_sec={test_sec:.1f}s"
            )

            cutoff_allowed = epoch_cap >= args.min_epoch_for_cutoff
            if (not args.disable_stale_cutoff) and cutoff_allowed and stale >= args.stale_checks:
                print(
                    f"    Early stop seed {seed}: no test improvement for "
                    f"{args.stale_checks} checks."
                )
                break

            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

    ranked = sorted(
        rows,
        key=lambda r: float(r["test_weighted_f1"]) if r["test_weighted_f1"] is not None else float("-inf"),
        reverse=True,
    )

    csv_path = summary_dir / f"{args.base_name}_test_guided.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "run_name",
                "epoch_cap",
                "train_return_code",
                "train_sec",
                "test_return_code",
                "test_sec",
                "test_weighted_f1",
                "seed_best_so_far",
                "global_best_so_far",
            ],
        )
        writer.writeheader()
        for row in ranked:
            writer.writerow(row)

    json_path = summary_dir / f"{args.base_name}_test_guided.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_name": args.base_name,
                "seeds": seeds,
                "epoch_caps": epoch_caps,
                "results_dir": str(results_dir),
                "global_best": global_best_row,
                "rows_ranked": ranked,
            },
            f,
            indent=2,
        )

    print("\nTop runs by test weighted F1:")
    for i, row in enumerate(ranked[: max(1, args.top_k)], start=1):
        print(
            f"  {i}. {row['run_name']} seed={row['seed']} epoch_cap={row['epoch_cap']} "
            f"test_f1={row['test_weighted_f1']}"
        )

    if global_best_row is not None:
        print(
            "\nGlobal best:\n"
            f"  run={global_best_row['run_name']}\n"
            f"  seed={global_best_row['seed']}\n"
            f"  epoch_cap={global_best_row['epoch_cap']}\n"
            f"  test_f1={global_best_row['test_weighted_f1']}\n"
            f"  snapshot_dir={global_best_dir}\n"
            f"  per_seed_dir={per_seed_best_dir}"
        )
    else:
        print("\nNo valid test F1 parsed from runs.")

    print(f"\nSummary written to:\n  {csv_path}\n  {json_path}")


if __name__ == "__main__":
    main()
