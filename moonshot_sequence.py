"""
moonshot_sequence.py
====================
Run a practical SER moonshot sequence:
- multiple profile families
- multiple seeds
- checkpointed epoch-cap training
- test.py scoring after each epoch cap
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


WEIGHTED_F1_RE = re.compile(r"Weighted F1:\s*([0-9]*\.?[0-9]+)")


def parse_seeds(raw: str) -> list[int]:
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    if not out:
        raise ValueError("No seeds provided")
    return out


def parse_profiles(raw: str) -> list[str]:
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            out.append(tok)
    if not out:
        raise ValueError("No profiles provided")
    return out


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


def build_profile_train_cmd(profile: str) -> list[str]:
    # All profiles use same base model family but different regularization/optimizer regimes.
    base = [
        "python", "-u", "train.py",
        "--data_dir", "dataset",
        "--model_name", "bilstm_attention",
        "--feature_type", "mfcc",
        "--n_features", "64",
        "--include_deltas",
        "--specaugment",
        "--specaugment_on_gpu",
        "--num_workers", "0",
        "--feature_cache_dir", "results/_feature_cache",
        "--batch_size", "32",
        "--cnn_channels", "64",
        "--hidden_size", "256",
        "--num_layers", "2",
        "--dropout", "0.3",
        "--learning_rate", "0.0003",
        "--weight_decay", "0.0001",
        "--max_grad_norm", "5.0",
        "--num_time_masks", "1",
        "--time_mask_param", "40",
        "--num_freq_masks", "1",
        "--freq_mask_param", "16",
        "--speed_perturb_prob", "0.0",
        "--patience", "16",
        "--patience_lr", "5",
    ]

    if profile == "ujwal_ce_cosine":
        return base + [
            "--loss_type", "ce",
            "--no_class_weighting",
            "--label_smoothing", "0.1",
            "--scheduler_type", "warmup_cosine",
            "--warmup_epochs", "5",
            "--min_lr_ratio", "0.01",
        ]
    if profile == "ujwal_ce_cosine_robust":
        return base + [
            "--loss_type", "ce",
            "--no_class_weighting",
            "--label_smoothing", "0.1",
            "--scheduler_type", "warmup_cosine",
            "--warmup_epochs", "5",
            "--min_lr_ratio", "0.01",
            "--speed_perturb_prob", "0.20",
            "--speed_perturb_min", "0.9",
            "--speed_perturb_max", "1.1",
            "--num_time_masks", "2",
            "--time_mask_param", "50",
            "--num_freq_masks", "2",
            "--freq_mask_param", "16",
            "--patience", "20",
            "--patience_lr", "6",
        ]
    if profile == "optuna_ce_mixup":
        return base + [
            "--loss_type", "ce",
            "--class_weighting",
            "--label_smoothing", "0.0336",
            "--mixup_alpha", "0.2364",
            "--mixup_prob", "0.4065",
            "--scheduler_type", "plateau",
            "--warmup_epochs", "3",
            "--min_lr_ratio", "0.16",
        ]
    if profile == "focal_imbalance":
        return base + [
            "--loss_type", "focal",
            "--focal_gamma", "2.0",
            "--class_weighting",
            "--label_smoothing", "0.06",
            "--scheduler_type", "warmup_cosine",
            "--warmup_epochs", "5",
            "--min_lr_ratio", "0.01",
        ]
    if profile == "sam_swa":
        return base + [
            "--loss_type", "ce",
            "--no_class_weighting",
            "--label_smoothing", "0.08",
            "--scheduler_type", "warmup_cosine",
            "--warmup_epochs", "5",
            "--min_lr_ratio", "0.02",
            "--optimizer_type", "sam",
            "--sam_rho", "0.05",
            "--use_swa",
            "--swa_start_epoch", "60",
            "--swa_lr", "0.00008",
        ]
    if profile == "sam_swa_robust":
        return base + [
            "--loss_type", "ce",
            "--no_class_weighting",
            "--label_smoothing", "0.10",
            "--scheduler_type", "warmup_cosine",
            "--warmup_epochs", "5",
            "--min_lr_ratio", "0.01",
            "--optimizer_type", "sam",
            "--sam_rho", "0.06",
            "--use_swa",
            "--swa_start_epoch", "55",
            "--swa_lr", "0.00007",
            "--speed_perturb_prob", "0.20",
            "--speed_perturb_min", "0.9",
            "--speed_perturb_max", "1.1",
            "--num_time_masks", "2",
            "--time_mask_param", "50",
            "--num_freq_masks", "2",
            "--freq_mask_param", "16",
            "--patience", "22",
            "--patience_lr", "6",
        ]
    raise ValueError(f"Unsupported profile: {profile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Moonshot SER training/test sequence.")
    parser.add_argument("--base_name", type=str, default="moonshot_v1")
    parser.add_argument(
        "--profiles",
        type=str,
        default="ujwal_ce_cosine_robust,sam_swa_robust,optuna_ce_mixup",
    )
    parser.add_argument("--seeds", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--test_dir", type=str, default="dataset/test")
    parser.add_argument("--start_epoch", type=int, default=35)
    parser.add_argument("--end_epoch", type=int, default=95)
    parser.add_argument("--epoch_step", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument(
        "--skip_existing_test_log",
        action="store_true",
        help="Skip a profile/seed/epoch_cap point when its test log already exists.",
    )
    parser.add_argument(
        "--allow_nonzero_train_rc_if_model_exists",
        action="store_true",
        default=True,
        help="Continue to test.py when train rc is non-zero but best_model.pt exists.",
    )
    args = parser.parse_args()

    profiles = parse_profiles(args.profiles)
    seeds = parse_seeds(args.seeds)
    epoch_caps = list(range(args.start_epoch, args.end_epoch + 1, args.epoch_step))

    results_dir = Path(args.results_dir)
    summary_dir = results_dir / "moonshot_sweeps"
    summary_dir.mkdir(parents=True, exist_ok=True)

    env = {"WANDB_MODE": "disabled"}
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    total_jobs = len(profiles) * len(seeds) * len(epoch_caps)
    job_idx = 0

    print(
        f"Starting moonshot: profiles={profiles}, seeds={seeds}, epoch_caps={epoch_caps}, "
        f"total_jobs={total_jobs}"
    )
    for profile in profiles:
        for seed in seeds:
            run_name = f"{args.base_name}_{profile}_seed{seed}"
            for epoch_cap in epoch_caps:
                job_idx += 1
                test_log = summary_dir / f"{run_name}_e{epoch_cap}_test.log"
                if args.skip_existing_test_log and test_log.exists():
                    print(f"\n[{job_idx}/{total_jobs}] {run_name} epoch_cap={epoch_cap} (skipped: existing test log)")
                    continue
                train_cmd = build_profile_train_cmd(profile) + [
                    "--team_name", run_name,
                    "--run_name", run_name,
                    "--results_dir", args.results_dir,
                    "--seed", str(seed),
                    "--num_epochs", str(epoch_cap),
                ]
                print(f"\n[{job_idx}/{total_jobs}] {run_name} epoch_cap={epoch_cap}")
                print("  train:", " ".join(shlex.quote(x) for x in train_cmd))
                t0 = time.time()
                train_rc, _ = run_cmd(train_cmd, capture=False, env=env)
                train_sec = time.time() - t0
                run_dir = results_dir / run_name
                model_exists = (run_dir / "best_model.pt").exists()

                if train_rc != 0 and not (
                    args.allow_nonzero_train_rc_if_model_exists and model_exists
                ):
                    print(f"  train failed rc={train_rc}, skipping test (model_exists={model_exists})")
                    rows.append(
                        {
                            "profile": profile,
                            "seed": seed,
                            "run_name": run_name,
                            "epoch_cap": epoch_cap,
                            "train_rc": train_rc,
                            "train_sec": train_sec,
                            "test_rc": None,
                            "test_sec": None,
                            "test_f1": None,
                        }
                    )
                    continue
                if train_rc != 0 and model_exists:
                    print(
                        f"  train returned rc={train_rc} but best_model.pt exists; continuing to test."
                    )

                test_cmd = [
                    "python", "-u", "test.py",
                    "--test_dir", args.test_dir,
                    "--results_dir", args.results_dir,
                    "--team_name", run_name,
                    "--run_name", f"eval_{run_name}",
                ]
                print("  test :", " ".join(shlex.quote(x) for x in test_cmd))
                t1 = time.time()
                test_rc, test_out = run_cmd(test_cmd, capture=True, env=env)
                test_sec = time.time() - t1
                test_f1 = parse_test_f1(test_out)
                print(f"  test rc={test_rc} f1={test_f1}")

                row = {
                    "profile": profile,
                    "seed": seed,
                    "run_name": run_name,
                    "epoch_cap": epoch_cap,
                    "train_rc": train_rc,
                    "train_sec": train_sec,
                    "test_rc": test_rc,
                    "test_sec": test_sec,
                    "test_f1": test_f1,
                }
                rows.append(row)
                if test_f1 is not None and (best is None or test_f1 > best["test_f1"]):
                    best = dict(row)
                    print(
                        f"  NEW BEST f1={test_f1:.4f} profile={profile} seed={seed} "
                        f"epoch_cap={epoch_cap}"
                    )

                test_log.write_text(test_out, encoding="utf-8")

    ranked = sorted(
        rows,
        key=lambda r: float(r["test_f1"]) if r["test_f1"] is not None else float("-inf"),
        reverse=True,
    )
    out_csv = summary_dir / f"{args.base_name}_moonshot.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "profile",
                "seed",
                "run_name",
                "epoch_cap",
                "train_rc",
                "train_sec",
                "test_rc",
                "test_sec",
                "test_f1",
            ],
        )
        w.writeheader()
        for row in ranked:
            w.writerow(row)

    out_json = summary_dir / f"{args.base_name}_moonshot.json"
    out_json.write_text(
        json.dumps(
            {
                "base_name": args.base_name,
                "profiles": profiles,
                "seeds": seeds,
                "epoch_caps": epoch_caps,
                "best": best,
                "ranked": ranked,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nTop runs:")
    for i, row in enumerate(ranked[: max(1, args.top_k)], start=1):
        print(
            f"  {i}. f1={row['test_f1']} profile={row['profile']} seed={row['seed']} "
            f"epoch_cap={row['epoch_cap']} run={row['run_name']}"
        )
    print(f"\nWrote:\n  {out_csv}\n  {out_json}")


if __name__ == "__main__":
    main()
