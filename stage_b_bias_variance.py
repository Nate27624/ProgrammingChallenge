"""
stage_b_bias_variance.py
=======================
Post-Stage-B selection that balances bias and variance.

It re-runs top completed Optuna Stage-B trials across fixed seeds and ranks
configs by: score = mean_val_f1 - lambda_std * std_val_f1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from model import BidirectionalMambaSER
from optuna_search import (
    N_MFCC,
    N_MELS,
    prepare_data_bundle,
    set_global_seed,
    make_dataloaders_for_trial,
    train_one_epoch,
    validate,
)

try:
    import optuna
except Exception as exc:
    raise RuntimeError("Optuna is not installed. Install with: pip install optuna") from exc


def parse_seeds(raw: str) -> List[int]:
    seeds = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        seeds.append(int(tok))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def trial_to_config(params: Dict[str, Any]) -> Dict[str, Any]:
    if "arch_triplet" in params:
        frontend_type, fusion_type, pooling_type = str(params["arch_triplet"]).split(":")
    else:
        frontend_type = str(params["frontend_type"])
        fusion_type = str(params["fusion_type"])
        pooling_type = str(params["pooling_type"])

    return {
        "mamba_d_model": int(params["mamba_d_model"]),
        "cnn_channels": int(params["cnn_channels"]),
        "mamba_d_state": int(params["mamba_d_state"]),
        "num_layers": int(params["num_layers"]),
        "mamba_expand": int(params["mamba_expand"]),
        "mamba_d_conv": int(params["mamba_d_conv"]),
        "frontend_type": frontend_type,
        "fusion_type": fusion_type,
        "pooling_type": pooling_type,
        "dropout": float(params["dropout"]),
        "weight_decay": float(params["weight_decay"]),
        "label_smoothing": float(params["label_smoothing"]),
        "learning_rate": float(params["learning_rate"]),
        "batch_size": int(params["batch_size"]),
        "num_time_masks": int(params["num_time_masks"]),
        "time_mask_param": int(params["time_mask_param"]),
        "num_freq_masks": int(params["num_freq_masks"]),
        "freq_mask_param": int(params["freq_mask_param"]),
    }


def run_single_seed(
    config: Dict[str, Any],
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    bundle: Any,
) -> float:
    set_global_seed(seed)

    train_loader, val_loader = make_dataloaders_for_trial(
        bundle=bundle,
        batch_size=int(config["batch_size"]),
        include_deltas=True,
        num_workers=args.num_workers,
        num_time_masks=int(config["num_time_masks"]),
        time_mask_param=int(config["time_mask_param"]),
        num_freq_masks=int(config["num_freq_masks"]),
        freq_mask_param=int(config["freq_mask_param"]),
    )

    model = BidirectionalMambaSER(
        in_channels=3,
        n_features=args.n_features,
        cnn_channels=int(config["cnn_channels"]),
        d_model=int(config["mamba_d_model"]),
        d_state=int(config["mamba_d_state"]),
        d_conv=int(config["mamba_d_conv"]),
        expand=int(config["mamba_expand"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        frontend_type=str(config["frontend_type"]),
        fusion_type=str(config["fusion_type"]),
        pooling_type=str(config["pooling_type"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience_lr
    )

    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    best_val_f1 = float("-inf")
    best_val_loss = float("inf")
    patience_counter = 0

    try:
        for _ in range(args.max_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler, amp_enabled
            )
            val_loss, val_acc, val_f1 = validate(
                model, val_loader, criterion, device, amp_enabled
            )
            _ = train_loss
            _ = train_acc
            _ = val_acc
            scheduler.step(val_loss)

            is_better_f1 = val_f1 > best_val_f1 + 1e-6
            is_tie_better_loss = abs(val_f1 - best_val_f1) <= 1e-6 and val_loss < best_val_loss
            if is_better_f1 or is_tie_better_loss:
                best_val_f1 = val_f1
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop_patience:
                    break
    finally:
        del model, optimizer, scheduler, scaler, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return float(best_val_f1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bias/variance-aware top-config selection for Stage B.")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--study_name", type=str, required=True,
                        help="Existing Stage B study name under results/optuna/<study_name>.")
    parser.add_argument("--feature_type", type=str, choices=["mfcc", "mel"], default="mfcc")
    parser.add_argument("--n_features", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Fixed split seed shared by all reruns.")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--seeds", type=str, default="42,1337",
                        help="Comma-separated rerun seeds, e.g. 42,1337,2026")
    parser.add_argument("--lambda_std", type=float, default=0.5,
                        help="Selection score = mean - lambda_std * std")
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--early_stop_patience", type=int, default=6)
    parser.add_argument("--patience_lr", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    args = parser.parse_args()
    if args.max_epochs < 1:
        raise ValueError("--max_epochs must be >= 1")

    args.n_features = args.n_features or (N_MFCC if args.feature_type == "mfcc" else N_MELS)
    seeds = parse_seeds(args.seeds)

    out_dir = Path(args.results_dir) / "optuna" / args.study_name
    if not out_dir.exists():
        raise FileNotFoundError(f"Study dir not found: {out_dir}")
    db_path = out_dir / f"{args.study_name}.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Study DB not found: {db_path}")

    storage = f"sqlite:///{db_path.as_posix()}"
    study = optuna.load_study(study_name=args.study_name, storage=storage)

    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not completed:
        raise RuntimeError("No completed trials found in the specified study.")

    completed.sort(key=lambda t: float(t.value), reverse=True)
    top_trials = completed[:max(1, args.top_k)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Evaluating top {len(top_trials)} trials across seeds={seeds}")
    bundle = prepare_data_bundle(
        data_dir=Path(args.data_dir),
        feature_type=args.feature_type,
        n_features=args.n_features,
        include_deltas=True,
        val_split=args.val_split,
        random_seed=args.split_seed,
    )

    run_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for rank, trial in enumerate(top_trials, start=1):
        cfg = trial_to_config(trial.params)
        vals: List[float] = []
        for seed in seeds:
            try:
                f1 = run_single_seed(cfg, seed=seed, args=args, device=device, bundle=bundle)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    f1 = float("nan")
                else:
                    raise
            vals.append(f1)
            run_rows.append(
                {
                    "rank_by_stageb": rank,
                    "trial_number": trial.number,
                    "seed": seed,
                    "val_f1": f1,
                    "frontend_type": cfg["frontend_type"],
                    "fusion_type": cfg["fusion_type"],
                    "pooling_type": cfg["pooling_type"],
                    "mamba_d_model": cfg["mamba_d_model"],
                    "cnn_channels": cfg["cnn_channels"],
                    "mamba_d_state": cfg["mamba_d_state"],
                    "num_layers": cfg["num_layers"],
                    "dropout": cfg["dropout"],
                    "weight_decay": cfg["weight_decay"],
                    "label_smoothing": cfg["label_smoothing"],
                    "learning_rate": cfg["learning_rate"],
                    "batch_size": cfg["batch_size"],
                }
            )

        valid_vals = [v for v in vals if not math.isnan(v)]
        if not valid_vals:
            mean_f1 = float("nan")
            std_f1 = float("nan")
            score = float("-inf")
        else:
            mean_f1 = float(sum(valid_vals) / len(valid_vals))
            std_f1 = float(statistics.pstdev(valid_vals)) if len(valid_vals) > 1 else 0.0
            score = mean_f1 - args.lambda_std * std_f1

        summary_rows.append(
            {
                "rank_by_stageb": rank,
                "trial_number": trial.number,
                "stageb_val_f1": float(trial.value),
                "num_valid_seeds": len(valid_vals),
                "mean_val_f1": mean_f1,
                "std_val_f1": std_f1,
                "score_mean_minus_lambda_std": score,
                "frontend_type": cfg["frontend_type"],
                "fusion_type": cfg["fusion_type"],
                "pooling_type": cfg["pooling_type"],
                "mamba_d_model": cfg["mamba_d_model"],
                "cnn_channels": cfg["cnn_channels"],
                "mamba_d_state": cfg["mamba_d_state"],
                "num_layers": cfg["num_layers"],
                "mamba_expand": cfg["mamba_expand"],
                "mamba_d_conv": cfg["mamba_d_conv"],
                "dropout": cfg["dropout"],
                "weight_decay": cfg["weight_decay"],
                "label_smoothing": cfg["label_smoothing"],
                "learning_rate": cfg["learning_rate"],
                "batch_size": cfg["batch_size"],
                "num_time_masks": cfg["num_time_masks"],
                "time_mask_param": cfg["time_mask_param"],
                "num_freq_masks": cfg["num_freq_masks"],
                "freq_mask_param": cfg["freq_mask_param"],
            }
        )

        print(
            f"[trial {trial.number}] stageb={float(trial.value):.4f} "
            f"mean={mean_f1:.4f} std={std_f1:.4f} score={score:.4f}"
        )

    summary_rows.sort(key=lambda r: r["score_mean_minus_lambda_std"], reverse=True)

    runs_csv = out_dir / "bias_variance_runs.csv"
    summary_csv = out_dir / "bias_variance_summary.csv"
    summary_json = out_dir / "bias_variance_summary.json"

    if run_rows:
        with open(runs_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
            writer.writeheader()
            writer.writerows(run_rows)

    if summary_rows:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    best = summary_rows[0] if summary_rows else None
    payload = {
        "study_name": args.study_name,
        "seeds": seeds,
        "lambda_std": args.lambda_std,
        "top_k": args.top_k,
        "best_by_bias_variance": best,
        "ranking": summary_rows,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if best is not None:
        retrain_cmd = [
            "python train.py",
            f"--data_dir {args.data_dir}",
            f"--results_dir {args.results_dir}",
            f"--team_name {args.study_name}_bv_best",
            f"--run_name {args.study_name}_bv_best",
            "--model_name mamba",
            f"--feature_type {args.feature_type}",
            "--include_deltas",
            "--specaugment",
            f"--mamba_d_model {best['mamba_d_model']}",
            f"--cnn_channels {best['cnn_channels']}",
            f"--mamba_d_state {best['mamba_d_state']}",
            f"--num_layers {best['num_layers']}",
            f"--mamba_expand {best['mamba_expand']}",
            f"--mamba_d_conv {best['mamba_d_conv']}",
            f"--frontend_type {best['frontend_type']}",
            f"--fusion_type {best['fusion_type']}",
            f"--pooling_type {best['pooling_type']}",
            f"--dropout {best['dropout']}",
            f"--learning_rate {best['learning_rate']}",
            f"--weight_decay {best['weight_decay']}",
            f"--label_smoothing {best['label_smoothing']}",
            f"--batch_size {best['batch_size']}",
            f"--num_time_masks {best['num_time_masks']}",
            f"--time_mask_param {best['time_mask_param']}",
            f"--num_freq_masks {best['num_freq_masks']}",
            f"--freq_mask_param {best['freq_mask_param']}",
        ]
        with open(out_dir / "bias_variance_retrain_command.txt", "w", encoding="utf-8") as f:
            f.write(" \\\n  ".join(retrain_cmd) + "\n")

    print(f"Wrote: {runs_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {summary_json}")
    if best is not None:
        print(f"Recommended trial #{best['trial_number']} with score={best['score_mean_minus_lambda_std']:.4f}")
        print(f"Wrote: {out_dir / 'bias_variance_retrain_command.txt'}")
    del bundle


if __name__ == "__main__":
    main()
