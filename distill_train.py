"""
distill_train.py
================
Train a single student model via knowledge distillation from multiple teacher runs.

Teacher runs must contain:
  results/<run_name>/best_model.pt
  results/<run_name>/norm_stats.pt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm

from baseline import BaselineLSTM
from dataloader import N_MELS, N_MFCC, get_dataloaders
from model import BidirectionalMambaSER, CNNBiLSTMAttentionSER, TemporalFrequencyMambaSER


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, cfg: dict, in_channels: int, n_features: int, device: torch.device):
    if model_name == "mamba":
        model = BidirectionalMambaSER(
            in_channels=in_channels,
            n_features=n_features,
            cnn_channels=int(cfg.get("cnn_channels", 64)),
            d_model=int(cfg.get("mamba_d_model", 128)),
            d_state=int(cfg.get("mamba_d_state", 32)),
            d_conv=int(cfg.get("mamba_d_conv", 4)),
            expand=int(cfg.get("mamba_expand", 2)),
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.2)),
            frontend_type=str(cfg.get("frontend_type", "basic_cnn")),
            fusion_type=str(cfg.get("fusion_type", "concat")),
            pooling_type=str(cfg.get("pooling_type", "meanmax")),
        )
    elif model_name == "tf_mamba":
        model = TemporalFrequencyMambaSER(
            in_channels=in_channels,
            n_features=n_features,
            d_model=int(cfg.get("mamba_d_model", 128)),
            d_state=int(cfg.get("mamba_d_state", 32)),
            d_conv=int(cfg.get("mamba_d_conv", 4)),
            expand=int(cfg.get("mamba_expand", 2)),
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.2)),
            pooling_type=str(cfg.get("pooling_type", "meanmax")),
        )
    elif model_name == "bilstm_attention":
        model = CNNBiLSTMAttentionSER(
            in_channels=in_channels,
            n_features=n_features,
            cnn_channels=int(cfg.get("cnn_channels", 64)),
            hidden_size=int(cfg.get("hidden_size", 256)),
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.3)),
        )
    else:
        model = BaselineLSTM(
            input_size=n_features * in_channels,
            hidden_size=int(cfg.get("hidden_size", 128)),
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    return model.to(device)


def load_teacher(results_dir: Path, run_name: str, device: torch.device):
    run_dir = results_dir / run_name
    stats = torch.load(run_dir / "norm_stats.pt", map_location="cpu")
    cfg = stats.get("model_config", {})
    model_name = stats.get("model_name", "baseline")
    feature_type = stats.get("feature_type", "mel")
    n_features = int(stats.get("n_features", N_MFCC if feature_type == "mfcc" else N_MELS))
    include_deltas = bool(stats.get("include_deltas", False))
    model = build_model(model_name, cfg, 3 if include_deltas else 1, n_features, device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()
    return {
        "run_name": run_name,
        "model": model,
        "model_name": model_name,
        "feature_type": feature_type,
        "n_features": n_features,
        "include_deltas": include_deltas,
    }


def validate(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for specs, labels in loader:
            specs = specs.to(device)
            labels = labels.to(device)
            logits = model(specs)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * specs.size(0)
            n += specs.size(0)
            preds_all.extend(logits.argmax(dim=1).cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
    f1 = f1_score(labels_all, preds_all, average="weighted")
    acc = float(np.mean(np.array(preds_all) == np.array(labels_all)))
    return total_loss / max(1, n), acc, f1


def main():
    p = argparse.ArgumentParser(description="Knowledge distillation trainer.")
    p.add_argument("--data_dir", type=str, default="dataset")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--teacher_runs", type=str, required=True, help="Comma-separated run names under results/")
    p.add_argument("--team_name", type=str, default="distilled_student")
    p.add_argument("--student_model", type=str, default="bilstm_attention", choices=["baseline", "bilstm_attention", "mamba", "tf_mamba"])
    p.add_argument("--student_hidden_size", type=int, default=192)
    p.add_argument("--student_num_layers", type=int, default=1)
    p.add_argument("--student_dropout", type=float, default=0.2)
    p.add_argument("--student_cnn_channels", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=60)
    p.add_argument("--learning_rate", type=float, default=8e-4)
    p.add_argument("--weight_decay", type=float, default=2.4e-5)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha_kd", type=float, default=0.7, help="Weight for distillation KL term.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--feature_cache_dir", type=str, default="results/_feature_cache")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    out_dir = results_dir / args.team_name
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher_names = [x.strip() for x in args.teacher_runs.split(",") if x.strip()]
    if not teacher_names:
        raise ValueError("No teacher runs provided.")

    teachers = [load_teacher(results_dir, n, device) for n in teacher_names]
    base = teachers[0]
    for t in teachers[1:]:
        if (
            t["feature_type"] != base["feature_type"]
            or t["n_features"] != base["n_features"]
            or t["include_deltas"] != base["include_deltas"]
        ):
            raise ValueError("All teachers must share feature_type/n_features/include_deltas.")

    train_loader, val_loader, _test_loader, mean, std = get_dataloaders(
        data_dir=args.data_dir,
        val_split=0.15,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=args.seed,
        include_deltas=base["include_deltas"],
        apply_specaugment=True,
        num_time_masks=1,
        time_mask_param=24,
        num_freq_masks=2,
        freq_mask_param=10,
        feature_type=base["feature_type"],
        n_features=base["n_features"],
        speed_perturb_prob=0.0,
        feature_cache_dir=args.feature_cache_dir,
    )

    student_cfg = {
        "hidden_size": args.student_hidden_size,
        "num_layers": args.student_num_layers,
        "dropout": args.student_dropout,
        "cnn_channels": args.student_cnn_channels,
    }
    in_channels = 3 if base["include_deltas"] else 1
    student = build_model(args.student_model, student_cfg, in_channels, base["n_features"], device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_f1 = -1.0
    best_loss = float("inf")
    bad_epochs = 0
    T = float(args.temperature)
    alpha = float(args.alpha_kd)

    print(f"Device: {device}")
    print(f"Teachers: {teacher_names}")
    print(f"Student: {args.student_model}")

    for epoch in range(args.num_epochs):
        student.train()
        tr_loss = 0.0
        tr_n = 0
        tr_correct = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{args.num_epochs}", leave=False)
        for specs, labels in pbar:
            specs = specs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                t_logits = []
                for t in teachers:
                    t_logits.append(t["model"](specs))
                teacher_logits = torch.stack(t_logits, dim=0).mean(dim=0)

            s_logits = student(specs)
            ce = F.cross_entropy(s_logits, labels)
            kd = F.kl_div(
                F.log_softmax(s_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction="batchmean",
            ) * (T * T)
            loss = alpha * kd + (1.0 - alpha) * ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * specs.size(0)
            tr_n += specs.size(0)
            tr_correct += (s_logits.argmax(dim=1) == labels).sum().item()

        tr_loss /= max(1, tr_n)
        tr_acc = tr_correct / max(1, tr_n)
        val_loss, val_acc, val_f1 = validate(student, val_loader, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:03d}/{args.num_epochs} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} lr={lr:.2e}"
        )

        better = val_f1 > best_f1 + 1e-6 or (abs(val_f1 - best_f1) <= 1e-6 and val_loss < best_loss)
        if better:
            best_f1 = val_f1
            best_loss = val_loss
            bad_epochs = 0
            torch.save(student.state_dict(), out_dir / "best_model.pt")
            print(f"  --> New best distilled model saved (val_f1={best_f1:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Save normalization and model config for test.py compatibility
    torch.save(
        {
            "mean": mean,
            "std": std,
            "include_deltas": base["include_deltas"],
            "feature_type": base["feature_type"],
            "n_features": base["n_features"],
            "model_name": args.student_model,
            "model_config": student_cfg,
            "teacher_runs": teacher_names,
            "distillation": {"temperature": T, "alpha_kd": alpha},
        },
        out_dir / "norm_stats.pt",
    )
    (out_dir / "distill_config.json").write_text(
        json.dumps(
            {
                "team_name": args.team_name,
                "teacher_runs": teacher_names,
                "student_model": args.student_model,
                "student_cfg": student_cfg,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "temperature": T,
                "alpha_kd": alpha,
                "best_val_f1": best_f1,
                "best_val_loss": best_loss,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Best distilled model saved to: {out_dir / 'best_model.pt'}")
    print(f"Norm stats saved to: {out_dir / 'norm_stats.pt'}")


if __name__ == "__main__":
    main()
