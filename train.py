"""
train.py
========
Training and validation pipeline for the speech emotion recognition challenge.

Trains the baseline model, saves the best checkpoint, and plots loss curves.
Integrates Weights & Biases (wandb) for experiment tracking.

Usage:
    python train.py --data_dir dataset --results_dir results --team_name "Team Awesome"

    # Optional: give the run a custom name
    python train.py --data_dir dataset --results_dir results --team_name "Team Awesome" --run_name baseline_experiment

Outputs saved to <results_dir>/<team_name>/:
    best_model.pt     <- model weights with the lowest validation loss
    checkpoint.pt     <- latest checkpoint for resuming interrupted runs
    norm_stats.pt     <- training set normalisation statistics (needed by test.py)
    loss_curve.png    <- training and validation loss/accuracy curves
"""

import argparse
import uuid
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score

from dataloader import get_dataloaders, N_MELS, N_MFCC
from baseline import BaselineLSTM
from model import BidirectionalMambaSER
from wandb_compat import wandb


# ── Default hyperparameters ────────────────────────────────────────────────────
CONFIG = {
    "model_name": "mamba",
    "feature_type": "mfcc",
    "n_features": N_MFCC,
    "hidden_size":   128,
    "num_layers":    2,
    "dropout":       0.35,
    "mamba_d_model": 96,
    "mamba_d_state": 32,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
    "cnn_channels": 48,
    "frontend_type": "basic_cnn",
    "fusion_type": "concat",
    "pooling_type": "meanmax",
    "batch_size":    64,
    "learning_rate": 3e-4,
    "weight_decay":  1e-4,
    "label_smoothing": 0.05,
    "num_epochs":    100,
    "patience":      10,      # early stopping patience (epochs)
    "patience_lr":   5,      # ReduceLROnPlateau patience (epochs)
    "val_split":     0.15,
    "include_deltas": True,
    "specaugment": True,
    "num_time_masks": 2,
    "time_mask_param": 24,
    "num_freq_masks": 2,
    "freq_mask_param": 8,
}


# ── One training epoch ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch. Logs step-level loss to wandb."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for specs, labels in pbar:
        specs, labels = specs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(specs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * specs.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += specs.size(0)

        # Step-level logging
        wandb.log({"train/step_loss": loss.item()})

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / total, correct / total


# ── Validation epoch ───────────────────────────────────────────────────────────
def validate(model, loader, criterion, device):
    """Run one validation epoch."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Val  ", leave=False)
    with torch.no_grad():
        for specs, labels in pbar:
            specs, labels = specs.to(device), labels.to(device)
            logits = model(specs)
            loss   = criterion(logits, labels)

            total_loss += loss.item() * specs.size(0)
            preds = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += specs.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / total, correct / total, weighted_f1


# ── Loss curve plotting ────────────────────────────────────────────────────────
def plot_curves(train_losses, val_losses, train_accs, val_accs,
                save_path, stopped_epoch=None):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, train_losses, "b-o", markersize=4, label="Train")
    ax1.plot(epochs, val_losses,   "r-o", markersize=4, label="Validation")
    if stopped_epoch:
        ax1.axvline(x=stopped_epoch, color="gray", linestyle="--",
                    label=f"Early stop (epoch {stopped_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in train_accs], "b-o", markersize=4,
             label="Train")
    ax2.plot(epochs, [a * 100 for a in val_accs],   "r-o", markersize=4,
             label="Validation")
    if stopped_epoch:
        ax2.axvline(x=stopped_epoch, color="gray", linestyle="--",
                    label=f"Early stop (epoch {stopped_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss curves saved to: {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    config = CONFIG.copy()
    config["model_name"] = args.model_name
    config["feature_type"] = args.feature_type
    if args.n_features is None:
        config["n_features"] = N_MFCC if args.feature_type == "mfcc" else N_MELS
    else:
        config["n_features"] = args.n_features
    config["include_deltas"] = args.include_deltas
    config["specaugment"] = args.specaugment
    config["num_time_masks"] = args.num_time_masks
    config["time_mask_param"] = args.time_mask_param
    config["num_freq_masks"] = args.num_freq_masks
    config["freq_mask_param"] = args.freq_mask_param
    config["hidden_size"] = args.hidden_size
    config["num_layers"] = args.num_layers
    config["dropout"] = args.dropout
    config["mamba_d_model"] = args.mamba_d_model
    config["mamba_d_state"] = args.mamba_d_state
    config["mamba_d_conv"] = args.mamba_d_conv
    config["mamba_expand"] = args.mamba_expand
    config["cnn_channels"] = args.cnn_channels
    config["frontend_type"] = args.frontend_type
    config["fusion_type"] = args.fusion_type
    config["pooling_type"] = args.pooling_type
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.learning_rate
    config["weight_decay"] = args.weight_decay
    config["label_smoothing"] = args.label_smoothing
    config["num_epochs"] = args.num_epochs
    config["patience"] = args.patience
    config["patience_lr"] = args.patience_lr
    config["val_split"] = args.val_split

    output_dir = Path(args.results_dir) / args.team_name.replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "checkpoint.pt"
    best_model_path = output_dir / "best_model.pt"
    norm_stats_path = output_dir / "norm_stats.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_loader, val_loader, _, mean, std = get_dataloaders(
        data_dir   = args.data_dir,
        val_split  = config["val_split"],
        batch_size = config["batch_size"],
        include_deltas = config["include_deltas"],
        apply_specaugment = config["specaugment"],
        num_time_masks = config["num_time_masks"],
        time_mask_param = config["time_mask_param"],
        num_freq_masks = config["num_freq_masks"],
        freq_mask_param = config["freq_mask_param"],
        feature_type = config["feature_type"],
        n_features = config["n_features"],
    )
    torch.save(
        {
            "mean": mean,
            "std": std,
            "include_deltas": config["include_deltas"],
            "feature_type": config["feature_type"],
            "n_features": config["n_features"],
            "model_name": config["model_name"],
            "model_config": {
                "hidden_size": config["hidden_size"],
                "num_layers": config["num_layers"],
                "dropout": config["dropout"],
                "mamba_d_model": config["mamba_d_model"],
                "mamba_d_state": config["mamba_d_state"],
                "mamba_d_conv": config["mamba_d_conv"],
                "mamba_expand": config["mamba_expand"],
                "cnn_channels": config["cnn_channels"],
                "frontend_type": config["frontend_type"],
                "fusion_type": config["fusion_type"],
                "pooling_type": config["pooling_type"],
                "weight_decay": config["weight_decay"],
                "label_smoothing": config["label_smoothing"],
            },
        },
        norm_stats_path
    )

    # ── Model, optimiser, scheduler ───────────────────────────────────────────
    in_channels = 3 if config["include_deltas"] else 1
    if config["model_name"] == "mamba":
        model = BidirectionalMambaSER(
            in_channels=in_channels,
            n_features=config["n_features"],
            cnn_channels=config["cnn_channels"],
            d_model=config["mamba_d_model"],
            d_state=config["mamba_d_state"],
            d_conv=config["mamba_d_conv"],
            expand=config["mamba_expand"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            frontend_type=config["frontend_type"],
            fusion_type=config["fusion_type"],
            pooling_type=config["pooling_type"],
        ).to(device)
    elif config["model_name"] == "baseline":
        input_size = config["n_features"] * in_channels
        model = BaselineLSTM(
            input_size=input_size,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_name: {config['model_name']}")

    print(f"\nModel: {model.__class__.__name__}")
    print(f"Trainable parameters: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config["patience_lr"]
    )

    # ── Resume logic ──────────────────────────────────────────────────────────
    start_epoch      = 0
    best_val_loss    = float("inf")
    best_val_f1      = float("-inf")
    patience_counter = 0
    stopped_epoch    = None
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    if hasattr(wandb, "util") and hasattr(wandb.util, "generate_id"):
        wandb_id = wandb.util.generate_id()
    else:
        wandb_id = uuid.uuid4().hex[:8]

    if checkpoint_path.exists():
        print(f"\nFound checkpoint at {checkpoint_path}. Resuming training...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch      = ckpt["epoch"] + 1
        best_val_loss    = ckpt["best_val_loss"]
        best_val_f1      = ckpt.get("best_val_f1", float("-inf"))
        patience_counter = ckpt["patience_counter"]
        train_losses     = ckpt["train_losses"]
        val_losses       = ckpt["val_losses"]
        train_accs       = ckpt["train_accs"]
        val_accs         = ckpt["val_accs"]
        wandb_id         = ckpt["wandb_id"]
        print(f"Resuming from epoch {start_epoch} | "
              f"best val loss so far: {best_val_loss:.4f}")
    else:
        print("No checkpoint found. Starting from scratch.")

    # ── Wandb initialisation ──────────────────────────────────────────────────
    # resume="allow" appends to the existing run when wandb_id matches
    run_name = args.run_name or (
        f"{config['model_name']}-{config['feature_type']}-lr{config['learning_rate']}"
    )
    wandb.init(
        project = "CSE 5526 - Programming Challenge",
        name    = run_name,
        config  = config,
        id      = wandb_id,
        resume  = "allow",
    )
    # Define epoch as the x-axis for all epoch-level metrics
    wandb.define_metric("epoch")
    wandb.define_metric("epoch/*", step_metric="epoch")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for up to {config['num_epochs']} epochs "
          f"(early stopping patience = {config['patience']})...\n")

    for epoch in range(start_epoch, config["num_epochs"]):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        lr = optimizer.param_groups[0]["lr"]

        # Epoch-level wandb logging
        wandb.log({
            "epoch":             epoch,
            "epoch/train_loss":  train_loss,
            "epoch/val_loss":    val_loss,
            "epoch/train_acc":   train_acc * 100,
            "epoch/val_acc":     val_acc   * 100,
            "epoch/val_f1":      val_f1,
            "epoch/lr":          lr,
        })

        print(f"Epoch {epoch + 1:>3}/{config['num_epochs']}  "
              f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f}  "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}  "
              f"val_f1: {val_f1:.4f}  "
              f"lr: {lr:.2e}")

        # Save best model by weighted F1; use loss as tiebreaker
        is_better_f1 = val_f1 > best_val_f1 + 1e-6
        is_tie_better_loss = abs(val_f1 - best_val_f1) <= 1e-6 and val_loss < best_val_loss
        if is_better_f1 or is_tie_better_loss:
            best_val_f1      = val_f1
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(
                "  --> New best model saved "
                f"(val_f1: {best_val_f1:.4f}, val_loss: {best_val_loss:.4f})"
            )
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                stopped_epoch = epoch + 1
                print(f"\nEarly stopping triggered at epoch {stopped_epoch}.")
                break

        # Save latest checkpoint for resuming interrupted runs
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss":        best_val_loss,
            "best_val_f1":          best_val_f1,
            "patience_counter":     patience_counter,
            "train_losses":         train_losses,
            "val_losses":           val_losses,
            "train_accs":           train_accs,
            "val_accs":             val_accs,
            "wandb_id":             wandb_id,
        }, checkpoint_path)

    # ── Post-training ──────────────────────────────────────────────────────────
    plot_curves(
        train_losses, val_losses,
        train_accs,   val_accs,
        save_path     = output_dir / "loss_curve.png",
        stopped_epoch = stopped_epoch,
    )

    # Upload final loss curve image to wandb
    wandb.log({"loss_curve": wandb.Image(str(output_dir / "loss_curve.png"))})

    print(f"\nBest model saved to : {best_model_path}")
    print(f"Norm stats saved to : {norm_stats_path}")
    wandb.finish()


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the speech emotion recognition baseline."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset",
        help="Root dataset directory containing train/ and test/ "
             "(default: dataset)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Root directory where results are saved (default: results)",
    )
    parser.add_argument(
        "--team_name",
        type=str,
        default="baseline",
        help="Team name — output is saved to <results_dir>/<team_name>/ "
             "(default: baseline)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default='baseline_train',
        help="Optional wandb run name (default: auto-generated from config)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["baseline", "mamba"],
        default="mamba",
        help="Model type to train (default: mamba).",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        choices=["mel", "mfcc"],
        default="mfcc",
        help="Input feature type (default: mfcc).",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=None,
        help="Number of feature bins (e.g., MFCC coeffs or mel bins).",
    )
    parser.add_argument(
        "--include_deltas",
        action="store_true",
        default=True,
        help="Use delta + delta-delta channels (default: enabled).",
    )
    parser.add_argument(
        "--no_include_deltas",
        action="store_false",
        dest="include_deltas",
        help="Disable delta + delta-delta channels.",
    )
    parser.add_argument(
        "--specaugment",
        action="store_true",
        default=True,
        help="Apply SpecAugment to train split only (default: enabled).",
    )
    parser.add_argument(
        "--no_specaugment",
        action="store_false",
        dest="specaugment",
        help="Disable SpecAugment.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Baseline LSTM hidden size (used when --model_name baseline).",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of recurrent/SSM layers (default: 2).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.35,
        help="Dropout probability (default: 0.35).",
    )
    parser.add_argument(
        "--mamba_d_model",
        type=int,
        default=96,
        help="Mamba hidden model width (default: 96).",
    )
    parser.add_argument(
        "--mamba_d_state",
        type=int,
        default=32,
        help="Mamba state dimension N (default: 32).",
    )
    parser.add_argument(
        "--mamba_d_conv",
        type=int,
        default=4,
        help="Mamba local convolution width (default: 4).",
    )
    parser.add_argument(
        "--mamba_expand",
        type=int,
        default=2,
        help="Mamba expansion factor E (default: 2).",
    )
    parser.add_argument(
        "--cnn_channels",
        type=int,
        default=48,
        help="CNN front-end output channels (default: 48).",
    )
    parser.add_argument(
        "--frontend_type",
        type=str,
        choices=["basic_cnn", "residual_cnn"],
        default="basic_cnn",
        help="CNN frontend variant for mamba model (default: basic_cnn).",
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        choices=["concat", "gated"],
        default="concat",
        help="Bidirectional fusion strategy for mamba model (default: concat).",
    )
    parser.add_argument(
        "--pooling_type",
        type=str,
        choices=["meanmax", "attention"],
        default="meanmax",
        help="Temporal pooling strategy for mamba model (default: meanmax).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Mini-batch size (default: 64).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Adam weight decay (default: 1e-4).",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.05,
        help="Cross-entropy label smoothing (default: 0.05).",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Maximum training epochs (default: 100).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10).",
    )
    parser.add_argument(
        "--patience_lr",
        type=int,
        default=5,
        help="ReduceLROnPlateau patience (default: 5).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Validation split fraction (default: 0.15).",
    )
    parser.add_argument(
        "--num_time_masks",
        type=int,
        default=2,
        help="Number of time masks for SpecAugment (default: 2).",
    )
    parser.add_argument(
        "--time_mask_param",
        type=int,
        default=24,
        help="Maximum width for each time mask (default: 24).",
    )
    parser.add_argument(
        "--num_freq_masks",
        type=int,
        default=2,
        help="Number of frequency masks for SpecAugment (default: 2).",
    )
    parser.add_argument(
        "--freq_mask_param",
        type=int,
        default=8,
        help="Maximum width for each frequency mask (default: 8).",
    )
    main(parser.parse_args())
