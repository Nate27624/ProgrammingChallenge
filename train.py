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

import os
import argparse
import wandb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from dataloader import get_dataloaders
from baseline import BaselineLSTM


# ── Default hyperparameters ────────────────────────────────────────────────────
CONFIG = {
    "hidden_size":   128,
    "num_layers":    2,
    "dropout":       0.0,
    "batch_size":    64,
    "learning_rate": 1e-2,
    "num_epochs":    100,
    "patience":      10,      # early stopping patience (epochs)
    "patience_lr":   5,      # ReduceLROnPlateau patience (epochs)
    "val_split":     0.15,
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

    pbar = tqdm(loader, desc="  Val  ", leave=False)
    with torch.no_grad():
        for specs, labels in pbar:
            specs, labels = specs.to(device), labels.to(device)
            logits = model(specs)
            loss   = criterion(logits, labels)

            total_loss += loss.item() * specs.size(0)
            correct    += (logits.argmax(dim=1) == labels).sum().item()
            total      += specs.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / total, correct / total


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
        val_split  = CONFIG["val_split"],
        batch_size = CONFIG["batch_size"],
    )
    torch.save({"mean": mean, "std": std}, norm_stats_path)

    # ── Model, optimiser, scheduler ───────────────────────────────────────────
    model = BaselineLSTM(
        hidden_size = CONFIG["hidden_size"],
        num_layers  = CONFIG["num_layers"],
        dropout     = CONFIG["dropout"],
    ).to(device)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Trainable parameters: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=CONFIG["patience_lr"]
    )

    # ── Resume logic ──────────────────────────────────────────────────────────
    start_epoch      = 0
    best_val_loss    = float("inf")
    patience_counter = 0
    stopped_epoch    = None
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    wandb_id = wandb.util.generate_id()   # new ID by default

    if checkpoint_path.exists():
        print(f"\nFound checkpoint at {checkpoint_path}. Resuming training...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch      = ckpt["epoch"] + 1
        best_val_loss    = ckpt["best_val_loss"]
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
        f"baseline-lr{CONFIG['learning_rate']}-h{CONFIG['hidden_size']}"
    )
    wandb.init(
        project = "CSE 5526 - Programming Challenge",
        name    = run_name,
        config  = CONFIG,
        id      = wandb_id,
        resume  = "allow",
    )
    # Define epoch as the x-axis for all epoch-level metrics
    wandb.define_metric("epoch")
    wandb.define_metric("epoch/*", step_metric="epoch")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for up to {CONFIG['num_epochs']} epochs "
          f"(early stopping patience = {CONFIG['patience']})...\n")

    for epoch in range(start_epoch, CONFIG["num_epochs"]):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(
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
            "epoch/lr":          lr,
        })

        print(f"Epoch {epoch + 1:>3}/{CONFIG['num_epochs']}  "
              f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f}  "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}  "
              f"lr: {lr:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> New best model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
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
    main(parser.parse_args())