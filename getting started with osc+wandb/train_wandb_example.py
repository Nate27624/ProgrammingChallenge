"""
train_wandb_example.py
----------------------
A minimal but complete example of training a linear regression model
with PyTorch, logging metrics to Weights & Biases (wandb).

What gets logged:
  - train_loss        : MSE loss on training batch each epoch
  - val_loss          : MSE loss on held-out validation set
  - learning_rate     : LR value after each scheduler step
  - grad_norm         : Gradient norm (useful for debugging)
  - epoch             : Current epoch

Usage:
  pip install torch wandb
  export WANDB_API_KEY=your_key_here
  python train_wandb_example.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import wandb


# ─── CONFIG ──────────────────────────────────────────────────────────────────

CONFIG = {
    "project":      "osc-linear-demo",
    "n_samples":    1000,
    "n_features":   10,
    "hidden_size":  32,           # small hidden layer so it's not purely linear
    "batch_size":   64,
    "epochs":       60,
    "lr":           0.05,
    "weight_decay": 1e-4,
    "lr_step_size": 10,           # StepLR: decay every N epochs
    "lr_gamma":     0.5,          # StepLR: multiply LR by this factor
    "val_fraction": 0.2,
    "seed":         42,
}


# ─── DATA GENERATION ─────────────────────────────────────────────────────────

def make_dataset(n_samples, n_features, seed=42):
    """
    Generate a simple linear regression dataset:
        y = X @ true_weights + bias + noise
    """
    torch.manual_seed(seed)
    X = torch.randn(n_samples, n_features)
    true_w = torch.randn(n_features, 1)
    true_b = torch.tensor([3.0])
    noise  = 0.3 * torch.randn(n_samples, 1)
    y = X @ true_w + true_b + noise
    return TensorDataset(X, y)


# ─── MODEL ───────────────────────────────────────────────────────────────────

class SmallNet(nn.Module):
    """
    Two-layer network: Linear → ReLU → Linear
    Simple enough to converge fast; complex enough to show wandb features.
    """
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)


# ─── TRAINING ────────────────────────────────────────────────────────────────

def train():
    # 1. Initialize wandb run
    run = wandb.init(
        project=CONFIG["project"],
        config=CONFIG,          # logs all hyperparameters automatically
        name="linear-demo-run",
    )
    cfg = wandb.config       # use wandb.config so sweeps can override values

    # 2. Reproducibility
    torch.manual_seed(cfg.seed)

    # 3. Data
    dataset = make_dataset(cfg.n_samples, cfg.n_features, cfg.seed)
    val_size   = int(len(dataset) * cfg.val_fraction)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=cfg.batch_size)

    # 4. Model, loss, optimizer, scheduler
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = SmallNet(cfg.n_features, cfg.hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # StepLR: multiply LR by gamma every step_size epochs
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma
    )

    # 5. Optional: watch model to log gradients & parameter histograms
    wandb.watch(model, criterion, log="all", log_freq=10)

    print(f"Training on {device} | {train_size} train / {val_size} val samples")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'LR':>10}")
    print("-" * 45)

    # 6. Training loop
    for epoch in range(1, cfg.epochs + 1):

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()

            # Compute gradient norm before stepping (useful diagnostic)
            grad_norm = sum(
                p.grad.detach().norm() ** 2
                for p in model.parameters() if p.grad is not None
            ).sqrt().item()

            optimizer.step()
            train_loss += loss.item() * len(X_batch)

        train_loss /= train_size

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds    = model(X_batch)
                val_loss += criterion(preds, y_batch).item() * len(X_batch)
        val_loss /= val_size

        # ── Scheduler step ─────────────────────────────────────────────────
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── Log to wandb ───────────────────────────────────────────────────
        wandb.log({
            "epoch":         epoch,
            "train_loss":    train_loss,
            "val_loss":      val_loss,
            "learning_rate": current_lr,
            "grad_norm":     grad_norm,
        })

        # ── Console print every 10 epochs ──────────────────────────────────
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}  {train_loss:>12.4f}  {val_loss:>10.4f}  {current_lr:>10.6f}")

    print("-" * 45)
    print("Training complete.")

    # 7. Save model as a wandb artifact
    torch.save(model.state_dict(), "model_final.pt")
    artifact = wandb.Artifact("linear-model", type="model")
    artifact.add_file("model_final.pt")
    run.log_artifact(artifact)

    wandb.finish()


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
