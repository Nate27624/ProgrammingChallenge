"""
train.py
========
Training script for the CNN+BiLSTM+Attention SER model.

Usage:
    python train.py --data_dir dataset --results_dir results --team_name MyTeam

Saves to results/<team_name>/:
    best_model.pt   - weights of the best validation epoch
    checkpoint.pt   - rolling checkpoint for resuming
    norm_stats.pt   - mean/std computed from training data
    loss_curve.png  - training curves
"""

import os
import random
import argparse
import wandb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score

from dataloader import get_dataloaders
from model import SERModel

# set to offline so wandb doesn't crash if no API key
os.environ.setdefault('WANDB_MODE', 'offline')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# hyperparameters
CONFIG = {
    'batch_size': 32,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'epochs': 80,
    'patience': 15,
    'label_smoothing': 0.1,
    'grad_clip': 5.0,
    'val_split': 0.15,
    # cosine annealing with warm restarts
    'T_0': 10,
    'T_mult': 2,
}


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for specs, labels in tqdm(loader, desc='  train', leave=False):
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type,
                                enabled=(device.type == 'cuda')):
            logits = model(specs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * specs.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        try:
            wandb.log({'train/step_loss': loss.item()})
        except Exception:
            pass

    avg_loss = total_loss / len(loader.dataset)
    wf1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return avg_loss, wf1


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for specs, labels in tqdm(loader, desc='  val  ', leave=False):
            specs, labels = specs.to(device), labels.to(device)
            with torch.amp.autocast(device_type=device.type,
                                    enabled=(device.type == 'cuda')):
                logits = model(specs)
                loss = criterion(logits, labels)

            total_loss += loss.item() * specs.size(0)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    wf1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return avg_loss, wf1


def plot_curves(train_losses, val_losses, train_f1s, val_f1s, path, stopped=None):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_losses, label='train')
    ax1.plot(epochs, val_losses, label='val')
    if stopped:
        ax1.axvline(stopped, ls='--', color='gray', label=f'stopped @ {stopped}')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_f1s, label='train')
    ax2.plot(epochs, val_f1s, label='val')
    if stopped:
        ax2.axvline(stopped, ls='--', color='gray', label=f'stopped @ {stopped}')
    ax2.set_title('Weighted F1'); ax2.set_xlabel('Epoch')
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'curves saved -> {path}')


def main(args):
    set_seed()
    out_dir = Path(args.results_dir) / args.team_name.replace(' ', '_')
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / 'checkpoint.pt'
    best_path = out_dir / 'best_model.pt'
    stats_path = out_dir / 'norm_stats.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    print('\nloading data...')
    train_loader, val_loader, _, mean, std = get_dataloaders(
        data_dir=args.data_dir,
        val_split=CONFIG['val_split'],
        batch_size=CONFIG['batch_size'],
    )
    torch.save({'mean': mean, 'std': std}, stats_path)

    model = SERModel(dropout=0.3).to(device)
    print(f'model params: {model.count_parameters():,}')

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CONFIG['lr'],
                                  weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=CONFIG['T_0'], T_mult=CONFIG['T_mult'], eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(device=device.type,
                                  enabled=(device.type == 'cuda'))

    # state for training loop
    start_epoch = 0
    best_f1 = 0.0
    patience_cnt = 0
    stopped_epoch = None
    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    wandb_id = wandb.util.generate_id()

    # resume from checkpoint if one exists
    if ckpt_path.exists():
        print(f'resuming from {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_f1 = ckpt['best_f1']
        patience_cnt = ckpt['patience_cnt']
        train_losses = ckpt['train_losses']
        val_losses = ckpt['val_losses']
        train_f1s = ckpt['train_f1s']
        val_f1s = ckpt['val_f1s']
        wandb_id = ckpt['wandb_id']
        print(f'epoch {start_epoch}, best val f1 so far: {best_f1:.4f}')

    run_name = args.run_name or 'cnn-bilstm-attn'
    wandb.init(project='CSE 5526 - Programming Challenge', name=run_name,
               config=CONFIG, id=wandb_id, resume='allow')
    wandb.define_metric('epoch')
    wandb.define_metric('epoch/*', step_metric='epoch')

    print(f'\ntraining for up to {CONFIG["epochs"]} epochs '
          f'(patience={CONFIG["patience"]})...\n')

    for epoch in range(start_epoch, CONFIG['epochs']):
        train_loss, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_f1 = val_epoch(model, val_loader, criterion, device)

        scheduler.step(epoch + 1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        lr = optimizer.param_groups[0]['lr']
        wandb.log({
            'epoch': epoch,
            'epoch/train_loss': train_loss, 'epoch/val_loss': val_loss,
            'epoch/train_f1': train_f1, 'epoch/val_f1': val_f1,
            'epoch/lr': lr,
        })

        print(f'epoch {epoch+1:>3}  '
              f'train_loss={train_loss:.4f}  train_f1={train_f1:.4f}  '
              f'val_loss={val_loss:.4f}  val_f1={val_f1:.4f}  lr={lr:.2e}')

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_cnt = 0
            torch.save(model.state_dict(), best_path)
            print(f'  -> saved best (val_f1={best_f1:.4f})')
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG['patience']:
                stopped_epoch = epoch + 1
                print(f'\nearly stopping at epoch {stopped_epoch}')
                break

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_f1': best_f1,
            'patience_cnt': patience_cnt,
            'train_losses': train_losses, 'val_losses': val_losses,
            'train_f1s': train_f1s, 'val_f1s': val_f1s,
            'wandb_id': wandb_id,
        }, ckpt_path)

    curve_path = out_dir / 'loss_curve.png'
    plot_curves(train_losses, val_losses, train_f1s, val_f1s,
                curve_path, stopped_epoch)
    wandb.log({'loss_curve': wandb.Image(str(curve_path))})
    print(f'\nbest val f1: {best_f1:.4f}')
    print(f'model saved to: {best_path}')
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--team_name', type=str, default='baseline')
    parser.add_argument('--run_name', type=str, default=None)
    main(parser.parse_args())