"""
test.py
=======
Evaluate a trained model on the test set and write the submission CSV.

Usage:
    python test.py \
        --test_dir dataset/test \
        --results_dir results \
        --team_name MyTeam

Outputs:
    - weighted F1 and per-class breakdown printed to stdout
    - results/<team_name>/<team_name>.csv  (leaderboard submission)
    - metrics logged to W&B
"""

import argparse
import csv
import wandb
import torch
import torchaudio.transforms as T
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from dataloader import (SpeechEmotionDataset, EMOTION_LABELS, IDX_TO_EMOTION,
                        SAMPLE_RATE, WIN_SIZE, HOP_SIZE, N_MELS, MAX_FRAMES, N_FFT)
from model import SERModel


def run_inference(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for specs, lbls in loader:
            specs = specs.to(device)
            with torch.amp.autocast(device_type=device.type,
                                    enabled=(device.type == 'cuda')):
                out = model(specs)
            preds.extend(out.argmax(1).cpu().tolist())
            labels.extend(lbls.tolist())
    return preds, labels


def print_results(preds, labels):
    names = [IDX_TO_EMOTION[i] for i in range(len(EMOTION_LABELS))]
    wf1 = f1_score(labels, preds, average='weighted', zero_division=0)
    per_class = f1_score(labels, preds, average=None,
                         labels=list(range(len(EMOTION_LABELS))), zero_division=0)

    print(f'\n{"="*50}')
    print(f'Weighted F1: {wf1:.4f}')
    print('\nPer-class F1:')
    for name, score in zip(names, per_class):
        bar = '█' * int(score * 20)
        print(f'  {name:<10} {score:.4f}  {bar}')
    print('\n' + classification_report(labels, preds, target_names=names,
                                      zero_division=0))
    return wf1, per_class, names


def log_wandb(preds, labels, wf1, per_class, names):
    f1_table = wandb.Table(columns=['emotion', 'f1'])
    for name, score in zip(names, per_class):
        f1_table.add_data(name, round(float(score), 4))

    cm = confusion_matrix(labels, preds, labels=list(range(len(names))))
    cm_table = wandb.Table(columns=['actual', 'predicted', 'count'])
    for i, a in enumerate(names):
        for j, p in enumerate(names):
            cm_table.add_data(a, p, int(cm[i][j]))

    wandb.log({
        'test/weighted_f1': wf1,
        'test/per_class_f1': f1_table,
        'test/confusion_matrix_table': cm_table,
        'test/confusion_matrix': wandb.plot.confusion_matrix(
            probs=None, y_true=labels, preds=preds, class_names=names),
    })
    wandb.summary['test/weighted_f1'] = wf1
    for name, score in zip(names, per_class):
        wandb.summary[f'test/f1_{name}'] = round(float(score), 4)


def save_csv(team_name, clip_ids, preds, out_dir):
    fname = out_dir / (team_name.replace(' ', '_') + '.csv')
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'predicted_emotion'])
        for cid, pred in zip(clip_ids, preds):
            writer.writerow([cid, IDX_TO_EMOTION[pred]])
    print(f'\nsubmission saved -> {fname}')


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.results_dir) / args.team_name.replace(' ', '_')
    model_path = out_dir / 'best_model.pt'
    stats_path = out_dir / 'norm_stats.pt'

    print(f'device: {device}')

    stats = torch.load(stats_path, map_location='cpu')
    mean, std = stats['mean'], stats['std']

    test_dir = Path(args.test_dir)
    csv_files = list(test_dir.glob('*_labels.csv'))
    if not csv_files:
        raise FileNotFoundError(f'no labels csv found in {test_dir}')
    labels_csv = csv_files[0]

    mel_tfm = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_SIZE,
        n_mels=N_MELS, power=2.0,
    )
    test_ds = SpeechEmotionDataset(
        audio_dir=test_dir / 'audio',
        labels_csv=labels_csv,
        transform=mel_tfm, max_frames=MAX_FRAMES,
        mean=mean, std=std, augment=False,
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    print(f'test clips: {len(test_ds)}')

    model = SERModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f'loaded model from {model_path}')

    run_name = args.run_name or out_dir.name
    wandb.init(project='CSE 5526 - Programming Challenge', name=run_name,
               config={'test_dir': str(test_dir), 'results_dir': str(out_dir)})

    preds, labels = run_inference(model, test_loader, device)
    wf1, per_class, names = print_results(preds, labels)
    log_wandb(preds, labels, wf1, per_class, names)
    wandb.finish()

    clip_ids = [cid for cid, _ in test_ds.samples]
    save_csv(args.team_name, clip_ids, preds, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default='dataset/test')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--team_name', type=str, default='baseline')
    parser.add_argument('--run_name', type=str, default='test')
    main(parser.parse_args())