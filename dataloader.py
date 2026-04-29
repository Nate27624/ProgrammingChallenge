"""
dataloader.py
=============
Loads the speech emotion dataset and returns PyTorch DataLoaders
for training, validation, and testing.

Each audio clip is:
  1. Loaded as a waveform and resampled to 16 kHz
  2. Converted to a 64-bin Mel-spectrogram
  3. Padded or truncated to a fixed number of time frames
  4. Normalised (mean 0, std 1) using training set statistics

The DataLoader returns:
  - spectrogram : (batch, 1, n_mels, max_frames)  float32
  - label       : (batch,)                         int64

Usage:
    from dataloader import get_dataloaders

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir   = "Dataset",
        val_split  = 0.15,
        batch_size = 64,
    )
"""

import torch
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split


# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000    # Hz — all clips resampled to this
N_MELS        = 64       # number of Mel filterbanks
WIN_SIZE_MS   = 25       # STFT window size in milliseconds
HOP_SIZE_MS   = 10       # STFT hop size in milliseconds
MAX_DURATION  = 3.0      # seconds — clips padded/truncated to this length

EMOTION_LABELS = {
    "anger":   0,
    "disgust": 1,
    "fear":    2,
    "happy":   3,
    "neutral": 4,
    "sad":     5,
}
IDX_TO_EMOTION = {v: k for k, v in EMOTION_LABELS.items()}

# Derived constants
WIN_SIZE   = int(SAMPLE_RATE * WIN_SIZE_MS / 1000)
HOP_SIZE   = int(SAMPLE_RATE * HOP_SIZE_MS / 1000)
MAX_FRAMES = int(np.ceil(MAX_DURATION * SAMPLE_RATE / HOP_SIZE))


# ── Dataset ────────────────────────────────────────────────────────────────────
class SpeechEmotionDataset(Dataset):
    """PyTorch Dataset for speech emotion recognition.

    Args:
        audio_dir  : path to folder containing .wav files
        labels_csv : path to CSV with columns [clip_id, emotion]
        transform  : torchaudio MelSpectrogram transform
        max_frames : fixed spectrogram length along the time axis
        mean       : (1, n_mels, 1) tensor for normalisation — computed on train
        std        : (1, n_mels, 1) tensor for normalisation — computed on train
    """

    def __init__(self, audio_dir, labels_csv, transform,
                 max_frames=MAX_FRAMES, mean=None, std=None):
        self.audio_dir  = Path(audio_dir)
        self.transform  = transform
        self.max_frames = max_frames
        self.mean       = mean
        self.std        = std

        df = pd.read_csv(labels_csv)
        df["label"] = df["emotion"].map(EMOTION_LABELS)
        if df["label"].isna().any():
            unknown = df[df["label"].isna()]["emotion"].unique().tolist()
            raise ValueError(f"Unknown emotion labels found: {unknown}")
        self.samples = list(zip(
            df["clip_id"].tolist(),
            df["label"].astype(int).tolist()
        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_id, label = self.samples[idx]
        wav_path = self.audio_dir / f"{clip_id}.wav"

        waveform, sr = torchaudio.load(wav_path)

        # Resample if needed
        if sr != SAMPLE_RATE:
            waveform = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)

        # Stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Mel-spectrogram: (1, n_mels, time_frames)
        spec = self.transform(waveform)

        # Log compression
        spec = T.AmplitudeToDB()(spec)

        # Pad or truncate to max_frames
        n_frames = spec.shape[-1]
        if n_frames < self.max_frames:
            spec = torch.nn.functional.pad(spec, (0, self.max_frames - n_frames))
        else:
            spec = spec[..., :self.max_frames]

        # Normalise
        if self.mean is not None and self.std is not None:
            spec = (spec - self.mean) / (self.std + 1e-8)

        return spec, label


# ── Normalisation ──────────────────────────────────────────────────────────────
def compute_mean_std(dataset):
    """Compute per-mel-bin mean and std over a dataset.

    Returns tensors of shape (1, n_mels, 1) suitable for broadcasting.
    """
    print("Computing normalisation statistics from training set...")
    all_specs = [dataset[i][0] for i in range(len(dataset))]
    stacked   = torch.stack(all_specs, dim=0)          # (N, 1, n_mels, T)
    mean      = stacked.mean(dim=(0, 3), keepdim=True).squeeze(0)   # (1, n_mels, 1)
    std       = stacked.std(dim=(0, 3),  keepdim=True).squeeze(0)   # (1, n_mels, 1)
    print(f"  Done. Mean range: [{mean.min():.2f}, {mean.max():.2f}]")
    return mean, std


# ── Main entry point ───────────────────────────────────────────────────────────
def get_dataloaders(data_dir, val_split=0.15, batch_size=64,
                    num_workers=0, max_frames=MAX_FRAMES, random_seed=42):
    """Build DataLoaders for train, validation, and test splits.

    The training set is split into train and validation subsets using
    val_split. The test set is the provided test directory (test).

    Args:
        data_dir    : root dataset directory containing train/ and test/
        val_split   : fraction of training data used for validation (default 0.15)
        batch_size  : mini-batch size (default 64)
        num_workers : DataLoader worker processes (default 0)
        max_frames  : fixed spectrogram time length (default MAX_FRAMES)
        random_seed : seed for reproducible train/val split (default 42)

    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir  = data_dir / "test"

    mel_transform = T.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft       = WIN_SIZE,
        hop_length  = HOP_SIZE,
        n_mels      = N_MELS,
    )

    # Build full training dataset (no normalisation yet) to compute stats
    full_train_raw = SpeechEmotionDataset(
        audio_dir  = train_dir / "audio",
        labels_csv = train_dir / "train_labels.csv",
        transform  = mel_transform,
        max_frames = max_frames,
    )

    # Train / validation split
    n_total = len(full_train_raw)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(random_seed)
    train_subset, val_subset = random_split(
        full_train_raw, [n_train, n_val], generator=generator
    )

    # Compute normalisation statistics on the training portion only
    # Build a temporary dataset of just the training indices
    train_only = SpeechEmotionDataset(
        audio_dir  = train_dir / "audio",
        labels_csv = train_dir / "train_labels.csv",
        transform  = mel_transform,
        max_frames = max_frames,
    )
    # Subset to training indices for stats computation
    train_indices = train_subset.indices
    train_stats_dataset = torch.utils.data.Subset(train_only, train_indices)
    mean, std = compute_mean_std(train_stats_dataset)

    # Rebuild all three splits with normalisation applied
    train_dataset = SpeechEmotionDataset(
        audio_dir  = train_dir / "audio",
        labels_csv = train_dir / "train_labels.csv",
        transform  = mel_transform,
        max_frames = max_frames,
        mean       = mean,
        std        = std,
    )
    train_final, val_final = random_split(
        train_dataset, [n_train, n_val], generator=generator
    )

    test_dataset = SpeechEmotionDataset(
        audio_dir  = test_dir / "audio",
        labels_csv = test_dir / "test_labels.csv",
        transform  = mel_transform,
        max_frames = max_frames,
        mean       = mean,
        std        = std,
    )

    train_loader = DataLoader(
        train_final, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_final, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    print(f"\nDataset split:")
    print(f"  Train:      {n_train:>5} clips")
    print(f"  Validation: {n_val:>5} clips  ({val_split*100:.0f}% of train)")
    print(f"  Test:       {len(test_dataset):>5} clips")
    print(f"\nSpectrogram shape: (1, {N_MELS}, {max_frames})")

    return train_loader, val_loader, test_loader, mean, std