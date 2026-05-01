"""
dataloader.py
=============
Loads the speech emotion dataset and returns PyTorch DataLoaders
for training, validation, and testing.

Each audio clip is:
  1. Loaded as a waveform and resampled to 16 kHz
  2. Converted to either MFCC or Mel features
  3. Optionally expanded with delta and delta-delta channels
  4. Padded or truncated to a fixed number of time frames
  5. Normalised (mean 0, std 1) using training set statistics

The DataLoader returns:
  - features : (batch, channels, n_features, max_frames)  float32
  - label    : (batch,)                                   int64

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
import torchaudio.functional as AF
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split


# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000    # Hz — all clips resampled to this
N_MELS        = 64       # number of Mel filterbanks
N_MFCC        = 40       # number of MFCC coefficients
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
        transform  : torchaudio feature transform
        max_frames : fixed spectrogram length along the time axis
        mean       : (channels, n_mels, 1) tensor for normalisation
        std        : (channels, n_mels, 1) tensor for normalisation
    """

    def __init__(self, audio_dir, labels_csv, transform,
                 max_frames=MAX_FRAMES, mean=None, std=None,
                 include_deltas=False, apply_specaugment=False,
                 num_time_masks=2, time_mask_param=24,
                 num_freq_masks=2, freq_mask_param=8,
                 apply_db=False, normalize_waveform_peak=True):
        self.audio_dir  = Path(audio_dir)
        self.transform  = transform
        self.max_frames = max_frames
        self.mean       = mean
        self.std        = std
        self.include_deltas = include_deltas
        self.apply_specaugment = apply_specaugment
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.apply_db = apply_db
        self.normalize_waveform_peak = normalize_waveform_peak
        self.to_db = T.AmplitudeToDB()
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)

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

        try:
            import soundfile as sf

            waveform_np, sr = sf.read(str(wav_path), dtype="float32")
            if waveform_np.ndim == 1:
                waveform = torch.from_numpy(waveform_np).unsqueeze(0)
            else:
                waveform = torch.from_numpy(waveform_np.T)
        except Exception:
            waveform, sr = torchaudio.load(wav_path)

        # Resample if needed
        if sr != SAMPLE_RATE:
            waveform = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)

        # Stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if self.normalize_waveform_peak:
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak

        # Features: (1, n_features, time_frames)
        spec = self.transform(waveform)

        # Log compression for magnitude features (e.g., Mel-spectrogram)
        if self.apply_db:
            spec = self.to_db(spec)

        # Pad or truncate to max_frames
        n_frames = spec.shape[-1]
        if n_frames < self.max_frames:
            spec = torch.nn.functional.pad(spec, (0, self.max_frames - n_frames))
        else:
            spec = spec[..., :self.max_frames]

        if self.include_deltas:
            delta = AF.compute_deltas(spec)
            delta2 = AF.compute_deltas(delta)
            spec = torch.cat([spec, delta, delta2], dim=0)

        # Normalise
        if self.mean is not None and self.std is not None:
            spec = (spec - self.mean) / (self.std + 1e-8)

        if self.apply_specaugment:
            for _ in range(self.num_freq_masks):
                spec = self.freq_mask(spec)
            for _ in range(self.num_time_masks):
                spec = self.time_mask(spec)

        return spec, label


# ── Normalisation ──────────────────────────────────────────────────────────────
def compute_mean_std(dataset):
    """Compute per-channel, per-mel-bin mean and std over a dataset.

    Returns tensors of shape (channels, n_mels, 1) suitable for broadcasting.
    """
    print("Computing normalisation statistics from training set...")
    all_specs = [dataset[i][0] for i in range(len(dataset))]
    stacked   = torch.stack(all_specs, dim=0)          # (N, C, n_mels, T)
    mean      = stacked.mean(dim=(0, 3), keepdim=True).squeeze(0)   # (C, n_mels, 1)
    std       = stacked.std(dim=(0, 3),  keepdim=True).squeeze(0)   # (C, n_mels, 1)
    print(f"  Done. Mean range: [{mean.min().item():.2f}, {mean.max().item():.2f}]")
    return mean, std


# ── Main entry point ───────────────────────────────────────────────────────────
def get_dataloaders(data_dir, val_split=0.15, batch_size=64,
                    num_workers=0, max_frames=MAX_FRAMES, random_seed=42,
                    include_deltas=False, apply_specaugment=False,
                    num_time_masks=2, time_mask_param=24,
                    num_freq_masks=2, freq_mask_param=8,
                    feature_type="mel", n_features=None):
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

    if n_features is None:
        n_features = N_MFCC if feature_type == "mfcc" else N_MELS

    if feature_type == "mfcc":
        feature_transform = T.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=n_features,
            melkwargs={
                "n_fft": WIN_SIZE,
                "hop_length": HOP_SIZE,
                "n_mels": N_MELS,
            },
        )
        apply_db = False
    elif feature_type == "mel":
        feature_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=WIN_SIZE,
            hop_length=HOP_SIZE,
            n_mels=n_features,
        )
        apply_db = True
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    # Build full training dataset (no normalisation yet) to compute stats
    full_train_raw = SpeechEmotionDataset(
        audio_dir  = train_dir / "audio",
        labels_csv = train_dir / "train_labels.csv",
        transform  = feature_transform,
        max_frames = max_frames,
        include_deltas = include_deltas,
        apply_db = apply_db,
    )

    # Stratified train / validation split for stable validation signal
    n_total = len(full_train_raw)
    all_indices = np.arange(n_total)
    all_labels = np.array([label for _, label in full_train_raw.samples])
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=val_split,
        random_state=random_seed,
        shuffle=True,
        stratify=all_labels,
    )
    train_indices = train_indices.tolist()
    val_indices = val_indices.tolist()
    n_train = len(train_indices)
    n_val = len(val_indices)

    # Compute normalisation statistics on the training portion only
    train_stats_dataset = Subset(full_train_raw, train_indices)
    mean, std = compute_mean_std(train_stats_dataset)

    # Rebuild train/validation splits with shared mean/std.
    # SpecAugment is applied only on the training split.
    train_dataset = SpeechEmotionDataset(
        audio_dir  = train_dir / "audio",
        labels_csv = train_dir / "train_labels.csv",
        transform  = feature_transform,
        max_frames = max_frames,
        mean       = mean,
        std        = std,
        include_deltas = include_deltas,
        apply_specaugment = apply_specaugment,
        num_time_masks = num_time_masks,
        time_mask_param = time_mask_param,
        num_freq_masks = num_freq_masks,
        freq_mask_param = freq_mask_param,
        apply_db = apply_db,
    )
    val_dataset = SpeechEmotionDataset(
        audio_dir  = train_dir / "audio",
        labels_csv = train_dir / "train_labels.csv",
        transform  = feature_transform,
        max_frames = max_frames,
        mean       = mean,
        std        = std,
        include_deltas = include_deltas,
        apply_db = apply_db,
    )
    train_final = Subset(train_dataset, train_indices)
    val_final = Subset(val_dataset, val_indices)

    test_dataset = SpeechEmotionDataset(
        audio_dir  = test_dir / "audio",
        labels_csv = test_dir / "test_labels.csv",
        transform  = feature_transform,
        max_frames = max_frames,
        mean       = mean,
        std        = std,
        include_deltas = include_deltas,
        apply_db = apply_db,
    )

    train_loader = DataLoader(
        train_final, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True,
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
    n_channels = 3 if include_deltas else 1
    print(
        f"\nFeature shape: ({n_channels}, {n_features}, {max_frames}) | "
        f"type={feature_type}"
    )
    if apply_specaugment:
        print(
            "SpecAugment (train only): "
            f"time_masks={num_time_masks}, time_param={time_mask_param}, "
            f"freq_masks={num_freq_masks}, freq_param={freq_mask_param}"
        )

    return train_loader, val_loader, test_loader, mean, std
