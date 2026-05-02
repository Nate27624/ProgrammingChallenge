"""
dataloader.py
=============
Dataset + DataLoader for the speech emotion recognition challenge.

Loads audio clips, converts to log-mel spectrograms, stacks delta
and delta-delta features as extra channels, then applies SpecAugment
during training to avoid overfitting.

Output shape per sample: (3, n_mels, max_frames)
  - channel 0: log-mel spectrogram
  - channel 1: delta (first temporal derivative)
  - channel 2: delta-delta (second temporal derivative)
"""

import torch
import soundfile as sf
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

# audio settings
SAMPLE_RATE = 16000
N_MELS = 64
WIN_SIZE_MS = 25
HOP_SIZE_MS = 10
N_FFT = 1024         # larger fft gives more freq bins for the mel filterbank
MAX_DURATION = 5.0  # seconds - clips longer than this get cut

EMOTION_LABELS = {
    "anger":   0,
    "disgust": 1,
    "fear":    2,
    "happy":   3,
    "neutral": 4,
    "sad":     5,
}
IDX_TO_EMOTION = {v: k for k, v in EMOTION_LABELS.items()}

WIN_SIZE = int(SAMPLE_RATE * WIN_SIZE_MS / 1000)  # kept for compat but n_fft overrides
HOP_SIZE = int(SAMPLE_RATE * HOP_SIZE_MS / 1000)
MAX_FRAMES = int(np.ceil(MAX_DURATION * SAMPLE_RATE / HOP_SIZE))


class SpeechEmotionDataset(Dataset):
    def __init__(self, audio_dir, labels_csv, transform,
                 max_frames=MAX_FRAMES, mean=None, std=None, augment=False):
        self.audio_dir = Path(audio_dir)
        self.transform = transform
        self.max_frames = max_frames
        self.mean = mean
        self.std = std
        self.augment = augment

        # specaugment - time and frequency masking
        self.time_mask = T.TimeMasking(time_mask_param=40)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=16)

        df = pd.read_csv(labels_csv)
        df["label"] = df["emotion"].map(EMOTION_LABELS)
        if df["label"].isna().any():
            bad = df[df["label"].isna()]["emotion"].unique().tolist()
            raise ValueError(f"Unknown labels in CSV: {bad}")

        self.samples = list(zip(
            df["clip_id"].tolist(),
            df["label"].astype(int).tolist()
        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_id, label = self.samples[idx]
        wav_path = self.audio_dir / f"{clip_id}.wav"

        # soundfile returns (samples, channels) as numpy, dtype float64
        data, sr = sf.read(str(wav_path), always_2d=True)
        # convert to (channels, samples) float32 tensor
        waveform = torch.tensor(data.T, dtype=torch.float32)

        if sr != SAMPLE_RATE:
            waveform = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)

        # mix down to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # log-mel spectrogram: (1, n_mels, T)
        spec = self.transform(waveform)
        spec = T.AmplitudeToDB(stype="power", top_db=80)(spec)

        # pad/truncate to fixed length
        T_len = spec.shape[-1]
        if T_len < self.max_frames:
            spec = torch.nn.functional.pad(spec, (0, self.max_frames - T_len))
        else:
            spec = spec[..., :self.max_frames]

        # normalize using training stats
        if self.mean is not None and self.std is not None:
            spec = (spec - self.mean) / (self.std + 1e-8)

        # delta and delta-delta features
        delta = torchaudio.functional.compute_deltas(spec)
        delta2 = torchaudio.functional.compute_deltas(delta)

        # stack into 3 channels
        feat = torch.cat([spec, delta, delta2], dim=0)  # (3, n_mels, T)

        # apply specaugment during training
        if self.augment:
            feat = self.freq_mask(feat)
            feat = self.time_mask(feat)

        return feat, label


def compute_norm_stats(dataset, indices):
    """Compute mean and std from training samples (log-mel channel only)."""
    print("Computing normalization stats...")
    specs = []
    for i in indices:
        feat, _ = dataset[i]
        specs.append(feat[0:1])  # just the log-mel channel

    stacked = torch.stack(specs, dim=0)  # (N, 1, n_mels, T)
    mean = stacked.mean(dim=(0, 3), keepdim=True).squeeze(0)  # (1, n_mels, 1)
    std = stacked.std(dim=(0, 3), keepdim=True).squeeze(0)
    print(f"  mean range: [{mean.min():.2f}, {mean.max():.2f}]")
    return mean, std


def get_dataloaders(data_dir, val_split=0.15, batch_size=32,
                    num_workers=0, max_frames=MAX_FRAMES, random_seed=42):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    mel_tfm = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_SIZE,
        n_mels=N_MELS,
        power=2.0,
    )

    # build raw dataset first (no norm, no augment) just to compute stats
    raw_ds = SpeechEmotionDataset(
        audio_dir=train_dir / "audio",
        labels_csv=train_dir / "train_labels.csv",
        transform=mel_tfm,
        max_frames=max_frames,
    )

    # stratified split so all classes are proportionally represented
    all_labels = [lab for _, lab in raw_ds.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split,
                                 random_state=random_seed)
    train_idx, val_idx = next(sss.split(all_labels, all_labels))

    mean, std = compute_norm_stats(raw_ds, train_idx)

    # rebuild datasets with normalization
    train_ds = SpeechEmotionDataset(
        audio_dir=train_dir / "audio",
        labels_csv=train_dir / "train_labels.csv",
        transform=mel_tfm, max_frames=max_frames,
        mean=mean, std=std, augment=True,
    )
    val_ds = SpeechEmotionDataset(
        audio_dir=train_dir / "audio",
        labels_csv=train_dir / "train_labels.csv",
        transform=mel_tfm, max_frames=max_frames,
        mean=mean, std=std, augment=False,
    )
    test_ds = SpeechEmotionDataset(
        audio_dir=test_dir / "audio",
        labels_csv=test_dir / "test_labels.csv",
        transform=mel_tfm, max_frames=max_frames,
        mean=mean, std=std, augment=False,
    )

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"\nSplit sizes:")
    print(f"  train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_ds)}")
    print(f"  feature shape: (3, {N_MELS}, {max_frames})")

    return train_loader, val_loader, test_loader, mean, std