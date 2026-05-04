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

import hashlib
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as AF
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold


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
                 apply_db=False, normalize_waveform_peak=True,
                 speed_perturb_prob=0.0, speed_perturb_min=0.9, speed_perturb_max=1.1,
                 feature_cache_dir=None, cache_tag=None):
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
        self.speed_perturb_prob = float(speed_perturb_prob)
        self.speed_perturb_min = float(speed_perturb_min)
        self.speed_perturb_max = float(speed_perturb_max)
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        self.cache_tag = str(cache_tag) if cache_tag else "default"
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
        if self.feature_cache_dir is not None:
            self.feature_cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.samples)

    def _cache_path_for(self, clip_id: str) -> Path | None:
        if self.feature_cache_dir is None:
            return None
        # Features with speed perturbation are stochastic and should not be cached.
        if self.speed_perturb_prob > 0.0:
            return None
        cache_key = hashlib.sha1(f"{clip_id}|{self.cache_tag}".encode("utf-8")).hexdigest()
        return self.feature_cache_dir / f"{cache_key}.pt"

    def _build_feature_tensor(self, wav_path: Path) -> torch.Tensor:
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

        # Waveform-domain speed perturbation (train augmentation).
        if (
            self.speed_perturb_prob > 0.0
            and torch.rand(1).item() < self.speed_perturb_prob
            and self.speed_perturb_min > 0.0
            and self.speed_perturb_max >= self.speed_perturb_min
        ):
            rate = torch.empty(1).uniform_(
                self.speed_perturb_min, self.speed_perturb_max
            ).item()
            perturbed_sr = max(1, int(round(SAMPLE_RATE * rate)))
            waveform = AF.resample(waveform, SAMPLE_RATE, perturbed_sr)
            waveform = AF.resample(waveform, perturbed_sr, SAMPLE_RATE)

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
        return spec

    def __getitem__(self, idx):
        clip_id, label = self.samples[idx]
        wav_path = self.audio_dir / f"{clip_id}.wav"
        cache_path = self._cache_path_for(clip_id)
        spec = None
        if cache_path is not None and cache_path.exists():
            spec = torch.load(cache_path, map_location="cpu")
        if spec is None:
            spec = self._build_feature_tensor(wav_path)
            if cache_path is not None:
                torch.save(spec, cache_path)

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
    n_frames_total = 0
    channel_sum = 0.0
    channel_sum_sq = 0.0
    
    for i in range(len(dataset)):
        spec = dataset[i][0]
        channel_sum += spec.sum(dim=-1)
        channel_sum_sq += (spec ** 2).sum(dim=-1)
        n_frames_total += spec.shape[-1]

    mean = (channel_sum / n_frames_total).unsqueeze(-1)
    var = (channel_sum_sq / n_frames_total) - (mean.squeeze(-1) ** 2)
    std = torch.sqrt(torch.clamp(var, min=1e-8)).unsqueeze(-1)
    
    print(f"  Done. Mean range: [{mean.min().item():.2f}, {mean.max().item():.2f}]")
    return mean, std


# ── Main entry point ───────────────────────────────────────────────────────────
def get_dataloaders(data_dir, val_split=0.15, batch_size=64,
                    num_workers=0, max_frames=MAX_FRAMES, random_seed=42,
                    include_deltas=False, apply_specaugment=False,
                    num_time_masks=2, time_mask_param=24,
                    num_freq_masks=2, freq_mask_param=8,
                    feature_type="mel", n_features=None,
                    speed_perturb_prob=0.0, speed_perturb_min=0.9, speed_perturb_max=1.1,
                    feature_cache_dir=None, num_folds=None, fold_index=None):
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

    cache_root = Path(feature_cache_dir) if feature_cache_dir else None
    cache_tag = (
        f"{feature_type}_{n_features}_frames{max_frames}_"
        f"{'deltas' if include_deltas else 'nodeltas'}_"
        f"{'db' if apply_db else 'lin'}"
    )

    # Build full training dataset (no normalisation yet) to compute stats
    full_train_raw = SpeechEmotionDataset(
        audio_dir  = train_dir / "audio",
        labels_csv = train_dir / "train_labels.csv",
        transform  = feature_transform,
        max_frames = max_frames,
        include_deltas = include_deltas,
        apply_db = apply_db,
        feature_cache_dir = cache_root / "train_raw" if cache_root else None,
        cache_tag = cache_tag,
    )

    # Stratified train / validation split for stable validation signal.
    # Optional explicit K-fold mode for robust cross-validation.
    n_total = len(full_train_raw)
    all_indices = np.arange(n_total)
    all_labels = np.array([label for _, label in full_train_raw.samples])
    use_kfold = num_folds is not None and fold_index is not None and int(num_folds) > 1
    if use_kfold:
        num_folds = int(num_folds)
        fold_index = int(fold_index)
        if fold_index < 0 or fold_index >= num_folds:
            raise ValueError(f"fold_index must be in [0, {num_folds - 1}], got {fold_index}")
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
        splits = list(skf.split(all_indices, all_labels))
        train_indices, val_indices = splits[fold_index]
    else:
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
        speed_perturb_prob = speed_perturb_prob,
        speed_perturb_min = speed_perturb_min,
        speed_perturb_max = speed_perturb_max,
        feature_cache_dir = cache_root / "train_aug" if cache_root else None,
        cache_tag = cache_tag,
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
        feature_cache_dir = cache_root / "train_val" if cache_root else None,
        cache_tag = cache_tag,
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
        feature_cache_dir = cache_root / "test" if cache_root else None,
        cache_tag = cache_tag,
    )

    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_final, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_final, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    print(f"\nDataset split:")
    print(f"  Train:      {n_train:>5} clips")
    if use_kfold:
        print(f"  Validation: {n_val:>5} clips  (fold {fold_index + 1}/{num_folds})")
    else:
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
    if speed_perturb_prob > 0.0:
        print(
            "Speed perturbation (train only): "
            f"prob={speed_perturb_prob:.2f}, range=[{speed_perturb_min:.2f}, {speed_perturb_max:.2f}]"
        )
    if cache_root is not None:
        print(f"Feature cache: {cache_root}")

    return train_loader, val_loader, test_loader, mean, std
