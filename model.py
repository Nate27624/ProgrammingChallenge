"""
model.py
========
SER model definitions currently in active use.

The maintained architecture is the CNN + BiLSTM + attention classifier used by
the final training pipeline.
"""

import torch
import torch.nn as nn

from dataloader import EMOTION_LABELS


class _AttentionPool(nn.Module):
    """Additive attention pooling over temporal steps."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        weights = torch.softmax(self.fc(x), dim=1)
        return (weights * x).sum(dim=1)


class CNNBiLSTMAttentionSER(nn.Module):
    """
    CNN + BiLSTM + attention model for SER.

    This mirrors the "emotion peaks matter" hypothesis by learning frame-level
    importance weights after temporal modeling.

    High-level data flow:
      (B, C, F, T)
        -> CNN front-end (local time-frequency pattern extraction + downsampling)
      (B, C', F', T')
        -> reshape to sequence
      (B, T', C'*F')
        -> BiLSTM (bidirectional temporal modeling)
      (B, T', 2H)
        -> attention pooling (learn which frames matter most)
      (B, 2H)
        -> classification head
      (B, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_features: int = 40,
        num_classes: int = len(EMOTION_LABELS),
        cnn_channels: int = 64,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        # CNN channel plan:
        # - c1 starts moderately wide to avoid early over-parameterization.
        # - c2 is the configured backbone width.
        # - c3 expands capacity before sequence modeling.
        # Using progressive widening improves feature richness without a huge
        # first-layer compute spike.
        c1 = max(cnn_channels // 2, 16)
        c2 = cnn_channels
        c3 = cnn_channels * 2

        # CNN block stack:
        # Each stage does Conv -> BN -> SiLU -> Pool -> Dropout2d.
        # Pooling halves both frequency and time dimensions at each stage.
        # After 3 poolings, feature/time axes are each reduced by ~8x.
        self.cnn = nn.Sequential(
            # Stage 1: learn low-level spectral edges/formants/prosody cues.
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
            # Stage 2: learn higher-level local patterns.
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
            # Stage 3: final compact representation before recurrent modeling.
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
        )

        # Frequency bins after 3x MaxPool2d(2) ~ n_features // 8.
        # max(1, ...) keeps the model valid for very small feature counts.
        reduced_features = max(1, n_features // 8)
        # Every time step fed into LSTM contains all channels at one frame.
        lstm_input = c3 * reduced_features

        # BiLSTM details:
        # - bidirectional=True captures context from past and future frames.
        # - batch_first=True keeps tensor layout intuitive: (B, T, D).
        # - LSTM dropout is only active when num_layers > 1 (PyTorch behavior).
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # BiLSTM outputs forward+backward states, so feature dim is 2*hidden_size.
        attn_dim = hidden_size * 2
        # Attention pooling learns a per-frame importance weight.
        # This is useful in SER because emotional salience is often concentrated
        # in specific regions rather than uniformly distributed.
        self.attn = _AttentionPool(attn_dim)
        # Classification head:
        # LayerNorm stabilizes optimization across different sequence statistics.
        # Two dropout points improve regularization on this relatively small
        # dataset regime.
        self.head = nn.Sequential(
            nn.LayerNorm(attn_dim),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        # Convolution layers use Kaiming init for SiLU/ReLU-like activations.
        # Linear layers use Xavier for stable variance through dense projections.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # LSTM-specific init:
        # - recurrent weights orthogonal for better long-sequence stability
        # - input weights Xavier for balanced signal scaling
        # - biases zeroed for neutral start
        for name, param in self.lstm.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input spectrogram tensor:
        #   B = batch, C = channels (MFCC + optional deltas), F = frequency bins, T = frames
        # Shape: (B, C, F, T)
        x = self.cnn(x)
        # After CNN/downsampling: (B, C', F', T')
        bsz, channels, freq_bins, time_steps = x.shape
        # Convert image-like map to sequence for recurrent modeling:
        #   (B, C', F', T') -> (B, T', C'*F')
        x = x.permute(0, 3, 1, 2).contiguous().view(bsz, time_steps, channels * freq_bins)
        # Temporal modeling over downsampled frame sequence.
        x, _ = self.lstm(x)
        # Attention-weighted aggregation across frames.
        x = self.attn(x)
        # Final emotion logits.
        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
