"""
baseline.py
===========
Baseline model for the speech emotion recognition challenge.

Architecture:
  - Input : Mel-spectrogram  (batch, 1, n_mels, max_frames)
  - Reshape to sequence      (batch, max_frames, n_mels)
  - 2-layer LSTM, 128 hidden units
  - Last hidden state        (batch, 128)
  - Fully connected layer    (batch, num_classes)

This baseline is a starting point. You are free to modify the
architecture, features, and training strategy to improve performance.
"""

import torch
import torch.nn as nn
from dataloader import N_MELS, EMOTION_LABELS


class BaselineLSTM(nn.Module):
    """2-layer LSTM baseline for speech emotion recognition.

    Args:
        input_size  : feature dimension per time step (default: n_mels = 64)
        hidden_size : number of LSTM hidden units per layer (default: 128)
        num_layers  : number of stacked LSTM layers (default: 2)
        num_classes : number of emotion classes (default: 6)
        dropout     : dropout between LSTM layers — only applies when
                      num_layers > 1 (default: 0.0)
    """

    def __init__(self,
                 input_size=N_MELS,
                 hidden_size=128,
                 num_layers=2,
                 num_classes=len(EMOTION_LABELS),
                 dropout=0.0):
        super(BaselineLSTM, self).__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialisation for all weight matrices."""
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x):
        """Forward pass.

        Args:
            x : (batch, 1, n_mels, max_frames)

        Returns:
            logits : (batch, num_classes)
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input (batch, channels, n_mels, time), got {x.shape}"
            )

        # (batch, channels, n_mels, time) -> (batch, time, channels * n_mels)
        bsz, channels, n_mels, time_steps = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(
            bsz, time_steps, channels * n_mels
        )
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"LSTM input mismatch: got {x.shape[-1]} features per frame, "
                f"expected {self.input_size}. Check include_deltas setting."
            )

        # LSTM: (batch, max_frames, n_mels) -> (batch, max_frames, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take the last time step: (batch, hidden_size)
        last_out = lstm_out[:, -1, :]

        # Output layer: (batch, num_classes)
        logits = self.fc(last_out)

        return logits

    def count_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNBiLSTMAttentionSER(nn.Module):
    """CNN + BiLSTM + attention classifier used for stronger SER performance."""

    def __init__(
        self,
        input_channels=3,
        n_features=64,
        cnn_channels=64,
        hidden_size=192,
        num_layers=1,
        dropout=0.2,
        num_classes=len(EMOTION_LABELS),
    ):
        super().__init__()
        mid = max(cnn_channels // 2, 16)
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(mid, cnn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True),
        )

        reduced_features = max(n_features // 4, 1)
        lstm_input_size = cnn_channels * reduced_features
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        # x: (B, C, F, T)
        x = self.cnn(x)  # (B, C', F', T')
        bsz, channels, freq_bins, time_steps = x.shape

        # (B, C', F', T') -> (B, T', C' * F')
        x = x.permute(0, 3, 1, 2).contiguous().view(bsz, time_steps, channels * freq_bins)
        x, _ = self.lstm(x)  # (B, T', 2H)

        attn_scores = self.attn(x).squeeze(-1)          # (B, T')
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T')
        context = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (B, 2H)

        return self.head(context)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = BaselineLSTM()
    print(model)
    print(f"\nTrainable parameters: {model.count_parameters():,}")

    from dataloader import MAX_FRAMES
    dummy = torch.randn(8, 1, N_MELS, MAX_FRAMES)
    out   = model(dummy)
    print(f"\nInput shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}  (expected [8, {len(EMOTION_LABELS)}])")
