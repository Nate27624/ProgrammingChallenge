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
        # (batch, 1, n_mels, max_frames) -> (batch, max_frames, n_mels)
        x = x.squeeze(1).transpose(1, 2)

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


if __name__ == "__main__":
    model = BaselineLSTM()
    print(model)
    print(f"\nTrainable parameters: {model.count_parameters():,}")

    from dataloader import MAX_FRAMES
    dummy = torch.randn(8, 1, N_MELS, MAX_FRAMES)
    out   = model(dummy)
    print(f"\nInput shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}  (expected [8, {len(EMOTION_LABELS)}])")