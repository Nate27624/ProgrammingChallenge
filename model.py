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
        c1 = max(cnn_channels // 2, 16)
        c2 = cnn_channels
        c3 = cnn_channels * 2

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
        )

        reduced_features = max(1, n_features // 8)
        lstm_input = c3 * reduced_features

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        attn_dim = hidden_size * 2
        self.attn = _AttentionPool(attn_dim)
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
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for name, param in self.lstm.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        x = self.cnn(x)  # (B, C', F', T')
        bsz, channels, freq_bins, time_steps = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(bsz, time_steps, channels * freq_bins)
        x, _ = self.lstm(x)
        x = self.attn(x)
        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
