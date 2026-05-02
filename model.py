"""
model.py
========
CNN + BiLSTM + attention model for speech emotion recognition.

The CNN front-end pulls local spectro-temporal features from the
3-channel input (log-mel + delta + delta-delta). The BiLSTM then
models the temporal sequence, and attention pools over time so the
model can focus on the frames that matter most for emotion.

Rough parameter count: ~3.5M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import N_MELS, EMOTION_LABELS


class ConvBlock(nn.Module):
    """basic conv block: conv -> bn -> relu -> pool -> dropout"""
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """simple additive attention over the time axis"""
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (B, T, D) -> weights: (B, T, 1)
        w = torch.softmax(self.fc(x), dim=1)
        return (w * x).sum(dim=1)


class SERModel(nn.Module):
    def __init__(self, n_mels=N_MELS, num_classes=len(EMOTION_LABELS), dropout=0.3):
        super().__init__()

        # 3 conv blocks; each halves the freq and time dims
        self.cnn = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )

        # after 3 maxpool(2) layers: freq_dim = n_mels // 8
        freq_dim = n_mels // 8
        lstm_in = 128 * freq_dim

        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.attn = Attention(512)  # 256 * 2 directions

        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # orthogonal init for LSTM recurrent weights
        for name, p in self.lstm.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, x):
        # x: (B, 3, n_mels, T)
        x = self.cnn(x)  # (B, 128, n_mels//8, T//8)

        B, C, F, T = x.shape
        # flatten freq into channels and transpose to (B, T, C*F)
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)

        x, _ = self.lstm(x)   # (B, T, 512)
        x = self.attn(x)       # (B, 512)
        return self.head(x)    # (B, num_classes)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    from dataloader import MAX_FRAMES
    m = SERModel()
    print(m)
    print(f'params: {m.count_parameters():,}')
    x = torch.randn(4, 3, N_MELS, MAX_FRAMES)
    print(f'output: {m(x).shape}')
