"""
model.py
========
SER model zoo: baseline-ready Mamba/CNN-BiLSTM variants.
"""

import torch
import torch.nn as nn
from dataloader import EMOTION_LABELS


def _resolve_mamba_block_factory():
    """Return a callable that builds one Mamba block from available backends.

    Resolution order:
    1) local minimal implementation (`mamba_minimal.py`)
    2) official `mamba_ssm`
    3) `mambapy` fallback
    """
    try:
        from mamba_minimal import MambaMinimalBlock  # type: ignore

        return lambda d_model, d_state, d_conv, expand: MambaMinimalBlock(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
    except ImportError:
        pass

    try:
        # Preferred package backend when installed.
        from mamba_ssm import Mamba  # type: ignore
        return lambda d_model, d_state, d_conv, expand: Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
    except ImportError:
        try:
            from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore
            return lambda d_model, d_state, d_conv, expand: Mamba(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )
        except ImportError:
            try:
                from mambapy.mamba import Mamba, MambaConfig  # type: ignore

                def _factory(d_model, d_state, d_conv, expand):
                    cfg = MambaConfig(
                        d_model=d_model,
                        n_layers=1,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand_factor=expand,
                        use_cuda=False,
                    )
                    return Mamba(cfg)

                return _factory
            except ImportError as exc:
                raise ImportError(
                    "No compatible Mamba backend found. Install one of:\n"
                    "1) local mamba_minimal.py in this repository\n"
                    "2) pip install mamba-ssm causal-conv1d\n"
                    "3) pip install mambapy"
                ) from exc


class ConvFrontEnd(nn.Module):
    """2D dilated CNN front-end for local time-frequency extraction."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_channels = max(out_channels // 2, 16)
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=(1, 2),
                dilation=(1, 2),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=(1, 4),
                dilation=(1, 4),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualConvFrontEnd(nn.Module):
    """Residual 2D CNN front-end with the same downsample factor as ConvFrontEnd."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_channels = max(out_channels // 2, 16)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.stem_skip = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.stem_act = nn.SiLU()

        self.block1 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=(1, 2), dilation=(1, 2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.block1_skip = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.block1_act = nn.SiLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=(1, 4), dilation=(1, 4), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.block2_act = nn.SiLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x) + self.stem_skip(x)
        x = self.stem_act(x)

        x = self.block1(x) + self.block1_skip(x)
        x = self.block1_act(x)
        x = self.pool1(x)

        x = self.block2(x) + x
        x = self.block2_act(x)
        x = self.pool2(x)
        return x


class BidirectionalMambaSER(nn.Module):
    """CNN + bidirectional Mamba encoder for 6-class emotion classification."""

    def __init__(
        self,
        in_channels: int = 3,
        n_features: int = 40,
        num_classes: int = len(EMOTION_LABELS),
        cnn_channels: int = 64,
        d_model: int = 128,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
        frontend_type: str = "basic_cnn",
        fusion_type: str = "concat",
        pooling_type: str = "meanmax",
    ):
        super().__init__()
        mamba_factory = _resolve_mamba_block_factory()
        self.frontend_type = frontend_type
        self.fusion_type = fusion_type
        self.pooling_type = pooling_type

        if frontend_type == "basic_cnn":
            self.frontend = ConvFrontEnd(in_channels=in_channels, out_channels=cnn_channels)
        elif frontend_type == "residual_cnn":
            self.frontend = ResidualConvFrontEnd(in_channels=in_channels, out_channels=cnn_channels)
        else:
            raise ValueError(f"Unsupported frontend_type: {frontend_type}")

        reduced_features = n_features // 4
        if reduced_features < 1:
            raise ValueError(f"n_features must be >= 4, got {n_features}")
        self.proj = nn.Linear(cnn_channels * reduced_features, d_model)

        self.fwd_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.bwd_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.fwd_blocks = nn.ModuleList(
            [mamba_factory(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_layers)]
        )
        self.bwd_blocks = nn.ModuleList(
            [mamba_factory(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)
        if fusion_type == "gated":
            self.fusion_gate = nn.Linear(d_model * 2, d_model)
            fused_dim = d_model
        elif fusion_type == "concat":
            self.fusion_gate = None
            fused_dim = d_model * 2
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        if pooling_type == "attention":
            self.attn_pool = nn.Linear(fused_dim, 1)
            pooled_dim = fused_dim
        elif pooling_type == "meanmax":
            self.attn_pool = None
            pooled_dim = fused_dim * 2
        else:
            raise ValueError(f"Unsupported pooling_type: {pooling_type}")

        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _run_directional_stack(
        self,
        x: torch.Tensor,
        blocks: nn.ModuleList,
        norms: nn.ModuleList,
        reverse: bool,
    ) -> torch.Tensor:
        if reverse:
            x = torch.flip(x, dims=[1])
        for block, norm in zip(blocks, norms):
            x = x + self.dropout(block(norm(x)))
        if reverse:
            x = torch.flip(x, dims=[1])
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        x = self.frontend(x)  # (B, C', F', T')
        bsz, channels, freq_bins, time_steps = x.shape

        # Convert feature map to sequence: (B, T', C' * F')
        x = x.permute(0, 3, 1, 2).contiguous().view(bsz, time_steps, channels * freq_bins)
        x = self.proj(x)

        x_fwd = self._run_directional_stack(
            x, self.fwd_blocks, self.fwd_norms, reverse=False
        )
        x_bwd = self._run_directional_stack(
            x, self.bwd_blocks, self.bwd_norms, reverse=True
        )

        if self.fusion_type == "gated":
            gate = torch.sigmoid(self.fusion_gate(torch.cat([x_fwd, x_bwd], dim=-1)))
            seq = gate * x_fwd + (1.0 - gate) * x_bwd
        else:
            seq = torch.cat([x_fwd, x_bwd], dim=-1)

        if self.pooling_type == "attention":
            attn_logits = self.attn_pool(seq).squeeze(-1)  # (B, T)
            attn = torch.softmax(attn_logits, dim=1).unsqueeze(-1)  # (B, T, 1)
            pooled = (seq * attn).sum(dim=1)
        else:
            pooled = torch.cat([seq.mean(dim=1), seq.max(dim=1).values], dim=1)

        return self.classifier(pooled)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
