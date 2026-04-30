"""
model.py
========
CNN + Bidirectional Mamba model for speech emotion recognition.
"""

import torch
import torch.nn as nn
from dataloader import EMOTION_LABELS


def _resolve_mamba_block_factory():
    try:
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
                    "1) pip install mamba-ssm causal-conv1d\n"
                    "2) pip install mambapy"
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
    ):
        super().__init__()
        mamba_factory = _resolve_mamba_block_factory()

        self.frontend = ConvFrontEnd(in_channels=in_channels, out_channels=cnn_channels)
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
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
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

        # Flatten sequence representation via pooled summaries from both directions
        pooled = torch.cat(
            [
                x_fwd.mean(dim=1),
                x_fwd.max(dim=1).values,
                x_bwd.mean(dim=1),
                x_bwd.max(dim=1).values,
            ],
            dim=1,
        )
        return self.classifier(pooled)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
