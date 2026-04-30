"""
mamba_minimal.py
================
Minimal pure-PyTorch Mamba block.

This implementation follows the high-level Mamba block structure:
input projection -> depthwise causal conv -> selective scan -> output projection.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaMinimalBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            bias=True,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Per-channel state transition and skip parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.zeros_(self.dt_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.uniform_(self.conv1d.weight, -0.01, 0.01)
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)

    def _selective_scan(self, u: torch.Tensor, delta: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
        # u, delta: (B, L, d_inner)
        # B, C: (B, L, d_state)
        batch_size, seq_len, _ = u.shape
        x_state = torch.zeros(
            batch_size, self.d_inner, self.d_state, device=u.device, dtype=u.dtype
        )
        A = -torch.exp(self.A_log).to(dtype=u.dtype)  # (d_inner, d_state)
        D = self.D.to(dtype=u.dtype)  # (d_inner,)

        ys = []
        for t in range(seq_len):
            u_t = u[:, t, :]               # (B, d_inner)
            dt_t = delta[:, t, :]          # (B, d_inner)
            B_t = B[:, t, :]               # (B, d_state)
            C_t = C[:, t, :]               # (B, d_state)

            deltaA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, d_state)
            deltaB_u = (dt_t.unsqueeze(-1) * B_t.unsqueeze(1)) * u_t.unsqueeze(-1)
            x_state = deltaA * x_state + deltaB_u

            y_t = torch.sum(x_state * C_t.unsqueeze(1), dim=-1) + D.unsqueeze(0) * u_t
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (B, L, d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        xz = self.in_proj(x)
        x_part, z_part = xz.chunk(2, dim=-1)

        # depthwise causal conv
        x_conv = self.conv1d(x_part.transpose(1, 2))[:, :, : x_part.size(1)].transpose(1, 2)
        x_conv = F.silu(x_conv)

        ssm_params = self.x_proj(x_conv)
        dt, B, C = torch.split(ssm_params, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))

        y = self._selective_scan(x_conv, delta, B, C)
        y = y * F.silu(z_part)
        return self.out_proj(y)
