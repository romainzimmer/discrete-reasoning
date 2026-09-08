from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from encoding import GRID_SIZE, NUM_VOCAB, SEQ_LEN


def _ffn_intermediate_dim(width: int) -> int:
    hidden = int(2 * (4 * width) / 3)
    return ((hidden + 7) // 8) * 8


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class StateEncoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.digit_embed = nn.Embedding(NUM_VOCAB, width)
        self.clue_cell_embed = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.clue_cell_embed, std=0.02)

    def encode_input(self, digit_id: torch.Tensor, clue_pin: torch.Tensor) -> torch.Tensor:
        """digit_id, clue_pin: (B, 9, 9) -> (B, 81, D)."""
        b = digit_id.size(0)
        h = self.digit_embed(digit_id.reshape(b, SEQ_LEN))
        return h + clue_pin.reshape(b, SEQ_LEN).unsqueeze(-1).float() * self.clue_cell_embed


class MixerBlock(nn.Module):
    """Pre-norm RMSNorm + token-mix Linear(81, 81) + channel-mix SwiGLU."""

    def __init__(self, seq_len: int, width: int):
        super().__init__()
        hidden = _ffn_intermediate_dim(width)
        self.norm1 = RMSNorm(width)
        self.token_mix = nn.Linear(seq_len, seq_len, bias=False)
        self.norm2 = RMSNorm(width)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm1(x)
        t = self.token_mix(t.transpose(1, 2)).transpose(1, 2)
        x = x + t
        h = self.norm2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class UnembedHead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = RMSNorm(width)
        self.proj = nn.Linear(width, NUM_VOCAB, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


@dataclass
class ModelOutput:
    cell_embed: torch.Tensor
    logits: torch.Tensor


class MixerNextStateModel(nn.Module):
    """Looped MLP-Mixer: h_{t+1} = M(h_t + P), logits from h."""

    def __init__(self, *, width: int = 512, num_blocks: int = 2):
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.encoder = StateEncoder(width)
        self.blocks = nn.ModuleList(MixerBlock(SEQ_LEN, width) for _ in range(num_blocks))
        self.unembed = UnembedHead(width)
        self.width = width
        self.num_blocks = num_blocks

    def encode_input(self, digit_id: torch.Tensor, clue_pin: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_input(digit_id, clue_pin)

    def forward(
        self,
        *,
        input_embed: torch.Tensor,
        cell_embed: torch.Tensor | None = None,
    ) -> ModelOutput:
        b = input_embed.size(0)
        h = cell_embed.reshape(b, SEQ_LEN, self.width) if cell_embed is not None else 0
        x = h + input_embed
        for block in self.blocks:
            x = block(x)
        logits = self.unembed(x).view(b, GRID_SIZE, GRID_SIZE, NUM_VOCAB)
        return ModelOutput(cell_embed=x.view(b, GRID_SIZE, GRID_SIZE, self.width), logits=logits)
