from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ema import ema_combine, validate_ema_alpha
from encoding import GRID_SIZE, NUM_VOCAB, SEQ_LEN

SWIGLU_EXPANSION = 4


def _round_up_multiple(value: int, multiple: int) -> int:
    return (-(value // -multiple)) * multiple


def _swiglu_hidden_dim(dim: int, *, expansion: float = SWIGLU_EXPANSION, multiple: int = 256) -> int:
    """TRM/HRM-style SwiGLU width: round(expansion * dim * 2/3) to a hardware multiple."""
    return _round_up_multiple(round(expansion * dim * 2 / 3), multiple)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class StateEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.digit_embed = nn.Embedding(NUM_VOCAB, dim)
        self.clue_type_embed = nn.Embedding(2, dim)

    def encode_input(self, digit_id: torch.Tensor, clue_pin: torch.Tensor) -> torch.Tensor:
        """digit_id, clue_pin: (B, 9, 9) -> (B, 81, D). clue_pin: 0 = non-clue, 1 = clue."""
        b = digit_id.size(0)
        clue_type = clue_pin.reshape(b, SEQ_LEN).long()
        h = self.digit_embed(digit_id.reshape(b, SEQ_LEN))
        return h + self.clue_type_embed(clue_type)


class MixerBlock(nn.Module):
    """Pre-norm RMSNorm + token-mix Linear(81, 81) + channel-mix SwiGLU."""

    def __init__(self, seq_len: int, dim: int):
        super().__init__()
        hidden = _swiglu_hidden_dim(dim)
        self.norm1 = RMSNorm(dim)
        self.token_mix = nn.Linear(seq_len, seq_len, bias=False)
        self.norm2 = RMSNorm(dim)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm1(x)
        t = self.token_mix(t.transpose(1, 2)).transpose(1, 2)
        x = x + t
        h = self.norm2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class UnembedHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.proj = nn.Linear(dim, NUM_VOCAB, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


@dataclass
class ModelOutput:
    cell_embed: torch.Tensor
    logits: torch.Tensor
    halt_logit: torch.Tensor  # (B,)


class MixerNextStateModel(nn.Module):
    """Looped MLP-Mixer with dual readout: h_{t+1} = LN_h(z), logits = unembed(LN_o(z)), z = M(P + h_t)."""

    def __init__(self, *, dim: int, num_blocks: int):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.encoder = StateEncoder(dim)
        self.blocks = nn.ModuleList(MixerBlock(SEQ_LEN, dim) for _ in range(num_blocks))
        self.norm_memory = RMSNorm(dim)
        self.unembed = UnembedHead(dim)
        self.halt_head = nn.Linear(dim, 1, bias=True)
        self.dim = dim

    def encode_input(self, digit_id: torch.Tensor, clue_pin: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_input(digit_id, clue_pin)

    def forward(
        self,
        *,
        input_embed: torch.Tensor,
        cell_embed: torch.Tensor | None = None,
        ema_embed: torch.Tensor | None,
        ema_alpha: float,
    ) -> ModelOutput:
        b = input_embed.size(0)
        h = cell_embed.reshape(b, SEQ_LEN, self.dim) if cell_embed is not None else 0
        if ema_alpha >= 1.0:
            state = h
        else:
            validate_ema_alpha(ema_alpha)
            if ema_embed is None:
                raise ValueError("ema_embed is required when ema_alpha < 1")
            ema = ema_embed.reshape(b, SEQ_LEN, self.dim)
            state = ema_combine(h, ema, ema_alpha)
        z = input_embed + state
        for block in self.blocks:
            z = block(z)
        memory = self.norm_memory(z)
        cell_embed = memory.view(b, GRID_SIZE, GRID_SIZE, self.dim)
        logits = self.unembed(z).view(b, GRID_SIZE, GRID_SIZE, NUM_VOCAB)
        halt_logit = self.halt_head(z.mean(dim=1)).squeeze(-1)
        return ModelOutput(cell_embed=cell_embed, logits=logits, halt_logit=halt_logit)
