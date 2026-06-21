"""
blocks.py — FFN block classes that plug into each transformer layer.

Interface contract: every block is an nn.Module whose forward() takes
    x : Tensor of shape (B, T, d_model)
and returns a Tensor of the same shape.  The surrounding layer handles
the residual add and the pre-norm.

Current blocks
──────────────
  PlainMLP          — standard FFN, the baseline to beat.

Coming soon (same interface, same slot)
──────────────────────────────────────
  ComposingExpertsBlock   — modulate → N MLPs → composition attention → pool
  AveragingExpertsBlock   — same structure but attention weights forced to 1/N
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PlainMLP(nn.Module):
    """
    Standard FFN sublayer: Linear → GELU → Linear.
    ffn_dim is the hidden (expansion) dimension; typically 4 × d_model.
    """

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1     = nn.Linear(d_model, ffn_dim)
        self.fc2     = nn.Linear(ffn_dim,  d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))

    # ── utility used by the FLOP-matching logic later ────────────────────────
    def flops_per_token(self, d_model: int) -> int:
        """
        Approximate FLOPs for one token (2 × matmuls, ignoring non-linearities).
        Used when param/FLOP-matching the experts blocks.
        """
        return 2 * (d_model * self.fc1.out_features + self.fc1.out_features * d_model)


# ── Placeholder stubs so imports don't break when we add them ─────────────────

class ComposingExpertsBlock(nn.Module):
    """Not yet implemented — will replace the FFN with N-expert binding."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("ComposingExpertsBlock — coming soon")


class AveragingExpertsBlock(nn.Module):
    """Not yet implemented — same as ComposingExperts but uniform attention (1/N)."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("AveragingExpertsBlock — coming soon")


# ── Registry for clean config-driven construction ────────────────────────────

BLOCK_REGISTRY = {
    "plain_mlp":          PlainMLP,
    "composing_experts":  ComposingExpertsBlock,
    "averaging_experts":  AveragingExpertsBlock,
}


def make_block(block_type: str, d_model: int, ffn_dim: int, dropout: float, **kwargs) -> nn.Module:
    """
    Construct a block by name.  Extra kwargs are forwarded to the constructor
    (e.g. n_experts for the expert blocks).
    """
    cls = BLOCK_REGISTRY[block_type]
    return cls(d_model=d_model, ffn_dim=ffn_dim, dropout=dropout, **kwargs)
