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

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1, **_):
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


class ComposingExpertsBlock(nn.Module):
    """
    Drop-in FFN replacement. Five steps per forward pass:

      1. Modulate  — x_i = h + address_i
                     Each expert adds its own learned address vector.

      2. Route     — routing_scores = softmax-free QK over stacked modulated inputs
                     Captures which experts' addresses are relevant to each other
                     BEFORE the MLPs run. Scores are saved raw (no softmax yet)
                     and cashed in at the pooling step.

      3. Transform — o_i = MLP_i(x_i)
                     N independent MLPs run in parallel.

      4. Compose   — O' = O + CompAttn(LayerNorm(O))
                     Expert outputs attend to each other (binding step).

      5. Pool      — weighted sum of O' using routing_scores
                     Column-sum routing_scores to get one vote-count per expert,
                     softmax to normalise, then weighted sum over expert outputs.
                     Experts that were collectively pointed at by others contribute
                     more to the output token.

    Instrumentation
    ───────────────
      last_attn_weights    (N × N) — composition attention weights, mean over B×T
      last_routing_weights (N,)    — final pooling weights, mean over B×T
      Both updated every forward pass. Check that neither collapses to uniform.
    """

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1, n_experts: int = 4):
        super().__init__()
        self.n_experts = n_experts
        expert_ffn     = ffn_dim // n_experts

        # ── 1. Addresses ──────────────────────────────────────────────────────
        self.addresses = nn.Parameter(torch.zeros(n_experts, d_model))
        nn.init.normal_(self.addresses, std=0.02)

        # ── 2. Routing attention (pre-MLP) ────────────────────────────────────
        # Q and K only — no V, no output projection, no softmax.
        # Raw scores are saved and used in the pooling step.
        self.route_q = nn.Linear(d_model, d_model, bias=False)
        self.route_k = nn.Linear(d_model, d_model, bias=False)

        # ── 3. Expert MLPs ────────────────────────────────────────────────────
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_ffn),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_ffn, d_model),
            )
            for _ in range(n_experts)
        ])

        # ── 4. Composition attention (post-MLP) ───────────────────────────────
        self.comp_norm = nn.LayerNorm(d_model)
        self.comp_q    = nn.Linear(d_model, d_model, bias=False)
        self.comp_k    = nn.Linear(d_model, d_model, bias=False)
        self.comp_v    = nn.Linear(d_model, d_model, bias=False)
        self.comp_out  = nn.Linear(d_model, d_model, bias=False)
        self.dropout   = nn.Dropout(dropout)

        # ── Instrumentation ───────────────────────────────────────────────────
        self.last_attn_weights:    torch.Tensor = None   # (N, N)
        self.last_routing_weights: torch.Tensor = None   # (N,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        N = self.n_experts

        # ── 1. Modulate ───────────────────────────────────────────────────────
        modulated = [x + self.addresses[i] for i in range(N)]  # N × (B, T, D)

        # ── 2. Routing attention ──────────────────────────────────────────────
        # Stack modulated inputs → (B*T, N, D) for the routing QK computation.
        X_mod = torch.stack(modulated, dim=2).view(B * T, N, D)

        Q_r = self.route_q(X_mod)                                          # (B*T, N, D)
        K_r = self.route_k(X_mod)                                          # (B*T, N, D)
        routing_scores = torch.bmm(Q_r, K_r.transpose(1, 2)) * (D ** -0.5)  # (B*T, N, N)
        # No softmax — raw scores are saved and cashed in at pooling.

        # ── 3. Expert MLPs ────────────────────────────────────────────────────
        expert_outs = [self.experts[i](modulated[i]) for i in range(N)]  # N × (B, T, D)

        # ── 4. Composition attention ──────────────────────────────────────────
        O = torch.stack(expert_outs, dim=2).view(B * T, N, D)  # (B*T, N, D)

        normed  = self.comp_norm(O)
        Q       = self.comp_q(normed)
        K       = self.comp_k(normed)
        V       = self.comp_v(normed)
        scores  = torch.bmm(Q, K.transpose(1, 2)) * (D ** -0.5)
        weights = scores.softmax(dim=-1)

        self.last_attn_weights = weights.detach().mean(dim=0)  # (N, N)

        composed = self.comp_out(torch.bmm(weights, V))
        O = O + self.dropout(composed)                         # (B*T, N, D)

        # ── 5. Weighted pool ──────────────────────────────────────────────────
        # Column-sum: for each expert j, total attention pointed *at* j by all others.
        # High value = many experts found j's address-modulated input relevant.
        col_sums        = routing_scores.sum(dim=1)          # (B*T, N)
        routing_weights = col_sums.softmax(dim=-1)           # (B*T, N)

        self.last_routing_weights = routing_weights.detach().mean(dim=0)  # (N,)

        out = torch.einsum('bn,bnd->bd', routing_weights, O)  # (B*T, D)
        return out.view(B, T, D)


class AveragingExpertsBlock(nn.Module):
    """Not yet implemented — same as ComposingExperts but attention weights forced to 1/N."""
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