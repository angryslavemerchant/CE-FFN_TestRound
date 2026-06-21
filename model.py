"""
model.py — Decoder-only transformer with a swappable FFN block.

Architecture
────────────
  Embedding + learned positional embedding
  → N × TransformerLayer (pre-norm, MHA + FFN block)
  → LayerNorm
  → Linear projection to vocab (tied to embedding)

The FFN sublayer in every TransformerLayer is the one swappable slot.
Changing `block_type` in make_model() is the only thing that differs
across the three experimental variants.
"""

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import make_block


class TransformerLayer(nn.Module):
    """
    One transformer layer with pre-norm (more stable for small models).

    Residual structure:
        x = x + Dropout(MHA(LayerNorm(x)))
        x = x + Dropout(block(LayerNorm(x)))
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        block: nn.Module,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.block   = block
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ── Token-mixing attention ──────────────────────────────────────────
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(
            normed, normed, normed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)

        # ── FFN block (the swappable slot) ──────────────────────────────────
        x = x + self.dropout(self.block(self.norm2(x)))

        return x


class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only (causal) language model.

    forward() signature
    ───────────────────
    input_ids        : (B, T) LongTensor — packed [BOS src SEP tgt EOS PAD...] sequence
    loss_mask        : (B, T) FloatTensor — 1.0 at answer positions (tgt + EOS), 0.0 elsewhere
    key_padding_mask : (B, T) BoolTensor  — True where input_ids == PAD (optional; computed if None)

    Returns
    ───────
    logits : (B, T, vocab_size)
    loss   : scalar Tensor if loss_mask provided, else None
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        n_layers: int,
        max_seq_len: int,
        block_type: str,
        ffn_dim: int,
        pad_idx: int,
        dropout: float = 0.1,
        block_kwargs: dict = None,
    ):
        super().__init__()
        self.d_model   = d_model
        self.pad_idx   = pad_idx

        block_kwargs = block_kwargs or {}

        # ── Input representation ────────────────────────────────────────────
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding   = nn.Embedding(max_seq_len, d_model)
        self.drop_in         = nn.Dropout(dropout)

        # ── Transformer stack ───────────────────────────────────────────────
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                nhead=nhead,
                block=make_block(block_type, d_model, ffn_dim, dropout, **block_kwargs),
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # ── Output projection (tied to token embedding) ─────────────────────
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight   # weight tying

        self._init_weights()

    # ── Initialisation ──────────────────────────────────────────────────────

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "norm" in name or "bias" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Small init for embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight,   std=0.02)

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        B, T = input_ids.shape
        device = input_ids.device

        # Padding mask (True = ignore this position)
        if key_padding_mask is None:
            key_padding_mask = (input_ids == self.pad_idx)

        # Embeddings
        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        x = self.drop_in(
            self.token_embedding(input_ids) + self.pos_embedding(positions)
        )

        # Causal mask: bool upper-triangular (True = "ignore this position")
        # Using bool dtype matches key_padding_mask and avoids a PyTorch deprecation warning.
        causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask=causal, key_padding_mask=key_padding_mask)

        x = self.norm(x)
        logits = self.lm_head(x)   # (B, T, V)

        # ── Loss ────────────────────────────────────────────────────────────
        loss = None
        if loss_mask is not None:
            # Shift: logits[:, t] predicts input_ids[:, t+1]
            shift_logits  = logits[:, :-1].contiguous().view(-1, logits.size(-1))
            shift_targets = input_ids[:, 1:].contiguous().view(-1)
            # shift_mask[t] = loss_mask[t+1]  (the target we want to predict)
            shift_mask = loss_mask[:, 1:].contiguous().view(-1)

            per_token = F.cross_entropy(shift_logits, shift_targets, reduction="none")
            loss = (per_token * shift_mask).sum() / shift_mask.sum().clamp(min=1.0)

        return logits, loss

    # ── Greedy decoding ─────────────────────────────────────────────────────

    @torch.no_grad()
    def greedy_decode(
        self,
        src_encoded: list,          # list of List[int] (each = vocab.encode(src_tokens))
        bos_idx: int,
        sep_idx: int,
        eos_idx: int,
        max_gen_len: int = 100,
    ) -> list:
        """
        Batched greedy decoding.

        src_encoded : list of B int-lists (the src token ids, NOT including BOS/SEP)
        Returns     : list of B int-lists — the predicted answer token ids (EOS stripped)

        Strategy: right-pad all prefixes to the same length; at each step take
        the argmax at the last real position for each unfinished sequence.
        """
        self.eval()
        device = next(self.parameters()).device
        B = len(src_encoded)

        # Build prefixes:  BOS + src + SEP
        prefixes = [[bos_idx] + src + [sep_idx] for src in src_encoded]
        prefix_lens = [len(p) for p in prefixes]

        # Working sequences (will grow)
        generated = [p[:] for p in prefixes]
        done      = [False] * B

        for _ in range(max_gen_len):
            # Pad all generated sequences to the same length (right-pad)
            max_len = max(len(g) for g in generated)
            input_ids = torch.full((B, max_len), self.pad_idx, dtype=torch.long, device=device)
            for j, g in enumerate(generated):
                input_ids[j, :len(g)] = torch.tensor(g, dtype=torch.long, device=device)

            kpm = (input_ids == self.pad_idx)
            logits, _ = self(input_ids, key_padding_mask=kpm)

            # Pick next token for each unfinished sequence
            any_new = False
            for j in range(B):
                if done[j]:
                    continue
                last_pos   = len(generated[j]) - 1
                next_token = logits[j, last_pos].argmax().item()
                generated[j].append(next_token)
                if next_token == eos_idx:
                    done[j] = True
                any_new = True

            if not any_new or all(done):
                break

        # Extract answer spans (everything after SEP, before EOS)
        answers = []
        for j, g in enumerate(generated):
            plen   = prefix_lens[j]
            answer = g[plen:]   # after [SEP]
            if eos_idx in answer:
                answer = answer[:answer.index(eos_idx)]
            answers.append(answer)

        return answers


def make_model(
    vocab_size: int,
    pad_idx: int,
    d_model: int     = 128,
    nhead: int       = 4,
    n_layers: int    = 4,
    ffn_dim: int     = 512,
    max_seq_len: int = 128,
    dropout: float   = 0.1,
    block_type: str  = "plain_mlp",
    block_kwargs: dict = None,
) -> DecoderOnlyTransformer:
    """Convenience constructor — all config in one place."""
    return DecoderOnlyTransformer(
        vocab_size   = vocab_size,
        d_model      = d_model,
        nhead        = nhead,
        n_layers     = n_layers,
        max_seq_len  = max_seq_len,
        block_type   = block_type,
        ffn_dim      = ffn_dim,
        pad_idx      = pad_idx,
        dropout      = dropout,
        block_kwargs = block_kwargs or {},
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
