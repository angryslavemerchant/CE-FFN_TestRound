"""
enc_dec.py — Plain cross-attention encoder–decoder (Vaswani seq2seq).

This is the STAGE-2 CONTROL for the MHMP experiment. It exists to answer one
question: if MHMP beats the decoder-only transformer, is that because of the
masked perceive/reason loop, or merely because an encoder–decoder structure
suits PCFG better than decoder-only?

It is deliberately identical to MHMP everywhere EXCEPT the one variable under test:

    MHMP decoder cross-attends to  M looped, compressed latents  (the mechanism)
    this  decoder cross-attends to  the full N encoder memory     (no loop, no compression)

Same packed (input_ids, loss_mask) interface, same src/tgt split at [SEP], same
greedy_decode signature, and it REUSES MHMP's EncoderLayer / DecoderLayer / FFN
so the building blocks are byte-identical — any difference is the memory the
decoder reads from, nothing else.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from mhmp import EncoderLayer, DecoderLayer  # identical layer implementations


class EncDecTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        sep_idx: int,
        d_model: int = 128,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        nhead: int = 4,
        ffn_dim: int = 512,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.sep_idx = sep_idx
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.drop_in = nn.Dropout(dropout)

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, ffn_dim, dropout) for _ in range(n_enc_layers)
        ])
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, ffn_dim, dropout) for _ in range(n_dec_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # tie

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "norm" in name or "bias" in name or "embedding" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    # ── Encode src into full N memory (bidirectional, NOT compressed, NO loop) ─
    def _encode(self, src_ids):
        B, Ns = src_ids.shape
        device = src_ids.device
        src_pad = (src_ids == self.pad_idx)
        pos = torch.arange(Ns, device=device).unsqueeze(0)
        x = self.drop_in(self.token_embedding(src_ids) + self.pos_embedding(pos))
        for layer in self.enc_layers:
            x = layer(x, key_padding_mask=src_pad)
        return x, src_pad                                    # memory (B, Ns, d), pad mask

    # ── Decode tgt, cross-attending to the full encoder memory ────────────────
    def _decode(self, tgt_in, memory, mem_pad):
        B, Nt = tgt_in.shape
        device = tgt_in.device
        tgt_pad = (tgt_in == self.pad_idx)
        pos = torch.arange(Nt, device=device).unsqueeze(0)
        y = self.drop_in(self.token_embedding(tgt_in) + self.pos_embedding(pos))
        causal = torch.triu(torch.ones(Nt, Nt, device=device, dtype=torch.bool), diagonal=1)
        for layer in self.dec_layers:
            y = layer(y, memory, causal_mask=causal,
                      tgt_key_padding_mask=tgt_pad, memory_key_padding_mask=mem_pad)
        return self.lm_head(self.norm_out(y))

    # ── Same src/tgt split as MHMP (SEP doubles as decoder-start token) ───────
    def _split_packed(self, input_ids):
        B, T = input_ids.shape
        device = input_ids.device
        sep_pos = torch.argmax((input_ids == self.sep_idx).int(), dim=1)
        lengths = (input_ids != self.pad_idx).sum(dim=1)
        src_l, tin_l, tout_l = [], [], []
        for b in range(B):
            sp, ln = int(sep_pos[b]), int(lengths[b])
            src_l.append(input_ids[b, :sp + 1])              # [BOS ... SEP]
            tin_l.append(input_ids[b, sp:ln - 1])            # [SEP t0 ... tm]
            tout_l.append(input_ids[b, sp + 1:ln])           # [t0 ... tm EOS]

        def pad_stack(seqs):
            mx = max(s.numel() for s in seqs)
            out = torch.full((B, mx), self.pad_idx, dtype=torch.long, device=device)
            for i, s in enumerate(seqs):
                out[i, :s.numel()] = s
            return out

        src_ids = pad_stack(src_l)
        tgt_in = pad_stack(tin_l)
        tgt_out = pad_stack(tout_l)
        tgt_loss = (tgt_out != self.pad_idx).float()
        return src_ids, tgt_in, tgt_out, tgt_loss

    def forward(self, input_ids, loss_mask=None, key_padding_mask=None):
        src_ids, tgt_in, tgt_out, tgt_loss = self._split_packed(input_ids)
        memory, mem_pad = self._encode(src_ids)
        logits = self._decode(tgt_in, memory, mem_pad)

        loss = None
        if loss_mask is not None:
            per_tok = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                      tgt_out.reshape(-1), reduction="none")
            m = tgt_loss.reshape(-1)
            loss = (per_tok * m).sum() / m.sum().clamp(min=1.0)
        return logits, loss

    @torch.no_grad()
    def greedy_decode(self, src_encoded: List[List[int]], bos_idx: int,
                      sep_idx: int, eos_idx: int, max_gen_len: int = 100) -> List[List[int]]:
        self.eval()
        device = next(self.parameters()).device
        B = len(src_encoded)
        src_seqs = [[bos_idx] + s + [sep_idx] for s in src_encoded]
        mx = max(len(s) for s in src_seqs)
        src_ids = torch.full((B, mx), self.pad_idx, dtype=torch.long, device=device)
        for i, s in enumerate(src_seqs):
            src_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        memory, mem_pad = self._encode(src_ids)             # encode once

        gens = [[sep_idx] for _ in range(B)]
        done = [False] * B
        for _ in range(max_gen_len):
            mxt = max(len(g) for g in gens)
            if mxt >= self.max_seq_len:
                break
            tgt_in = torch.full((B, mxt), self.pad_idx, dtype=torch.long, device=device)
            for i, g in enumerate(gens):
                tgt_in[i, :len(g)] = torch.tensor(g, dtype=torch.long, device=device)
            logits = self._decode(tgt_in, memory, mem_pad)
            for i in range(B):
                if done[i]:
                    continue
                nxt = logits[i, len(gens[i]) - 1].argmax().item()
                gens[i].append(nxt)
                if nxt == eos_idx:
                    done[i] = True
            if all(done):
                break

        answers = []
        for g in gens:
            a = g[1:]
            if eos_idx in a:
                a = a[:a.index(eos_idx)]
            answers.append(a)
        return answers


def make_enc_dec(vocab_size, pad_idx, sep_idx, **kwargs) -> EncDecTransformer:
    return EncDecTransformer(vocab_size=vocab_size, pad_idx=pad_idx, sep_idx=sep_idx, **kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
