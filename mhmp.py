"""
mhmp.py — Multi-Head Masked Perceiver Loop (encoder–decoder).

A latent-space reasoning architecture, sibling to DecoderOnlyTransformer
(NOT a swappable FFN block — it has its own latent array and compresses
the input N -> M, so it cannot live in the BLOCK_REGISTRY slot).

Shape of the model
──────────────────
    ENCODER      embed(src) + pos  -> X_base  (B, N, d), frozen across the loop
    LATENT LOOP  M latents, split into H groups, looped T times:
                   per head h (bound to latent group h):
                     FiLM   : group h -> (gamma, beta) -> illuminate X_base
                     MASK   : group h queries the illuminated view -> (M/H, N) weights
                     READ   : weights gather from values -> (M/H, d)
                   regroup reads -> (M, d)
                   REASON : latents cross-attend to reads, self-attend, FFN
    DECODER      tgt tokens self-attend causally + cross-attend to final
                 latents L_T -> next-token logits

Interface contract (so the shared training loop / evaluate() stay model-agnostic)
─────────────────────────────────────────────────────────────────────────────
    forward(input_ids, loss_mask=None, key_padding_mask=None) -> (logits, loss)
        Takes the SAME packed [BOS] src [SEP] tgt [EOS] batch the transformer
        baseline takes, and splits src/tgt internally at [SEP]. logits are
        returned in the packed layout (B, T, V) so loss_mask lines up exactly
        with the baseline's — identical loss positions, honest comparison.

    greedy_decode(src_encoded, bos_idx, sep_idx, eos_idx, max_gen_len)
        Same signature as DecoderOnlyTransformer.greedy_decode.

Instrumentation (read after a forward pass; analogues of CE's routing logs)
──────────────────────────────────────────────────────────────────────────
    last_mask_entropy      (T, H)  — per-loop, per-head entropy of the masks
    last_mask_frac_zero    (T, H)  — fraction of positions receiving ~zero weight
    last_head_similarity   (T,)    — mean pairwise cosine sim of per-head mask
                                      vectors (collapse monitor; lower is healthier)
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  Small reusable pieces
# ─────────────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    """Standard pre-norm-friendly FFN: Linear -> GELU -> Linear."""
    def __init__(self, d_model: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """Bidirectional (non-causal) self-attention + FFN, pre-norm.

    The encoder reads the WHOLE src — no causal mask. Only padding is masked.
    Keep n_enc_layers small; with 0 layers X_base is just embeddings (the
    pure-Perceiver / sub-quadratic encode), with >0 it contextualises src
    at the cost of an N^2 term (cheap on short PCFG sequences, the thing to
    watch as N grows).
    """
    def __init__(self, d_model, nhead, ffn_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = FFN(d_model, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        n = self.norm1(x)
        a, _ = self.attn(n, n, n, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  The core: one perceive→reason loop iteration, weight-shared across T
# ─────────────────────────────────────────────────────────────────────────────

class PerceiveReason(nn.Module):
    """One iteration of the latent loop. Weight-shared and called T times.

    PERCEIVE (specialised, per-head):
        latent group h drives FiLM over X_base, queries the illuminated view,
        gathers a read. Heads are forced apart structurally — each is driven by
        a different latent group through a different projection.

    REASON (integrated, all latents):
        cross-attention pulls the reads into the latents (every latent sees
        every read), self-attention mixes latents, FFN thinks.
    """
    def __init__(self, d_model, M, H, mask_dim, film_hidden, reason_nhead,
                 ffn_dim, dropout):
        super().__init__()
        assert M % H == 0, f"M={M} must be divisible by H={H}"
        self.d, self.M, self.H = d_model, M, H
        self.g = M // H                      # latents per group
        self.mask_dim = mask_dim

        # ── Per-head FiLM: pooled group summary (d) -> (gamma, beta) (2d) ──────
        # Stacked weights so all H heads run in one einsum.
        self.film_w1 = nn.Parameter(torch.empty(H, d_model, film_hidden))
        self.film_b1 = nn.Parameter(torch.zeros(H, film_hidden))
        self.film_w2 = nn.Parameter(torch.empty(H, film_hidden, 2 * d_model))
        self.film_b2 = nn.Parameter(torch.zeros(H, 2 * d_model))

        # ── Per-head mask projections (query from latents, key from view) ─────
        self.WQ = nn.Parameter(torch.empty(H, d_model, mask_dim))
        self.WK = nn.Parameter(torch.empty(H, d_model, mask_dim))
        self.WV = nn.Parameter(torch.empty(H, d_model, d_model))   # value -> model dim

        # ── Reasoning block ───────────────────────────────────────────────────
        self.cross = nn.MultiheadAttention(d_model, reason_nhead, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, reason_nhead, dropout=dropout, batch_first=True)
        self.ffn = FFN(d_model, ffn_dim, dropout)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_s = nn.LayerNorm(d_model)
        self.norm_f = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        self._init()

        # instrumentation buffers (filled each forward, detached)
        self.last_entropy: Optional[torch.Tensor] = None      # (H,)
        self.last_frac_zero: Optional[torch.Tensor] = None    # (H,)
        self.last_head_similarity: Optional[torch.Tensor] = None  # scalar

    def _init(self):
        for w in (self.film_w1, self.film_w2, self.WQ, self.WK, self.WV):
            for h in range(w.shape[0]):
                nn.init.xavier_uniform_(w[h])
        # FiLM second layer small -> gamma~0, beta~0 at init => near-identity
        # illumination, so the loop starts close to a plain Perceiver read.
        with torch.no_grad():
            self.film_w2.mul_(0.01)

    def forward(self, L, X_base, key_padding_mask=None):
        """
        L          : (B, M, d)   current latents
        X_base     : (B, N, d)   frozen input encoding
        key_padding_mask : (B, N) bool, True = pad position to ignore
        returns updated L : (B, M, d)
        """
        B, M, d = L.shape
        H, g, N = self.H, self.g, X_base.shape[1]

        # group view of latents: (B, H, g, d)
        Lg = L.view(B, H, g, d)

        # ── FiLM: per-group pooled summary -> per-head (gamma, beta) ───────────
        s = Lg.mean(dim=2)                                   # (B, H, d)
        fh = torch.einsum('bhd,hde->bhe', s, self.film_w1) + self.film_b1   # (B,H,film_hidden)
        fh = F.gelu(fh)
        film = torch.einsum('bhe,hef->bhf', fh, self.film_w2) + self.film_b2  # (B,H,2d)
        gamma, beta = film[..., :d], film[..., d:]           # (B,H,d) each

        # ── Illuminate X_base per head: (B,H,N,d) ─────────────────────────────
        # X_h = X_base * (1 + gamma_h) + beta_h   (broadcast over N)
        Xh = X_base.unsqueeze(1) * (1.0 + gamma.unsqueeze(2)) + beta.unsqueeze(2)

        # ── Mask scores: group h latents query head h's illuminated view ──────
        # Q: (B,H,g,mask_dim)  K: (B,H,N,mask_dim)
        Q = torch.einsum('bhgd,hdm->bhgm', Lg, self.WQ)
        K = torch.einsum('bhnd,hdm->bhnm', Xh, self.WK)
        V = torch.einsum('bhnd,hde->bhne', Xh, self.WV)      # (B,H,N,d)
        scores = torch.einsum('bhgm,bhnm->bhgn', Q, K) * (self.mask_dim ** -0.5)  # (B,H,g,N)

        if key_padding_mask is not None:
            # mask out pad positions before softmax
            pad = key_padding_mask.view(B, 1, 1, N)          # (B,1,1,N)
            scores = scores.masked_fill(pad, float('-inf'))

        mask = scores.softmax(dim=-1)                        # (B,H,g,N)  row-stochastic
        mask = torch.nan_to_num(mask)                        # rows that were all-pad -> 0

        # ── Read: gather values -> (B,H,g,d) -> regroup to (B,M,d) ────────────
        reads = torch.einsum('bhgn,bhne->bhge', mask, V)     # (B,H,g,d)
        reads = reads.reshape(B, M, d)                       # aligned with latents

        # ── Reasoning: absorb reads (cross), integrate (self), think (FFN) ────
        Lq = self.norm_q(L)
        Lkv = self.norm_kv(reads)
        c, _ = self.cross(Lq, Lkv, Lkv, need_weights=False)
        L = L + self.drop(c)
        Ls = self.norm_s(L)
        a, _ = self.self_attn(Ls, Ls, Ls, need_weights=False)
        L = L + self.drop(a)
        L = L + self.drop(self.ffn(self.norm_f(L)))

        self._instrument(mask)
        return L

    @torch.no_grad()
    def _instrument(self, mask):
        # mask: (B,H,g,N). Average diagnostics over batch and group.
        B, H, g, N = mask.shape
        p = mask.clamp_min(1e-12)
        ent = -(p * p.log()).sum(-1).mean(dim=(0, 2))        # (H,)
        # mean fraction of near-zero entries per head (sparsity monitor):
        frac_zero = (mask < (1.0 / (10 * N))).float().mean(dim=(0, 2, 3))  # (H,)
        # head similarity: flatten each head's mask to a vector, mean pairwise cosine
        flat = mask.mean(dim=2).reshape(B, H, N)             # (B,H,N) avg over group
        flat = F.normalize(flat, dim=-1)
        sim = torch.einsum('bhn,bkn->bhk', flat, flat)       # (B,H,H)
        eye = torch.eye(H, device=mask.device).bool()
        off = sim.masked_select(~eye.unsqueeze(0)).mean() if H > 1 else torch.tensor(0.0)
        self.last_entropy = ent.detach()
        self.last_frac_zero = frac_zero.detach()
        self.last_head_similarity = off.detach()


# ─────────────────────────────────────────────────────────────────────────────
#  Decoder: causal self-attn over tgt + cross-attn to final latents
# ─────────────────────────────────────────────────────────────────────────────

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, ffn_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = FFN(d_model, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, y, memory, causal_mask, tgt_key_padding_mask=None):
        n = self.norm1(y)
        a, _ = self.self_attn(n, n, n, attn_mask=causal_mask,
                              key_padding_mask=tgt_key_padding_mask, need_weights=False)
        y = y + self.drop(a)
        n = self.norm2(y)
        c, _ = self.cross(n, memory, memory, need_weights=False)
        y = y + self.drop(c)
        y = y + self.drop(self.ffn(self.norm3(y)))
        return y


# ─────────────────────────────────────────────────────────────────────────────
#  Full model
# ─────────────────────────────────────────────────────────────────────────────

class MHMP(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        sep_idx: int,
        d_model: int = 128,
        # latent loop
        n_latents: int = 64,      # M
        n_mask_heads: int = 4,    # H
        n_loops: int = 6,         # T
        mask_dim: int = 64,
        film_hidden: int = 64,
        reason_nhead: int = 4,
        # encoder / decoder depth
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        enc_nhead: int = 4,
        dec_nhead: int = 4,
        ffn_dim: int = 256,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        latent_init: str = "learned",   # "learned" | "input_seeded"
    ):
        super().__init__()
        assert n_latents % n_mask_heads == 0
        self.pad_idx = pad_idx
        self.sep_idx = sep_idx
        self.d_model = d_model
        self.M, self.H, self.T = n_latents, n_mask_heads, n_loops
        self.max_seq_len = max_seq_len
        self.latent_init = latent_init

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.drop_in = nn.Dropout(dropout)

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, enc_nhead, ffn_dim, dropout) for _ in range(n_enc_layers)
        ])

        self.latents = nn.Parameter(torch.empty(n_latents, d_model))
        nn.init.normal_(self.latents, std=0.02)
        if latent_init == "input_seeded":
            self.seed_attn = nn.MultiheadAttention(d_model, reason_nhead, dropout=dropout, batch_first=True)

        self.loop = PerceiveReason(
            d_model, n_latents, n_mask_heads, mask_dim, film_hidden,
            reason_nhead, ffn_dim, dropout,
        )

        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, dec_nhead, ffn_dim, dropout) for _ in range(n_dec_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # tie

        self._init_weights()

        # instrumentation: list over loop iterations, filled each forward
        self.last_loop_entropy: Optional[torch.Tensor] = None       # (T, H)
        self.last_loop_frac_zero: Optional[torch.Tensor] = None     # (T, H)
        self.last_loop_head_sim: Optional[torch.Tensor] = None      # (T,)

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "norm" in name or "bias" in name or "embedding" in name:
                continue
            if "film" in name or name.startswith("loop.W"):
                continue  # handled in PerceiveReason._init
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    # ── Encode src into frozen X_base + run the latent loop -> L_T ────────────
    def _think(self, src_ids):
        """src_ids: (B, Ns) padded. Returns (L_T (B,M,d), src_pad (B,Ns))."""
        B, Ns = src_ids.shape
        device = src_ids.device
        src_pad = (src_ids == self.pad_idx)
        pos = torch.arange(Ns, device=device).unsqueeze(0)
        x = self.drop_in(self.token_embedding(src_ids) + self.pos_embedding(pos))
        for layer in self.enc_layers:
            x = layer(x, key_padding_mask=src_pad)
        X_base = x  # frozen across the loop

        L = self.latents.unsqueeze(0).expand(B, -1, -1).contiguous()
        if self.latent_init == "input_seeded":
            L0, _ = self.seed_attn(L, X_base, X_base, key_padding_mask=src_pad, need_weights=False)
            L = L + L0

        ent, fz, sim = [], [], []
        for _ in range(self.T):
            L = self.loop(L, X_base, key_padding_mask=src_pad)
            ent.append(self.loop.last_entropy)
            fz.append(self.loop.last_frac_zero)
            sim.append(self.loop.last_head_similarity)
        self.last_loop_entropy = torch.stack(ent)       # (T,H)
        self.last_loop_frac_zero = torch.stack(fz)       # (T,H)
        self.last_loop_head_sim = torch.stack(sim)       # (T,)
        return L, src_pad

    # ── Decode tgt given memory L_T ───────────────────────────────────────────
    def _decode(self, tgt_in, memory, mem_pad=None):
        """tgt_in: (B, Nt). Returns logits (B, Nt, V)."""
        B, Nt = tgt_in.shape
        device = tgt_in.device
        tgt_pad = (tgt_in == self.pad_idx)
        pos = torch.arange(Nt, device=device).unsqueeze(0)
        y = self.drop_in(self.token_embedding(tgt_in) + self.pos_embedding(pos))
        causal = torch.triu(torch.ones(Nt, Nt, device=device, dtype=torch.bool), diagonal=1)
        for layer in self.dec_layers:
            y = layer(y, memory, causal_mask=causal, tgt_key_padding_mask=tgt_pad)
        y = self.norm_out(y)
        return self.lm_head(y)

    # ── Split a packed [BOS] src [SEP] tgt [EOS] batch into src / tgt ─────────
    def _split_packed(self, input_ids):
        """
        Returns:
          src_ids   (B, max_src)  = [BOS ... SEP]   (padded)
          tgt_in    (B, max_tgt)  = [SEP t0 ... tm] (decoder input, padded)
          tgt_out   (B, max_tgt)  = [t0 ... tm EOS] (targets, padded)
          tgt_loss  (B, max_tgt)  = 1.0 on real target positions
        The SEP token doubles as the decoder start symbol. Targets are the
        teacher-forcing shift of tgt_in.
        """
        B, T = input_ids.shape
        device = input_ids.device
        is_sep = (input_ids == self.sep_idx)
        # first SEP position per row
        sep_pos = torch.argmax(is_sep.int(), dim=1)          # (B,)
        not_pad = (input_ids != self.pad_idx)
        lengths = not_pad.sum(dim=1)                         # (B,) real length incl EOS

        src_list, tin_list, tout_list = [], [], []
        for b in range(B):
            sp = int(sep_pos[b].item())
            ln = int(lengths[b].item())
            src = input_ids[b, :sp + 1]                      # [BOS ... SEP]
            tin = input_ids[b, sp:ln - 1]                    # [SEP t0 ... t_{m}]  (drop EOS)
            tout = input_ids[b, sp + 1:ln]                   # [t0 ... tm EOS]
            src_list.append(src)
            tin_list.append(tin)
            tout_list.append(tout)

        def pad_stack(seqs, value):
            mx = max(s.numel() for s in seqs)
            out = torch.full((B, mx), value, dtype=torch.long, device=device)
            for i, s in enumerate(seqs):
                out[i, :s.numel()] = s
            return out

        src_ids = pad_stack(src_list, self.pad_idx)
        tgt_in = pad_stack(tin_list, self.pad_idx)
        tgt_out = pad_stack(tout_list, self.pad_idx)
        tgt_loss = (tgt_out != self.pad_idx).float()
        return src_ids, tgt_in, tgt_out, tgt_loss

    # ── Forward: same packed interface as DecoderOnlyTransformer ──────────────
    def forward(self, input_ids, loss_mask=None, key_padding_mask=None):
        src_ids, tgt_in, tgt_out, tgt_loss = self._split_packed(input_ids)
        L_T, src_pad = self._think(src_ids)
        logits = self._decode(tgt_in, L_T)                   # (B, Nt, V)

        loss = None
        if loss_mask is not None:
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_tgt = tgt_out.reshape(-1)
            flat_mask = tgt_loss.reshape(-1)
            per_tok = F.cross_entropy(flat_logits, flat_tgt, reduction="none")
            loss = (per_tok * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)
        return logits, loss

    # ── Greedy decode — same signature as the baseline ────────────────────────
    @torch.no_grad()
    def greedy_decode(self, src_encoded: List[List[int]], bos_idx: int,
                      sep_idx: int, eos_idx: int, max_gen_len: int = 100) -> List[List[int]]:
        self.eval()
        device = next(self.parameters()).device
        B = len(src_encoded)

        # Encoder reads [BOS] src [SEP]; think ONCE.
        src_seqs = [[bos_idx] + s + [sep_idx] for s in src_encoded]
        mx = max(len(s) for s in src_seqs)
        src_ids = torch.full((B, mx), self.pad_idx, dtype=torch.long, device=device)
        for i, s in enumerate(src_seqs):
            src_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        L_T, _ = self._think(src_ids)

        # Decode autoregressively from fixed memory. Start token = SEP.
        gens = [[sep_idx] for _ in range(B)]
        done = [False] * B
        for _ in range(max_gen_len):
            mxt = max(len(g) for g in gens)
            if mxt >= self.max_seq_len:
                break
            tgt_in = torch.full((B, mxt), self.pad_idx, dtype=torch.long, device=device)
            for i, g in enumerate(gens):
                tgt_in[i, :len(g)] = torch.tensor(g, dtype=torch.long, device=device)
            logits = self._decode(tgt_in, L_T)
            any_new = False
            for i in range(B):
                if done[i]:
                    continue
                nxt = logits[i, len(gens[i]) - 1].argmax().item()
                gens[i].append(nxt)
                if nxt == eos_idx:
                    done[i] = True
                any_new = True
            if not any_new or all(done):
                break

        answers = []
        for g in gens:
            a = g[1:]  # drop the leading SEP
            if eos_idx in a:
                a = a[:a.index(eos_idx)]
            answers.append(a)
        return answers


def make_mhmp(vocab_size, pad_idx, sep_idx, **kwargs) -> MHMP:
    return MHMP(vocab_size=vocab_size, pad_idx=pad_idx, sep_idx=sep_idx, **kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
