"""
compute_mhmp.py — Params + analytic FLOPs for MHMP, and matched-baseline finder.

Why analytic FLOPs (not the torch profiler used in compute_stats.py)?
MHMP is implemented almost entirely in einsum, which torch's with_flops profiler
under-counts. Here every matmul is counted by hand as a function of the knobs,
which is both accurate AND decomposable — you see which term dominates as you
turn M / H / T / N. Params are exact (counted off the real constructed modules).

Convention: one (a×b)·(b×c) matmul = a·b·c MACs = 2·a·b·c FLOPs (matches the
"2 × matmuls" convention in compute_stats.py).

Usage
─────
  python compute_mhmp.py --d_model 128 --n_latents 64 --n_mask_heads 4 --n_loops 6 \
                         --seq_len 64 --ans_len 16
  python compute_mhmp.py ... --match_d_model 128 --match_n_layers 4   # baseline to size
"""

import argparse
import torch

from mhmp import make_mhmp, count_params as count_params_mhmp
from model import make_model, count_params as count_params_tf


def fmt_p(n):
    return f"{n/1e3:.1f} K" if n < 1e6 else f"{n/1e6:.3f} M"

def fmt_f(n):
    if n < 1e6:   return f"{n/1e3:.1f} K"
    if n < 1e9:   return f"{n/1e6:.3f} M"
    return f"{n/1e9:.3f} G"


# ─────────────────────────────────────────────────────────────────────────────
#  Analytic FLOP counts (FLOPs = 2 × MACs), per single example (B=1)
# ─────────────────────────────────────────────────────────────────────────────

def attn_flops(n_q, n_kv, d):
    """Full attention: qkv proj + scores + a·v + out proj."""
    qkv   = (n_q + 2 * n_kv) * d * d        # q from n_q, k&v from n_kv
    score = n_q * n_kv * d
    av    = n_q * n_kv * d
    out   = n_q * d * d
    return 2 * (qkv + score + av + out)

def ffn_flops(n, d, f):
    return 2 * (2 * n * d * f)              # two matmuls

def mhmp_flops(cfg, N, A, vocab):
    """
    N = encoder input length (BOS + src + SEP)
    A = decoder length (SEP + answer)
    Returns dict of component -> FLOPs (single example).
    """
    d, M, H, T = cfg.d_model, cfg.n_latents, cfg.n_mask_heads, cfg.n_loops
    g = M // H
    md, fh, F = cfg.mask_dim, cfg.film_hidden, cfg.ffn_dim
    out = {}

    # ── Encoder ──
    enc = cfg.n_enc_layers * (attn_flops(N, N, d) + ffn_flops(N, d, F))
    out["encoder"] = enc

    # ── Loop (×T) ──
    # FiLM: per-head MLP on H pooled summaries (d->fh->2d)
    film  = H * 2 * (d * fh + fh * 2 * d)
    illum = 2 * (H * N * d)                       # elementwise scale+shift, counted as MACs-ish
    qproj = 2 * (M * d * md)                       # all heads: H·g·d·md = M·d·md
    kproj = 2 * (H * N * d * md)
    vproj = 2 * (H * N * d * d)
    score = 2 * (M * N * md)                       # H·g·N·md
    read  = 2 * (M * N * d)                        # H·g·N·d
    perceive = film + illum + qproj + kproj + vproj + score + read
    reason = attn_flops(M, M, d) + attn_flops(M, M, d) + ffn_flops(M, d, F)  # cross + self + ffn
    loop_one = perceive + reason
    out["loop_perceive"] = T * perceive
    out["loop_reason"]   = T * reason
    out["loop_total"]    = T * loop_one

    # ── Decoder (×Ld) ──
    dec_self  = attn_flops(A, A, d)
    dec_cross = attn_flops(A, M, d)
    dec_ffn   = ffn_flops(A, d, F)
    dec = cfg.n_dec_layers * (dec_self + dec_cross + dec_ffn)
    out["decoder"] = dec

    out["lm_head"] = 2 * (A * d * vocab)
    out["TOTAL"] = enc + out["loop_total"] + dec + out["lm_head"]
    return out


def transformer_flops(d, n_layers, ffn_dim, N, vocab):
    """Decoder-only transformer over full packed length N (single example)."""
    per_layer = attn_flops(N, N, d) + ffn_flops(N, d, ffn_dim)
    return n_layers * per_layer + 2 * (N * d * vocab)


def enc_dec_flops(d, n_enc, n_dec, ffn_dim, N, A, vocab):
    """Plain cross-attn encoder-decoder. N = encoder src length, A = decoder length.
    Decoder cross-attends to the FULL N memory (contrast: MHMP cross-attends to M)."""
    enc = n_enc * (attn_flops(N, N, d) + ffn_flops(N, d, ffn_dim))
    dec = n_dec * (attn_flops(A, A, d)        # causal self
                   + attn_flops(A, N, d)      # cross to full N memory
                   + ffn_flops(A, d, ffn_dim))
    return enc + dec + 2 * (A * d * vocab)


# ─────────────────────────────────────────────────────────────────────────────
#  Baseline matching: size a transformer to MHMP's FLOPs / params
# ─────────────────────────────────────────────────────────────────────────────

def find_flop_match(target_flops, d, n_layers, N, vocab, ffn_range=range(16, 4097, 16)):
    best, best_err = None, float("inf")
    for f in ffn_range:
        fl = transformer_flops(d, n_layers, f, N, vocab)
        err = abs(fl - target_flops)
        if err < best_err:
            best, best_err = f, err
    return best, transformer_flops(d, n_layers, best, N, vocab)

def find_param_match(target_params, d, n_layers, vocab, pad, max_seq_len,
                     ffn_range=range(16, 4097, 16)):
    best, best_err, best_p = None, float("inf"), None
    for f in ffn_range:
        m = make_model(vocab, pad, d_model=d, nhead=max(1, d // 32),
                       n_layers=n_layers, ffn_dim=f, max_seq_len=max_seq_len,
                       block_type="plain_mlp")
        p = count_params_tf(m)
        err = abs(p - target_params)
        if err < best_err:
            best, best_err, best_p = f, err, p
    return best, best_p


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

class C:  # lightweight config holder
    pass

def main(a):
    vocab = a.vocab_size
    cfg = C()
    for k in ("d_model", "n_latents", "n_mask_heads", "n_loops", "mask_dim",
              "film_hidden", "ffn_dim", "n_enc_layers", "n_dec_layers"):
        setattr(cfg, k, getattr(a, k))

    N, A = a.seq_len, a.ans_len

    # ── Build the real model for an exact param count ──
    m = make_mhmp(vocab, pad_idx=0, sep_idx=3,
                  d_model=a.d_model, n_latents=a.n_latents, n_mask_heads=a.n_mask_heads,
                  n_loops=a.n_loops, mask_dim=a.mask_dim, film_hidden=a.film_hidden,
                  reason_nhead=a.reason_nhead, n_enc_layers=a.n_enc_layers,
                  n_dec_layers=a.n_dec_layers, enc_nhead=a.enc_nhead, dec_nhead=a.dec_nhead,
                  ffn_dim=a.ffn_dim, max_seq_len=max(128, N + A))
    mhmp_params = count_params_mhmp(m)
    fl = mhmp_flops(cfg, N, A, vocab)

    print(f"\n{'─'*64}")
    print(f"  MHMP  d={a.d_model}  M={a.n_latents}  H={a.n_mask_heads}  T={a.n_loops}  "
          f"mask_dim={a.mask_dim}")
    print(f"        enc_layers={a.n_enc_layers}  dec_layers={a.n_dec_layers}  ffn={a.ffn_dim}")
    print(f"        N(enc)={N}  A(dec)={A}  vocab={vocab}")
    print(f"{'─'*64}\n")

    print(f"  PARAMS (exact): {fmt_p(mhmp_params)}\n")
    print(f"  FLOPs / forward (analytic, single example):")
    order = ["encoder", "loop_perceive", "loop_reason", "loop_total", "decoder", "lm_head", "TOTAL"]
    for k in order:
        share = fl[k] / fl["TOTAL"] * 100
        indent = "    " if k in ("loop_perceive", "loop_reason") else "  "
        label = k if k not in ("loop_perceive", "loop_reason") else f"↳ {k.split('_')[1]}"
        print(f"  {indent}{label:<16} {fmt_f(fl[k]):>12}   ({share:4.1f}%)")

    # ── The tension, made explicit ──
    d, M, H, T = a.d_model, a.n_latents, a.n_mask_heads, a.n_loops
    print(f"\n  Mask cost scales as H·M·N per loop  (here {H}·{M}·{N}·T={H*M*N*T:,} units).")
    print(f"  It pays off vs an N²-attention baseline only when N is large relative")
    print(f"  to H·M·T.  At N={N} the loop is {fl['loop_total']/fl['TOTAL']*100:.0f}% of FLOPs.")

    # ── Matched baselines ──
    if a.match_d_model:
        md, ml = a.match_d_model, a.match_n_layers
        print(f"\n{'─'*64}")
        print(f"  MATCHED CLASSIC TRANSFORMER  (d={md}, n_layers={ml})\n")

        f_flop, flop_fl = find_flop_match(fl["TOTAL"], md, ml, N + A, vocab)
        m_fl = make_model(vocab, 0, d_model=md, nhead=max(1, md // 32), n_layers=ml,
                          ffn_dim=f_flop, max_seq_len=max(128, N + A), block_type="plain_mlp")
        print(f"  FLOP-matched : ffn_dim={f_flop}")
        print(f"                 FLOPs {fmt_f(flop_fl)}  ({flop_fl/fl['TOTAL']*100:.1f}% of MHMP)")
        print(f"                 params {fmt_p(count_params_tf(m_fl))}  (MHMP: {fmt_p(mhmp_params)})")

        f_par, par_p = find_param_match(mhmp_params, md, ml, vocab, 0, max(128, N + A))
        par_fl = transformer_flops(md, ml, f_par, N + A, vocab)
        print(f"\n  Param-matched: ffn_dim={f_par}")
        print(f"                 params {fmt_p(par_p)}  ({par_p/mhmp_params*100:.1f}% of MHMP)")
        print(f"                 FLOPs  {fmt_f(par_fl)}  ({par_fl/fl['TOTAL']*100:.1f}% of MHMP)")

        print(f"\n  → FLOP-match and param-match diverge: pick FLOP-match as the primary")
        print(f"    comparison (CE's lesson was compute, not params), param-match as footnote.")

        # ── Plain enc-dec control, FLOP-matched (the stage-2 disentangler) ────
        from enc_dec import make_enc_dec, count_params as count_params_ed
        ne, nd = a.match_n_enc, a.match_n_dec
        best_f, best_err = None, float("inf")
        for f in range(16, 4097, 16):
            fl_ed = enc_dec_flops(md, ne, nd, f, N, A, vocab)
            if abs(fl_ed - fl["TOTAL"]) < best_err:
                best_f, best_err = f, abs(fl_ed - fl["TOTAL"])
        ed_fl = enc_dec_flops(md, ne, nd, best_f, N, A, vocab)
        m_ed = make_enc_dec(vocab, 0, 3, d_model=md, n_enc_layers=ne, n_dec_layers=nd,
                            nhead=max(1, md // 32), ffn_dim=best_f, max_seq_len=max(128, N + A))
        print(f"\n  PLAIN ENC-DEC CONTROL  (d={md}, enc={ne}, dec={nd})")
        print(f"  FLOP-matched : ffn_dim={best_f}")
        print(f"                 FLOPs {fmt_f(ed_fl)}  ({ed_fl/fl['TOTAL']*100:.1f}% of MHMP)")
        print(f"                 params {fmt_p(count_params_ed(m_ed))}  (MHMP: {fmt_p(mhmp_params)})")
        print(f"  → isolates 'masked loop helps' from 'enc-dec structure helps'.")

        print(f"\n  Run commands:")
        print(f"    python train_exp2.py --model mhmp --d_model {a.d_model} "
              f"--n_latents {a.n_latents} --n_mask_heads {a.n_mask_heads} --n_loops {a.n_loops} "
              f"--mask_dim {a.mask_dim} --ffn_dim {a.ffn_dim}")
        print(f"    python train_exp2.py --model transformer --d_model {md} "
              f"--n_layers {ml} --ffn_dim_plain {f_flop}    # FLOP-matched")
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_size", type=int, default=535)
    p.add_argument("--seq_len", type=int, default=128, help="encoder input length N")
    p.add_argument("--ans_len", type=int, default=16, help="decoder length A")
    # mhmp knobs
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_latents", type=int, default=64)
    p.add_argument("--n_mask_heads", type=int, default=8)
    p.add_argument("--n_loops", type=int, default=6)
    p.add_argument("--mask_dim", type=int, default=64)
    p.add_argument("--film_hidden", type=int, default=64)
    p.add_argument("--reason_nhead", type=int, default=4)
    p.add_argument("--n_enc_layers", type=int, default=2)
    p.add_argument("--n_dec_layers", type=int, default=2)
    p.add_argument("--enc_nhead", type=int, default=4)
    p.add_argument("--dec_nhead", type=int, default=4)
    p.add_argument("--ffn_dim", type=int, default=128)
    # baseline to match (optional)
    p.add_argument("--match_d_model", type=int, default=128)
    p.add_argument("--match_n_layers", type=int, default=4)
    p.add_argument("--match_n_enc", type=int, default=4, help="enc-dec control: encoder layers")
    p.add_argument("--match_n_dec", type=int, default=4, help="enc-dec control: decoder layers")
    main(p.parse_args())