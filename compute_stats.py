"""
compute_stats.py — FLOPs and parameters for each block, plus matched plain ffn_dim.

Usage
─────
python compute_stats.py                                     # defaults
python compute_stats.py --d_model 256 --n_experts 4
python compute_stats.py --d_model 128 --n_experts 8
"""

import argparse
import sys
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, ".")
from blocks import PlainMLP, ComposingExpertsBlock
from model import make_model, count_params


# ── Helpers ───────────────────────────────────────────────────────────────────

def measure_flops(module: nn.Module, *input_tensors) -> int:
    module.eval()
    with profile(activities=[ProfilerActivity.CPU], with_flops=True, record_shapes=True) as prof:
        with torch.no_grad():
            module(*input_tensors)
    return sum(e.flops for e in prof.key_averages() if e.flops)

def fmt_p(n):
    return f"{n/1e3:.1f} K" if n < 1e6 else f"{n/1e6:.3f} M"

def fmt_f(n):
    return f"{n/1e3:.1f} K" if n < 1e6 else f"{n/1e6:.3f} M"


# ── Main ──────────────────────────────────────────────────────────────────────

def main(d_model, ffn_dim, n_experts, n_layers, seq_len):
    B, T = 1, seq_len
    x = torch.randn(B, T, d_model)

    print(f"\n{'─'*62}")
    print(f"  d_model={d_model}  ffn_dim_experts={ffn_dim}  n_experts={n_experts}  n_layers={n_layers}")
    print(f"{'─'*62}\n")

    # ── Block-level stats ─────────────────────────────────────────────────────
    plain   = PlainMLP(d_model, ffn_dim)
    experts = ComposingExpertsBlock(d_model, ffn_dim, n_experts=n_experts)

    plain_p   = count_params(plain)
    experts_p = count_params(experts)
    plain_f   = measure_flops(plain,   x)
    experts_f = measure_flops(experts, x)

    plain_fpt   = plain_f   // T
    experts_fpt = experts_f // T

    print(f"  {'BLOCK':<32}  {'PARAMS':>10}  {'FLOPs/token':>14}")
    print(f"  {'─'*32}  {'─'*10}  {'─'*14}")
    print(f"  {'PlainMLP (ffn_dim=' + str(ffn_dim) + ')':<32}  {fmt_p(plain_p):>10}  {fmt_f(plain_fpt):>14}")
    print(f"  {'ComposingExperts (N=' + str(n_experts) + ')':<32}  {fmt_p(experts_p):>10}  {fmt_f(experts_fpt):>14}  ← {experts_fpt/plain_fpt:.1f}× plain")

    # ── ComposingExperts param breakdown ──────────────────────────────────────
    print(f"\n  ComposingExperts breakdown:")
    for name, child in experts.named_children():
        p = sum(x.numel() for x in child.parameters())
        if p > 0:
            print(f"    {name:<18}  {fmt_p(p):>8}  ({p/experts_p*100:.0f}%)")
    direct = sum(p.numel() for n, p in experts.named_parameters() if "." not in n)
    if direct:
        print(f"    {'addresses':<18}  {fmt_p(direct):>8}  ({direct/experts_p*100:.0f}%)")

    # ── Matched plain ffn_dim ─────────────────────────────────────────────────
    # PlainMLP FLOPs/token = 4 × d_model × ffn_dim_plain
    # Solve for ffn_dim_plain such that FLOPs match CE:
    # PlainMLP FLOPs/token = 4 × d_model × ffn_dim_plain  →  ffn_dim_plain = FLOPs / (4 × d_model)
    matched_ffn_raw = experts_fpt // (4 * d_model)
    matched_ffn     = (matched_ffn_raw // 4) * 4   # round down to nearest multiple of 4
    matched_plain = PlainMLP(d_model, matched_ffn)
    matched_f     = measure_flops(matched_plain, x)
    matched_fpt   = matched_f // T
    matched_p     = count_params(matched_plain)

    print(f"\n{'─'*62}")
    print(f"  MATCHED PLAIN BASELINE\n")
    print(f"  CE FLOPs/token     : {fmt_f(experts_fpt)}")
    print(f"  Matched ffn_dim    : {matched_ffn}   (plain ffn_dim to use)")
    print(f"  Verification       : {fmt_f(matched_fpt)} FLOPs/token  ({matched_fpt/experts_fpt*100:.1f}% of CE)")
    print(f"  Matched params     : {fmt_p(matched_p)}  (CE: {fmt_p(experts_p)})")

    # ── Full model stats ───────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  FULL MODEL  ({n_layers} layers, vocab=535)\n")

    ids = torch.randint(1, 535, (B, T))
    configs = [
        ("plain_mlp",         ffn_dim,      {},                       f"PlainMLP  ffn={ffn_dim}"),
        ("plain_mlp",         matched_ffn,  {},                       f"PlainMLP  ffn={matched_ffn}  (matched)"),
        ("composing_experts", ffn_dim,      {"n_experts": n_experts}, f"ComposingExperts  N={n_experts}  ffn={ffn_dim}"),
    ]
    nheads = max(1, d_model // 32)

    print(f"  {'MODEL':<42}  {'PARAMS':>10}  {'FLOPs/token':>14}")
    print(f"  {'─'*42}  {'─'*10}  {'─'*14}")
    for btype, fd, bkw, label in configs:
        m = make_model(535, 0, d_model=d_model, nhead=nheads,
                       n_layers=n_layers, ffn_dim=fd,
                       block_type=btype, block_kwargs=bkw)
        f = measure_flops(m, ids)
        p = count_params(m)
        print(f"  {label:<42}  {fmt_p(p):>10}  {fmt_f(f//T):>14}")

    print(f"\n  Run commands:")
    print(f"    python train.py --block composing_experts --n_experts {n_experts} --ffn_dim_experts {ffn_dim}")
    print(f"    python train.py --block plain_mlp --ffn_dim_plain {matched_ffn}    # compute-matched")
    print(f"    python train.py --block plain_mlp --ffn_dim_plain {ffn_dim}        # param-matched (smaller)\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--d_model",   type=int, default=128)
    p.add_argument("--ffn_dim",   type=int, default=512,
                   help="ffn_dim used for the experts block")
    p.add_argument("--n_experts", type=int, default=8)
    p.add_argument("--n_layers",  type=int, default=4)
    p.add_argument("--seq_len",   type=int, default=32)
    a = p.parse_args()
    main(a.d_model, a.ffn_dim, a.n_experts, a.n_layers, a.seq_len)
