"""
train_grokking.py — Grokking experiment on modular addition (a + b mod p).

Usage
─────
python train_grokking.py --block plain_mlp
python train_grokking.py --block composing_experts --n_experts 4
python train_grokking.py --block plain_mlp --schedule cosine
"""

import argparse
import math
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from data import build_modular_vocab, generate_modular_addition, PCFGDataset, make_loader, Vocabulary
from model import make_model, count_params


@dataclass
class Config:
    p:           int   = 113        # modulus (prime)
    train_frac:  float = 0.3        # fraction of all p² pairs used for training
    output_dir:  str   = "runs"
    schedule:    str   = "constant" # "constant" or "cosine"
    block_type:  str   = "plain_mlp"
    looped:      bool  = False
    block_first: bool  = False
    n_experts:   int   = 4
    comp_dim:    int   = 64
    d_model:     int   = 128
    nhead:       int   = 4
    n_layers:    int   = 4
    ffn_dim_plain:   int = 512
    ffn_dim_experts: int = 512
    dropout:     float = 0.0        # grokking is sensitive to dropout; default off
    batch_size:  int   = 512
    lr:          float = 1e-3
    weight_decay: float = 1.0       # strong weight decay is essential for grokking
    warmup_steps: int  = 500
    max_steps:   int   = 100_000
    grad_clip:   float = 1.0
    seed:        int   = 42
    eval_every:  int   = 1_000
    log_every:   int   = 200
    save_every:  int   = 50_000
    eval_batch:  int   = 512


def make_scheduler(optimizer, warmup_steps: int, max_steps: int, schedule: str) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if schedule == "constant":
            return 1.0
        # cosine
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, dataset: PCFGDataset, vocab: Vocabulary,
             device, batch_size: int = 512) -> float:
    model.eval()
    n_correct, n_total = 0, 0
    for _, _, srcs, tgts in make_loader(dataset, batch_size, shuffle=False):
        preds = model.greedy_decode(
            [vocab.encode(s) for s in srcs],
            vocab.bos_idx, vocab.sep_idx, vocab.eos_idx,
        )
        for pred, tgt in zip(preds, tgts):
            n_correct += (pred == vocab.encode(tgt))
            n_total   += 1
    model.train()
    return n_correct / n_total if n_total else 0.0


def train(cfg: Config):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loop_tag  = "_looped"     if cfg.looped      else ""
    order_tag = "_blockfirst" if cfg.block_first  else ""
    run_name  = (
        f"{cfg.block_type}{loop_tag}{order_tag}"
        f"_p{cfg.p}_frac{cfg.train_frac}"
        f"_dm{cfg.d_model}_L{cfg.n_layers}"
        f"_wd{cfg.weight_decay}_seed{cfg.seed}"
    )

    wandb.init(project="grokking-ce-ffn", name=run_name, config=cfg.__dict__)

    vocab = build_modular_vocab(cfg.p)
    (train_srcs, train_tgts), (test_srcs, test_tgts) = generate_modular_addition(
        cfg.p, cfg.train_frac, cfg.seed
    )

    max_seq_len = 16   # [BOS a b SEP c EOS] = 6 tokens; 16 is generous headroom
    train_dataset = PCFGDataset(train_srcs, train_tgts, vocab)
    test_dataset  = PCFGDataset(test_srcs,  test_tgts,  vocab)
    train_loader  = make_loader(train_dataset, cfg.batch_size, shuffle=True)

    ffn_dim = cfg.ffn_dim_plain if cfg.block_type == "plain_mlp" else cfg.ffn_dim_experts

    model = make_model(
        vocab_size=len(vocab), pad_idx=vocab.pad_idx,
        d_model=cfg.d_model, nhead=cfg.nhead, n_layers=cfg.n_layers,
        ffn_dim=ffn_dim, max_seq_len=max_seq_len,
        dropout=cfg.dropout, block_type=cfg.block_type,
        block_kwargs={"n_experts": cfg.n_experts, "comp_dim": cfg.comp_dim},
        looped=cfg.looped,
        block_first=cfg.block_first,
    ).to(device)

    print(f"{run_name}  |  params: {count_params(model):,}  |  device: {device}")
    print(f"  train: {len(train_dataset)}  test: {len(test_dataset)}  vocab: {len(vocab)}")

    optimizer = AdamW(model.parameters(), lr=cfg.lr,
                      weight_decay=cfg.weight_decay, betas=(0.9, 0.98), eps=1e-9)
    scheduler = make_scheduler(optimizer, cfg.warmup_steps, cfg.max_steps, cfg.schedule)

    run_dir = os.path.join(cfg.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    step, epoch, loss_accum = 0, 0, 0.0
    pbar = tqdm(total=cfg.max_steps, unit="step", dynamic_ncols=True)

    while step < cfg.max_steps:
        epoch += 1
        for input_ids, loss_mask, _, _ in train_loader:
            if step >= cfg.max_steps:
                break

            _, loss = model(input_ids.to(device), loss_mask=loss_mask.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            loss_accum = loss_accum + loss.detach()
            pbar.update(1)

            if step % cfg.log_every == 0:
                avg_loss = loss_accum.item() / cfg.log_every
                lr_now   = scheduler.get_last_lr()[0]
                wandb.log({"train/loss": avg_loss, "train/lr": lr_now}, step=step)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr_now:.2e}")
                loss_accum = 0.0

                if cfg.block_type == "composing_experts":
                    log = {}
                    for i, layer in enumerate(model.layers):
                        b = layer.block
                        N = b.n_experts
                        if b.last_routing_weights is not None:
                            rw = b.last_routing_weights.cpu()
                            log[f"routing/L{i}/deviation_from_uniform"] = (rw - 1/N).abs().mean().item()
                            log[f"routing/L{i}/max_weight"]             = rw.max().item()
                            log[f"routing/L{i}/min_weight"]             = rw.min().item()
                        if b.last_attn_weights is not None:
                            aw = b.last_attn_weights.cpu()
                            log[f"comp_attn/L{i}/deviation_from_uniform"] = (aw - 1/N).abs().mean().item()
                    if log:
                        wandb.log(log, step=step)

            if step % cfg.eval_every == 0:
                train_acc = evaluate(model, train_dataset, vocab, device, cfg.eval_batch)
                test_acc  = evaluate(model, test_dataset,  vocab, device, cfg.eval_batch)
                wandb.log({"eval/train": train_acc, "eval/test": test_acc}, step=step)
                pbar.write(f"  step {step:6d}  train: {train_acc:.4f}  test: {test_acc:.4f}")

            if step % cfg.save_every == 0:
                torch.save({"step": step, "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "config": cfg.__dict__},
                           f"{run_dir}/ckpt_{step}.pt")

    pbar.close()

    train_acc = evaluate(model, train_dataset, vocab, device, cfg.eval_batch)
    test_acc  = evaluate(model, test_dataset,  vocab, device, cfg.eval_batch)
    wandb.log({"eval_final/train": train_acc, "eval_final/test": test_acc})
    print(f"\nFinal:  train: {train_acc:.4f}  test: {test_acc:.4f}")
    wandb.finish()


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--p",            type=int,   default=113)
    p.add_argument("--train_frac",   type=float, default=0.3)
    p.add_argument("--output_dir",   default="runs")
    p.add_argument("--schedule",     default="constant", choices=["constant", "cosine"])
    p.add_argument("--block",        default="plain_mlp",
                   choices=["plain_mlp", "composing_experts"], dest="block_type")
    p.add_argument("--looped",       action="store_true")
    p.add_argument("--block_first",  action="store_true")
    p.add_argument("--n_experts",    type=int,   default=4)
    p.add_argument("--comp_dim",     type=int,   default=64)
    p.add_argument("--d_model",      type=int,   default=128)
    p.add_argument("--nhead",        type=int,   default=4)
    p.add_argument("--n_layers",     type=int,   default=4)
    p.add_argument("--ffn_dim_plain",    type=int, default=512)
    p.add_argument("--ffn_dim_experts",  type=int, default=512)
    p.add_argument("--dropout",      type=float, default=0.0)
    p.add_argument("--batch_size",   type=int,   default=512)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int,   default=500)
    p.add_argument("--max_steps",    type=int,   default=100_000)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--eval_every",   type=int,   default=1_000)
    p.add_argument("--log_every",    type=int,   default=200)
    p.add_argument("--save_every",   type=int,   default=50_000)
    p.add_argument("--eval_batch",   type=int,   default=512)
    a = p.parse_args()
    kwargs = {k: v for k, v in vars(a).items() if k in Config.__dataclass_fields__}
    return Config(**kwargs)


if __name__ == "__main__":
    train(parse_args())
