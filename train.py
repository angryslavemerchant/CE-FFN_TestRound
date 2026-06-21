"""
train.py — Training loop and CLI entry point.

Usage
─────
python train.py --block plain_mlp --train_split pcfgset --eval_splits pcfgset
python train.py --block plain_mlp --train_split pcfgset --eval_splits pcfgset systematicity productivity
"""

import argparse
import math
import os
import random
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from data import build_vocab, load_pairs, PCFGDataset, make_loader, Vocabulary
from model import make_model, count_params


def ensure_data(data_dir: str):
    """Clone am-i-compositional if data_dir doesn't exist."""
    if os.path.exists(data_dir):
        return
    repo_dir = "am-i-compositional"
    if not os.path.exists(repo_dir):
        print("Data not found. Cloning am-i-compositional...")
        subprocess.run(
            ["git", "clone", "https://github.com/i-machine-think/am-i-compositional.git"],
            check=True,
        )
    else:
        print("Repo already cloned, skipping.")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Cloned repo but still can't find: {data_dir}\n"
            f"Check the folder structure inside am-i-compositional/"
        )
    print(f"Data ready at: {data_dir}\n")


@dataclass
class Config:
    data_dir:    str       = "am-i-compositional/data/pcfgset"
    output_dir:  str       = "runs"
    train_split: str       = "pcfgset"
    eval_splits: List[str] = field(default_factory=lambda: ["pcfgset"])
    max_seq_len: int       = 128
    block_type:  str       = "plain_mlp"
    d_model:     int       = 128
    nhead:       int       = 4
    n_layers:    int       = 4
    ffn_dim:     int       = 512
    dropout:     float     = 0.1
    batch_size:  int       = 256
    lr:          float     = 3e-4
    weight_decay: float    = 0.01
    warmup_steps: int      = 1000
    max_steps:   int       = 50_000
    grad_clip:   float     = 1.0
    seed:        int       = 42
    eval_every:  int       = 2_000
    log_every:   int       = 200
    save_every:  int       = 10_000
    eval_batch:  int       = 128
    max_eval_batches: Optional[int] = 50


def make_scheduler(optimizer, warmup_steps: int, max_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, dataset: PCFGDataset, vocab: Vocabulary,
             device, batch_size: int = 128,
             max_batches: Optional[int] = None) -> float:
    model.eval()
    n_correct, n_total = 0, 0
    for i, (_, _, srcs, tgts) in enumerate(make_loader(dataset, batch_size, shuffle=False)):
        if max_batches is not None and i >= max_batches:
            break
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

    ensure_data(cfg.data_dir)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{cfg.block_type}_dm{cfg.d_model}_L{cfg.n_layers}_seed{cfg.seed}"

    wandb.init(project="composing-experts", name=run_name, config=cfg.__dict__)

    vocab = build_vocab(cfg.data_dir)

    train_srcs, train_tgts = load_pairs(
        f"{cfg.data_dir}/{cfg.train_split}/train.src",
        f"{cfg.data_dir}/{cfg.train_split}/train.tgt",
        cfg.max_seq_len,
    )
    train_loader = make_loader(PCFGDataset(train_srcs, train_tgts, vocab), cfg.batch_size, shuffle=True)

    eval_datasets = {}
    for split in cfg.eval_splits:
        fname = "dev" if split == "pcfgset" else "test"
        src_p = f"{cfg.data_dir}/{split}/{fname}.src"
        tgt_p = f"{cfg.data_dir}/{split}/{fname}.tgt"
        if not os.path.exists(src_p):
            print(f"[warn] {src_p} not found, skipping")
            continue
        srcs, tgts = load_pairs(src_p, tgt_p, cfg.max_seq_len)
        eval_datasets[split] = PCFGDataset(srcs, tgts, vocab)

    model = make_model(
        vocab_size=len(vocab), pad_idx=vocab.pad_idx,
        d_model=cfg.d_model, nhead=cfg.nhead, n_layers=cfg.n_layers,
        ffn_dim=cfg.ffn_dim, max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout, block_type=cfg.block_type,
    ).to(device)

    print(f"{run_name}  |  params: {count_params(model):,}  |  device: {device}")

    optimizer = AdamW(model.parameters(), lr=cfg.lr,
                      weight_decay=cfg.weight_decay, betas=(0.9, 0.98), eps=1e-9)
    scheduler = make_scheduler(optimizer, cfg.warmup_steps, cfg.max_steps)

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
            loss_accum = loss_accum + loss.detach()  # stays on GPU, no sync per step
            pbar.update(1)

            if step % cfg.log_every == 0:
                avg_loss = loss_accum.item() / cfg.log_every  # one sync per log interval
                lr_now   = scheduler.get_last_lr()[0]
                wandb.log({"train/loss": avg_loss, "train/lr": lr_now}, step=step)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr_now:.2e}")
                loss_accum = 0.0  # back to float after the sync

            if step % cfg.eval_every == 0:
                scores = {
                    split: evaluate(model, ds, vocab, device, cfg.eval_batch, cfg.max_eval_batches)
                    for split, ds in eval_datasets.items()
                }
                wandb.log({f"eval/{s}": v for s, v in scores.items()}, step=step)
                pbar.write("  " + "  ".join(f"{s}: {v:.4f}" for s, v in scores.items()))

            if step % cfg.save_every == 0:
                torch.save({"step": step, "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "config": cfg.__dict__},
                           f"{run_dir}/ckpt_{step}.pt")

    pbar.close()

    # Final full eval
    scores = {
        split: evaluate(model, ds, vocab, device, cfg.eval_batch, max_batches=None)
        for split, ds in eval_datasets.items()
    }
    wandb.log({f"eval_final/{s}": v for s, v in scores.items()})
    print("\nFinal:  " + "  ".join(f"{s}: {v:.4f}" for s, v in scores.items()))
    wandb.finish()


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="am-i-compositional/data/pcfgset")
    p.add_argument("--output_dir",  default="runs")
    p.add_argument("--train_split", default="pcfgset")
    p.add_argument("--eval_splits", nargs="+", default=["pcfgset"])
    p.add_argument("--block",       default="plain_mlp",
                   choices=["plain_mlp", "composing_experts", "averaging_experts"], dest="block_type")
    p.add_argument("--d_model",      type=int,   default=128)
    p.add_argument("--nhead",        type=int,   default=4)
    p.add_argument("--n_layers",     type=int,   default=4)
    p.add_argument("--ffn_dim",      type=int,   default=512)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int,   default=1000)
    p.add_argument("--max_steps",    type=int,   default=50_000)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--eval_every",   type=int,   default=2_000)
    p.add_argument("--log_every",    type=int,   default=200)
    p.add_argument("--save_every",   type=int,   default=10_000)
    p.add_argument("--eval_batch",   type=int,   default=128)
    p.add_argument("--max_eval_batches", type=int, default=50)
    a = p.parse_args()
    kwargs = {k: v for k, v in vars(a).items() if k in Config.__dataclass_fields__}
    kwargs["max_eval_batches"] = a.max_eval_batches if a.max_eval_batches > 0 else None
    return Config(**kwargs)


if __name__ == "__main__":
    train(parse_args())