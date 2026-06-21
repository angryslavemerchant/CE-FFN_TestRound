"""
train.py — Training loop, evaluation, and CLI entry point.

Usage
─────
# Sanity check: plain MLP on the random split (do this first)
python train.py --block plain_mlp --train_split pcfgset --eval_splits pcfgset

# Full run with all three splits evaluated
python train.py --block plain_mlp --train_split pcfgset --eval_splits pcfgset systematicity productivity

# Override model size / training budget
python train.py --block plain_mlp --d_model 256 --n_layers 6 --max_steps 100000
"""

import argparse
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from data import (
    build_vocab, load_pairs, PCFGDataset, make_loader, Vocabulary
)
from model import make_model, count_params


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Paths
    data_dir: str = "am-i-compositional/data/pcfgset"
    output_dir:   str  = "runs"

    # Data
    train_split:  str  = "pcfgset"          # which folder's train.src/tgt to use
    eval_splits:  List[str] = field(default_factory=lambda: ["pcfgset"])
    max_seq_len:  int  = 128

    # Model
    block_type:   str  = "plain_mlp"
    d_model:      int  = 128
    nhead:        int  = 4
    n_layers:     int  = 4
    ffn_dim:      int  = 512
    dropout:      float = 0.1

    # Training
    batch_size:   int   = 256
    lr:           float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int   = 1000
    max_steps:    int   = 50_000
    grad_clip:    float = 1.0
    seed:         int   = 42

    # Evaluation / logging
    eval_every:   int   = 2_000    # steps between eval runs
    log_every:    int   = 200      # steps between loss logs
    save_every:   int   = 10_000   # steps between checkpoints
    eval_batch:   int   = 128      # batch size for greedy decode eval
    max_eval_batches: int = 50     # cap eval at this many batches (None = full)


# ── LR schedule ───────────────────────────────────────────────────────────────

def make_scheduler(optimizer, warmup_steps: int, max_steps: int) -> LambdaLR:
    """Linear warmup → cosine decay to 0."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:    nn.Module,
    dataset:  PCFGDataset,
    vocab:    Vocabulary,
    device:   torch.device,
    batch_size: int = 128,
    max_batches: Optional[int] = None,
) -> dict:
    """
    Returns a dict with:
        exact_match : fraction of examples where the full predicted answer matches ground truth
        n_evaluated : how many examples were decoded
    """
    model.eval()
    n_correct = 0
    n_total   = 0

    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for batch_idx, (input_ids, loss_mask, srcs, tgts) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Encode srcs for greedy_decode (ids only, without BOS/SEP)
        src_encoded = [vocab.encode(src) for src in srcs]

        preds = model.greedy_decode(
            src_encoded = src_encoded,
            bos_idx     = vocab.bos_idx,
            sep_idx     = vocab.sep_idx,
            eos_idx     = vocab.eos_idx,
            max_gen_len = 100,
        )

        for pred_ids, tgt_tokens in zip(preds, tgts):
            tgt_ids = vocab.encode(tgt_tokens)
            if pred_ids == tgt_ids:
                n_correct += 1
            n_total += 1

    return {
        "exact_match": n_correct / n_total if n_total > 0 else 0.0,
        "n_evaluated": n_total,
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg: Config):

    ensure_data(cfg.data_dir)

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    run_name = f"{cfg.block_type}_dm{cfg.d_model}_L{cfg.n_layers}_seed{cfg.seed}"
    run_dir  = os.path.join(cfg.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run: {run_name}")

    # ── Vocabulary ────────────────────────────────────────────────────────────
    print("Building vocabulary...")
    vocab = build_vocab(cfg.data_dir)
    print(f"Vocab size: {len(vocab)}")

    # ── Training data ─────────────────────────────────────────────────────────
    train_src_path = os.path.join(cfg.data_dir, cfg.train_split, "train.src")
    train_tgt_path = os.path.join(cfg.data_dir, cfg.train_split, "train.tgt")
    print(f"Loading training data from: {cfg.train_split}/")
    train_srcs, train_tgts = load_pairs(train_src_path, train_tgt_path, cfg.max_seq_len)
    print(f"  {len(train_srcs)} training examples")
    train_dataset = PCFGDataset(train_srcs, train_tgts, vocab)

    train_loader = make_loader(train_dataset, cfg.batch_size, shuffle=True)

    # ── Eval datasets ─────────────────────────────────────────────────────────
    eval_datasets = {}
    for split in cfg.eval_splits:
        # Use dev set for pcfgset random split, test set for others
        if split == "pcfgset":
            es_path = os.path.join(cfg.data_dir, split, "dev.src")
            et_path = os.path.join(cfg.data_dir, split, "dev.tgt")
        else:
            es_path = os.path.join(cfg.data_dir, split, "test.src")
            et_path = os.path.join(cfg.data_dir, split, "test.tgt")
        if not os.path.exists(es_path):
            print(f"  [warn] {es_path} not found, skipping eval on {split}")
            continue
        srcs, tgts = load_pairs(es_path, et_path, cfg.max_seq_len)
        eval_datasets[split] = PCFGDataset(srcs, tgts, vocab)
        print(f"  {split} eval: {len(srcs)} examples")

    # ── Model ────────────────────────────────────────────────────────────────
    model = make_model(
        vocab_size   = len(vocab),
        pad_idx      = vocab.pad_idx,
        d_model      = cfg.d_model,
        nhead        = cfg.nhead,
        n_layers     = cfg.n_layers,
        ffn_dim      = cfg.ffn_dim,
        max_seq_len  = cfg.max_seq_len,
        dropout      = cfg.dropout,
        block_type   = cfg.block_type,
    ).to(device)

    n_params = count_params(model)
    print(f"Parameters: {n_params:,}")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr           = cfg.lr,
        weight_decay = cfg.weight_decay,
        betas        = (0.9, 0.98),    # standard transformer betas
        eps          = 1e-9,
    )
    scheduler = make_scheduler(optimizer, cfg.warmup_steps, cfg.max_steps)

    # ── Training loop ─────────────────────────────────────────────────────────
    step        = 0
    epoch       = 0
    loss_accum  = 0.0
    t_start     = time.time()

    log_path = os.path.join(run_dir, "log.tsv")
    log_file = open(log_path, "w")
    log_file.write("step\tloss\t" + "\t".join(f"em_{s}" for s in eval_datasets) + "\n")

    print(f"\nStarting training for {cfg.max_steps} steps...")

    while step < cfg.max_steps:
        epoch += 1
        model.train()

        for input_ids, loss_mask, srcs, tgts in train_loader:
            if step >= cfg.max_steps:
                break

            input_ids = input_ids.to(device)
            loss_mask  = loss_mask.to(device)

            # Forward
            _, loss = model(input_ids, loss_mask=loss_mask)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            loss_accum += loss.item()

            # ── Log ──────────────────────────────────────────────────────────
            if step % cfg.log_every == 0:
                avg_loss = loss_accum / cfg.log_every
                elapsed  = time.time() - t_start
                lr_now   = scheduler.get_last_lr()[0]
                print(f"  step {step:6d} | loss {avg_loss:.4f} | lr {lr_now:.2e} | {elapsed:.0f}s")
                loss_accum = 0.0

            # ── Evaluate ─────────────────────────────────────────────────────
            if step % cfg.eval_every == 0:
                print(f"\n── Eval at step {step} ──")
                em_scores = {}
                for split_name, ds in eval_datasets.items():
                    result = evaluate(
                        model, ds, vocab, device,
                        batch_size=cfg.eval_batch,
                        max_batches=cfg.max_eval_batches,
                    )
                    em = result["exact_match"]
                    em_scores[split_name] = em
                    print(f"  {split_name:20s}  exact_match = {em:.4f}  (n={result['n_evaluated']})")
                print()

                # Log to file
                em_vals = "\t".join(f"{em_scores.get(s, 0):.4f}" for s in eval_datasets)
                avg_l   = loss_accum / max(1, step % cfg.log_every) if step % cfg.log_every != 0 else 0.0
                log_file.write(f"{step}\t{avg_l:.4f}\t{em_vals}\n")
                log_file.flush()

                model.train()

            # ── Checkpoint ───────────────────────────────────────────────────
            if step % cfg.save_every == 0:
                ckpt_path = os.path.join(run_dir, f"ckpt_{step}.pt")
                torch.save({
                    "step":        step,
                    "model":       model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "scheduler":   scheduler.state_dict(),
                    "config":      cfg.__dict__,
                    "vocab_size":  len(vocab),
                }, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n══ Final evaluation (full test sets) ══")
    for split_name, ds in eval_datasets.items():
        result = evaluate(model, ds, vocab, device, batch_size=cfg.eval_batch, max_batches=None)
        print(f"  {split_name:20s}  exact_match = {result['exact_match']:.4f}  (n={result['n_evaluated']})")

    log_file.close()
    print(f"\nLogs written to: {log_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="PCFG composing-experts experiment")

    parser.add_argument("--data_dir", default="am-i-compositional/data/pcfgset")
    parser.add_argument("--output_dir",   default="runs")
    parser.add_argument("--train_split",  default="pcfgset",
                        help="Which split's train set to use (pcfgset | systematicity | productivity)")
    parser.add_argument("--eval_splits",  nargs="+", default=["pcfgset"],
                        help="Splits to evaluate on (space-separated)")

    parser.add_argument("--block",        default="plain_mlp",
                        choices=["plain_mlp", "composing_experts", "averaging_experts"],
                        dest="block_type")
    parser.add_argument("--d_model",      type=int,   default=128)
    parser.add_argument("--nhead",        type=int,   default=4)
    parser.add_argument("--n_layers",     type=int,   default=4)
    parser.add_argument("--ffn_dim",      type=int,   default=512)
    parser.add_argument("--dropout",      type=float, default=0.1)

    parser.add_argument("--batch_size",   type=int,   default=256)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int,   default=1000)
    parser.add_argument("--max_steps",    type=int,   default=50_000)
    parser.add_argument("--grad_clip",    type=float, default=1.0)
    parser.add_argument("--seed",         type=int,   default=42)

    parser.add_argument("--eval_every",   type=int,   default=2_000)
    parser.add_argument("--log_every",    type=int,   default=200)
    parser.add_argument("--save_every",   type=int,   default=10_000)
    parser.add_argument("--eval_batch",   type=int,   default=128)
    parser.add_argument("--max_eval_batches", type=int, default=50,
                        help="Cap eval at N batches during training (None = full). "
                             "Use 0 for full eval every time.")

    args = parser.parse_args()

    # Convert max_eval_batches=0 to None
    max_eval_batches = args.max_eval_batches if args.max_eval_batches > 0 else None

    cfg = Config(
        data_dir         = args.data_dir,
        output_dir       = args.output_dir,
        train_split      = args.train_split,
        eval_splits      = args.eval_splits,
        block_type       = args.block_type,
        d_model          = args.d_model,
        nhead            = args.nhead,
        n_layers         = args.n_layers,
        ffn_dim          = args.ffn_dim,
        dropout          = args.dropout,
        batch_size       = args.batch_size,
        lr               = args.lr,
        weight_decay     = args.weight_decay,
        warmup_steps     = args.warmup_steps,
        max_steps        = args.max_steps,
        grad_clip        = args.grad_clip,
        seed             = args.seed,
        eval_every       = args.eval_every,
        log_every        = args.log_every,
        save_every       = args.save_every,
        eval_batch       = args.eval_batch,
        max_eval_batches = max_eval_batches,
    )
    return cfg



import subprocess

def ensure_data(data_dir: str):
    """Clone am-i-compositional if data_dir doesn't exist."""
    if os.path.exists(data_dir):
        return
    repo_dir = "am-i-compositional"
    if not os.path.exists(repo_dir):
        print("Data not found. Cloning am-i-compositional...")
        subprocess.run(
            ["git", "clone", "https://github.com/i-machine-think/am-i-compositional.git"],
            check=True
        )
    else:
        print("Repo already cloned, skipping.")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Cloned repo but still can't find: {data_dir}\n"
            f"Check the folder structure inside am-i-compositional/"
        )
    print(f"Data ready at: {data_dir}\n")





if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
