"""
train_exp2.py — Shared training loop for the MHMP-vs-classic-transformer comparison.

One protocol, two architectures. Every training-schedule knob (lr, warmup,
batch_size, max_steps, weight_decay, grad_clip, seed, eval cadence) is shared
and identical across both models — that is the whole point, so any difference
in results is the architecture, not the schedule. The `--model` switch selects
which architecture is built; the inner training step is identical for both
because MHMP.forward takes the same packed (input_ids, loss_mask) interface as
the baseline DecoderOnlyTransformer.

The classic baseline is imported from model.py UNTOUCHED — the exact model that
ran for CE — so the comparison can't be blamed on a re-implemented baseline.

Usage
─────
  python train_exp2.py --model transformer --d_model 128 --n_layers 4 --ffn_dim_plain 512
  python train_exp2.py --model mhmp --d_model 128 --n_latents 64 --n_mask_heads 4 --n_loops 6
  python train_exp2.py --model mhmp --smoke         # tiny synthetic data, no GPU/wandb needed
  python train_exp2.py ... --no_wandb               # disable logging
"""

import argparse
import math
import os
import random
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from data import (build_vocab, load_pairs, PCFGDataset, make_loader, Vocabulary,
                  PAD, BOS, EOS, SEP, SPECIAL_TOKENS)
from model import make_model, count_params as count_params_tf
from mhmp import make_mhmp, count_params as count_params_mhmp
from enc_dec import make_enc_dec, count_params as count_params_ed


# ─────────────────────────────────────────────────────────────────────────────
#  Optional wandb (no-op shim if disabled or unavailable)
# ─────────────────────────────────────────────────────────────────────────────

class _NoWandb:
    def init(self, *a, **k):  pass
    def log(self, *a, **k):   pass
    def finish(self, *a, **k): pass

def get_wandb(enabled: bool):
    if not enabled:
        return _NoWandb()
    try:
        import wandb
        return wandb
    except Exception:
        print("[warn] wandb unavailable — logging disabled.")
        return _NoWandb()


# ─────────────────────────────────────────────────────────────────────────────
#  Config — shared protocol flags + per-architecture flags
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── which architecture ───────────────────────────────────────────────────
    model: str = "transformer"          # "transformer" | "mhmp" | "enc_dec"

    # ── data / protocol (SHARED, identical across models) ─────────────────────
    data_dir:    str       = "am-i-compositional/data/pcfgset"
    output_dir:  str       = "runs"
    train_split: str       = "pcfgset"
    eval_splits: List[str] = field(default_factory=lambda: ["pcfgset", "systematicity"])
    max_seq_len: int       = 128
    dropout:     float     = 0.1
    batch_size:  int       = 256
    lr:          float     = 3e-4
    weight_decay: float    = 0.01
    warmup_steps: int      = 1000
    decay_frac:   float    = 0.0   # 0 = warmup+constant; >0 = linear tail decay over final frac
    max_steps:   int       = 50_000
    grad_clip:   float     = 1.0
    seed:        int       = 42
    eval_every:  int       = 2_000
    log_every:   int       = 200
    save_every:  int       = 10_000
    eval_batch:  int       = 128
    max_eval_batches: Optional[int] = 50

    # ── shared model dims ─────────────────────────────────────────────────────
    d_model:     int       = 128

    # ── transformer-only ──────────────────────────────────────────────────────
    nhead:          int    = 4
    n_layers:       int    = 4
    ffn_dim_plain:  int    = 512
    looped:         bool   = False
    block_first:    bool   = False

    # ── mhmp-only ──────────────────────────────────────────────────────────────
    n_latents:    int = 64      # M
    n_mask_heads: int = 4       # H
    n_loops:      int = 6       # T
    mask_dim:     int = 64
    film_hidden:  int = 64
    reason_nhead: int = 4
    n_enc_layers: int = 2
    n_dec_layers: int = 2
    enc_nhead:    int = 4
    dec_nhead:    int = 4
    ffn_dim:      int = 256     # FFN dim used inside MHMP (enc/dec/reason)
    latent_init:  str = "learned"   # "learned" | "input_seeded"

    # ── enc_dec-only (plain cross-attn encoder-decoder baseline) ─────────────
    # reuses n_enc_layers / n_dec_layers / ffn_dim / d_model; its attention heads:
    ed_nhead:     int = 4

    # ── runtime ────────────────────────────────────────────────────────────────
    use_wandb:    bool = True
    smoke:        bool = False
    resume:       str  = ""    # path to a ckpt_*.pt to continue training from
    compile:      bool = False   # torch.compile the training forward (GPU; see caveats)


# ─────────────────────────────────────────────────────────────────────────────
#  Build the selected model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: Config, vocab: Vocabulary, device):
    if cfg.model == "transformer":
        m = make_model(
            vocab_size=len(vocab), pad_idx=vocab.pad_idx,
            d_model=cfg.d_model, nhead=cfg.nhead, n_layers=cfg.n_layers,
            ffn_dim=cfg.ffn_dim_plain, max_seq_len=cfg.max_seq_len,
            dropout=cfg.dropout, block_type="plain_mlp",
            looped=cfg.looped, block_first=cfg.block_first,
        ).to(device)
        return m, count_params_tf(m)
    elif cfg.model == "mhmp":
        m = make_mhmp(
            vocab_size=len(vocab), pad_idx=vocab.pad_idx, sep_idx=vocab.sep_idx,
            d_model=cfg.d_model, n_latents=cfg.n_latents, n_mask_heads=cfg.n_mask_heads,
            n_loops=cfg.n_loops, mask_dim=cfg.mask_dim, film_hidden=cfg.film_hidden,
            reason_nhead=cfg.reason_nhead, n_enc_layers=cfg.n_enc_layers,
            n_dec_layers=cfg.n_dec_layers, enc_nhead=cfg.enc_nhead, dec_nhead=cfg.dec_nhead,
            ffn_dim=cfg.ffn_dim, max_seq_len=cfg.max_seq_len, dropout=cfg.dropout,
            latent_init=cfg.latent_init,
        ).to(device)
        return m, count_params_mhmp(m)
    elif cfg.model == "enc_dec":
        m = make_enc_dec(
            vocab_size=len(vocab), pad_idx=vocab.pad_idx, sep_idx=vocab.sep_idx,
            d_model=cfg.d_model, n_enc_layers=cfg.n_enc_layers, n_dec_layers=cfg.n_dec_layers,
            nhead=cfg.ed_nhead, ffn_dim=cfg.ffn_dim, max_seq_len=cfg.max_seq_len,
            dropout=cfg.dropout,
        ).to(device)
        return m, count_params_ed(m)
    raise ValueError(f"unknown model {cfg.model}")


# ─────────────────────────────────────────────────────────────────────────────
#  MHMP-specific instrumentation logging (analogue of CE's routing logs)
# ─────────────────────────────────────────────────────────────────────────────

def log_mhmp_instrumentation(model, wb, step):
    ent = model.last_loop_entropy        # (T, H)
    fz  = model.last_loop_frac_zero      # (T, H)
    sim = model.last_loop_head_sim       # (T,)
    if ent is None:
        return
    log = {}
    T = ent.shape[0]
    # summary scalars (mean over loop) + final-iteration values
    log["mask/entropy_mean"]        = ent.mean().item()
    log["mask/entropy_final_iter"]  = ent[-1].mean().item()
    log["mask/frac_zero_mean"]      = fz.mean().item()
    log["mask/frac_zero_final"]     = fz[-1].mean().item()
    log["mask/head_similarity_mean"] = sim.mean().item()   # collapse monitor: lower healthier
    log["mask/head_similarity_max"]  = sim.max().item()
    wb.log(log, step=step)


# ─────────────────────────────────────────────────────────────────────────────
#  Eval — shared, model-agnostic (both models expose greedy_decode)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataset: PCFGDataset, vocab: Vocabulary, device,
             batch_size: int = 128, max_batches: Optional[int] = None) -> float:
    model.eval()
    n_correct, n_total = 0, 0
    for i, (_, _, srcs, tgts) in enumerate(make_loader(dataset, batch_size, shuffle=False, num_workers=0)):
        if max_batches is not None and i >= max_batches:
            break
        preds = model.greedy_decode(
            [vocab.encode(s) for s in srcs],
            vocab.bos_idx, vocab.sep_idx, vocab.eos_idx,
        )
        for pred, tgt in zip(preds, tgts):
            n_correct += (pred == vocab.encode(tgt))
            n_total += 1
    model.train()
    return n_correct / n_total if n_total else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic data for --smoke (a tiny learnable src->tgt map; no files/GPU)
# ─────────────────────────────────────────────────────────────────────────────

def make_smoke_dataset(n=128, seed=0):
    """src = [a, b]; tgt = [b, a] (reverse) — trivially learnable, exercises
    the whole pipeline including variable answer length."""
    rng = random.Random(seed)
    toks = [f"x{i}" for i in range(6)]
    srcs, tgts = [], []
    for _ in range(n):
        a, b = rng.choice(toks), rng.choice(toks)
        srcs.append([a, b]); tgts.append([b, a])
    vocab = Vocabulary().build([srcs[0]] + srcs + tgts)
    return srcs, tgts, vocab


# ─────────────────────────────────────────────────────────────────────────────
#  Data setup (real PCFG)
# ─────────────────────────────────────────────────────────────────────────────

def ensure_data(data_dir: str):
    if os.path.exists(data_dir):
        return
    if not os.path.exists("am-i-compositional"):
        print("Data not found. Cloning am-i-compositional...")
        subprocess.run(["git", "clone",
                        "https://github.com/i-machine-think/am-i-compositional.git"], check=True)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Cloned repo but can't find {data_dir}")


def make_scheduler(optimizer, warmup_steps, max_steps, decay_frac=0.0):
    """Warmup -> constant -> (optional) linear tail decay.

    decay_frac = 0.0  : warmup then flat forever. No horizon to guess, identical
                        across models, safe to stop whenever loss plateaus. Default
                        for iteration / matched-comparison runs.
    decay_frac > 0.0  : WSD-style. Hold flat, then decay linearly to 0 over the final
                        `decay_frac` of max_steps. Use for a single best-case final run
                        on the winner (set max_steps to where you decided to stop).
    """
    decay_start = max_steps * (1.0 - decay_frac)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)        # linear warmup ramp
        if decay_frac <= 0.0 or step < decay_start:
            return 1.0                                 # flat
        p = (step - decay_start) / max(1, max_steps - decay_start)
        return max(0.0, 1.0 - p)                       # linear decay to 0
    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
#  Train
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: Config):
    random.seed(cfg.seed); torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wb = get_wandb(cfg.use_wandb and not cfg.smoke)

    # ── data ───────────────────────────────────────────────────────────────────
    if cfg.smoke:
        srcs, tgts, vocab = make_smoke_dataset()
        train_ds = PCFGDataset(srcs, tgts, vocab)
        eval_datasets = {"smoke": train_ds}
        cfg.max_steps = min(cfg.max_steps, 300)
        cfg.warmup_steps = 20; cfg.eval_every = 100; cfg.log_every = 50
        cfg.batch_size = 32; cfg.max_eval_batches = 4
    else:
        ensure_data(cfg.data_dir)
        vocab = build_vocab(cfg.data_dir)
        tr_s, tr_t = load_pairs(f"{cfg.data_dir}/{cfg.train_split}/train.src",
                                f"{cfg.data_dir}/{cfg.train_split}/train.tgt", cfg.max_seq_len)
        train_ds = PCFGDataset(tr_s, tr_t, vocab)

        # Generalization splits (systematicity, productivity, ...) are RE-SPLITS of
        # the training data: each holds specific combinations OUT of its own train set
        # and concentrates them in its test set. A test score is only a valid
        # generalization measure if you TRAINED on that same split's train set.
        # Evaluating, say, systematicity/test with a model trained on the i.i.d.
        # pcfgset/train is meaningless — the held-out combinations were NOT held out.
        GEN_SPLITS = {"systematicity", "productivity", "substitutivity",
                      "localism", "overgeneralisation"}
        for split in cfg.eval_splits:
            if split in GEN_SPLITS and split != cfg.train_split:
                print(f"  [!] eval split '{split}' is a GENERALIZATION test but you are "
                      f"training on '{cfg.train_split}'.\n"
                      f"      Its score is NOT a valid {split} measure — to test {split}, "
                      f"run with --train_split {split}.\n"
                      f"      (Keeping it as an extra eval, but read it as in-distribution, "
                      f"not generalization.)")

        eval_datasets = {}
        for split in cfg.eval_splits:
            # pcfgset has a dev set (the shared dev/health split); generalization
            # splits only have train/test, so eval their test set.
            fname = "dev" if split == "pcfgset" else "test"
            sp, tp = f"{cfg.data_dir}/{split}/{fname}.src", f"{cfg.data_dir}/{split}/{fname}.tgt"
            if not os.path.exists(sp):
                print(f"[warn] {sp} not found, skipping"); continue
            s, t = load_pairs(sp, tp, cfg.max_seq_len)
            eval_datasets[split] = PCFGDataset(s, t, vocab)

    nworkers = 0 if cfg.smoke else 4
    train_loader = make_loader(train_ds, cfg.batch_size, shuffle=True, num_workers=nworkers)

    # ── model ────────────────────────────────────────────────────────────────
    model, n_params = build_model(cfg, vocab, device)
    run_name = f"{cfg.model}_dm{cfg.d_model}_seed{cfg.seed}"
    if cfg.model == "mhmp":
        run_name += f"_M{cfg.n_latents}_H{cfg.n_mask_heads}_T{cfg.n_loops}"
    elif cfg.model == "enc_dec":
        run_name += f"_enc{cfg.n_enc_layers}_dec{cfg.n_dec_layers}_ffn{cfg.ffn_dim}"
    else:
        run_name += f"_L{cfg.n_layers}_ffn{cfg.ffn_dim_plain}"
    print(f"{run_name}  |  params: {n_params:,}  |  device: {device}")
    wb.init(project="mhmp-systematicity", name=run_name, config=cfg.__dict__)

    # torch.compile wraps the model for the TRAINING forward only. The wrapper
    # shares parameters with `model`, so the optimizer (built on model.parameters())
    # and eager eval (model.greedy_decode) both stay in sync automatically.
    # Eval/decode deliberately stays eager: greedy_decode grows the sequence one
    # token per step, so compiling it would trigger constant recompilation.
    # Caveats: (1) Windows + CUDA torch.compile is finicky — treat as best-effort;
    # if it errors at first step, drop --compile. (2) _split_packed has data-dependent
    # control flow (.item(), python loops) so the graph breaks there; the speedup
    # comes from the compiled encoder/loop/decoder, which is where the compute is.
    train_model = model
    if cfg.compile:
        if device.type != "cuda":
            print("[warn] --compile requested on non-CUDA device; benefit is on GPU.")
        print("[compile] torch.compile(model) — first step will be slow (tracing).")
        train_model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                      betas=(0.9, 0.98), eps=1e-9)
    scheduler = make_scheduler(optimizer, cfg.warmup_steps, cfg.max_steps, cfg.decay_frac)

    # ── Resume from a checkpoint (model always; optimizer/scheduler if present) ──
    start_step = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        # architecture must match — load_state_dict will raise clearly if not
        model.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print(f"  [resume] loaded optimizer state (AdamW moments preserved)")
        else:
            print(f"  [resume] checkpoint has NO optimizer state — AdamW moments reset to "
                  f"zero.\n             Expect a brief loss transient while moments re-estimate.")
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            # fast-forward the LR schedule to start_step (initial_lr already set at construction)
            scheduler.last_epoch = start_step - 1
            scheduler.step()
        print(f"  [resume] continuing '{cfg.resume}' from step {start_step} "
              f"(lr now {scheduler.get_last_lr()[0]:.2e}, target max_steps {cfg.max_steps})")
        if start_step >= cfg.max_steps:
            print(f"  [resume] !! start_step {start_step} >= max_steps {cfg.max_steps}: "
                  f"raise --max_steps to actually train further.")

    run_dir = os.path.join(cfg.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    step, loss_accum = start_step, 0.0
    pbar = tqdm(total=cfg.max_steps, initial=start_step, unit="step",
                dynamic_ncols=True, disable=cfg.smoke)

    while step < cfg.max_steps:
        for input_ids, loss_mask, _, _ in train_loader:
            if step >= cfg.max_steps:
                break
            # ── identical inner step for both models ──────────────────────────
            _, loss = train_model(input_ids.to(device), loss_mask=loss_mask.to(device))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step(); scheduler.step()
            step += 1
            loss_accum = loss_accum + loss.detach()
            pbar.update(1)

            if step % cfg.log_every == 0:
                avg = loss_accum.item() / cfg.log_every
                lr_now = scheduler.get_last_lr()[0]
                wb.log({"train/loss": avg, "train/lr": lr_now}, step=step)
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_now:.2e}")
                loss_accum = 0.0
                if cfg.model == "mhmp":
                    log_mhmp_instrumentation(model, wb, step)
                if cfg.smoke:
                    print(f"  step {step:4d}  loss {avg:.4f}")

            if step % cfg.eval_every == 0:
                scores = {s: evaluate(model, ds, vocab, device, cfg.eval_batch, cfg.max_eval_batches)
                          for s, ds in eval_datasets.items()}
                wb.log({f"eval/{s}": v for s, v in scores.items()}, step=step)
                msg = "  ".join(f"{s}: {v:.4f}" for s, v in scores.items())
                (print if cfg.smoke else pbar.write)("  " + msg)

            if step % cfg.save_every == 0 and not cfg.smoke:
                torch.save({"step": step, "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "config": cfg.__dict__}, f"{run_dir}/ckpt_{step}.pt")
    pbar.close()

    scores = {s: evaluate(model, ds, vocab, device, cfg.eval_batch, max_batches=None)
              for s, ds in eval_datasets.items()}
    wb.log({f"eval_final/{s}": v for s, v in scores.items()})
    print("\nFinal:  " + "  ".join(f"{s}: {v:.4f}" for s, v in scores.items()))
    wb.finish()
    return scores


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["transformer", "mhmp", "enc_dec"], default="transformer")
    # protocol
    p.add_argument("--data_dir", default="am-i-compositional/data/pcfgset")
    p.add_argument("--output_dir", default="runs")
    p.add_argument("--train_split", default="pcfgset")
    p.add_argument("--eval_splits", nargs="+", default=["pcfgset", "systematicity"],
                   help="splits to eval. 'pcfgset' = i.i.d. dev (health/divergence signal); "
                        "a generalization split (e.g. systematicity) is only a valid measure "
                        "if --train_split is set to that same split.")
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--decay_frac", type=float, default=0.0,
                   help="0 = warmup+constant (default, for comparison runs); "
                        ">0 = linear tail decay over final frac of max_steps (final best-case run)")
    p.add_argument("--max_steps", type=int, default=50_000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=2_000)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--save_every", type=int, default=10_000)
    p.add_argument("--eval_batch", type=int, default=128)
    p.add_argument("--max_eval_batches", type=int, default=50)
    # shared dim
    p.add_argument("--d_model", type=int, default=128)
    # transformer
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--ffn_dim_plain", type=int, default=512)
    p.add_argument("--looped", action="store_true")
    p.add_argument("--block_first", action="store_true")
    # mhmp
    p.add_argument("--n_latents", type=int, default=64)
    p.add_argument("--n_mask_heads", type=int, default=4)
    p.add_argument("--n_loops", type=int, default=6)
    p.add_argument("--mask_dim", type=int, default=64)
    p.add_argument("--film_hidden", type=int, default=64)
    p.add_argument("--reason_nhead", type=int, default=4)
    p.add_argument("--n_enc_layers", type=int, default=2)
    p.add_argument("--n_dec_layers", type=int, default=2)
    p.add_argument("--enc_nhead", type=int, default=4)
    p.add_argument("--dec_nhead", type=int, default=4)
    p.add_argument("--ffn_dim", type=int, default=256)
    p.add_argument("--latent_init", choices=["learned", "input_seeded"], default="learned")
    # enc_dec
    p.add_argument("--ed_nhead", type=int, default=4)
    # runtime
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", type=str, default="",
                   help="path to a ckpt_*.pt to continue from. Architecture flags must match "
                        "the checkpoint. Raise --max_steps above the ckpt's step to train further.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the training forward for speedup (GPU). "
                        "Eval stays eager. Windows+CUDA is finicky; drop if it errors.")

    a = p.parse_args()
    kwargs = {k: v for k, v in vars(a).items() if k in Config.__dataclass_fields__}
    kwargs["use_wandb"] = not a.no_wandb
    kwargs["max_eval_batches"] = a.max_eval_batches if a.max_eval_batches > 0 else None
    return Config(**kwargs)


if __name__ == "__main__":
    train(parse_args())