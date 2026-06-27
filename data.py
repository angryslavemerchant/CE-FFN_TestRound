"""
data.py — PCFG SET vocabulary, dataset, and dataloaders.

Each example is packed as a single decoder-only sequence:
    [BOS] src_token... [SEP] tgt_token... [EOS]

Loss is masked to only the answer span (SEP+1 through EOS inclusive).

All tensors are pre-computed at dataset construction time so __getitem__
is a pure list lookup — no string processing or tensor allocation on the
hot path.
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple

# ── Special tokens (indices 0–3 in every vocabulary) ─────────────────────────
PAD, BOS, EOS, SEP = "<pad>", "<bos>", "<eos>", "<sep>"
SPECIAL_TOKENS = [PAD, BOS, EOS, SEP]


class Vocabulary:
    def __init__(self):
        self.token2id: dict = {}
        self.id2token: list = []

    def build(self, token_lists: List[List[str]]) -> "Vocabulary":
        """Build from a list of token sequences (src and tgt together)."""
        for tok in SPECIAL_TOKENS:
            self.token2id[tok] = len(self.id2token)
            self.id2token.append(tok)
        all_tokens = set()
        for toks in token_lists:
            all_tokens.update(toks)
        for tok in sorted(all_tokens - set(SPECIAL_TOKENS)):
            self.token2id[tok] = len(self.id2token)
            self.id2token.append(tok)
        return self

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token2id[t] for t in tokens]

    def decode_ids(self, ids: List[int], strip_special: bool = True) -> List[str]:
        special = set(SPECIAL_TOKENS)
        out = []
        for i in ids:
            tok = self.id2token[i]
            if strip_special and tok in special:
                continue
            out.append(tok)
        return out

    @property
    def pad_idx(self) -> int:  return self.token2id[PAD]
    @property
    def bos_idx(self) -> int:  return self.token2id[BOS]
    @property
    def eos_idx(self) -> int:  return self.token2id[EOS]
    @property
    def sep_idx(self) -> int:  return self.token2id[SEP]

    def __len__(self) -> int:
        return len(self.id2token)


def build_vocab(data_dir: str) -> Vocabulary:
    """
    Build a shared vocabulary from the pcfgset (random) training set.
    Always use the random split so vocab is identical across all experiments.
    """
    src_path = os.path.join(data_dir, "pcfgset", "train.src")
    tgt_path = os.path.join(data_dir, "pcfgset", "train.tgt")
    token_lists = []
    for path in [src_path, tgt_path]:
        with open(path) as f:
            for line in f:
                token_lists.append(line.strip().split())
    return Vocabulary().build(token_lists)


def load_pairs(
    src_path: str,
    tgt_path: str,
    max_total_len: int = 128,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Load (src, tgt) string-token pairs from two parallel files.
    Drops examples where BOS+src+SEP+tgt+EOS > max_total_len.
    """
    srcs, tgts = [], []
    dropped = 0
    with open(src_path) as sf, open(tgt_path) as tf:
        for src_line, tgt_line in zip(sf, tf):
            src = src_line.strip().split()
            tgt = tgt_line.strip().split()
            if 1 + len(src) + 1 + len(tgt) + 1 <= max_total_len:
                srcs.append(src)
                tgts.append(tgt)
            else:
                dropped += 1
    if dropped:
        print(f"  [data] dropped {dropped} examples exceeding max_total_len={max_total_len}")
    return srcs, tgts


class PCFGDataset(Dataset):
    """
    Packs each (src, tgt) pair into a single sequence:
        [BOS] src... [SEP] tgt... [EOS]

    All input_ids and loss_mask tensors are pre-computed at init time.
    __getitem__ is a pure list lookup — zero allocation on the hot path.

    Memory cost: ~30MB for 82K examples (trivial).
    """

    def __init__(
        self,
        srcs: List[List[str]],
        tgts: List[List[str]],
        vocab: Vocabulary,
    ):
        assert len(srcs) == len(tgts)
        self.srcs = srcs   # kept for greedy eval
        self.tgts = tgts   # kept for greedy eval
        self.vocab = vocab

        # Pre-compute all tensors once
        self._ids:   List[torch.Tensor] = []
        self._masks: List[torch.Tensor] = []

        for src, tgt in zip(srcs, tgts):
            ids = [vocab.bos_idx] + vocab.encode(src) + [vocab.sep_idx] + vocab.encode(tgt) + [vocab.eos_idx]

            loss_mask = [0.0] * len(ids)
            answer_start = 1 + len(src) + 1
            for i in range(answer_start, len(ids)):
                loss_mask[i] = 1.0

            self._ids.append(torch.tensor(ids,       dtype=torch.long))
            self._masks.append(torch.tensor(loss_mask, dtype=torch.float))

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx):
        return self._ids[idx], self._masks[idx], self.srcs[idx], self.tgts[idx]


def collate(batch, pad_idx: int):
    """Pad a batch of variable-length sequences."""
    ids_list, mask_list, srcs, tgts = zip(*batch)
    ids_padded  = pad_sequence(ids_list,  batch_first=True, padding_value=pad_idx)
    mask_padded = pad_sequence(mask_list, batch_first=True, padding_value=0.0)
    return ids_padded, mask_padded, list(srcs), list(tgts)


def build_modular_vocab(p: int) -> Vocabulary:
    """Vocabulary for modular addition: integers 0..p-1 as string tokens."""
    return Vocabulary().build([[str(i) for i in range(p)]])


def generate_modular_addition(
    p: int,
    train_frac: float,
    seed: int,
) -> Tuple[Tuple[List[List[str]], List[List[str]]], Tuple[List[List[str]], List[List[str]]]]:
    """
    Generate all p² (a, b) pairs for modular addition (a + b) mod p.
    Returns ((train_srcs, train_tgts), (test_srcs, test_tgts)).
    src = [str(a), str(b)], tgt = [str((a+b) % p)]
    """
    rng = random.Random(seed)
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    rng.shuffle(all_pairs)
    n_train = int(len(all_pairs) * train_frac)
    train_pairs = all_pairs[:n_train]
    test_pairs  = all_pairs[n_train:]

    def to_src_tgt(pairs):
        srcs = [[str(a), str(b)] for a, b in pairs]
        tgts = [[str((a + b) % p)] for a, b in pairs]
        return srcs, tgts

    return to_src_tgt(train_pairs), to_src_tgt(test_pairs)


def make_loader(
    dataset: PCFGDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
) -> DataLoader:
    pad_idx = dataset.vocab.pad_idx
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate(b, pad_idx),
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4,
    )