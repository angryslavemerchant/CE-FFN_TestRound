"""
data.py — PCFG SET vocabulary, dataset, and dataloaders.

Each example is packed as a single decoder-only sequence:
    [BOS] src_token... [SEP] tgt_token... [EOS]

Loss is masked to only the answer span (SEP+1 through EOS inclusive).
"""

import os
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
        # Special tokens first — indices are stable
        for tok in SPECIAL_TOKENS:
            self.token2id[tok] = len(self.id2token)
            self.id2token.append(tok)
        # All other tokens, sorted for reproducibility
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
    We always use the random split to define vocab so it's the same across all experiments.
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
    Returns: (srcs, tgts) as parallel lists.
    """
    srcs, tgts = [], []
    dropped = 0
    with open(src_path) as sf, open(tgt_path) as tf:
        for src_line, tgt_line in zip(sf, tf):
            src = src_line.strip().split()
            tgt = tgt_line.strip().split()
            # full sequence length: 1(BOS) + len(src) + 1(SEP) + len(tgt) + 1(EOS)
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

    Returns:
        input_ids  : LongTensor  — the full packed sequence
        loss_mask  : FloatTensor — 1.0 at positions to predict (tgt tokens + EOS), 0.0 elsewhere
        src_tokens : List[str]   — raw src strings (for greedy eval)
        tgt_tokens : List[str]   — raw tgt strings (for greedy eval)
    """

    def __init__(
        self,
        srcs: List[List[str]],
        tgts: List[List[str]],
        vocab: Vocabulary,
    ):
        assert len(srcs) == len(tgts)
        self.srcs = srcs
        self.tgts = tgts
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.srcs)

    def __getitem__(self, idx):
        src, tgt = self.srcs[idx], self.tgts[idx]
        v = self.vocab

        # Packed sequence
        ids = [v.bos_idx] + v.encode(src) + [v.sep_idx] + v.encode(tgt) + [v.eos_idx]

        # Loss mask: 1 at every position the model should predict
        # In next-token prediction, position i produces a logit that predicts ids[i+1].
        # We want to predict tgt[0], tgt[1], ..., tgt[-1], EOS.
        # Those sit at positions 1+len(src)+1 through 1+len(src)+1+len(tgt) in ids.
        # The loss_mask marks these positions so that when we shift by 1 (logits[i] → ids[i+1]),
        # shift_mask = loss_mask[1:] correctly selects the right logit-target pairs.
        loss_mask = [0.0] * len(ids)
        answer_start = 1 + len(src) + 1   # first tgt token index
        for i in range(answer_start, len(ids)):
            loss_mask[i] = 1.0

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.float),
            src,
            tgt,
        )


def collate(batch, pad_idx: int):
    """Pad a batch of variable-length sequences."""
    ids_list, mask_list, srcs, tgts = zip(*batch)
    ids_padded  = pad_sequence(ids_list,  batch_first=True, padding_value=pad_idx)
    mask_padded = pad_sequence(mask_list, batch_first=True, padding_value=0.0)
    return ids_padded, mask_padded, list(srcs), list(tgts)


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
        num_workers=0,   # 0 = main process; avoids worker overhead for small data
        collate_fn=lambda b: collate(b, pad_idx),
        pin_memory=False,
    )
