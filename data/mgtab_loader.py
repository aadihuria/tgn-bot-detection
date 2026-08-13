"""
Loader for MGTAB, a real published Twitter bot detection benchmark
(arxiv 2301.01123). Unlike the synthetic generator, this is a static
multi-relational graph -- there's no timestamp on any edge, so it can only
validate the static baseline (GraphSAGE), not the TGN or the burst detector,
both of which need chronological order to mean anything.

Get the data yourself: https://github.com/GraphDetec/MGTAB has a direct
Google Drive link, no request form, no approval wait. Unzip it so the six
.pt files (features.pt, labels_bot.pt, labels_stance.pt, edge_index.pt,
edge_type.pt, edge_weight.pt) land in data/mgtab_raw/. That folder is
gitignored on purpose -- the dataset isn't ours to redistribute, same as
every other dataset this repo touches.
"""

import os

import numpy as np
import torch

RELATION_TYPES = ["followers", "friends", "mention", "reply", "quoted", "url", "hashtag"]

MGTAB_N_NODES = 10199


def mgtab_available(data_dir: str = "data/mgtab_raw") -> bool:
    required = ["features.pt", "labels_bot.pt", "edge_index.pt", "edge_type.pt"]
    return all(os.path.exists(os.path.join(data_dir, f)) for f in required)


def load_mgtab(data_dir: str = "data/mgtab_raw"):
    x = torch.load(os.path.join(data_dir, "features.pt"), weights_only=False)
    y = torch.load(os.path.join(data_dir, "labels_bot.pt"), weights_only=False)
    edge_index = torch.load(os.path.join(data_dir, "edge_index.pt"), weights_only=False)
    edge_type = torch.load(os.path.join(data_dir, "edge_type.pt"), weights_only=False)
    return x, y.long(), edge_index.long(), edge_type.long()


def make_mgtab_splits(y: torch.Tensor, seed: int = 13, train_frac: float = 0.7, val_frac: float = 0.15):
    """stratified so train/val/test each keep roughly the same bot ratio as the full set"""
    rng = np.random.default_rng(seed)
    n = y.shape[0]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    for label in torch.unique(y).tolist():
        idx = torch.where(y == label)[0].numpy()
        rng.shuffle(idx)
        n_train = int(train_frac * len(idx))
        n_val = int(val_frac * len(idx))
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train:n_train + n_val]] = True
        test_mask[idx[n_train + n_val:]] = True

    return train_mask, val_mask, test_mask
