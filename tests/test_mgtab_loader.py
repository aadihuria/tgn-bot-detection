import pytest
import torch

from data.mgtab_loader import mgtab_available, load_mgtab, make_mgtab_splits, MGTAB_N_NODES

pytestmark = pytest.mark.skipif(
    not mgtab_available(),
    reason="mgtab raw .pt files not present, download from github.com/GraphDetec/MGTAB",
)


def test_shapes_line_up():
    x, y, edge_index, edge_type = load_mgtab()
    assert x.shape[0] == MGTAB_N_NODES
    assert y.shape[0] == MGTAB_N_NODES
    assert edge_index.shape[0] == 2
    assert edge_type.shape[0] == edge_index.shape[1]


def test_labels_are_binary():
    _, y, _, _ = load_mgtab()
    assert set(torch.unique(y).tolist()) <= {0, 1}


def test_splits_are_disjoint_and_cover_everything():
    _, y, _, _ = load_mgtab()
    train_mask, val_mask, test_mask = make_mgtab_splits(y)

    overlap = (train_mask & val_mask) | (val_mask & test_mask) | (train_mask & test_mask)
    assert not overlap.any()

    covered = train_mask | val_mask | test_mask
    assert covered.sum().item() > 0.99 * MGTAB_N_NODES


def test_splits_stay_roughly_stratified():
    _, y, _, _ = load_mgtab()
    train_mask, _, test_mask = make_mgtab_splits(y)

    full_ratio = y.float().mean().item()
    train_ratio = y[train_mask].float().mean().item()
    test_ratio = y[test_mask].float().mean().item()

    assert abs(train_ratio - full_ratio) < 0.03
    assert abs(test_ratio - full_ratio) < 0.05
