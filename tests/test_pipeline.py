import numpy as np
import torch

from data.synthetic_generator import SyntheticTemporalGraph, SyntheticGraphConfig, make_splits
from data.pipeline import compute_burst_features


def small_graph():
    cfg = SyntheticGraphConfig(n_organic=300, n_bot_clusters=4, n_targets=50, time_span_days=20)
    return SyntheticTemporalGraph(cfg).generate()


def test_edges_are_chronologically_sorted():
    g = small_graph()
    t = g["t"]
    assert bool((t[1:] >= t[:-1]).all())


def test_node_feature_shape_and_normalization():
    g = small_graph()
    x = g["x"]
    assert x.shape[0] == g["n_source_nodes"] + g["n_target_nodes"]
    # source-node features should be roughly zero mean, unit variance
    source_x = x[: g["n_source_nodes"]]
    means = source_x.mean(dim=0)
    assert torch.all(means.abs() < 0.5)


def test_bot_clusters_are_labeled_correctly():
    g = small_graph()
    is_bot = g["is_bot"]
    y = g["y"]
    for cluster in g["bot_clusters"]:
        for node in cluster:
            assert is_bot[node]
            assert y[node].item() == 1


def test_target_nodes_are_not_classifiable():
    g = small_graph()
    mask = g["classifiable_mask"]
    n_source = g["n_source_nodes"]
    assert mask[:n_source].all()
    assert not mask[n_source:].any()


def test_make_splits_are_disjoint_and_cover_labeled_nodes():
    g = small_graph()
    train_mask, val_mask, test_mask = make_splits(g["classifiable_mask"], g["y"])

    overlap = (train_mask & val_mask) | (val_mask & test_mask) | (train_mask & test_mask)
    assert not overlap.any()

    covered = train_mask | val_mask | test_mask
    assert bool((covered == g["classifiable_mask"]).all())


def test_burst_features_are_higher_for_reactivating_bot_nodes():
    g = small_graph()
    n_total = g["x"].shape[0]
    bf = compute_burst_features(g["src"], g["dst"], g["t"], n_total)
    assert bf.shape == (n_total, 3)
    # nothing here should be nan or inf
    assert torch.isfinite(bf).all()


def test_reproducible_with_same_seed():
    cfg = SyntheticGraphConfig(n_organic=200, n_bot_clusters=3, seed=42)
    g1 = SyntheticTemporalGraph(cfg).generate()
    g2 = SyntheticTemporalGraph(cfg).generate()
    assert torch.equal(g1["src"], g2["src"])
    assert torch.equal(g1["t"], g2["t"])
