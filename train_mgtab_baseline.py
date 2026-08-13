"""
Runs the static GraphSAGE baseline on real MGTAB data. This is separate from
train.py on purpose -- MGTAB has no edge timestamps, so only the static
baseline applies here, not the tgn or the burst detector. Think of this as
a sanity check that the static-gnn side of the pipeline isn't just an
artifact of how the synthetic generator is built.
"""

import json
import os
import time

import torch

from data.mgtab_loader import load_mgtab, make_mgtab_splits, mgtab_available
from models.graphsage_baseline import train_baseline, evaluate


def main():
    if not mgtab_available():
        print("data/mgtab_raw/ not found -- grab MGTAB from "
              "https://github.com/GraphDetec/MGTAB (direct google drive link, "
              "no approval needed) and unzip the six .pt files there")
        return

    os.makedirs("results", exist_ok=True)

    x, y, edge_index, edge_type = load_mgtab()
    train_mask, val_mask, test_mask = make_mgtab_splits(y)

    print(f"mgtab: {x.shape[0]} accounts, {edge_index.shape[1]} edges, "
          f"{int(y.sum())} bots ({y.float().mean() * 100:.1f}%)")

    t0 = time.time()
    model, history = train_baseline(x, edge_index, y, train_mask, val_mask, epochs=100)
    print(f"trained in {time.time() - t0:.1f}s")

    test_metrics = evaluate(model, x, edge_index, y, test_mask)
    print(f"mgtab test: f1={test_metrics['f1']:.4f} auc={test_metrics['auc']:.4f}")

    result = {
        "n_nodes": int(x.shape[0]),
        "n_edges": int(edge_index.shape[1]),
        "n_bots": int(y.sum()),
        "bot_ratio": float(y.float().mean()),
        "test_f1": test_metrics["f1"],
        "test_auc": test_metrics["auc"],
    }
    with open("results/mgtab_baseline.json", "w") as f:
        json.dump(result, f, indent=2)

    print("wrote results/mgtab_baseline.json")


if __name__ == "__main__":
    main()
