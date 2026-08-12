"""
Runs the trained pipeline once more and dumps a small, github-pages-friendly
json bundle: a handful of example account risk scores and the flagged
coordination clusters. This is what docs/index.html reads -- the demo page
has no backend, it's just rendering real numbers produced by this script.
"""

import json
import os

import torch
from torch_geometric.data import TemporalData

from data.synthetic_generator import SyntheticTemporalGraph, SyntheticGraphConfig, make_splits
from data.pipeline import compute_burst_features
from models.tgn_bot_detector import train_tgn, evaluate_tgn_on_slice
from detectors.burst_detector import BurstCoordinationDetector


def main():
    device = torch.device("cpu")
    generator = SyntheticTemporalGraph(SyntheticGraphConfig())
    graph = generator.generate()

    x = graph["x"]
    src, dst, t = graph["src"], graph["dst"], graph["t"]
    y = graph["y"].clone()
    y[y < 0] = 0

    burst_feats = compute_burst_features(src, dst, t, x.shape[0])
    train_mask, val_mask, test_mask = make_splits(graph["classifiable_mask"], graph["y"])

    temporal_data = TemporalData(src=src, dst=dst, t=t, msg=graph["edge_attr"], x=x, y=y)
    model, neighbor_loader, history, threshold = train_tgn(
        temporal_data, burst_feats, train_mask, val_mask, epochs=8, batch_size=200
    )

    temporal_data.test_mask = test_mask.to(device)
    model.eval()
    model.memory.reset_state()
    neighbor_loader.reset_state()

    from models.tgn_bot_detector import _run_chronological_pass
    preds, probs, labels = _run_chronological_pass(
        model, neighbor_loader, temporal_data, burst_feats.to(device),
        slice(0, src.shape[0]), device, 200, "test_mask", threshold=threshold,
    )

    detector = BurstCoordinationDetector()
    cofollow_graph = detector.build_cofollow_graph(src.numpy(), dst.numpy(), t.numpy())
    flagged_clusters = detector.detect_coordinated_clusters(cofollow_graph, src.numpy(), t.numpy())

    example_scores = []
    for i in range(min(12, len(probs))):
        p = probs[i]
        risk = "high" if p > 0.6 else "medium" if p > 0.3 else "low"
        example_scores.append({
            "account_id": f"acct_{i:04d}",
            "bot_probability": round(float(p), 3),
            "actual_label": "bot" if labels[i] == 1 else "human",
            "risk_level": risk,
        })

    demo_clusters = []
    for c in flagged_clusters[:8]:
        demo_clusters.append({
            "cluster_size": c["cluster_size"],
            "median_inter_arrival_sec": c["median_inter_arrival_sec"],
            "coordination_score": c["coordination_score"],
            "n_events": c["n_events"],
        })

    with open("results/summary.json") as f:
        summary = json.load(f)

    bundle = {
        "summary": summary,
        "example_scores": example_scores,
        "flagged_clusters": demo_clusters,
    }

    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/demo.json", "w") as f:
        json.dump(bundle, f, indent=2)

    print("wrote docs/data/demo.json")


if __name__ == "__main__":
    main()
