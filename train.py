"""
End to end run: generate the synthetic benchmark, train the static baseline,
train the tgn, run the burst detector, run the early-detection comparison,
and dump everything to results/. This is what actually produced the numbers
quoted in the README -- rerun it and you get the same story (seeded).
"""

import json
import os
import time

import torch
from torch_geometric.data import TemporalData

from data.synthetic_generator import SyntheticTemporalGraph, SyntheticGraphConfig, make_splits
from data.pipeline import compute_burst_features
from models.graphsage_baseline import train_baseline, evaluate as evaluate_baseline
from models.tgn_bot_detector import train_tgn, evaluate_tgn_on_slice
from detectors.burst_detector import BurstCoordinationDetector
from evaluation.early_detection import run_early_detection_eval, plot_early_detection_curve


def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("\n=== generating synthetic benchmark ===")
    t0 = time.time()
    generator = SyntheticTemporalGraph(SyntheticGraphConfig())
    graph = generator.generate()
    print(f"generated in {time.time() - t0:.2f}s | "
          f"{graph['x'].shape[0]} nodes, {graph['src'].shape[0]} edges, "
          f"{int(graph['is_bot'].sum())} bots across {len(graph['bot_clusters'])} clusters")

    x = graph["x"]
    src, dst, t = graph["src"], graph["dst"], graph["t"]
    y = graph["y"].clone()
    y[y < 0] = 0

    n_total = x.shape[0]
    burst_feats = compute_burst_features(src, dst, t, n_total)

    train_mask, val_mask, test_mask = make_splits(graph["classifiable_mask"], graph["y"])

    print("\n=== training graphsage baseline (static snapshot) ===")
    edge_index = torch.stack([src, dst], dim=0)
    t0 = time.time()
    baseline_model, baseline_history = train_baseline(x, edge_index, y, train_mask, val_mask, epochs=80)
    print(f"baseline trained in {time.time() - t0:.2f}s")
    baseline_test = evaluate_baseline(baseline_model, x, edge_index, y, test_mask)
    print(f"baseline test: f1={baseline_test['f1']:.4f} auc={baseline_test['auc']:.4f}")

    print("\n=== training tgn ===")
    temporal_data = TemporalData(src=src, dst=dst, t=t, msg=graph["edge_attr"], x=x, y=y)
    t0 = time.time()
    tgn_model, neighbor_loader, tgn_history, tgn_threshold = train_tgn(
        temporal_data, burst_feats, train_mask, val_mask, epochs=8, batch_size=200
    )
    print(f"tgn trained in {time.time() - t0:.2f}s")

    temporal_data.test_mask = test_mask.to(device)
    tgn_test = evaluate_tgn_on_slice(
        tgn_model, neighbor_loader, temporal_data, burst_feats.to(device),
        slice(0, src.shape[0]), device, mask_attr="test_mask", threshold=tgn_threshold,
    )
    print(f"tgn test: f1={tgn_test['f1']:.4f} auc={tgn_test['auc']:.4f}")

    print("\n=== burst coordination detector ===")
    detector = BurstCoordinationDetector()
    t0 = time.time()
    cofollow_graph = detector.build_cofollow_graph(src.numpy(), dst.numpy(), t.numpy())
    flagged_clusters = detector.detect_coordinated_clusters(cofollow_graph, src.numpy(), t.numpy())
    print(f"burst detector ran in {time.time() - t0:.2f}s, flagged {len(flagged_clusters)} clusters")

    bot_clusters = graph["bot_clusters"]
    matched = 0
    for bc in bot_clusters:
        bset = set(bc)
        best = max((len(bset & set(c["node_ids"])) / len(bset) for c in flagged_clusters), default=0.0)
        if best > 0.5:
            matched += 1
    is_bot_arr = graph["is_bot"]
    false_positives = sum(
        1 for c in flagged_clusters
        if sum(1 for n in c["node_ids"] if n < len(is_bot_arr) and not is_bot_arr[n]) / len(c["node_ids"]) > 0.5
    )
    burst_summary = {
        "n_true_bot_clusters": len(bot_clusters),
        "n_flagged_clusters": len(flagged_clusters),
        "n_bot_clusters_caught": matched,
        "recall": matched / len(bot_clusters) if bot_clusters else 0.0,
        "false_positive_flagged_clusters": false_positives,
    }
    print(f"burst detector: caught {matched}/{len(bot_clusters)} true bot clusters "
          f"({burst_summary['recall']*100:.1f}% recall)")

    print("\n=== early detection eval ===")
    t0 = time.time()
    early_results = run_early_detection_eval(
        baseline_model, tgn_model, neighbor_loader, temporal_data, burst_feats,
        x.to(device), src.to(device), dst.to(device), y.to(device), test_mask.to(device),
        device, n_checkpoints=6, tgn_threshold=tgn_threshold,
    )
    plot_early_detection_curve(early_results)
    print(f"early detection eval ran in {time.time() - t0:.2f}s")

    summary = {
        "n_nodes": n_total,
        "n_edges": int(src.shape[0]),
        "n_bots": int(graph["is_bot"].sum()),
        "n_bot_clusters": len(bot_clusters),
        "baseline_test": baseline_test,
        "tgn_test": tgn_test,
        "burst_detector": burst_summary,
        "early_detection": early_results,
    }
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\ndone. results written to results/summary.json, results/early_detection.json, "
          "results/early_detection_curve.png")


if __name__ == "__main__":
    main()
