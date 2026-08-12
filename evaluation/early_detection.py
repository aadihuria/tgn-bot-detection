"""
The evaluation that actually matters for this project: given only the first
X% of the edge stream, how well can each model flag accounts that are
eventually confirmed bots? A platform doesn't get to wait for the full
history before deciding whether to act -- it needs a score after a few
days, not a few months. This measures exactly that tradeoff for both the
static baseline (rebuilt from a truncated edge list at each checkpoint) and
the TGN (walked chronologically up to the checkpoint).
"""

import json
import os

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from models.graphsage_baseline import GraphSAGEBot
from models.tgn_bot_detector import evaluate_tgn_on_slice


def _static_eval_at_cutoff(x, src, dst, y, mask, cutoff, model_ctor_kwargs, trained_state, threshold):
    edge_index = torch.stack([src[:cutoff], dst[:cutoff]], dim=0)
    model = GraphSAGEBot(**model_ctor_kwargs)
    model.load_state_dict(trained_state)
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        probs = torch.softmax(out[mask], dim=1)[:, 1].cpu()
        pred = (probs >= threshold).long()
        true = y[mask].cpu()
    f1 = f1_score(true, pred, average="macro")
    auc = roc_auc_score(true, probs) if len(set(true.tolist())) > 1 else 0.5
    return f1, auc


def run_early_detection_eval(baseline_model, tgn_model, tgn_neighbor_loader,
                              temporal_data, burst_feats, x, src, dst, y, eval_mask,
                              device, n_checkpoints: int = 6, out_dir: str = "results",
                              tgn_threshold: float = 0.5):
    n_edges = src.shape[0]
    checkpoints = np.linspace(1.0 / n_checkpoints, 1.0, n_checkpoints)
    temporal_data.eval_mask_early = eval_mask

    baseline_kwargs = {"in_channels": x.shape[1]}
    baseline_state = baseline_model.state_dict()
    baseline_threshold = getattr(baseline_model, "decision_threshold", 0.5)

    results = {"checkpoints": [], "tgn_f1": [], "tgn_auc": [], "static_f1": [], "static_auc": []}

    for frac in checkpoints:
        cutoff = max(int(frac * n_edges), 2)

        static_f1, static_auc = _static_eval_at_cutoff(
            x, src, dst, y, eval_mask, cutoff, baseline_kwargs, baseline_state, baseline_threshold
        )

        tgn_metrics = evaluate_tgn_on_slice(
            tgn_model, tgn_neighbor_loader, temporal_data, burst_feats,
            slice(0, cutoff), device, mask_attr="eval_mask_early", threshold=tgn_threshold,
        )

        results["checkpoints"].append(float(frac))
        results["static_f1"].append(static_f1)
        results["static_auc"].append(static_auc)
        results["tgn_f1"].append(tgn_metrics["f1"])
        results["tgn_auc"].append(tgn_metrics["auc"])

        print(f"checkpoint {frac*100:5.1f}% | static f1 {static_f1:.4f} auc {static_auc:.4f} "
              f"| tgn f1 {tgn_metrics['f1']:.4f} auc {tgn_metrics['auc']:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "early_detection.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def plot_early_detection_curve(results: dict, out_path: str = "results/early_detection_curve.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    checkpoints = [c * 100 for c in results["checkpoints"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(checkpoints, results["tgn_f1"], "o-", label="tgn (temporal)", linewidth=2)
    ax1.plot(checkpoints, results["static_f1"], "s--", label="graphsage (static)", linewidth=2)
    ax1.set_xlabel("% of temporal stream observed")
    ax1.set_ylabel("f1 score (macro)")
    ax1.set_title("early detection: f1 vs stream observed")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(checkpoints, results["tgn_auc"], "o-", label="tgn (temporal)", linewidth=2)
    ax2.plot(checkpoints, results["static_auc"], "s--", label="graphsage (static)", linewidth=2)
    ax2.set_xlabel("% of temporal stream observed")
    ax2.set_ylabel("roc-auc")
    ax2.set_title("early detection: auc vs stream observed")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
