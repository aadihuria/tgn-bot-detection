"""
Temporal Graph Network for bot detection, built on PyG's TGN building blocks
(Rossi et al., 2020). The difference from the static baseline: every node
carries a memory vector that gets updated as edges arrive, so a node's
representation is a function of its history, not just its current
neighborhood. A bot cluster's memory should encode "this account just took
part in a synchronized burst of follows," which a snapshot model can't see
at all since it throws away ordering.

Training has to walk the edge stream in chronological order -- shuffling
batches would let the memory update on future information, silently leaking
labels. This is the main way TGN training code differs from a normal GNN
training loop.
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve


class TGNBotDetector(nn.Module):
    def __init__(self, num_nodes: int, node_feat_dim: int, edge_feat_dim: int,
                 memory_dim: int = 64, time_dim: int = 64, embedding_dim: int = 64,
                 burst_feat_dim: int = 3):
        super().__init__()

        self.memory = TGNMemory(
            num_nodes, edge_feat_dim, memory_dim, time_dim,
            message_module=IdentityMessage(edge_feat_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )

        self.gnn = TransformerConv(
            in_channels=memory_dim + node_feat_dim,
            out_channels=embedding_dim,
            heads=2, dropout=0.1, edge_dim=edge_feat_dim, concat=False,
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim + burst_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def embed(self, n_id, edge_index, edge_attr, node_feats, burst_feats):
        z, _ = self.memory(n_id)
        node_input = torch.cat([z, node_feats[n_id]], dim=1)
        z = self.gnn(node_input, edge_index, edge_attr)
        return torch.cat([z, burst_feats[n_id]], dim=1)

    def forward(self, n_id, edge_index, edge_attr, node_feats, burst_feats):
        z_with_burst = self.embed(n_id, edge_index, edge_attr, node_feats, burst_feats)
        return self.classifier(z_with_burst)


def best_f1_threshold(labels, probs) -> float:
    """picks the decision threshold that maximizes f1 on the given labels/probs"""
    if len(set(labels)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1s = 2 * precision * recall / (precision + recall + 1e-12)
    if len(thresholds) == 0:
        return 0.5
    best_idx = np.argmax(f1s[:-1]) if len(f1s) > len(thresholds) else np.argmax(f1s)
    return float(thresholds[best_idx])


def _run_chronological_pass(model, neighbor_loader, temporal_data, burst_feats,
                             edge_slice, device, batch_size, mask_attr, optimizer=None, criterion=None,
                             threshold=0.5):
    """
    walks one slice of the edge stream in order, updating memory after every
    batch. if optimizer is given, does a training step on any labeled nodes
    seen in the batch; otherwise just collects predictions for eval.
    """
    src_all = temporal_data.src[edge_slice]
    dst_all = temporal_data.dst[edge_slice]
    t_all = temporal_data.t[edge_slice]
    msg_all = temporal_data.msg[edge_slice]

    mask_tensor = getattr(temporal_data, mask_attr)

    all_preds, all_probs, all_labels = [], [], []
    total_loss, n_batches = 0.0, 0

    for i in range(0, len(src_all), batch_size):
        src = src_all[i:i + batch_size]
        dst = dst_all[i:i + batch_size]
        t = t_all[i:i + batch_size].long()  # memory's last_update buffer is int64
        msg = msg_all[i:i + batch_size]

        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)

        labeled = mask_tensor[n_id]
        if labeled.sum() == 0:
            model.memory.update_state(src, dst, t, msg)
            neighbor_loader.insert(src, dst)
            continue

        if optimizer is not None:
            optimizer.zero_grad()
            out = model(n_id, edge_index, msg_all[e_id] if e_id.numel() > 0 else msg_all[:0],
                        temporal_data.x, burst_feats)
            loss = criterion(out[labeled], temporal_data.y[n_id][labeled])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.memory.detach()
            total_loss += loss.item()
            n_batches += 1
        else:
            with torch.no_grad():
                out = model(n_id, edge_index, msg_all[e_id] if e_id.numel() > 0 else msg_all[:0],
                            temporal_data.x, burst_feats)
                probs = torch.softmax(out[labeled], dim=1)[:, 1]
                preds = (probs >= threshold).long()
                labels = temporal_data.y[n_id][labeled]
                all_probs.extend(probs.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        model.memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)

    if optimizer is not None:
        return total_loss / max(n_batches, 1)
    return all_preds, all_probs, all_labels


def train_tgn(temporal_data, burst_feats, train_mask, val_mask, epochs: int = 8,
              batch_size: int = 200, lr: float = 1e-3, neighbor_size: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_nodes = temporal_data.x.shape[0]
    model = TGNBotDetector(
        num_nodes=num_nodes,
        node_feat_dim=temporal_data.x.shape[1],
        edge_feat_dim=temporal_data.msg.shape[1],
    ).to(device)

    neighbor_loader = LastNeighborLoader(num_nodes, size=neighbor_size, device=device)

    n_pos = temporal_data.y[train_mask].clamp(min=0).sum().item()
    n_neg = train_mask.sum().item() - n_pos
    weight = torch.tensor([1.0, max(n_neg, 1) / max(n_pos, 1)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    temporal_data.train_mask = train_mask
    temporal_data.val_mask = val_mask
    temporal_data = temporal_data.to(device)
    burst_feats = burst_feats.to(device)

    n_edges = temporal_data.src.shape[0]
    full_slice = slice(0, n_edges)

    history = []
    for epoch in range(epochs):
        model.train()
        model.memory.reset_state()
        neighbor_loader.reset_state()

        avg_loss = _run_chronological_pass(
            model, neighbor_loader, temporal_data, burst_feats,
            full_slice, device, batch_size, "train_mask",
            optimizer=optimizer, criterion=criterion,
        )

        model.eval()
        model.memory.reset_state()
        neighbor_loader.reset_state()
        preds, probs, labels = _run_chronological_pass(
            model, neighbor_loader, temporal_data, burst_feats,
            full_slice, device, batch_size, "val_mask",
        )

        if labels:
            val_f1 = f1_score(labels, preds, average="macro")
            val_auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5
        else:
            val_f1, val_auc = 0.0, 0.5

        history.append({"epoch": epoch, "loss": avg_loss, "val_f1": val_f1, "val_auc": val_auc})
        print(f"epoch {epoch:2d} | loss {avg_loss:.4f} | val f1 {val_f1:.4f} | val auc {val_auc:.4f}")

    # best_f1_threshold() is available above but not used here -- with a
    # ~100-node val split the pr-curve threshold pick was unstable (kept
    # landing near 0 or 1 depending on the run). sticking with 0.5 and
    # reporting auc as the headline metric instead, see README
    threshold = 0.5

    return model, neighbor_loader, history, threshold


def evaluate_tgn_on_slice(model, neighbor_loader, temporal_data, burst_feats,
                           edge_slice, device, mask_attr="test_mask", batch_size: int = 200,
                           threshold: float = 0.5):
    model.eval()
    model.memory.reset_state()
    neighbor_loader.reset_state()
    preds, probs, labels = _run_chronological_pass(
        model, neighbor_loader, temporal_data, burst_feats,
        edge_slice, device, batch_size, mask_attr, threshold=threshold,
    )
    if not labels:
        return {"f1": 0.0, "auc": 0.5, "n": 0}
    f1 = f1_score(labels, preds, average="macro")
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5
    return {"f1": f1, "auc": auc, "n": len(labels)}
