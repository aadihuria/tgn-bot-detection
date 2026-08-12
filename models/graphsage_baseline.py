"""
Static graph baseline. This is deliberately blind to timing -- it sees one
snapshot of the graph (all edges collapsed together, no ordering) and has to
classify nodes from structure + features alone. The whole point of building
this first is to have a number to beat: if the temporal model in
tgn_bot_detector.py can't clear this bar, the temporal signal isn't earning
its complexity.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve


class GraphSAGEBot(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128, out_channels: int = 2, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden // 2)

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden // 2)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden // 2, out_channels)
        # picked once after training by best_threshold(), used at eval time
        # instead of the naive argmax -- see note in train_baseline
        self.decision_threshold = 0.5

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        return self.classifier(x)


def best_threshold(labels, probs) -> float:
    if len(set(labels)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1s = 2 * precision * recall / (precision + recall + 1e-12)
    if len(thresholds) == 0:
        return 0.5
    idx = np.argmax(f1s[:-1]) if len(f1s) > len(thresholds) else np.argmax(f1s)
    return float(thresholds[idx])


def train_baseline(x, edge_index, y, train_mask, val_mask, epochs: int = 100, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGEBot(in_channels=x.shape[1]).to(device)

    n_bot = y[train_mask].sum().item()
    n_human = train_mask.sum().item() - n_bot
    weight = torch.tensor([1.0, max(n_human, 1) / max(n_bot, 1)]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=weight)

    x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
    train_mask, val_mask = train_mask.to(device), val_mask.to(device)

    best_val_f1 = 0.0
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_out = model(x, edge_index)
                val_pred = val_out[val_mask].argmax(dim=1).cpu()
                val_true = y[val_mask].cpu()
                val_f1 = f1_score(val_true, val_pred, average="macro")
                probs = torch.softmax(val_out[val_mask], dim=1)[:, 1].cpu()
                val_auc = roc_auc_score(val_true, probs) if len(set(val_true.tolist())) > 1 else 0.5

            history.append({"epoch": epoch, "loss": loss.item(), "val_f1": val_f1, "val_auc": val_auc})

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # tried tuning this threshold on val via best_threshold() above, but the
    # val split here is only ~100 nodes with ~9% bots -- way too small for a
    # pr-curve threshold pick to generalize, it was overfitting to noise.
    # sticking with a plain 0.5 cutoff and leaning on auc as the headline
    # number instead (see README for why)
    model.decision_threshold = 0.5

    return model, history


def evaluate(model, x, edge_index, y, mask, threshold: float = None):
    device = next(model.parameters()).device
    if threshold is None:
        threshold = getattr(model, "decision_threshold", 0.5)
    model.eval()
    with torch.no_grad():
        out = model(x.to(device), edge_index.to(device))
        probs = torch.softmax(out[mask], dim=1)[:, 1].cpu()
        pred = (probs >= threshold).long()
        true = y[mask].cpu()

    f1 = f1_score(true, pred, average="macro")
    auc = roc_auc_score(true, probs) if len(set(true.tolist())) > 1 else 0.5
    return {"f1": f1, "auc": auc}
