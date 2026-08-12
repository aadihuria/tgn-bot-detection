"""
Loader for the real TwiBot-20 / TwiBot-22 file format.

Not exercised against real data yet -- access to both datasets requires
emailing the authors (shangbin@cs.washington.edu) and waiting for approval,
which hasn't come through. This is written against the documented file
layout so that once the files exist locally, swapping the synthetic
generator for this loader in train.py is a one-line change.

Expected files, all in one data_dir:
    user.json    node features per account
    edge.csv     source_id, target_id, relation_type, timestamp
    label.csv    id, label ('bot' or 'human')
    split.csv    id, split ('train' / 'val' / 'test')
"""

import json
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import TemporalData

RELATION_TYPES = [
    "followers", "following", "mention", "reply",
    "retweet", "quoted", "listed", "liked",
]


class TwiBotDataPipeline:
    """converts raw twibot-20/22 files into a pyg TemporalData object"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}

    def _safe_log(self, x, default=0.0) -> float:
        val = float(x) if x else default
        return float(np.log1p(max(val, 0)))

    def _account_age_days(self, created_at: str) -> float:
        if not created_at:
            return 0.0
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            reference = datetime(2022, 1, 1, tzinfo=dt.tzinfo)
            age_days = (reference - dt).days
            return float(np.log1p(max(age_days, 0)))
        except (ValueError, TypeError):
            return 0.0

    def load_nodes(self) -> Tuple[torch.Tensor, list]:
        with open(os.path.join(self.data_dir, "user.json")) as f:
            users = json.load(f)

        feature_rows = []
        user_ids = []

        for i, user in enumerate(users):
            uid = user.get("id", str(i))
            self.user_to_idx[uid] = i
            self.idx_to_user[i] = uid
            user_ids.append(uid)

            profile = user.get("public_metrics", {})
            row = [
                self._safe_log(profile.get("followers_count", 0)),
                self._safe_log(profile.get("following_count", 0)),
                self._safe_log(profile.get("tweet_count", 0)),
                self._safe_log(profile.get("listed_count", 0)),
                1.0 if user.get("verified", False) else 0.0,
                self._account_age_days(user.get("created_at", "")),
                1.0 if user.get("description", "") else 0.0,
                1.0 if user.get("url", "") else 0.0,
                float(len(user.get("name", ""))),
                float(len(user.get("username", ""))),
                1.0 if str(user.get("profile_image_url", "")).endswith("default_profile.png") else 0.0,
            ]
            feature_rows.append(row)

        x = torch.tensor(feature_rows, dtype=torch.float)
        mean = x.mean(dim=0)
        std = x.std(dim=0) + 1e-8
        x = (x - mean) / std
        return x, user_ids

    def load_edges(self):
        edges_df = pd.read_csv(os.path.join(self.data_dir, "edge.csv"))
        edges_df["ts_unix"] = pd.to_datetime(edges_df["timestamp"], utc=True).astype(np.int64) // 10**9
        edges_df = edges_df.sort_values("ts_unix").reset_index(drop=True)

        edges_df["src_idx"] = edges_df["source_id"].map(self.user_to_idx)
        edges_df["dst_idx"] = edges_df["target_id"].map(self.user_to_idx)
        edges_df = edges_df.dropna(subset=["src_idx", "dst_idx"])
        edges_df["src_idx"] = edges_df["src_idx"].astype(int)
        edges_df["dst_idx"] = edges_df["dst_idx"].astype(int)

        relation_dummies = pd.get_dummies(
            edges_df["relation_type"].fillna("unknown")
        ).reindex(columns=RELATION_TYPES, fill_value=0)

        ts = pd.to_datetime(edges_df["ts_unix"], unit="s", utc=True)
        hour_sin = np.sin(2 * np.pi * ts.dt.hour / 24)
        hour_cos = np.cos(2 * np.pi * ts.dt.hour / 24)
        dow_sin = np.sin(2 * np.pi * ts.dt.dayofweek / 7)

        edge_attr = np.column_stack([relation_dummies.values, hour_sin, hour_cos, dow_sin])

        src = torch.tensor(edges_df["src_idx"].values, dtype=torch.long)
        dst = torch.tensor(edges_df["dst_idx"].values, dtype=torch.long)
        t = torch.tensor(edges_df["ts_unix"].values, dtype=torch.float)
        edge_attr = torch.tensor(edge_attr.astype(np.float64), dtype=torch.float)
        return src, dst, t, edge_attr

    def load_labels(self, user_ids: list) -> Tuple[torch.Tensor, torch.Tensor]:
        labels_df = pd.read_csv(os.path.join(self.data_dir, "label.csv"))
        label_map = dict(zip(labels_df["id"], labels_df["label"]))

        labels = torch.zeros(len(user_ids), dtype=torch.long)
        mask = torch.zeros(len(user_ids), dtype=torch.bool)
        for i, uid in enumerate(user_ids):
            if uid in label_map:
                labels[i] = 1 if label_map[uid] == "bot" else 0
                mask[i] = True
        return labels, mask

    def load_splits(self, user_ids: list):
        split_path = os.path.join(self.data_dir, "split.csv")
        n = len(user_ids)
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)

        splits_df = pd.read_csv(split_path)
        split_map = dict(zip(splits_df["id"], splits_df["split"]))
        for i, uid in enumerate(user_ids):
            split = split_map.get(uid)
            if split == "train":
                train_mask[i] = True
            elif split == "val":
                val_mask[i] = True
            elif split == "test":
                test_mask[i] = True

        return train_mask, val_mask, test_mask

    def build_temporal_data(self) -> TemporalData:
        x, user_ids = self.load_nodes()
        src, dst, t, edge_attr = self.load_edges()
        labels, labeled_mask = self.load_labels(user_ids)
        train_mask, val_mask, test_mask = self.load_splits(user_ids)

        return TemporalData(
            src=src, dst=dst, t=t, msg=edge_attr,
            x=x, y=labels,
            labeled_mask=labeled_mask,
            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        )


def compute_burst_features(src: torch.Tensor, dst: torch.Tensor, t: torch.Tensor,
                            n_nodes: int, window_hours: float = 6.0) -> torch.Tensor:
    """
    per-node burst signal, shared by both the synthetic and real pipelines.

    three numbers per node:
      0. log(max connections formed in any window_hours-wide window)
      1. burstiness coefficient (sigma - mu) / (sigma + mu) of inter-arrival times
      2. that max-window count normalized by the node's total edge count
    """
    window_sec = window_hours * 3600
    burst_features = torch.zeros(n_nodes, 3)

    node_times: Dict[int, list] = {}
    src_np = src.numpy()
    t_np = t.numpy()
    for i in range(len(src_np)):
        s = int(src_np[i])
        node_times.setdefault(s, []).append(float(t_np[i]))

    for node_idx, times in node_times.items():
        if node_idx >= n_nodes or len(times) < 2:
            continue
        times = sorted(times)
        times_arr = np.array(times)

        max_window_count = 0
        for i, start_t in enumerate(times):
            end = np.searchsorted(times_arr, start_t + window_sec, side="right")
            count = end - i
            if count > max_window_count:
                max_window_count = int(count)

        burst_features[node_idx, 0] = float(np.log1p(max_window_count))

        inter_arrivals = np.diff(times_arr)
        if len(inter_arrivals) > 0:
            mu = inter_arrivals.mean()
            sigma = inter_arrivals.std() + 1e-8
            burst_features[node_idx, 1] = float((sigma - mu) / (sigma + mu))

        burst_features[node_idx, 2] = max_window_count / max(len(times), 1)

    return burst_features
