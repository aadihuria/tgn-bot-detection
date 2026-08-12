"""
Synthetic temporal social graph generator.

TwiBot-20 and TwiBot-22 both require emailing the dataset authors and waiting
for manual approval, so this generator stands in for that data while access
is pending. It's not trying to be a perfect simulation of Twitter -- it's
trying to reproduce the one structural property this whole project is built
around: bots give themselves away through *when* they connect, not what
their profile looks like.

Two populations of nodes get generated:

- organic accounts connect to targets following a time-inhomogeneous Poisson
  process. the rate is modulated by a circadian curve (higher during waking
  hours) so inter-arrival times look like a person who sleeps, and by a
  slow-varying "interest" drift so someone's attention wanders over weeks
  rather than jumping around uniformly.

- bot cluster accounts sit dormant, then the whole cluster activates inside
  a short window (minutes to a few hours) and fires a batch of follows at
  a small set of shared targets, then goes back to sleep for a while before
  the next activation. this is the "50 accounts all follow the same person
  within six hours" pattern described in the coordinated-inauthentic-behavior
  literature (see e.g. the temporal-graph framing in Rossi et al., TGN,
  2020, and the graph-based bot benchmark discussion in the TwiBot-22
  NeurIPS 2022 paper) -- none of it is copied from either dataset, it's a
  generative model built to have the same *shape* of signal.

account-level features (follower counts, account age, etc.) are drawn from
distributions loosely shaped like what's reported for bot vs human accounts
in the bot-detection literature, but the numbers themselves are made up.
"""

import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class SyntheticGraphConfig:
    n_organic: int = 6000
    n_bot_clusters: int = 25
    cluster_size_min: int = 8
    cluster_size_max: int = 35
    n_targets: int = 400
    time_span_days: float = 60.0
    organic_follows_per_day: float = 0.7
    burst_window_hours: float = 4.0
    cluster_reactivations_min: int = 2
    cluster_reactivations_max: int = 5
    seed: int = 13


class SyntheticTemporalGraph:
    """generates node features, a chronological edge stream, and labels."""

    def __init__(self, config: SyntheticGraphConfig = None):
        self.cfg = config or SyntheticGraphConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def _circadian_rate(self, hour_of_day: np.ndarray) -> np.ndarray:
        # peaks around 8pm, dips around 4am -- rough approximation of when
        # people actually pick up their phone and follow someone
        return 0.35 + 0.65 * np.clip(np.sin((hour_of_day - 5) / 24 * 2 * np.pi), 0, None)

    def _organic_edges_for_node(self, node_id: int, start_t: float) -> list:
        cfg = self.cfg
        total_seconds = cfg.time_span_days * 86400
        base_rate_per_sec = cfg.organic_follows_per_day / 86400
        # each person has their own baseline activity level
        personal_multiplier = self.rng.lognormal(mean=0.0, sigma=0.6)

        edges = []
        t = start_t
        while t < total_seconds:
            hour = (t / 3600) % 24
            rate = base_rate_per_sec * personal_multiplier * self._circadian_rate(np.array([hour]))[0]
            rate = max(rate, 1e-8)
            dt = self.rng.exponential(1.0 / rate)
            t += dt
            if t >= total_seconds:
                break
            target = self.rng.integers(0, self.cfg.n_targets)
            edges.append((node_id, target, t))
        return edges

    def _bot_cluster_edges(self, node_ids: list, start_t: float) -> list:
        cfg = self.cfg
        total_seconds = cfg.time_span_days * 86400
        n_reactivations = self.rng.integers(
            cfg.cluster_reactivations_min, cfg.cluster_reactivations_max + 1
        )
        # pick a handful of shared targets this cluster exists to boost
        shared_targets = self.rng.choice(
            cfg.n_targets, size=self.rng.integers(2, 6), replace=False
        )

        edges = []
        activation_times = np.sort(
            self.rng.uniform(start_t, total_seconds * 0.9, size=n_reactivations)
        )
        for activation_t in activation_times:
            window_sec = cfg.burst_window_hours * 3600
            for node_id in node_ids:
                # not every account fires on every reactivation, but most do
                if self.rng.random() < 0.85:
                    jitter = self.rng.uniform(0, window_sec)
                    t = activation_t + jitter
                    if t >= total_seconds:
                        continue
                    target = self.rng.choice(shared_targets)
                    edges.append((node_id, int(target), t))
        return edges

    def _node_features(self, is_bot: np.ndarray) -> torch.Tensor:
        n = len(is_bot)
        rng = self.rng

        followers = np.where(
            is_bot,
            rng.lognormal(mean=2.5, sigma=1.0, size=n),
            rng.lognormal(mean=4.5, sigma=1.8, size=n),
        )
        following = np.where(
            is_bot,
            rng.lognormal(mean=4.0, sigma=0.7, size=n),
            rng.lognormal(mean=3.5, sigma=1.3, size=n),
        )
        tweets = np.where(
            is_bot,
            rng.lognormal(mean=3.0, sigma=1.2, size=n),
            rng.lognormal(mean=5.0, sigma=1.5, size=n),
        )
        listed = np.where(
            is_bot,
            rng.lognormal(mean=0.3, sigma=0.8, size=n),
            rng.lognormal(mean=1.2, sigma=1.4, size=n),
        )
        verified = np.where(
            is_bot, rng.random(n) < 0.002, rng.random(n) < 0.03
        ).astype(float)
        account_age_days = np.where(
            is_bot,
            rng.uniform(5, 250, size=n),
            rng.uniform(60, 4000, size=n),
        )
        has_description = np.where(
            is_bot, rng.random(n) < 0.55, rng.random(n) < 0.9
        ).astype(float)
        has_url = np.where(
            is_bot, rng.random(n) < 0.15, rng.random(n) < 0.35
        ).astype(float)
        name_len = np.where(
            is_bot,
            rng.integers(4, 16, size=n).astype(float),
            rng.integers(3, 22, size=n).astype(float),
        )
        username_len = np.where(
            is_bot,
            rng.integers(8, 15, size=n).astype(float),
            rng.integers(4, 15, size=n).astype(float),
        )
        default_profile_image = np.where(
            is_bot, rng.random(n) < 0.25, rng.random(n) < 0.02
        ).astype(float)

        def log1p_safe(x):
            return np.log1p(np.clip(x, 0, None))

        raw = np.column_stack([
            log1p_safe(followers),
            log1p_safe(following),
            log1p_safe(tweets),
            log1p_safe(listed),
            verified,
            log1p_safe(account_age_days),
            has_description,
            has_url,
            name_len,
            username_len,
            default_profile_image,
        ])

        x = torch.tensor(raw, dtype=torch.float)
        mean = x.mean(dim=0)
        std = x.std(dim=0) + 1e-8
        x = (x - mean) / std
        return x

    def generate(self):
        cfg = self.cfg
        n_bots_total = 0
        cluster_assignments = []

        node_counter = cfg.n_organic
        clusters = []
        for _ in range(cfg.n_bot_clusters):
            size = self.rng.integers(cfg.cluster_size_min, cfg.cluster_size_max + 1)
            ids = list(range(node_counter, node_counter + size))
            node_counter += size
            clusters.append(ids)
            n_bots_total += size

        n_total = node_counter
        is_bot = np.zeros(n_total, dtype=bool)
        for cluster in clusters:
            is_bot[cluster] = True

        x = self._node_features(is_bot)
        y = torch.tensor(is_bot.astype(np.int64))

        all_edges = []
        for node_id in range(cfg.n_organic):
            start_t = self.rng.uniform(0, cfg.time_span_days * 86400 * 0.3)
            all_edges.extend(self._organic_edges_for_node(node_id, start_t))

        for cluster in clusters:
            start_t = self.rng.uniform(0, cfg.time_span_days * 86400 * 0.5)
            all_edges.extend(self._bot_cluster_edges(cluster, start_t))

        all_edges.sort(key=lambda e: e[2])

        src = torch.tensor([e[0] for e in all_edges], dtype=torch.long)
        dst = torch.tensor([e[1] + n_total for e in all_edges], dtype=torch.long)
        t = torch.tensor([e[2] for e in all_edges], dtype=torch.float)

        n_relation_types = 1
        hour = (t / 3600) % 24
        dow = (t / 86400) % 7
        hour_sin = torch.sin(2 * np.pi * hour / 24)
        hour_cos = torch.cos(2 * np.pi * hour / 24)
        dow_sin = torch.sin(2 * np.pi * dow / 7)
        relation_onehot = torch.ones(len(src), n_relation_types)
        edge_attr = torch.cat([
            relation_onehot,
            hour_sin.unsqueeze(1),
            hour_cos.unsqueeze(1),
            dow_sin.unsqueeze(1),
        ], dim=1)

        # target nodes get placeholder (zero) features -- they're only ever
        # destinations in this synthetic graph, not classified
        target_features = torch.zeros(cfg.n_targets, x.shape[1])
        x_full = torch.cat([x, target_features], dim=0)
        y_full = torch.cat([y, torch.full((cfg.n_targets,), -1, dtype=torch.long)])

        node_is_classifiable = torch.zeros(n_total + cfg.n_targets, dtype=torch.bool)
        node_is_classifiable[:n_total] = True

        return {
            "x": x_full,
            "y": y_full,
            "src": src,
            "dst": dst,
            "t": t,
            "edge_attr": edge_attr,
            "classifiable_mask": node_is_classifiable,
            "n_source_nodes": n_total,
            "n_target_nodes": cfg.n_targets,
            "bot_clusters": clusters,
            "is_bot": is_bot,
        }


def make_splits(classifiable_mask: torch.Tensor, y: torch.Tensor, seed: int = 13):
    """chronology-agnostic node split -- train/val/test over labeled nodes only"""
    rng = np.random.default_rng(seed)
    idx = torch.where(classifiable_mask)[0].numpy()
    rng.shuffle(idx)

    n = len(idx)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    def mask_from(indices, total):
        m = torch.zeros(total, dtype=torch.bool)
        m[indices] = True
        return m

    total = classifiable_mask.shape[0]
    return (
        mask_from(train_idx, total),
        mask_from(val_idx, total),
        mask_from(test_idx, total),
    )
