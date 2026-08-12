"""
Coordination detector that operates on clusters of accounts, not individual
nodes. The TGN scores individual accounts; this catches something different
-- groups of accounts that never even follow each other directly, but all
pile onto the same handful of targets inside a suspiciously tight window.

Method: build a co-follow graph where two accounts get an edge if they both
followed the same target within `window_hours` of each other (and did it
more than once, so it's not just coincidence), then look at each connected
component's combined activity. The median inter-arrival time across the
cluster's events is what actually separates bots from noise in practice --
a coordinated cluster's follows land close together in time even though the
cluster goes dormant for long stretches between reactivations, and the
median isn't dragged around by those dormant gaps the way a mean is. KL
divergence against a modeled organic timing distribution is computed too
and reported alongside as a secondary signal, but the median threshold is
what gates a flag.
"""

from collections import defaultdict
from typing import Dict, List

import networkx as nx
import numpy as np
from scipy.stats import entropy


class BurstCoordinationDetector:
    def __init__(self, window_hours: float = 4.0, median_inter_arrival_threshold_sec: float = 2500.0,
                 min_cluster_size: int = 8, min_shared_events: int = 2, min_events: int = 10):
        self.window_sec = window_hours * 3600
        self.min_cluster_size = min_cluster_size
        # need enough events to get a meaningful median/kl estimate at all --
        # tunable because the live api deals with much smaller subgraphs
        # than the benchmark run does
        self.min_events = min_events
        # two accounts landing on the same target once, within a few hours,
        # happens by pure chance all the time in a population this size --
        # only draw a co-follow edge once a pair has done it
        # min_shared_events times, which is much less likely organically
        self.min_shared_events = min_shared_events
        # median (not mean) inter-arrival time within a cluster's combined
        # activity is what actually separates bot bursts from organic
        # traffic in practice -- mean gets dragged around by the long dormant
        # gaps between a bot cluster's reactivations, median doesn't. this
        # threshold was picked by looking at where the two populations split
        # on the synthetic benchmark, see notebooks / README for the numbers
        self.median_threshold = median_inter_arrival_threshold_sec

    def build_cofollow_graph(self, src: np.ndarray, dst: np.ndarray, t: np.ndarray) -> nx.Graph:
        target_followers: Dict[int, List] = defaultdict(list)
        for i in range(len(src)):
            target_followers[int(dst[i])].append((int(src[i]), float(t[i])))

        pair_weight: Dict[tuple, int] = defaultdict(int)
        for target, followers in target_followers.items():
            if len(followers) < 2:
                continue
            followers.sort(key=lambda x: x[1])
            for i in range(len(followers)):
                for j in range(i + 1, len(followers)):
                    u, t_u = followers[i]
                    v, t_v = followers[j]
                    if t_v - t_u > self.window_sec:
                        break
                    if u == v:
                        continue
                    key = (u, v) if u < v else (v, u)
                    pair_weight[key] += 1

        G = nx.Graph()
        for (u, v), weight in pair_weight.items():
            if weight >= self.min_shared_events:
                G.add_edge(u, v, weight=weight)
        return G

    def _organic_inter_arrival_histogram(self, n_samples: int = 8000) -> np.ndarray:
        """
        rough model of what "a bunch of real people happen to follow the
        same popular account" looks like timing-wise: circadian-modulated
        poisson process, same shape used by the synthetic data generator's
        organic accounts (kept independent on purpose -- this isn't allowed
        to just memorize the generator's exact parameters).
        """
        base_rate = 4.0 / 86400
        times, t = [], 0.0
        for _ in range(n_samples):
            hour = (t / 3600) % 24
            rate = base_rate * (3.0 if 8 <= hour <= 22 else 0.4)
            dt = np.random.exponential(1.0 / rate)
            times.append(dt)
            t += dt

        bins = np.logspace(0, 7, 40)
        hist, _ = np.histogram(times, bins=bins, density=True)
        return hist + 1e-10

    def coordination_score(self, inter_arrival_times: np.ndarray) -> float:
        if len(inter_arrival_times) < 5:
            return 0.0
        organic = self._organic_inter_arrival_histogram()
        bins = np.logspace(0, 7, 40)
        observed, _ = np.histogram(inter_arrival_times, bins=bins, density=True)
        observed = observed + 1e-10
        return float(entropy(observed, organic))

    def detect_coordinated_clusters(self, G: nx.Graph, src: np.ndarray, t: np.ndarray) -> List[Dict]:
        node_times: Dict[int, List[float]] = defaultdict(list)
        for i in range(len(src)):
            node_times[int(src[i])].append(float(t[i]))

        results = []
        for component in nx.connected_components(G):
            if len(component) < self.min_cluster_size:
                continue

            all_times = []
            for node in component:
                all_times.extend(node_times.get(node, []))
            if len(all_times) < self.min_events:
                continue

            all_times = sorted(all_times)
            inter_arrivals = np.diff(all_times)
            median_ia = float(np.median(inter_arrivals))

            if median_ia < self.median_threshold:
                kl_score = self.coordination_score(inter_arrivals)
                results.append({
                    "node_ids": sorted(component),
                    "cluster_size": len(component),
                    "median_inter_arrival_sec": round(median_ia, 2),
                    "mean_inter_arrival_sec": round(float(np.mean(inter_arrivals)), 2),
                    "coordination_score": round(kl_score, 3),
                    "n_events": len(all_times),
                })

        results.sort(key=lambda x: x["median_inter_arrival_sec"])
        return results
