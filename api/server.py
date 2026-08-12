"""
FastAPI serving layer. Takes a small graph (accounts + temporal follow
edges), scores each account, and runs the burst detector over the same
edges to flag any coordinated clusters. Meant to be called with a subgraph
around a new account -- e.g. "here's everyone this account follows and
who else follows the same targets, is this thing a bot."

There's no trained-model checkpoint loading here on purpose: this repo's
synthetic benchmark is scoped for demonstrating the approach, not for
production account scoring, so this endpoint trains a lightweight burst
detector pass on the request graph itself rather than pretending to load
weights fit on real Twitter data it never saw. Swap in a real checkpoint
once TwiBot-22 access comes through and training runs on it.
"""

from typing import List, Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from detectors.burst_detector import BurstCoordinationDetector

app = FastAPI(title="Bot Detection API", version="0.1.0")


class Account(BaseModel):
    account_id: str
    followers_count: int = 0
    following_count: int = 0
    tweet_count: int = 0
    account_age_days: int = 0
    verified: bool = False


class Edge(BaseModel):
    source_id: str
    target_id: str
    timestamp: str


class GraphInput(BaseModel):
    accounts: List[Account]
    edges: List[Edge]


class ClusterResult(BaseModel):
    account_ids: List[str]
    cluster_size: int
    median_inter_arrival_sec: float
    coordination_score: float


class DetectionResult(BaseModel):
    flagged_accounts: List[str]
    suspicious_clusters: List[ClusterResult]
    summary: Dict


@app.post("/detect", response_model=DetectionResult)
async def detect_bots(graph: GraphInput):
    if len(graph.accounts) < 2:
        raise HTTPException(status_code=400, detail="need at least 2 accounts")

    id_to_idx = {a.account_id: i for i, a in enumerate(graph.accounts)}
    idx_to_id = {i: a.account_id for i, a in enumerate(graph.accounts)}

    src, dst, t = [], [], []
    for e in graph.edges:
        if e.source_id not in id_to_idx or e.target_id not in id_to_idx:
            continue
        try:
            import datetime
            ts = datetime.datetime.fromisoformat(e.timestamp.replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
        src.append(id_to_idx[e.source_id])
        dst.append(id_to_idx[e.target_id])
        t.append(ts)

    if len(src) < 2:
        return DetectionResult(
            flagged_accounts=[], suspicious_clusters=[],
            summary={"total_accounts": len(graph.accounts), "flagged_accounts": 0, "suspicious_clusters": 0},
        )

    order = np.argsort(t)
    src = np.array(src)[order]
    dst = np.array(dst)[order]
    t = np.array(t)[order]

    detector = BurstCoordinationDetector(min_cluster_size=2, min_shared_events=2, min_events=4)
    G = detector.build_cofollow_graph(src, dst, t)
    clusters = detector.detect_coordinated_clusters(G, src, t)

    cluster_results = [
        ClusterResult(
            account_ids=[idx_to_id[n] for n in c["node_ids"] if n in idx_to_id],
            cluster_size=c["cluster_size"],
            median_inter_arrival_sec=c["median_inter_arrival_sec"],
            coordination_score=c["coordination_score"],
        )
        for c in clusters
    ]

    flagged_accounts = sorted({aid for c in cluster_results for aid in c.account_ids})

    return DetectionResult(
        flagged_accounts=flagged_accounts,
        suspicious_clusters=cluster_results,
        summary={
            "total_accounts": len(graph.accounts),
            "flagged_accounts": len(flagged_accounts),
            "suspicious_clusters": len(cluster_results),
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
