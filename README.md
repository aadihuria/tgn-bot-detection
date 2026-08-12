# temporal graph network for coordinated inauthentic behavior detection

live demo: https://aadihuria.github.io/tgn-bot-detection/

## the problem

Meta reported removing 687 million fake accounts in a single quarter of 2025, and fake
account prevalence has stayed roughly flat at 3-4% of monthly active users despite that —
generative AI made it cheap to spin up accounts that pass individual review. Most detection
systems still score one account at a time: profile photo, post frequency, follower ratio.
A bot operator who knows what those checks look for can build an account that clears every
one of them.

What's harder to fake is the *shape* of a network's behavior over time. A real account
accumulates connections gradually, through overlapping interests and organic interaction.
A bot cluster gets deployed, and thirty accounts follow the same handful of targets inside
a six-hour window, go quiet for two weeks, then reactivate together the moment something
starts trending. No single account in that cluster looks suspicious. The synchronized
timing across the group is the signal, and it's invisible to anything that evaluates
accounts in isolation or looks at a single static snapshot of the graph.

This project builds two things that go after that timing signal directly:

1. a **temporal graph network (TGN)** that scores individual accounts using a memory
   vector updated as connection events arrive, so a node's representation depends on its
   *history*, not just its current neighborhood
2. a **burst coordination detector** that looks at clusters of accounts and flags groups
   whose combined connection timing is statistically implausible for organic behavior

## about the data — read this before the results below

TwiBot-20 and TwiBot-22, the standard benchmarks for this task, both require emailing the
dataset authors and waiting for manual approval. That access hasn't come through as of this
writing. Rather than wait, everything in this repo was built and evaluated against a
**synthetic temporal graph generator** (`data/synthetic_generator.py`) written for this
project. It's seeded and reproducible, and it's deliberately built so account-level features
overlap heavily between bots and humans — the whole premise here is that per-account checks
should *not* be enough to solve this, so the generator doesn't hand that signal to the
model. Bot clusters get away with looking normal individually; only their timing gives them
away.

`data/pipeline.py` is a second loader written against the real TwiBot-20/22 file format
(`user.json` / `edge.csv` / `label.csv` / `split.csv`) so that swapping in the real dataset,
once access arrives, is a matter of pointing `train.py` at it instead of the generator — the
rest of the pipeline (feature engineering, burst features, model code) doesn't change.

**Every number below came from an actual run of `train.py` on this synthetic benchmark.**
Nothing here is projected or estimated. If the eventual TwiBot-22 numbers come out
differently — better or worse — that's what will get reported when that happens.

## architecture

```
synthetic generator / twibot loader
              │
              ▼
   feature engineering + burst features
   (data/pipeline.py — log-normalized counts,
   circadian time features, sliding-window burst stats)
              │
     ┌────────┴────────┐
     ▼                  ▼
graphsage baseline   tgn (memory + attention)
(static snapshot,    (chronological training,
 no time ordering)    node memory per account)
     │                  │
     └────────┬─────────┘
              ▼
   early detection eval
   (checkpoints at 10-100%
   of the observed stream)

burst coordination detector (separate path)
   co-follow graph → connected components →
   median inter-arrival gate → flagged clusters
              │
              ▼
       fastapi server (api/server.py)
       + static demo (docs/index.html)
```

## results (synthetic benchmark, seed=13)

7,028 accounts, 123,690 temporal edges, 628 bots across 25 coordinated clusters.

| metric | graphsage (static) | tgn (temporal) |
|---|---|---|
| f1 (macro, full stream) | 0.488 | 0.521 |
| roc-auc (full stream) | 0.597 | 0.959 |

auc is the metric to trust here over f1 — with bots at ~9% of labeled nodes, f1 at a fixed
0.5 decision threshold is a noisy statistic regardless of how good the underlying ranking
is. auc doesn't have that problem, and the gap there (0.60 static vs 0.96 temporal) is the
actual story: a model that can't see event ordering barely beats random on this benchmark,
one that can gets close to a clean separation.

(the tgn test-set count in `results/summary.json` is larger than the raw node count because
metrics are computed over every time a labeled node shows up in the chronological walk, not
deduplicated to one score per node — a account gets re-scored each time new edges touch it.)

**burst detector:** caught 18 of 25 injected bot clusters (72% recall) with 2 false
positives out of 19 flagged clusters, using nothing but connection timing — no node
features, no model weights, just the co-follow graph and a median inter-arrival threshold.

**early detection**, checking the tgn against the same static baseline at increasing
fractions of the observed edge stream:

![early detection curve](docs/assets/early_detection_curve.png)

the tgn's auc is already at 0.96+ by the 17% checkpoint and stays roughly flat from there —
on this benchmark, the coordinated bursts that give bot clusters away happen early and
repeatedly, so there isn't much of a "wait longer to be more sure" curve to show. that's
itself a reasonably interesting result and is worth re-checking once this runs on TwiBot-22,
where bot behavior is less scripted than a synthetic generator's.

## repo layout

```
data/
  synthetic_generator.py   synthetic temporal graph + labels
  pipeline.py               real twibot-20/22 loader + shared feature engineering
models/
  graphsage_baseline.py     static gnn baseline
  tgn_bot_detector.py       temporal graph network
detectors/
  burst_detector.py         cofollow graph + coordination scoring
evaluation/
  early_detection.py        checkpointed temporal evaluation
  export_demo_data.py       dumps results into docs/data/demo.json
api/
  server.py                 fastapi serving layer
  Dockerfile
docs/                        github pages demo (static, no backend)
tests/
train.py                     runs the whole thing end to end
```

## running it

```
pip install -r requirements.txt
python train.py                        # generates data, trains both models, runs eval
python -m evaluation.export_demo_data  # regenerates docs/data/demo.json
pytest tests/
```

everything is CPU-only and finishes in a few minutes on a laptop — the synthetic benchmark
is sized at ~7k nodes / ~125k edges specifically so a full run doesn't require a GPU.

running the api locally:

```
uvicorn api.server:app --reload
```

`POST /detect` with a list of accounts and temporal edges runs the burst detector over the
submitted subgraph and returns flagged accounts and clusters.

## what's still open

- **real dataset.** waiting on TwiBot-22 access (emailed the dataset authors). the loader
  in `data/pipeline.py` is ready for it; nothing else in the pipeline needs to change.
- **hosted api.** `api/server.py` + `api/Dockerfile` + `railway.json` are ready to deploy,
  that just hasn't been done yet — it's one `railway up` away.
- **tgn checkpoint in the api.** the live `/detect` endpoint currently runs the burst
  detector only, not the trained tgn, since there's no model checkpoint persistence yet and
  the honest thing was to not fake loading weights that don't exist on disk.
