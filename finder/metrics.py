"""Information-retrieval metrics with binary relevance."""
import json
import os
import random

import numpy as np
import pandas as pd


def precision_at_k(ranked, rel, k):
    return sum(d in rel for d in ranked[:k]) / k


def recall_at_k(ranked, rel, k):
    return sum(d in rel for d in ranked[:k]) / len(rel)


def reciprocal_rank(ranked, rel):
    return next((1 / (i + 1) for i, d in enumerate(ranked) if d in rel), 0.0)


def average_precision(ranked, rel):
    hits, total = 0, 0.0
    for i, d in enumerate(ranked):
        if d in rel:
            hits += 1
            total += hits / (i + 1)
    return total / len(rel)


def ndcg_at_k(ranked, rel, k):
    dcg = sum(1 / np.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in rel)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal else 0.0


def evaluate_methods(engine, queries, judgments, methods=("keyword", "semantic", "hybrid"), k=10, depth=50):
    """Per-query metrics. Unjudged documents count as non-relevant (standard pooled-judgment practice)."""
    rows = []
    for q in queries:
        rel = {pid for pid, v in judgments.get(q["id"], {}).items() if v}
        if not rel:
            continue
        for m in methods:
            ranked = engine.rank_ids(q["text"], m, depth)
            rows.append({"query_id": q["id"], "query": q["text"], "method": m, "n_relevant": len(rel),
                         f"P@{k}": precision_at_k(ranked, rel, k), f"R@{k}": recall_at_k(ranked, rel, k),
                         "MRR": reciprocal_rank(ranked, rel), f"nDCG@{k}": ndcg_at_k(ranked, rel, k),
                         "MAP": average_precision(ranked, rel)})
    per_query = pd.DataFrame(rows)
    if per_query.empty:
        return per_query, per_query
    summary = per_query.drop(columns=["query_id", "query", "n_relevant"]).groupby("method").mean()
    return per_query, summary


def pool(engine, query_id, query, methods=("keyword", "semantic", "hybrid"), depth=10):
    """Union of each method's top results, shuffled so the annotator can't tell which method found what."""
    ids = list(dict.fromkeys(i for m in methods for i in engine.rank_ids(query, m, depth)))
    random.Random(query_id).shuffle(ids)
    return ids


def load_json(path, default):
    return json.load(open(path)) if os.path.exists(path) else default


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2)
