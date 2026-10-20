"""Compare keyword vs semantic vs hybrid search on judged queries.
Label relevance first in the app (Evaluation tab), then run:  python evaluate.py"""
import os

import pandas as pd

import config as C
from finder.engine import load_engine
from finder.metrics import evaluate_methods, load_json

K = 10
engine, meta = load_engine(C.INDEX_DIR)
queries = load_json(f"{C.EVAL_DIR}/queries.json", [])
judgments = load_json(f"{C.EVAL_DIR}/judgments.json", {})
per_query, summary = evaluate_methods(engine, queries, judgments, k=K)
if per_query.empty:
    raise SystemExit("No judged queries yet - label some results in the app's Evaluation tab first.")
pd.set_option("display.width", 200)
print(f"Queries evaluated: {per_query.query_id.nunique()} / {len(queries)}  |  model: {meta['model']}\n")
print(summary.round(3).to_string())
per_query.to_csv(f"{C.EVAL_DIR}/results_per_query.csv", index=False)
summary.to_csv(f"{C.EVAL_DIR}/results_summary.csv")
