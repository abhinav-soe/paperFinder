"""Embed every paper and save the vector index.
  python build_index.py                      # Sentence Transformers (recommended, downloads ~90MB model once)
  python build_index.py --backend tfidf      # lightweight offline fallback (lower quality, no PyTorch needed)"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.decomposition import PCA

import config as C
from finder.data import load_papers, paper_texts
from finder.embedders import make_embedder

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["st", "tfidf"], default="st")
ap.add_argument("--model", default=C.DEFAULT_MODEL)
ap.add_argument("--data", default=C.DATA_PATH)
a = ap.parse_args()

papers = load_papers(a.data)
texts = paper_texts(papers)
emb = make_embedder(a.backend, a.model)
t0 = time.time()
print(f"Embedding {len(texts)} papers with {emb.config()['model']} ...")
E = emb.fit(texts).encode(texts, show_progress=True)

os.makedirs(C.INDEX_DIR, exist_ok=True)
np.save(f"{C.INDEX_DIR}/embeddings.npy", E)
np.save(f"{C.INDEX_DIR}/coords.npy", PCA(2, random_state=0).fit_transform(E).astype("float32"))
papers.to_csv(f"{C.INDEX_DIR}/papers.csv", index=False)       # snapshot keeps row order aligned with vectors
emb.save(C.INDEX_DIR)
json.dump(dict(**emb.config(), n_papers=len(papers), dim=int(E.shape[1]),
               build_seconds=round(time.time() - t0, 1)), open(f"{C.INDEX_DIR}/meta.json", "w"), indent=2)
print(f"Done in {time.time() - t0:.0f}s -> {C.INDEX_DIR}/  (dim={E.shape[1]})")
