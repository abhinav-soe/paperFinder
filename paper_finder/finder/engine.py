import os
from collections import defaultdict

import numpy as np

from finder.data import load_papers, paper_texts
from finder.embedders import load_embedder
from finder.keyword import BM25
from finder.vector_index import VectorIndex

METHODS = ("semantic", "keyword", "hybrid")


class SearchEngine:
    def __init__(self, papers, embeddings, embedder, backend="auto"):
        self.papers = papers.reset_index(drop=True)
        self.embedder = embedder
        self.index = VectorIndex(embeddings, backend)
        self.bm25 = BM25(paper_texts(self.papers))

    def mask(self, year_range=None, fields=None, topics=None, venue="", min_citations=0):
        p, m = self.papers, np.ones(len(self.papers), bool)
        if year_range:
            m &= p.year.between(*year_range).values
        if fields:
            m &= p.field.isin(fields).values
        if topics:
            m &= p.topic.isin(topics).values
        if venue:
            m &= p.venue.str.contains(venue, case=False, regex=False).values
        if min_citations:
            m &= (p.cited_by_count >= min_citations).values
        return m

    def _semantic(self, query, k, mask):
        return self.index.search(self.embedder.encode([query])[0], k, mask)

    def _hybrid(self, query, k, mask, depth=100, rrf_k=60):
        """Reciprocal Rank Fusion of the semantic and BM25 rankings."""
        rrf = defaultdict(float)
        for ranking in (self._semantic(query, depth, mask)[0], self.bm25.search(query, depth, mask)[0]):
            for r, i in enumerate(ranking):
                rrf[int(i)] += 1 / (rrf_k + r + 1)
        order = sorted(rrf, key=rrf.get, reverse=True)[:k]
        return np.array(order, dtype=int), np.array([rrf[i] for i in order])

    def search(self, query, method="semantic", k=10, mask=None):
        fn = {"semantic": self._semantic, "hybrid": self._hybrid, "keyword": self.bm25.search}[method]
        if not query.strip():
            idx, sc = np.array([], dtype=int), np.array([])
        else:
            idx, sc = fn(query, k, mask)
        df = self.papers.iloc[idx].copy()
        df.insert(0, "rank", np.arange(1, len(idx) + 1))
        df.insert(1, "score", sc)
        df["paper_idx"] = idx
        return df.reset_index(drop=True)

    def rank_ids(self, query, method, k=50, mask=None):
        return self.search(query, method, k, mask).paper_id.tolist()


def load_engine(index_dir="index", backend="auto"):
    embedder, meta = load_embedder(index_dir)
    papers = load_papers(os.path.join(index_dir, "papers.csv"))
    emb = np.load(os.path.join(index_dir, "embeddings.npy"))
    return SearchEngine(papers, emb, embedder, backend), meta
