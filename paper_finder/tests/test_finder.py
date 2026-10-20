import numpy as np
import pandas as pd
import pytest

from fetch_openalex import clean, parse, rebuild_abstract
from finder.embedders import TfidfSvdEmbedder
from finder.engine import SearchEngine
from finder.keyword import BM25
from finder.metrics import (average_precision, ndcg_at_k, pool, precision_at_k, recall_at_k, reciprocal_rank)
from finder.vector_index import VectorIndex

TOPICS = {
    "news": "fake news misinformation detection social media posts credibility claims verification",
    "vision": "image segmentation convolutional network medical scans organs pixels labelling",
    "rl": "reinforcement learning agent reward policy environment trial exploration",
    "speech": "speech recognition audio acoustic transcription spoken words waveform",
}


def corpus():
    rows = []
    for t, (name, words) in enumerate(TOPICS.items()):
        w = words.split()
        for j in range(5):
            rows.append(dict(paper_id=f"{name}{j}", title=f"{name} study {j} " + " ".join(w[j:j + 3]),
                             abstract=" ".join(w[j:] + w[:j]) * 2 + f" extra{j}", authors="A", year=2018 + t + j % 2,
                             venue="V" + name, field="CS", topic=name, cited_by_count=10 * j, doi="", url="u"))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def engine():
    df = corpus()
    texts = (df.title + ". " + df.abstract).tolist()
    emb = TfidfSvdEmbedder(dim=16).fit(texts)
    return SearchEngine(df, emb.encode(texts), emb, backend="numpy")


def test_rebuild_abstract_and_parse():
    inv = {"Deep": [0], "learning": [1, 3], "is": [2], "fun": [4]}
    assert rebuild_abstract(inv) == "Deep learning is learning fun"
    assert rebuild_abstract(None) == ""
    w = {"id": "https://openalex.org/W42", "title": "T", "abstract_inverted_index": inv, "publication_year": 2020,
         "cited_by_count": 3, "authorships": [{"author": {"display_name": "Ada"}}],
         "primary_location": {"source": {"display_name": "Nature"}},
         "primary_topic": {"display_name": "AI", "field": {"display_name": "CS"}}}
    p = parse(w, "seed")
    assert p["paper_id"] == "W42" and p["venue"] == "Nature" and p["field"] == "CS" and p["authors"] == "Ada"


def test_clean_drops_short_and_duplicate():
    df = pd.DataFrame([dict(paper_id="a", title="X", abstract="word " * 50), dict(paper_id="a", title="X", abstract="word " * 50),
                       dict(paper_id="b", title="Y", abstract="short"), dict(paper_id="c", title="x!", abstract="word " * 50)])
    assert clean(df).paper_id.tolist() == ["a"]


def test_vector_index_mask_and_backends_agree():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(200, 16)).astype("float32")
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    q, mask = E[3], np.arange(200) % 2 == 0
    i_np, s_np = VectorIndex(E, "numpy").search(q, 5, mask)
    assert mask[i_np].all() and (np.diff(s_np) <= 1e-6).all()
    faiss = pytest.importorskip("faiss")
    i_fa, s_fa = VectorIndex(E, "faiss").search(q, 5, mask)
    assert list(i_np) == list(i_fa) and np.allclose(s_np, s_fa, atol=1e-5)


def test_bm25_prefers_matching_documents():
    bm = BM25(["fake news detection", "image segmentation", "speech audio transcription"])
    idx, _ = bm.search("detection of fake news", 2)
    assert idx[0] == 0


def test_semantic_search_finds_topic(engine):
    res = engine.search("misinformation claims verification", "semantic", 5)
    assert (res.topic == "news").sum() >= 4 and res.score.is_monotonic_decreasing


def test_filters_apply_to_all_methods(engine):
    m = engine.mask(year_range=(2018, 2019), venue="vnews")
    for method in ("semantic", "keyword", "hybrid"):
        res = engine.search("news detection", method, 10, m)
        assert len(res) > 0 and res.year.between(2018, 2019).all() and (res.topic == "news").all()


def test_hybrid_unique_and_sized(engine):
    res = engine.search("reinforcement learning reward", "hybrid", 8)
    assert len(res) == 8 and res.paper_id.is_unique


def test_empty_query_returns_empty(engine):
    assert engine.search("   ", "semantic", 5).empty


def test_ir_metrics_known_values():
    ranked, rel = ["a", "b", "c", "d"], {"a", "c"}
    assert precision_at_k(ranked, rel, 2) == 0.5 and recall_at_k(ranked, rel, 2) == 0.5
    assert reciprocal_rank(["x", "a"], rel) == 0.5
    assert average_precision(ranked, rel) == pytest.approx((1 + 2 / 3) / 2)
    assert ndcg_at_k(ranked, rel, 4) == pytest.approx(1.5 / (1 + 1 / np.log2(3)))


def test_pool_is_deterministic_and_unique(engine):
    a = pool(engine, "q1", "speech recognition audio")
    assert a == pool(engine, "q1", "speech recognition audio") and len(a) == len(set(a))
