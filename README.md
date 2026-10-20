# AI-Powered Research Paper Finder

Semantic search over research papers (OpenAlex): embeddings + cosine similarity, filters, and a measured
comparison against keyword (BM25) search.

## Setup & run
```bash
python -m pip install -r requirements.txt
python fetch_openalex.py --api-key YOUR_KEY     # free key: https://openalex.org/settings/api  (or set OPENALEX_API_KEY)
python build_index.py                           # embeds papers (downloads the model ~90MB on first run)
python -m streamlit run app.py
python -m pytest                                # tests
```
No PyTorch / install trouble? `python build_index.py --backend tfidf` builds a lightweight (lower quality) index.
Optional: `pip install faiss-cpu` - the app auto-uses FAISS, otherwise exact NumPy search (identical results).

## Evaluation workflow
1. Open the app's **Evaluation** tab, judge each test query (blind, pooled results), save.
2. `python evaluate.py` (or view the chart in the tab): Precision@10, Recall@10, MRR, nDCG@10, MAP for
   keyword vs semantic vs hybrid. Edit `eval/queries.json` or add queries in the app.

## Structure
| Path | Purpose |
|---|---|
| `fetch_openalex.py` | Download + clean metadata (cursor paging, retries, cost tracking) |
| `build_index.py` | Embed papers, save vectors, 2-D map coordinates |
| `finder/embedders.py` | Sentence-Transformers (default) and TF-IDF/SVD fallback |
| `finder/vector_index.py` | FAISS / NumPy cosine index with metadata filtering |
| `finder/keyword.py` | BM25 baseline |
| `finder/engine.py` | Semantic, keyword and hybrid (Reciprocal Rank Fusion) search + filters |
| `finder/metrics.py` | IR metrics and judgment pooling |
| `app.py` | Streamlit UI (search, compare, evaluation) |

## Design notes / limitations
- Only titles + abstracts are indexed (no full text). Model: `all-MiniLM-L6-v2`; swap via `--model`.
- Relevance judgments use pooling over the top results; unjudged papers count as non-relevant, and with ~12 queries
  the comparison is indicative rather than statistically conclusive.
- Next steps: re-ranking with a cross-encoder, SPECTER2 embeddings, a RAG Q&A layer over the abstracts.
