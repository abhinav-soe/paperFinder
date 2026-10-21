# AI-Powered Research Paper Finder

A research discovery tool that searches academic papers by **semantic meaning**, not just exact keyword matches. The application retrieves paper metadata from OpenAlex and compares keyword-based, embedding-based, and hybrid search approaches through information-retrieval metrics.

Built with Python, Sentence Transformers, FAISS/NumPy, BM25, and Streamlit, the project combines semantic search, metadata filtering, interactive relevance judgments, and quantitative search evaluation.

## Features

- **Semantic search:** Represent paper titles and abstracts as embeddings and rank papers using cosine similarity.
- **Keyword search:** Use BM25 as a traditional information-retrieval baseline.
- **Hybrid retrieval:** Combine semantic and keyword rankings using Reciprocal Rank Fusion (RRF).
- **Metadata filtering:** Narrow results using the metadata filters supported by the application.
- **Interactive evaluation:** Judge retrieved papers for test queries and compare retrieval methods.
- **Search metrics:** Calculate Precision@10, Recall@10, Mean Reciprocal Rank (MRR), nDCG@10, and Mean Average Precision (MAP).
- **Flexible indexing:** Use Sentence Transformers by default or a lightweight TF-IDF/SVD backend.
- **Efficient similarity search:** Use FAISS when installed, with an exact NumPy-based fallback.

## How It Works

The application follows a data collection, indexing, retrieval, and evaluation pipeline.

1. **Collect paper metadata:** `fetch_openalex.py` retrieves academic metadata from OpenAlex, including titles and abstracts where available. It supports cursor-based pagination, retries, and API cost tracking.
2. **Build the search index:** `build_index.py` processes the collected records and creates vector representations using the configured embedding backend.
3. **Retrieve relevant papers:** The search engine supports three retrieval strategies:
   - **Keyword:** BM25 ranks papers based on lexical matching.
   - **Semantic:** Sentence embeddings represent paper text, and cosine similarity ranks papers by meaning.
   - **Hybrid:** Reciprocal Rank Fusion combines the keyword and semantic rankings.
4. **Apply filters and display results:** The Streamlit interface lets users search, compare methods, and narrow results using supported metadata filters.
5. **Evaluate retrieval quality:** Users judge results for test queries, and the evaluation pipeline calculates standard information-retrieval metrics.

## Search Methods

| Method | Approach | Purpose |
|---|---|---|
| BM25 | Lexical term matching | Keyword-search baseline |
| Semantic | Text embeddings and cosine similarity | Retrieve conceptually related papers |
| Hybrid | Reciprocal Rank Fusion (RRF) | Combine lexical and semantic rankings |

## Tech Stack

- **Language:** Python
- **Data source:** OpenAlex API
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Similarity search:** FAISS or NumPy
- **Keyword retrieval:** BM25
- **Alternative embedding backend:** TF-IDF with SVD
- **Interface:** Streamlit
- **Evaluation:** Precision@10, Recall@10, MRR, nDCG@10, MAP
- **Testing:** pytest

## Prerequisites

- Python installed in a version compatible with the project's dependencies
- An OpenAlex API key for data collection
- Internet access to retrieve paper metadata and download the embedding model on first use

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

Replace the placeholders with your repository URL and local directory name.

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Configure OpenAlex access

Obtain an API key from [OpenAlex settings](https://openalex.org/settings/api).

You can supply the key directly to the collection script:

```bash
python fetch_openalex.py --api-key YOUR_KEY
```

Alternatively, configure the `OPENALEX_API_KEY` environment variable using your operating system's environment-variable mechanism.

Do not commit API keys or other secrets to GitHub.

### 4. Fetch the paper metadata

Run:

```bash
python fetch_openalex.py --api-key YOUR_KEY
```

The script downloads and cleans the metadata used by the search application. If you use the environment variable, omit the `--api-key YOUR_KEY` argument.

### 5. Build the search index

Using the default embedding backend:

```bash
python build_index.py
```

The first run downloads the Sentence Transformers model, approximately 90 MB, and builds the index and 2-D map coordinates.

If PyTorch installation or compatibility is a problem, use the lightweight alternative:

```bash
python build_index.py --backend tfidf
```

This alternative uses TF-IDF/SVD and may provide lower-quality semantic retrieval than the default embedding model.

### 6. Launch the application

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit in your terminal.

### 7. Run the tests

```bash
python -m pytest
```

## Evaluation Workflow

The project includes an interactive workflow for comparing keyword, semantic, and hybrid retrieval.

1. Start the application and open the **Evaluation** tab.
2. Review the pooled search results for each test query.
3. Judge the relevance of the retrieved papers and save the judgments.
4. Run the evaluation script:

```bash
python evaluate.py
```

The application can also display evaluation charts in the Evaluation tab.

### Metrics

| Metric | What it measures |
|---|---|
| Precision@10 | Proportion of the top 10 results judged relevant |
| Recall@10 | Proportion of known relevant papers retrieved in the top 10 |
| MRR | Rank of the first relevant result, averaged across queries |
| nDCG@10 | Ranking quality with greater weight for relevant results near the top |
| MAP | Average precision across the evaluated queries |

The comparison uses relevance judgments collected from pooled search results. See the limitations section for important caveats when interpreting the scores.

## Project Structure

| Path | Purpose |
|---|---|
| `fetch_openalex.py` | Download and clean metadata with cursor paging, retries, and cost tracking |
| `build_index.py` | Generate embeddings and save vectors and 2-D map coordinates |
| `finder/embedders.py` | Sentence Transformers and TF-IDF/SVD embedding backends |
| `finder/vector_index.py` | FAISS/NumPy cosine search and metadata filtering |
| `finder/keyword.py` | BM25 keyword-search baseline |
| `finder/engine.py` | Semantic, keyword, and hybrid retrieval with filters |
| `finder/metrics.py` | Information-retrieval metrics and judgment pooling |
| `app.py` | Streamlit search, comparison, and evaluation interface |
| `eval/queries.json` | Test queries used for search evaluation |
| `evaluate.py` | Run the retrieval evaluation workflow |
| `requirements.txt` | Python dependencies |

## Design Decisions and Limitations

- **Abstract-level retrieval:** The index uses paper titles and abstracts rather than full-text PDFs. Results depend on the available metadata.
- **Default embedding model:** The project uses `all-MiniLM-L6-v2`; the model can be changed using the supported `--model` option.
- **Optional FAISS:** Installing `faiss-cpu` enables FAISS-based search. Without it, the application falls back to exact NumPy search.
- **Evaluation scope:** Relevance judgments use pooled results, and unjudged papers count as non-relevant. With approximately 12 test queries, comparisons are indicative rather than statistically conclusive.
- **Alternative backend:** TF-IDF/SVD provides a lighter indexing option but may reduce retrieval quality compared with the default embedding model.

## Future Improvements

Potential extensions include:

- Re-ranking retrieved papers with a cross-encoder.
- Evaluating domain-specific embedding models such as SPECTER2.
- Expanding the test-query set and relevance judgments.
- Adding a retrieval-augmented question-answering layer over paper abstracts.
- Improving evaluation coverage and reproducibility across datasets.

## License and Attribution

Add the appropriate license information for this repository and follow the OpenAlex API's applicable terms and attribution requirements.
