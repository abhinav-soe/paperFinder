import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as C
from finder.engine import load_engine
from finder.metrics import evaluate_methods, load_json, pool, save_json

st.set_page_config(page_title="AI Research Paper Finder", page_icon="📚", layout="wide")
METHODS = {"Semantic (embeddings)": "semantic", "Keyword (BM25)": "keyword", "Hybrid (semantic + keyword)": "hybrid"}
SCORE_LABEL = {"semantic": "cosine", "keyword": "BM25", "hybrid": "RRF"}
K_EVAL = 10


@st.cache_resource
def get_engine():
    return load_engine(C.INDEX_DIR)


@st.cache_data
def get_coords():
    return np.load(f"{C.INDEX_DIR}/coords.npy")


@st.cache_data
def run_eval(_engine, queries_json, judg_json):
    return evaluate_methods(_engine, json.loads(queries_json), json.loads(judg_json), k=K_EVAL)


try:
    engine, meta = get_engine()
except FileNotFoundError:
    st.error("No index found. Run `python fetch_openalex.py`, then `python build_index.py` first.")
    st.stop()
papers = engine.papers

# ------------------------------ sidebar ------------------------------
st.sidebar.title("📚 Paper Finder")
method_label = st.sidebar.radio("Ranking method", list(METHODS))
method = METHODS[method_label]
top_k = st.sidebar.slider("Results", 5, 30, 10)
st.sidebar.markdown("**Filters**")
ymin, ymax = int(papers.year[papers.year > 0].min()), int(papers.year.max())
years = st.sidebar.slider("Publication year", ymin, ymax, (ymin, ymax)) if ymin < ymax else (ymin, ymax)
fields = st.sidebar.multiselect("Field", sorted(f for f in papers.field.unique() if f))
topics = st.sidebar.multiselect("Topic", papers.topic[papers.topic != ""].value_counts().head(60).index.tolist())
venue = st.sidebar.text_input("Venue contains")
min_cit = st.sidebar.number_input("Min. citations", 0, 100000, 0, step=10)
mask = engine.mask(years, fields, topics, venue, min_cit)
st.sidebar.caption(f"{int(mask.sum()):,} of {len(papers):,} papers match filters")
st.sidebar.markdown("---")
st.sidebar.caption(f"Model: `{meta['model']}` · dim {meta['dim']}\n\nIndex: {engine.index.backend_name}")

st.title("AI-Powered Research Paper Finder")
tab_search, tab_cmp, tab_eval = st.tabs(["🔎 Search", "⚖️ Compare methods", "📏 Evaluation"])
queries = load_json(f"{C.EVAL_DIR}/queries.json", [])

# ------------------------------ search ------------------------------
if "q" not in st.session_state:
    st.session_state.q = ""


def set_q(t):
    st.session_state.q = t


with tab_search:
    st.text_input("Describe what you are looking for", key="q",
                  placeholder="e.g. detecting fake news using multimodal deep learning")
    if queries:
        cols = st.columns(3)
        for i, q in enumerate(queries[:3]):
            cols[i].button(q["text"], on_click=set_q, args=(q["text"],), key=f"ex{i}", width="stretch")
    query = st.session_state.q
    res = engine.search(query, method, top_k, mask)
    if not query.strip():
        st.info("Type a query above, or click an example. Try rephrasing it in completely different words "
                "and compare Semantic vs Keyword in the sidebar.")
    elif res.empty:
        st.warning("No results. Loosen the filters or rephrase the query.")
    else:
        left, right = st.columns([3, 2])
        with left:
            st.caption(f"Top {len(res)} results · {method_label}")
            for _, r in res.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{r['rank']}. [{r.title}]({r.url})**")
                    st.caption(f"{r.authors or 'Unknown authors'} · {r.venue or 'Unknown venue'} · {r.year} · "
                               f"{r.cited_by_count:,} citations · {SCORE_LABEL[method]} {r.score:.3f}")
                    st.write(r.abstract[:320] + ("…" if len(r.abstract) > 320 else ""))
                    with st.expander("Full abstract"):
                        st.write(r.abstract)
                        st.caption(f"Field: {r.field or '-'} · Topic: {r.topic or '-'}" + (f" · {r.doi}" if r.doi else ""))
        with right:
            c = get_coords()
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=c[:, 0], y=c[:, 1], mode="markers", name="All papers", text=papers.title,
                                       hoverinfo="text", marker=dict(size=4, color="lightgray", opacity=0.5)))
            pts = c[res.paper_idx.values]
            fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers+text", name="Results",
                                     text=res["rank"].astype(str), textposition="top center", hovertext=res.title,
                                     hoverinfo="text", marker=dict(size=12, color="#d9534f")))
            fig.update_layout(title="Where results sit in the paper collection (2-D PCA of embeddings)", height=380,
                              margin=dict(l=0, r=0, t=40, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False),
                              legend=dict(orientation="h", y=-0.05))
            st.plotly_chart(fig, width="stretch")
            fig = px.histogram(res, x="year", nbins=min(15, res.year.nunique() + 1), title="Results by publication year")
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=40, b=0), yaxis_title="papers")
            st.plotly_chart(fig, width="stretch")

# ------------------------------ compare ------------------------------
with tab_cmp:
    q = st.session_state.q
    if not q.strip():
        st.info("Enter a query in the Search tab first.")
    else:
        st.markdown(f"**Query:** {q}")
        out = {m: engine.search(q, m, K_EVAL, mask) for m in ("keyword", "semantic", "hybrid")}
        cols = st.columns(3)
        for col, (m, name) in zip(cols, [("keyword", "Keyword (BM25)"), ("semantic", "Semantic"), ("hybrid", "Hybrid")]):
            col.subheader(name)
            for _, r in out[m].head(5).iterrows():
                col.markdown(f"{r['rank']}. {r.title} *({r.year})*")
        ids = {m: set(d.paper_id) for m, d in out.items()}
        jac = len(ids["keyword"] & ids["semantic"]) / max(len(ids["keyword"] | ids["semantic"]), 1)
        st.metric(f"Overlap of keyword vs semantic top {K_EVAL} (Jaccard)", f"{jac:.0%}")
        only_sem = out["semantic"][~out["semantic"].paper_id.isin(ids["keyword"])]
        only_kw = out["keyword"][~out["keyword"].paper_id.isin(ids["semantic"])]
        a, b = st.columns(2)
        a.markdown("**Found only by semantic search**")
        for t in only_sem.title:
            a.markdown(f"- {t}")
        b.markdown("**Found only by keyword search**")
        for t in only_kw.title:
            b.markdown(f"- {t}")
        st.caption("Papers found only by semantic search often use different wording for the same idea.")

# ------------------------------ evaluation ------------------------------
with tab_eval:
    judg = load_json(f"{C.EVAL_DIR}/judgments.json", {})
    st.subheader("1 · Judge relevance")
    st.caption("For each test query, tick the papers that are genuinely relevant. Results from all three methods "
               "are pooled and shuffled so you can't tell which method found which paper (blind judging).")
    with st.expander("Add a test query"):
        new_q = st.text_input("Query text", key="newq")
        if st.button("Add query") and new_q.strip():
            queries.append({"id": f"q{len(queries) + 1}", "text": new_q.strip()})
            save_json(f"{C.EVAL_DIR}/queries.json", queries)
            st.rerun()
    qmap = {q["id"]: q for q in queries}
    qid = st.selectbox("Query to judge", list(qmap), format_func=lambda i: f"{i} - {qmap[i]['text']}"
                       + (f"  ✅ {sum(judg[i].values())} relevant" if i in judg else ""))
    ids = pool(engine, qid, qmap[qid]["text"])
    sub = papers.set_index("paper_id").loc[ids]
    cur = judg.get(qid, {})
    df = pd.DataFrame({"relevant": [bool(cur.get(i, 0)) for i in ids], "title": sub.title.values,
                       "year": sub.year.values, "abstract": sub.abstract.str.slice(0, 350).values}, index=ids)
    edited = st.data_editor(df, key=f"ed_{qid}", hide_index=True, disabled=["title", "year", "abstract"],
                            column_config={"relevant": st.column_config.CheckboxColumn("Relevant?", width="small")})
    if st.button("💾 Save judgments for this query"):
        judg[qid] = {pid: int(v) for pid, v in edited.relevant.items()}
        save_json(f"{C.EVAL_DIR}/judgments.json", judg)
        st.success("Saved.")

    st.subheader("2 · Results")
    per_q, summ = run_eval(engine, json.dumps(queries), json.dumps(judg))
    if summ.empty:
        st.info("No judged queries with relevant papers yet. Judge a few queries above (aim for 10+).")
    else:
        st.caption(f"{per_q.query_id.nunique()} queries with at least one relevant paper · metrics at k={K_EVAL}")
        order = ["keyword", "semantic", "hybrid"]
        long = summ.loc[order].reset_index().melt("method", var_name="metric", value_name="score")
        fig = px.bar(long, x="metric", y="score", color="method", barmode="group",
                     color_discrete_map={"keyword": "#9aa5b1", "semantic": "#4c72b0", "hybrid": "#55a868"})
        fig.update_layout(height=380, yaxis_range=[0, 1], xaxis_title=None, legend_title=None)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(summ.loc[order].round(3), width="stretch")
        with st.expander("Per-query results"):
            st.dataframe(per_q.round(3), hide_index=True, width="stretch")
        st.caption("Caveats: relevance is judged only on pooled top results (unjudged = not relevant), and a handful of "
                   "queries is a small sample - report these as indicative, not definitive.")

with st.expander("How this works"):
    st.markdown("""
- **Semantic search:** each paper's title + abstract is embedded into a vector; your query is embedded the same way;
  papers are ranked by cosine similarity, so different wording with the same meaning still matches.
- **Keyword search (BM25):** classic term-frequency ranking - the baseline.
- **Hybrid:** merges both rankings with Reciprocal Rank Fusion.
- **Evaluation:** Precision@k, Recall@k, MRR, nDCG@k and MAP against your blind relevance judgments.
""")
