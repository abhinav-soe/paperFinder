"""Download paper metadata (title, abstract, authors, year, venue, topic, citations) from OpenAlex.
Needs a FREE API key: https://openalex.org/settings/api   (set OPENALEX_API_KEY or pass --api-key)"""
import argparse
import os
import time
#apikey="DzQ140djSWeQBkdgq2SSuM"
import pandas as pd
import requests

import config as C

API = "https://api.openalex.org/works"
SELECT = ("id,title,abstract_inverted_index,authorships,publication_year,primary_location,"
          "cited_by_count,primary_topic,doi")


def rebuild_abstract(inverted):
    """OpenAlex ships abstracts as {word: [positions]}; turn that back into text."""
    if not inverted:
        return ""
    pos = {i: w for w, idxs in inverted.items() for i in idxs}
    return " ".join(pos[i] for i in sorted(pos))


def parse(work, seed):
    src = ((work.get("primary_location") or {}).get("source") or {})
    topic = work.get("primary_topic") or {}
    names = [a["author"]["display_name"] for a in work.get("authorships") or [] if a.get("author")]
    authors = "; ".join(names[:8]) + (" et al." if len(names) > 8 else "")
    return dict(paper_id=work["id"].rsplit("/", 1)[-1], title=(work.get("title") or "").strip(),
                abstract=rebuild_abstract(work.get("abstract_inverted_index")), authors=authors,
                year=work.get("publication_year"), venue=src.get("display_name"),
                field=(topic.get("field") or {}).get("display_name"), topic=topic.get("display_name"),
                cited_by_count=work.get("cited_by_count") or 0, doi=work.get("doi"),
                url=work["id"], seed_topic=seed)


def get(params, retries=6):
    for attempt in range(retries):
        r = requests.get(API, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (401, 403):
            raise SystemExit("API key missing or rejected. Get a free key at https://openalex.org/settings/api")
        if r.status_code == 402:
            raise SystemExit("OpenAlex daily budget used up. Try again after midnight UTC.")
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    raise SystemExit("Too many failed requests - try again later.")


def fetch_topic(topic, n, from_year, key):
    rows, cursor, calls = [], "*", 0
    while len(rows) < n and cursor:
        params = dict(search=topic, per_page=100, cursor=cursor, select=SELECT, api_key=key,
                      filter=f"has_abstract:true,type:article,language:en,from_publication_date:{from_year}-01-01")
        data = get(params)
        calls += 1
        rows += [parse(w, topic) for w in data["results"]]
        cursor = data["meta"].get("next_cursor")
        time.sleep(0.2)
        if not data["results"]:
            break
    return rows[:n], calls


def clean(df, min_words=40):
    df = df.dropna(subset=["title", "abstract"])
    df = df[df.abstract.str.split().str.len() >= min_words]
    df = df.drop_duplicates("paper_id")
    df = df.assign(_t=df.title.str.lower().str.replace(r"\W+", " ", regex=True).str.strip()).drop_duplicates("_t")
    return df.drop(columns="_t").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY"))
    ap.add_argument("--per-topic", type=int, default=300)
    ap.add_argument("--from-year", type=int, default=2016)
    ap.add_argument("--out", default=C.DATA_PATH)
    a = ap.parse_args()
    if not a.api_key:
        raise SystemExit("Provide --api-key or set OPENALEX_API_KEY (free key: https://openalex.org/settings/api)")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    allrows, total_calls = [], 0
    for t in C.TOPICS:
        rows, calls = fetch_topic(t, a.per_topic, a.from_year, a.api_key)
        allrows += rows
        total_calls += calls
        print(f"{t:40s} {len(rows):4d} papers  ({calls} calls)")
    df = clean(pd.DataFrame(allrows))
    df.to_csv(a.out, index=False)
    print(f"\nSaved {len(df)} papers to {a.out}  | API calls: {total_calls} (~${total_calls * 0.001:.3f})")
