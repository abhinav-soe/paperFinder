import pandas as pd


def load_papers(path):
    df = pd.read_csv(path)
    for c in ["title", "abstract", "authors", "venue", "field", "topic", "doi", "url"]:
        if c in df:
            df[c] = df[c].fillna("")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["cited_by_count"] = pd.to_numeric(df["cited_by_count"], errors="coerce").fillna(0).astype(int)
    return df.reset_index(drop=True)


def paper_texts(df):
    return (df["title"] + ". " + df["abstract"]).tolist()
