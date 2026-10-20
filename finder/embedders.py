"""Embedding backends. Default = Sentence Transformers; TF-IDF+SVD (LSA) is a light offline fallback."""
import json
import os

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class SentenceTransformerEmbedder:
    backend = "st"

    def __init__(self, model_name):
        self.model_name, self._model = model_name, None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts):
        return self

    def encode(self, texts, batch_size=64, show_progress=False):
        v = self.model.encode(list(texts), batch_size=batch_size, show_progress_bar=show_progress,
                              normalize_embeddings=True, convert_to_numpy=True)
        return v.astype("float32")

    def save(self, d):
        pass

    def config(self):
        return dict(backend=self.backend, model=self.model_name)


class TfidfSvdEmbedder:
    backend = "tfidf"

    def __init__(self, dim=256):
        self.dim, self.vec, self.svd = dim, None, None

    def fit(self, texts):
        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                                   max_features=50000)
        X = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=max(2, min(self.dim, X.shape[1] - 1, X.shape[0] - 1)),
                                random_state=0).fit(X)
        return self

    def encode(self, texts, batch_size=0, show_progress=False):
        return normalize(self.svd.transform(self.vec.transform(list(texts)))).astype("float32")

    def save(self, d):
        joblib.dump((self.vec, self.svd), os.path.join(d, "tfidf_svd.joblib"))

    def config(self):
        return dict(backend=self.backend, model=f"tfidf-svd-{self.dim}")


def make_embedder(backend, model=None):
    return SentenceTransformerEmbedder(model) if backend == "st" else TfidfSvdEmbedder()


def load_embedder(index_dir):
    meta = json.load(open(os.path.join(index_dir, "meta.json")))
    if meta["backend"] == "st":
        return SentenceTransformerEmbedder(meta["model"]), meta
    e = TfidfSvdEmbedder()
    e.vec, e.svd = joblib.load(os.path.join(index_dir, "tfidf_svd.joblib"))
    return e, meta
