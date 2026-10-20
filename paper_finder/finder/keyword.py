"""BM25 keyword baseline (no stemming, so 'detect' != 'detecting' - exactly the weakness semantic search fixes)."""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.vec = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b")
        self.tf = self.vec.fit_transform(texts).tocsc()
        self.n = self.tf.shape[0]
        df = np.diff(self.tf.indptr)
        self.idf = np.log(1 + (self.n - df + 0.5) / (df + 0.5))
        self.dl = np.asarray(self.tf.sum(axis=1)).ravel()
        self.avgdl, self.k1, self.b = self.dl.mean(), k1, b
        self.analyzer = self.vec.build_analyzer()

    def scores(self, query):
        s = np.zeros(self.n)
        for t in set(self.analyzer(query)):
            j = self.vec.vocabulary_.get(t)
            if j is None:
                continue
            lo, hi = self.tf.indptr[j], self.tf.indptr[j + 1]
            rows, tf = self.tf.indices[lo:hi], self.tf.data[lo:hi]
            s[rows] += self.idf[j] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.dl[rows] / self.avgdl))
        return s

    def search(self, query, k, mask=None):
        s = self.scores(query)
        if mask is not None:
            s = np.where(mask, s, 0)
        k = min(k, int((s > 0).sum()))
        if k <= 0:
            return np.array([], dtype=int), np.array([])
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return top, s[top]
