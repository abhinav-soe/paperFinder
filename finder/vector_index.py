"""Cosine-similarity index. Uses FAISS (IndexFlatIP) when installed, otherwise exact NumPy.
Vectors are L2-normalised, so inner product == cosine similarity. Both backends give identical results."""
import numpy as np


class VectorIndex:
    def __init__(self, embeddings, backend="auto"):
        self.emb = np.ascontiguousarray(embeddings, dtype="float32")
        self.faiss = None
        if backend in ("auto", "faiss"):
            try:
                import faiss
                self.faiss = faiss.IndexFlatIP(self.emb.shape[1])
                self.faiss.add(self.emb)
            except ImportError:
                if backend == "faiss":
                    raise

    @property
    def backend_name(self):
        return "FAISS (IndexFlatIP)" if self.faiss is not None else "NumPy (exact)"

    def search(self, qvec, k, mask=None):
        """Top-k (indices, cosine scores). `mask` is a boolean array: only True rows may be returned."""
        q = np.ascontiguousarray(qvec, dtype="float32").ravel()
        n = len(self.emb)
        allowed = n if mask is None else int(mask.sum())
        k = min(k, allowed)
        if k <= 0:
            return np.array([], dtype=int), np.array([], dtype="float32")
        if self.faiss is None:
            s = self.emb @ q
            if mask is not None:
                s = np.where(mask, s, -np.inf)
            top = np.argpartition(-s, k - 1)[:k]
            top = top[np.argsort(-s[top])]
            return top, s[top]
        fetch = k if mask is None else min(n, max(k * 10, 100))
        while True:                                    # over-fetch, filter, widen if needed
            D, I = self.faiss.search(q[None, :], fetch)
            I, D = I[0], D[0]
            if mask is not None:
                keep = mask[I]
                I, D = I[keep], D[keep]
            if len(I) >= k or fetch >= n:
                return I[:k], D[:k]
            fetch = min(n, fetch * 4)
