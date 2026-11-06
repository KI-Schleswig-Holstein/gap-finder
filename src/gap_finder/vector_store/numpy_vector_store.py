import json
from pathlib import Path

import numpy as np

from gap_finder.vector_store.vector_store import SearchResult, VectorStore, VectorStoreContents

class NumpyVectorStore(VectorStore):
    """VectorStore backed by plain numpy (exact cosine similarity search, no FAISS dependency)."""

    def __init__(self, dim: int):
        """Initialize an empty store for `dim`-dimensional embeddings."""
        super().__init__(dim)
        self.vectors = np.empty((0, dim), dtype=np.float32)
        self.chunks = []

    def add(self, embeddings: np.ndarray, chunks: list[str]) -> None:
        normalized = self._normalize_L2(embeddings.astype(np.float32, copy=True))
        self.vectors = np.vstack([self.vectors, normalized])
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> SearchResult:
        """Return ``(scores, indices, embeddings, chunk_texts)`` like FAISS search."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = self._normalize_L2(query_embedding.astype(np.float32, copy=True))

        n = self.vectors.shape[0]

        D = np.full((1, k), -np.finfo(np.float32).max, dtype=np.float32)
        I = np.full((1, k), -1, dtype=np.int64)

        if n == 0:
            return D, I, [], []

        similarities = (self.vectors @ query_embedding[0]).astype(np.float32)

        n_hits = min(k, n)
        order = np.argsort(-similarities, kind="stable")[:n_hits]

        D[0, :n_hits] = similarities[order]
        I[0, :n_hits] = order

        return D, I, [self.vectors[i] for i in I[0]], [self.chunks[i] for i in I[0]]

    def get_all(self) -> VectorStoreContents:
        return self.vectors.copy(), self.chunks

    def save_to(self, folder_path: str | Path, file_name: str) -> None:
        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        base_path = folder_path / file_name
        np.save(f"{base_path}.npy", self.vectors)
        with open(folder_path / f"{file_name}.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from(cls, folder_path: str | Path, file_name: str):
        base_path = Path(folder_path) / file_name

        vectors = np.load(f"{base_path}.npy", allow_pickle=False)

        with open(folder_path / f"{file_name}.json", encoding="utf-8") as f:
            chunks = json.load(f)

        if vectors.ndim != 2:
            raise ValueError("Saved NumPy vector store must contain a 2-D array.")
        if not isinstance(chunks, list) or not all(
            isinstance(chunk, str) for chunk in chunks
        ):
            raise ValueError("Saved vector-store chunks must be a list of strings.")
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                "Saved NumPy vector and chunk counts do not match."
            )

        db = cls(dim=vectors.shape[1])
        db.vectors = vectors
        db.chunks = chunks

        return db

    @staticmethod
    def _normalize_L2(x: np.ndarray) -> np.ndarray:
        """L2-normalize each row of x in place, matching faiss.normalize_L2 (zero-norm rows untouched)."""
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        np.divide(x, norms, out=x, where=norms != 0)
        return x
