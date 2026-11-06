import json
from pathlib import Path

import faiss
import numpy as np

from gap_finder.vector_store.vector_store import SearchResult, VectorStore, VectorStoreContents

class FaissVectorStore(VectorStore):
    """VectorStore backed by a FAISS IndexFlatIP (exact cosine similarity search)."""

    def __init__(self, dim: int):
        """Initialize an empty index for `dim`-dimensional embeddings."""
        super().__init__(dim)
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks = []

    def add(self, embeddings: np.ndarray, chunks: list[str]) -> None:
        embeddings = embeddings.astype(np.float32, copy=True)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> SearchResult:
        """Return ``(scores, indices, embeddings, chunk_texts)`` like FAISS search."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32, copy=True)
        faiss.normalize_L2(query_embedding)

        if self.index.ntotal == 0:
            D = np.full((1, k), -np.finfo(np.float32).max, dtype=np.float32)
            I = np.full((1, k), -1, dtype=np.int64)
            return D, I, [], []

        D, I = self.index.search(query_embedding, k)
        return D, I, [self.index.reconstruct(int(i)) for i in I[0]], [self.chunks[i] for i in I[0]]

    def get_all(self) -> VectorStoreContents:
        n = self.index.ntotal
        embeddings = self.index.reconstruct_n(0, n)
        return embeddings, self.chunks

    def save_to(self, folder_path: str | Path, file_name: str) -> None:
        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        base_path = str(folder_path / f"{file_name}")
        faiss.write_index(self.index, f"{base_path}.index")
        with open(folder_path / f"{file_name}.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from(cls, folder_path: str | Path, file_name: str):
        base_path = str(Path(folder_path) / file_name)

        index = faiss.read_index(f"{base_path}.index")

        with open(folder_path / f"{file_name}.json", encoding="utf-8") as f:
            chunks = json.load(f)

        if not isinstance(chunks, list) or not all(
            isinstance(chunk, str) for chunk in chunks
        ):
            raise ValueError("Saved vector-store chunks must be a list of strings.")
        if index.ntotal != len(chunks):
            raise ValueError(
                "Saved FAISS vector and chunk counts do not match."
            )

        db = cls(dim=index.d)
        db.index = index
        db.chunks = chunks

        return db
