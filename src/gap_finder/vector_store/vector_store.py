from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeAlias

import numpy as np

SearchResult: TypeAlias = tuple[np.ndarray, np.ndarray, list[np.ndarray], list[str]]
VectorStoreContents: TypeAlias = tuple[np.ndarray, list[str]]


class VectorStore(ABC):
    """
    Common interface for vector stores: add embeddings + chunks, search by
    cosine similarity, and persist/restore to disk.

    FaissVectorStore and NumpyVectorStore both implement this with
    numerically identical outputs.
    """

    @abstractmethod
    def __init__(self, dim: int):
        """Initialize an empty store for `dim`-dimensional embeddings."""
        self.dim = dim

    @abstractmethod
    def add(self, embeddings: np.ndarray, chunks: list[str]) -> None:
        """Add embeddings (n, dim) and their corresponding chunk texts to the store."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5) -> SearchResult:
        """Return the top-k most similar chunks in FAISS-style order.

        Returns:
            tuple[np.ndarray, np.ndarray, list[np.ndarray], list[str]]:
                ``(scores, indices, embeddings, chunk_texts)``.

                ``scores`` is always the first value. It is a float array with
                shape ``(1, k)`` and descending cosine similarities. ``indices``
                has shape ``(1, k)`` and contains the matching corpus indices.
                Missing hits are represented like FAISS does: index ``-1`` and
                a large negative score.
        """
        raise NotImplementedError

    @abstractmethod
    def get_all(self) -> VectorStoreContents:
        """Return all stored (embeddings, chunk texts)."""
        raise NotImplementedError

    @abstractmethod
    def save_to(self, folder_path: str | Path, file_name: str) -> None:
        """Persist the store's embeddings and chunks to disk under `folder_path/file_name`."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_from(cls, folder_path: str | Path, file_name: str) -> "VectorStore":
        """Load a vector store previously saved via `save_to()`."""
        raise NotImplementedError

    def save_pdf_vector_store(self, folder_path: str | Path, embedding_model: str):
        "Convenience method to save a vector store specifically for PDFs using a given embedding model."
        self.save_to(folder_path, file_name=f"pdf_{embedding_model.replace('/', '_')}")

    @classmethod
    def load_pdf_vector_store(cls, folder_path: str | Path, embedding_model: str) -> "VectorStore":
        "Convenience method to load a vector store specifically for PDFs using a given embedding model."
        return cls.load_from(folder_path, file_name=f"pdf_{embedding_model.replace('/', '_')}")
