from typing import TYPE_CHECKING

from gap_finder.vector_store.vector_store import VectorStore
from gap_finder.vector_store.faiss_vector_store import FaissVectorStore
from gap_finder.vector_store.numpy_vector_store import NumpyVectorStore

if TYPE_CHECKING:
    from gap_finder.vector_store.pdf_chunker import PDFChunker

__all__ = ["VectorStore", "FaissVectorStore", "NumpyVectorStore", "PDFChunker"]


def __getattr__(name: str):
    if name == "PDFChunker":
        from gap_finder.vector_store.pdf_chunker import PDFChunker

        return PDFChunker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
