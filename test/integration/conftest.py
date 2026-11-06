"""Integration-test configuration loaded before test modules are imported."""

from pathlib import Path

import numpy as np
import pytest
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

PDF_DIR = Path(__file__).parent.parent / "data" / "pdf"
SYNTHETIC_PDF = PDF_DIR / "synthetic_language_systems.pdf"


@pytest.fixture(scope="session")
def pdf_dir():
    return PDF_DIR


@pytest.fixture(scope="session")
def embedding_model_name():
    from integration.dynamics import pdf_embedding_model_name

    return pdf_embedding_model_name()


@pytest.fixture(scope="session")
def embedding_model(embedding_model_name):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(embedding_model_name)


@pytest.fixture(scope="session")
def document_chunks():
    from gap_finder.vector_store.pdf_chunker import PDFChunker

    chunker = PDFChunker(chunk_size=1000, chunk_overlap=100)
    return chunker.chunk_pdf(SYNTHETIC_PDF)


@pytest.fixture(scope="session")
def document_embeddings(embedding_model, document_chunks):
    return embedding_model.encode(document_chunks).astype(np.float32)


@pytest.fixture
def populated_store(vector_store_cls, document_embeddings, document_chunks):
    """Build a real vector store from chunks and embeddings of a synthetic PDF."""
    store = vector_store_cls(dim=document_embeddings.shape[1])
    store.add(document_embeddings.copy(), list(document_chunks))
    return store, document_chunks
