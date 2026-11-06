from unittest.mock import Mock

import pytest

from gap_finder.embedding_model.embedding_model import EmbeddingModel
from gap_finder.language_model.language_model import LanguageModel
from gap_finder.vector_store.faiss_vector_store import FaissVectorStore
from gap_finder.vector_store.numpy_vector_store import NumpyVectorStore

VECTOR_STORE_CLASSES = [NumpyVectorStore, FaissVectorStore]


@pytest.fixture(params=VECTOR_STORE_CLASSES, ids=lambda cls: cls.__name__)
def vector_store_cls(request):
    return request.param


@pytest.fixture
def mock_language_model():
    """Language model mock; the factories pass ``LanguageModel`` instances through."""
    return Mock(spec=LanguageModel)


@pytest.fixture
def mock_embedding_model():
    """Embedding model mock; set ``encode.return_value`` for test-specific vectors."""
    return Mock(spec=EmbeddingModel)
