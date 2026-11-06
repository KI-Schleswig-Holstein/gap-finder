"""Integration tests for local and OpenAI-compatible embedding backends."""

import numpy as np
import pandas as pd
import pytest

from gap_finder.question_ranking import QuestionRanker
from gap_finder.vector_store.numpy_vector_store import NumpyVectorStore
from dynamics import (
    build_embedding_model,
    embedding_backend_params,
    embedding_model_name_for_backend,
    embedding_pytestmark,
)


pytestmark = embedding_pytestmark(
    "set GAP_FINDER_RUN_EMBEDDING_INTEGRATION=1 to run real embedding tests"
)


@pytest.mark.parametrize("embedding_backend", embedding_backend_params())
def test_embedding_backend_batches_and_supports_vector_search(embedding_backend):
    model_name = embedding_model_name_for_backend(embedding_backend)
    model = build_embedding_model(embedding_backend)
    documents = [
        "Training data is missing.",
        "Evaluation benchmarks are incomplete.",
        "The deployment environment is undocumented.",
    ]

    embeddings = model.encode(documents, batch_size=2, show_progress_bar=True)

    assert embeddings.dtype == np.float32
    assert embeddings.shape[0] == len(documents)
    assert embeddings.ndim == 2

    vector_store = NumpyVectorStore(dim=embeddings.shape[1])
    vector_store.add(embeddings, documents)
    ranker = QuestionRanker(
        vector_store=vector_store,
        embedding_model=model,
        top_k=1,
        batch_size=2,
    )
    rankings = ranker.rank(
        pd.DataFrame({"question": [documents[0], documents[1]]})
    )
    model.cleanup()

    print(f"\nEmbedding backend: {embedding_backend}, model: {model_name}")
    assert rankings["similarity_score"].map(lambda scores: scores[0] > 0.99).all()
