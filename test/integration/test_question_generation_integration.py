"""Integration tests for real question generation.

These tests build a chunks DataFrame from synthetic PDF fixtures and run
QuestionGenerator with explicit test configuration. The same generation flow
is exercised once with the small local integration-test model and once with
the default OpenAI-compatible API model.

They are skipped by default because they require real LLM infrastructure:

    GAP_FINDER_RUN_LLM_INTEGRATION=1 uv run pytest -s test/integration/test_question_generation_integration.py

Select one backend or override model/device/API settings when needed:

    GAP_FINDER_RUN_LLM_INTEGRATION=1 \
        GAP_FINDER_LLM_INTEGRATION_BACKEND=local \
        GAP_FINDER_LLM_INTEGRATION_LOCAL_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct \
        GAP_FINDER_LLM_INTEGRATION_DEVICE=cuda \
        uv run pytest -s test/integration/test_question_generation_integration.py

    GAP_FINDER_RUN_LLM_INTEGRATION=1 \
        GAP_FINDER_LLM_INTEGRATION_BACKEND=api \
        GAP_FINDER_LLM_INTEGRATION_API_MODEL_NAME=gpt-oss-20b \
        GAP_FINDER_LLM_INTEGRATION_API_BASE_URL=http://localhost:11434/v1 \
        GAP_FINDER_API_KEY=- \
        uv run pytest -s test/integration/test_question_generation_integration.py

The -s flag shows the generated questions printed by the test.
"""

import json

import numpy as np
import pandas as pd
import pytest

from gap_finder.question_generation import QuestionGenerator
from gap_finder.vector_store.faiss_vector_store import FaissVectorStore
from dynamics import (
    build_language_model_or_skip,
    language_backend_params,
    language_model_name_for_backend,
    env_int,
    llm_pytestmark,
)

pytestmark = [
    pytest.mark.vs,
    *llm_pytestmark(
        "set GAP_FINDER_RUN_LLM_INTEGRATION=1 to run real LLM integration tests"
    ),
]


def _build_faiss_store(
    document_embeddings: np.ndarray,
    document_chunks: list[str],
) -> FaissVectorStore:
    batch_size = env_int(
        "QUESTION_GENERATION_BATCH_SIZE",
        4,
    )
    chunk_count = min(len(document_chunks), batch_size)
    chunks = list(document_chunks[:chunk_count])
    embeddings = document_embeddings[:chunk_count]

    store = FaissVectorStore(dim=embeddings.shape[1])
    store.add(embeddings.copy(), chunks)
    return store


def _chunks_dataframe(vector_store: FaissVectorStore) -> pd.DataFrame:
    embeddings, texts = vector_store.get_all()
    return pd.DataFrame(
        {
            "id": range(len(texts)),
            "embedding": list(embeddings),
            "text": texts,
        }
    )


@pytest.mark.parametrize("language_backend", language_backend_params())
def test_question_generation_with_default_values_generates_questions(
    language_backend,
    document_embeddings,
    document_chunks,
):
    vector_store = _build_faiss_store(document_embeddings, document_chunks)
    chunks = _chunks_dataframe(vector_store)
    language_model_name = language_model_name_for_backend(language_backend)
    print(
        "\nSelected question generation integration language backend: "
        f"{language_backend}, model: {language_model_name}"
    )
    language_model = build_language_model_or_skip(language_backend)
    generator = QuestionGenerator(
        language_model=language_model,
        k_questions=env_int(
            "QUESTION_GENERATION_K_QUESTIONS",
            3,
        ),
        batch_size=env_int(
            "QUESTION_GENERATION_BATCH_SIZE",
            4,
        ),
    )

    generated_questions = generator.generate(chunks)

    questions = generated_questions["question"].tolist()
    print(
        f"\nGenerated questions with {language_backend} backend "
        f"({language_model_name}):\n"
        f"{json.dumps(questions, ensure_ascii=False, indent=2)}"
    )

    assert not generated_questions.empty
    assert list(generated_questions.columns) == ["id", "question", "chunk_id"]
    assert questions
