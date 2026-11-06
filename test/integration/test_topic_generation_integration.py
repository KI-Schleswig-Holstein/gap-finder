"""Integration tests for real topic generation.

Skipped by default because they use a real embedding backend and run BERTopic:

    GAP_FINDER_RUN_EMBEDDING_INTEGRATION=1 uv run pytest -s test/integration/test_topic_generation_integration.py
"""

import numpy as np
import pandas as pd
import pytest

from gap_finder import persistence
from gap_finder.topic_generation import TopicGenerator
from dynamics import (
    build_embedding_model,
    embedding_backend_params,
    embedding_model_name_for_backend,
    embedding_pytestmark,
    topic_generation_settings,
)


pytestmark = [
    *embedding_pytestmark(
        "set GAP_FINDER_RUN_EMBEDDING_INTEGRATION=1 to run real topic generation integration tests"
    ),
]


@pytest.mark.parametrize("embedding_backend", embedding_backend_params())
def test_topic_generator_groups_questions_and_saves_topic_words(
    embedding_backend,
    tmp_path,
):
    settings = topic_generation_settings()
    data_questions = [
        f"Welche Trainingsdaten Datensaetze Labels fehlen fuer Modell {index}?"
        for index in range(16)
    ]
    evaluation_questions = [
        f"Welche Evaluation Benchmark Metrik Validierung fehlt fuer Methode {index}?"
        for index in range(16)
    ]
    question_texts = [*data_questions, *evaluation_questions]
    questions = pd.DataFrame(
        {"id": range(len(question_texts)), "question": question_texts}
    )
    generator = TopicGenerator(
        language=settings.language,
        embedding_model=build_embedding_model(embedding_backend, topic=True),
    )

    topics, questions_with_topics, topic_model = generator.generate(
        questions,
        cleanup_after_generation=True,
    )

    model_name = embedding_model_name_for_backend(embedding_backend, topic=True)
    print(
        f"\nEmbedding backend: {embedding_backend}, model: {model_name}"
        f"\nGenerated topics:\n{topics}"
    )

    assert questions_with_topics["question"].tolist() == question_texts
    assert list(questions_with_topics.columns) == [
        "id",
        "question",
        "topic_id",
        "topic_prob",
    ]
    assert len(questions_with_topics) == len(questions)
    assert topics["topic_words"].map(lambda words: isinstance(words, list)).all()

    non_outlier_topics = sorted(
        topic for topic in questions_with_topics["topic_id"].unique() if topic != -1
    )
    assert len(non_outlier_topics) >= 2

    data_dominant_topic = (
        questions_with_topics["topic_id"].iloc[: len(data_questions)].mode().iloc[0]
    )
    evaluation_dominant_topic = (
        questions_with_topics["topic_id"].iloc[len(data_questions):].mode().iloc[0]
    )
    assert data_dominant_topic != evaluation_dominant_topic

    assert generator.probs is not None
    expected_probabilities = [prob.max() for prob in generator.probs]
    assert questions_with_topics["topic_prob"].tolist() == pytest.approx(
        expected_probabilities
    )
    assert np.array_equal(
        generator.topics,
        questions_with_topics["topic_id"].to_numpy(),
    )
    assert generator.probs.shape[0] == len(questions)
    assert topic_model is generator.topic_model
    assert topic_model.embedding_model is None

    if embedding_backend == "local":
        persistence.save_topic_model(
            topic_model,
            tmp_path,
            embedding_model_name=model_name,
        )
        restored_topic_model = persistence.restore_topic_model(
            tmp_path,
            embedding_model=model_name,
        )

        restored_topics, _ = restored_topic_model.transform(question_texts[:2])
        assert len(restored_topics) == 2
