from unittest.mock import Mock

import numpy as np
import pandas as pd

from gap_finder.main import GapFinder
from gap_finder.vector_store.numpy_vector_store import NumpyVectorStore


def test_find_creates_and_ranks_topics_without_external_models(
    mock_embedding_model, mock_language_model
):
    mock_embedding_model.encode.return_value = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    vector_store = NumpyVectorStore(dim=2)
    vector_store.add(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        ["chunk a", "chunk b"],
    )
    gap_finder = GapFinder(
        vector_store=vector_store,
        embedding_model=mock_embedding_model,
        language_model=mock_language_model,
        questions=["Frage zu A", "Frage zu B"],
    )
    topics = pd.DataFrame(
        {
            "id": [0, 1],
            "topic_words": [[("a", 1.0)], [("b", 1.0)]],
        }
    )
    questions_with_topics = pd.DataFrame(
        {
            "id": [0, 1],
            "question": ["Frage zu A", "Frage zu B"],
            "topic_id": [0, 1],
        }
    )
    topic_model = object()
    gap_finder._create_topics = Mock(
        return_value=(topics, questions_with_topics, topic_model)
    )
    gap_finder._explain_topics = Mock(
        side_effect=lambda *, questions_with_topics, topics: topics
    )
    gap_finder._create_report = Mock(side_effect=lambda *, topics: topics)

    ranked_topics = gap_finder.find()
    ranked_questions = gap_finder.questions

    assert ranked_topics[["id", "question_count", "avg_similarity"]].to_dict(
        "records"
    ) == [
        {"id": 0, "question_count": 1, "avg_similarity": 1.0},
        {"id": 1, "question_count": 1, "avg_similarity": 1.0},
    ]
    assert ranked_questions[
        ["id", "question", "topic_id", "similarity_score"]
    ].to_dict("records") == [
        {
            "id": 0,
            "question": "Frage zu A",
            "topic_id": 0,
            "similarity_score": [1.0],
        },
        {
            "id": 1,
            "question": "Frage zu B",
            "topic_id": 1,
            "similarity_score": [1.0],
        },
    ]
    assert gap_finder.questions is ranked_questions
    assert gap_finder.topics is ranked_topics
    assert gap_finder.topic_model is topic_model
    gap_finder._explain_topics.assert_called_once_with(
        questions_with_topics=ranked_questions,
        topics=ranked_topics,
    )
