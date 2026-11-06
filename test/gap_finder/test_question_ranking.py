from unittest.mock import Mock

import numpy as np
import pandas as pd

from gap_finder.question_ranking import QuestionRanker


def test_rank_encodes_questions_and_uses_vector_store_scores():
    class DummyModel:
        def __init__(self):
            self.cleanup = Mock()

        def encode(self, questions, show_progress_bar=False, batch_size=32):
            assert questions == ["Frage zu A", "Frage zu C"]
            assert show_progress_bar is True
            assert batch_size == 2
            return np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            )

    embedding_model = DummyModel()
    vector_store = Mock()
    vector_store.search.side_effect = [
        (np.array([[0.91, 0.42]], dtype=np.float32), None, None, None),
        (np.array([[0.83, 0.21]], dtype=np.float32), None, None, None),
    ]
    ranker = QuestionRanker(
        vector_store=vector_store,
        embedding_model=embedding_model,
        top_k=2,
        batch_size=2,
    )

    questions = pd.DataFrame(
        {
            "question": ["Frage zu A", "Frage zu C"],
            "topic_id": [0, 1],
        }
    )

    rankings = ranker.rank(questions)

    assert rankings["question"].tolist() == ["Frage zu A", "Frage zu C"]
    assert rankings["topic_id"].tolist() == [0, 1]
    assert rankings["similarity_score"].tolist() == [
        [0.9100000262260437, 0.41999998688697815],
        [0.8299999833106995, 0.20999999344348907],
    ]
    np.testing.assert_array_equal(
        np.stack(rankings["embedding"]),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    assert vector_store.search.call_count == 2
    np.testing.assert_array_equal(
        vector_store.search.call_args_list[0].args[0],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        vector_store.search.call_args_list[1].args[0],
        np.array([[0.0, 1.0]], dtype=np.float32),
    )
    assert vector_store.search.call_args_list[0].kwargs == {"k": 2}
    assert vector_store.search.call_args_list[1].kwargs == {"k": 2}


def test_rank_returns_empty_dataframe_for_empty_questions():
    model = Mock()
    ranker = QuestionRanker(
        vector_store=Mock(),
        embedding_model=model,
        top_k=1,
    )

    questions = pd.DataFrame({"question": pd.Series(dtype=str)})
    rankings = ranker.rank(questions)

    assert rankings.empty
    assert list(rankings.columns) == ["question", "embedding", "similarity_score"]
    model.encode.assert_not_called()


def test_ranker_does_not_take_ownership_of_embedding_model():
    model = Mock()
    ranker = QuestionRanker(
        vector_store=Mock(),
        embedding_model=model,
        top_k=1,
    )

    assert ranker.embedding_model is model
    model.cleanup.assert_not_called()
