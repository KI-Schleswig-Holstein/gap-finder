import numpy as np
import pandas as pd
import pytest

from gap_finder.topic_ranking import TopicRanker


def test_rank_enriches_topics_with_similarity_statistics():
    questions_with_topics = pd.DataFrame(
        {
            "question": ["q1", "q2", "q3"],
            "topic_id": [0, 0, 1],
            "similarity_score": [0.9, 0.7, 0.2],
        }
    )
    topics = pd.DataFrame(
        {
            "id": [0, 1],
            "topic_words": [[("a", 1.0)], [("b", 1.0)]],
        }
    )

    rankings = TopicRanker().rank(questions_with_topics, topics)

    assert list(rankings.columns) == [
        "id",
        "topic_words",
        "question_count",
        "avg_similarity",
        "std_similarity",
        "min_similarity",
        "max_similarity",
    ]
    records = rankings.to_dict("records")
    assert records[0] == {
        "id": 0,
        "topic_words": [("a", 1.0)],
        "question_count": 2,
        "avg_similarity": pytest.approx(0.8),
        "std_similarity": pytest.approx(0.1414213562),
        "min_similarity": pytest.approx(0.7),
        "max_similarity": pytest.approx(0.9),
    }
    assert records[1]["id"] == 1
    assert records[1]["question_count"] == 1
    assert records[1]["avg_similarity"] == pytest.approx(0.2)
    assert np.isnan(records[1]["std_similarity"])
    assert records[1]["min_similarity"] == pytest.approx(0.2)
    assert records[1]["max_similarity"] == pytest.approx(0.2)


def test_rank_uses_first_similarity_when_score_is_a_list():
    questions_with_topics = pd.DataFrame(
        {
            "question": ["q1", "q2"],
            "topic_id": [0, 0],
            "similarity_score": [[0.9], [0.7]],
        }
    )
    topics = pd.DataFrame({"id": [0]})

    rankings = TopicRanker().rank(questions_with_topics, topics)

    assert rankings.loc[0, "question_count"] == 2
    assert rankings.loc[0, "avg_similarity"] == pytest.approx(0.8)


def test_rank_keeps_topics_without_questions():
    questions_with_topics = pd.DataFrame(
        {
            "question": ["q1"],
            "topic_id": [0],
            "similarity_score": [0.9],
        }
    )
    topics = pd.DataFrame({"id": [0, 1]})

    rankings = TopicRanker().rank(questions_with_topics, topics)

    assert rankings.loc[1, "id"] == 1
    assert rankings.loc[1, "question_count"] == 0
    assert np.isnan(rankings.loc[1, "avg_similarity"])
