import sys
import types
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from gap_finder.topic_generation import TopicGenerator


def install_fake_bertopic(monkeypatch, topic_model: Mock) -> Mock:
    bertopic_module = types.ModuleType("bertopic")
    bertopic_constructor = Mock(return_value=topic_model)
    bertopic_module.BERTopic = bertopic_constructor
    monkeypatch.setitem(sys.modules, "bertopic", bertopic_module)
    return bertopic_constructor


def test_generate_happy_path_uses_bertopic_and_returns_dataframe(monkeypatch):
    questions = pd.DataFrame(
        {
            "id": [10, 11],
            "question": [
                "Welche Daten fehlen fuer diese Methode?",
                "Welche Evaluation wurde noch nicht gemacht?",
            ],
        }
    )
    topic_ids = [0, 1]
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    embedding_model = object()
    topic_model = Mock()
    topic_model.fit_transform.return_value = (topic_ids, probs)
    topic_model.get_topic_info.return_value = "topic info"
    topic_model.get_topic.side_effect = lambda topic_id: [
        (f"word-{topic_id}", 0.5),
    ]
    bertopic_constructor = install_fake_bertopic(monkeypatch, topic_model)
    generator = TopicGenerator(
        language="german",
        embedding_model=embedding_model,
    )

    topics, questions_with_topics, returned_topic_model = generator.generate(
        questions, verbose=True
    )

    assert topics.to_dict("records") == [
        {"id": 0, "topic_words": [("word-0", 0.5)]},
        {"id": 1, "topic_words": [("word-1", 0.5)]},
    ]
    assert questions_with_topics.to_dict("records") == [
        {
            "id": 10,
            "question": "Welche Daten fehlen fuer diese Methode?",
            "topic_id": 0,
            "topic_prob": 0.9,
        },
        {
            "id": 11,
            "question": "Welche Evaluation wurde noch nicht gemacht?",
            "topic_id": 1,
            "topic_prob": 0.8,
        },
    ]
    bertopic_constructor.assert_called_once_with(
        embedding_model=embedding_model,
        language="german",
        calculate_probabilities=True,
        verbose=True,
    )
    topic_model.fit_transform.assert_called_once_with(questions["question"].tolist())
    assert topic_model.get_topic.call_count == 2
    assert generator.topic_model is topic_model
    assert returned_topic_model is topic_model
    assert generator.topics == topic_ids
    assert np.array_equal(generator.probs, probs)


def test_generate_empty_questions_returns_empty_dataframe(monkeypatch):
    topic_model = Mock()
    topic_model.fit_transform.return_value = ([], None)
    topic_model.get_topic_info.return_value = "topic info"
    install_fake_bertopic(monkeypatch, topic_model)
    embedding_model = object()
    generator = TopicGenerator(
        language="german",
        embedding_model=embedding_model,
    )
    questions = pd.DataFrame(
        {"id": pd.Series(dtype="int64"), "question": pd.Series(dtype="object")}
    )

    topics, questions_with_topics, returned_topic_model = generator.generate(
        questions
    )

    assert topics.empty
    assert list(topics.columns) == ["id", "topic_words"]
    assert questions_with_topics.empty
    assert list(questions_with_topics.columns) == ["id", "question", "topic_id"]
    topic_model.fit_transform.assert_called_once_with([])
    topic_model.get_topic.assert_not_called()
    assert returned_topic_model is topic_model


def test_generate_calls_cleanup_when_requested(monkeypatch):
    generator = TopicGenerator(
        language="german",
        embedding_model=Mock(),
    )
    topic_model = Mock()
    topic_model.get_topic.return_value = [("gap", 1.0)]
    topics = [0]
    probs = np.array([[1.0]])
    generator._fit_topic_model = Mock(return_value=(topic_model, topics, probs))
    generator.cleanup_topic_model = Mock()

    questions = pd.DataFrame(
        {"id": [0], "question": ["Welche Forschungsluecke bleibt offen?"]}
    )
    generator.generate(questions, cleanup_after_generation=True)

    generator.cleanup_topic_model.assert_called_once_with()


def test_generate_cleans_up_when_postprocessing_fails():
    generator = TopicGenerator(
        language="german",
        embedding_model="topic-model",
    )
    topic_model = Mock()
    generator._fit_topic_model = Mock(
        return_value=(topic_model, [0], np.array([[1.0]]))
    )
    generator._topic_words_by_topic = Mock(
        side_effect=RuntimeError("postprocessing failed")
    )
    generator.cleanup_topic_model = Mock()
    questions = pd.DataFrame(
        {"id": [0], "question": ["Welche Forschungsluecke bleibt offen?"]}
    )

    with pytest.raises(RuntimeError, match="postprocessing failed"):
        generator.generate(questions, cleanup_after_generation=True)

    generator.cleanup_topic_model.assert_called_once_with()


def test_save_model_calls_bertopic_save(tmp_path):
    topic_model = Mock()
    generator = TopicGenerator(
        language="german",
        embedding_model=Mock(),
    )
    generator.topic_model = topic_model

    generator.save_model(str(tmp_path), "run-1")

    topic_model.save.assert_called_once_with(
        tmp_path / "run-1" / "bertopic_question_model",
        serialization="pytorch",
        save_ctfidf=True,
    )


def test_cleanup_detaches_backend_without_moving_caller_model(monkeypatch):
    embedding_model = Mock()
    backend = Mock(embedding_model=embedding_model)
    topic_model = Mock(embedding_model=backend)
    generator = TopicGenerator(
        language="german",
        embedding_model=embedding_model,
    )
    generator.topic_model = topic_model
    monkeypatch.setattr("torch.cuda.is_available", Mock(return_value=False))

    generator.cleanup_topic_model()

    embedding_model.to.assert_not_called()
    assert topic_model.embedding_model is None


def test_cleanup_offloads_owned_bertopic_embedding(monkeypatch):
    embedding_model = Mock()
    backend = Mock(embedding_model=embedding_model)
    topic_model = Mock(embedding_model=backend)
    generator = TopicGenerator(
        language="german",
        embedding_model="topic-model",
    )
    generator.topic_model = topic_model
    monkeypatch.setattr("torch.cuda.is_available", Mock(return_value=False))

    generator.cleanup_topic_model()

    embedding_model.to.assert_called_once_with("cpu")
    assert topic_model.embedding_model is None
