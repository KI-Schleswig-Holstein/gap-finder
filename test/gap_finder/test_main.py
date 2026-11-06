"""
Tests for gap_finder.main.GapFinder.

These tests cover pipeline wiring and persistence with mocks and synthetic
embeddings. Real model and PDF pipelines live under test/integration.
"""

import inspect
from typing import get_type_hints
from unittest.mock import Mock, call

import numpy as np
import pandas as pd
import pytest

from gap_finder.main import GapFinder, STANDARD_LANGUAGE_MODEL
from gap_finder.embedding_model.local_embedding_model import LocalEmbeddingModel
from gap_finder.embedding_model.embedding_model import EmbeddingModel
from gap_finder.language_model.language_model import LanguageModel
from gap_finder.language_model.openai_language_model import OpenAILanguageModel
from gap_finder.vector_store.faiss_vector_store import FaissVectorStore
from gap_finder.vector_store.numpy_vector_store import NumpyVectorStore


class TestGapFinderInitNetworkFree:
    def test_public_init_annotations_are_introspectable(self):
        hints = get_type_hints(GapFinder.__init__)

        assert "embedding_model" in hints
        assert "language_model" in hints
        assert "topic_embedding_model" in hints
        assert "embedding_batch_size" in inspect.signature(
            GapFinder.from_pdf
        ).parameters
        assert "embedding_batch_size" in inspect.signature(
            GapFinder.from_chunks
        ).parameters

    def test_init_wires_up_attributes(self):
        store = NumpyVectorStore(dim=4)
        gap_finder = GapFinder(store, "emb-model", "language-model", api_key="-")

        assert gap_finder.vector_store is store
        assert isinstance(gap_finder.embedding_model, LocalEmbeddingModel)
        assert gap_finder.embedding_model.model_name == "emb-model"
        assert isinstance(gap_finder.language_model, OpenAILanguageModel)
        assert gap_finder.language_model.client.model_name == "language-model"

    def test_saving_intermediate_results_requires_base_path(self):
        with pytest.raises(ValueError, match="base_path must be provided"):
            GapFinder(
                NumpyVectorStore(dim=4),
                "emb-model",
                "language-model",
                api_key="-",
                save_intermediate_results=True,
            )

    def test_intermediate_saving_is_disabled_by_default(self, monkeypatch):
        gap_finder = GapFinder(
            NumpyVectorStore(dim=4), "emb-model", "language-model", api_key="-"
        )
        file_handler = Mock(side_effect=AssertionError("must not be created"))
        monkeypatch.setattr("gap_finder.main.FileHandler", file_handler)

        assert gap_finder._create_file_handler() is None
        assert gap_finder.intermediate_results_path is None
        file_handler.assert_not_called()

    def test_each_find_call_creates_a_new_run_folder(self, tmp_path, monkeypatch):
        gap_finder = GapFinder(
            NumpyVectorStore(dim=4),
            "emb-model",
            "language-model",
            api_key="-",
            save_intermediate_results=True,
            base_path=tmp_path,
        )
        reported_topics = pd.DataFrame(
            {
                "id": [1],
                "report": ["## A gap\n\nDetails.\n"],
            }
        )
        topic_model = object()

        monkeypatch.setattr(
            gap_finder,
            "_load_chunks_and_embeddings",
            Mock(return_value=pd.DataFrame({"id": [0], "text": ["chunk"]})),
        )
        monkeypatch.setattr(
            gap_finder,
            "_get_or_create_questions",
            Mock(return_value=pd.DataFrame({"question": ["question"]})),
        )
        monkeypatch.setattr(
            gap_finder,
            "_create_topics",
            Mock(
                return_value=(
                    pd.DataFrame({"id": [1]}),
                    pd.DataFrame({"question": ["question"], "topic_id": [1]}),
                    topic_model,
                )
            ),
        )
        monkeypatch.setattr(
            gap_finder,
            "_rank_topics",
            Mock(
                return_value=(
                    pd.DataFrame({"id": [1], "avg_similarity": [0.2]}),
                    pd.DataFrame(
                        {"question": ["question"], "topic_id": [1], "similarity": [0.2]}
                    ),
                )
            ),
        )
        monkeypatch.setattr(
            gap_finder,
            "_explain_topics",
            Mock(
                return_value=pd.DataFrame(
                    {"id": [1], "avg_similarity": [0.2], "explanation": ["Details"]}
                )
            ),
        )
        monkeypatch.setattr(
            gap_finder, "_create_report", Mock(return_value=reported_topics)
        )

        assert gap_finder.find() is reported_topics
        assert gap_finder.topic_model is topic_model
        first_run_path = gap_finder.intermediate_results_path
        assert first_run_path is not None
        assert (first_run_path / "report.md").read_text(
            encoding="utf-8"
        ) == "## A gap\n\nDetails.\n"

        gap_finder.find()
        assert gap_finder.intermediate_results_path != first_run_path

    def test_create_questions_with_default_vector_store_and_language_model(
        self, monkeypatch, mock_embedding_model
    ):
        mock_embedding_model.encode.return_value = np.array(
            [[1.0, 0.0]], dtype=np.float32
        )
        language_model = Mock(spec=LanguageModel)
        create_language_model = Mock(return_value=language_model)
        question_generator = Mock()
        generated_questions = pd.DataFrame(
            {
                "id": [0],
                "question": ["Welche Forschungsluecke bleibt offen?"],
                "chunk_id": [0],
            }
        )
        question_generator.generate.return_value = generated_questions
        question_generator_cls = Mock(return_value=question_generator)

        monkeypatch.setattr(
            "gap_finder.main.language_model_factory.create_language_model",
            create_language_model,
        )
        monkeypatch.setattr("gap_finder.main.QuestionGenerator", question_generator_cls)

        gap_finder = GapFinder.from_chunks(
            chunks=["chunk 1"],
            embedding_model=mock_embedding_model,
        )

        questions = gap_finder.create_questions()

        assert isinstance(gap_finder.vector_store, NumpyVectorStore)
        assert gap_finder.language_model is language_model
        assert questions is generated_questions
        assert gap_finder.questions is generated_questions
        mock_embedding_model.encode.assert_called_once_with(
            ["chunk 1"], show_progress_bar=True, batch_size=32
        )
        language_model.activate.assert_called_once_with()
        language_model.deactivate.assert_not_called()
        create_language_model.assert_called_once_with(
            language_model=STANDARD_LANGUAGE_MODEL,
            base_url=None,
            api_key=None,
        )
        question_generator_cls.assert_called_once_with(
            language_model=language_model,
            k_questions=3,
            batch_size=1,
            summarize_chunks=True,
            domain_specific_questions=False,
            language="english",
            domain_description=None,
        )
        generated_chunks = question_generator.generate.call_args.kwargs["chunks"]
        assert generated_chunks["text"].tolist() == ["chunk 1"]
        assert generated_chunks["id"].tolist() == [0]
        np.testing.assert_array_equal(
            np.stack(generated_chunks["embedding"]),
            np.array([[1.0, 0.0]], dtype=np.float32),
        )

    def test_save_model_is_parameterless(self):
        assert list(inspect.signature(GapFinder.save_model).parameters) == ["self"]

    def test_create_language_model_uses_configured_model_settings(self, monkeypatch):
        language_model = object()
        create_language_model = Mock(return_value=language_model)
        monkeypatch.setattr(
            "gap_finder.main.language_model_factory.create_language_model",
            create_language_model,
        )

        gap_finder = GapFinder(
            NumpyVectorStore(dim=4),
            "emb-model",
            "language-model",
            base_url="http://example.test/v1",
            api_key="secret",
        )

        assert gap_finder.language_model is language_model
        create_language_model.assert_called_once_with(
            language_model="language-model",
            base_url="http://example.test/v1",
            api_key="secret",
        )

    def test_create_embedding_model_uses_configured_model_settings(self, monkeypatch):
        embedding_model = Mock(spec=EmbeddingModel)
        create_embedding_model = Mock(return_value=embedding_model)
        monkeypatch.setattr(
            "gap_finder.main.embedding_model_factory.create_embedding_model",
            create_embedding_model,
        )
        gap_finder = GapFinder(
            NumpyVectorStore(dim=4),
            "corpus-model",
            api_key="-",
        )

        assert gap_finder.embedding_model is embedding_model
        assert create_embedding_model.call_args_list == [
            call(embedding_model="corpus-model"),
        ]

    def test_from_chunks_uses_api_embedding_model_and_batch_contract(self, monkeypatch):
        indexing_model = Mock()
        indexing_model.encode.return_value = np.array(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        create_embedding_model = Mock(return_value=indexing_model)
        monkeypatch.setattr(
            "gap_finder.main.embedding_model_factory.create_embedding_model",
            create_embedding_model,
        )

        gap_finder = GapFinder.from_chunks(
            chunks=["first", "second"],
            embedding_model="api-embedding-model",
            embedding_batch_size=8,
            api_key="-",
        )

        assert create_embedding_model.call_args_list == [
            call(embedding_model="api-embedding-model"),
            call(embedding_model=indexing_model),
        ]
        indexing_model.encode.assert_called_once_with(
            ["first", "second"],
            show_progress_bar=True,
            batch_size=8,
        )
        indexing_model.cleanup.assert_not_called()
        indexing_model.deactivate.assert_not_called()
        assert gap_finder.embedding_model is indexing_model
        assert gap_finder.embedding_batch_size == 8
        assert gap_finder.vector_store.get_all()[1] == ["first", "second"]

    def test_from_chunks_deactivates_reused_model_when_cleanup_is_enabled(
        self, mock_embedding_model
    ):
        embedding_model = mock_embedding_model
        embedding_model.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)

        gap_finder = GapFinder.from_chunks(
            chunks=["first"],
            embedding_model=embedding_model,
            cleanup_models_after_usage=True,
            api_key="-",
        )

        assert gap_finder.embedding_model is embedding_model
        embedding_model.deactivate.assert_called_once_with()
        embedding_model.cleanup.assert_not_called()

    def test_from_chunks_cleans_up_model_when_construction_fails(
        self, mock_embedding_model
    ):
        embedding_model = mock_embedding_model
        embedding_model.encode.side_effect = RuntimeError("embedding failed")

        with pytest.raises(RuntimeError, match="embedding failed"):
            GapFinder.from_chunks(
                chunks=["first"],
                embedding_model=embedding_model,
            )

        embedding_model.cleanup.assert_called_once_with()
        embedding_model.deactivate.assert_not_called()

    def test_create_topics_uses_topic_embedding_model(self, monkeypatch):
        topic_generator = Mock()
        expected = (object(), object(), object())
        topic_generator.generate.return_value = expected
        topic_generator_cls = Mock(return_value=topic_generator)
        monkeypatch.setattr(
            "gap_finder.topic_generation.TopicGenerator",
            topic_generator_cls,
        )
        gap_finder = GapFinder(
            NumpyVectorStore(dim=4),
            "corpus-model",
            api_key="-",
            topic_embedding_model="topic-model",
            questions=["What is missing?"],
        )
        questions = pd.DataFrame({"question": ["What is missing?"]})

        assert gap_finder._create_topics(questions) == expected
        topic_generator_cls.assert_called_once_with(
            language="english",
            embedding_model="topic-model",
        )
        topic_generator.generate.assert_called_once_with(
            questions=questions,
            cleanup_after_generation=False,
        )

    def test_rank_topics_uses_corpus_embedding_model(self, monkeypatch):
        embedding_model = Mock(spec=EmbeddingModel)
        ranked_questions = object()
        ranked_topics = object()
        question_ranker = Mock()
        question_ranker.rank.return_value = ranked_questions
        question_ranker_cls = Mock(return_value=question_ranker)
        topic_ranker = Mock()
        topic_ranker.rank.return_value = ranked_topics
        topic_ranker_cls = Mock(return_value=topic_ranker)
        monkeypatch.setattr(
            "gap_finder.question_ranking.QuestionRanker",
            question_ranker_cls,
        )
        monkeypatch.setattr(
            "gap_finder.topic_ranking.TopicRanker",
            topic_ranker_cls,
        )
        gap_finder = GapFinder(
            NumpyVectorStore(dim=4),
            "corpus-model",
            api_key="-",
            embedding_batch_size=8,
        )
        gap_finder.embedding_model = embedding_model
        topics = object()
        questions = object()

        assert gap_finder._rank_topics(topics, questions) == (
            ranked_topics,
            ranked_questions,
        )
        question_ranker_cls.assert_called_once_with(
            vector_store=gap_finder.vector_store,
            embedding_model=embedding_model,
            top_k=1,
            batch_size=8,
        )
        question_ranker.rank.assert_called_once_with(questions=questions)
        embedding_model.activate.assert_called_once_with()
        embedding_model.deactivate.assert_not_called()
        topic_ranker.rank.assert_called_once_with(
            questions_with_topics=ranked_questions,
            topics=topics,
        )

    def test_model_step_deactivates_only_when_cleanup_is_enabled(self):
        gap_finder = GapFinder(
            NumpyVectorStore(dim=4),
            "emb-model",
            "language-model",
            api_key="-",
            cleanup_models_after_usage=True,
        )
        model = Mock(spec=LanguageModel)

        with gap_finder._model_step(model):
            model.invoke("first")
            model.invoke("second")

        model.activate.assert_called_once_with()
        model.deactivate.assert_called_once_with()
        model.cleanup.assert_not_called()

    def test_find_cleans_models_after_failure_when_enabled(self, monkeypatch):
        gap_finder = GapFinder(
            NumpyVectorStore(dim=4),
            "emb-model",
            "language-model",
            api_key="-",
            cleanup_models_after_usage=True,
        )
        embedding_model = Mock(spec=EmbeddingModel)
        language_model = Mock(spec=LanguageModel)
        gap_finder.embedding_model = embedding_model
        gap_finder.language_model = language_model
        monkeypatch.setattr(
            gap_finder,
            "_run_pipeline",
            Mock(side_effect=RuntimeError("pipeline failed")),
        )

        with pytest.raises(RuntimeError, match="pipeline failed"):
            gap_finder.find()

        embedding_model.cleanup.assert_called_once_with()
        language_model.cleanup.assert_called_once_with()

    def test_find_does_not_cleanup_models_when_disabled(self, monkeypatch):
        gap_finder = GapFinder(
            NumpyVectorStore(dim=4),
            "emb-model",
            "language-model",
            api_key="-",
        )
        embedding_model = Mock(spec=EmbeddingModel)
        language_model = Mock(spec=LanguageModel)
        gap_finder.embedding_model = embedding_model
        gap_finder.language_model = language_model
        expected = pd.DataFrame({"id": [1]})
        monkeypatch.setattr(gap_finder, "_run_pipeline", Mock(return_value=expected))

        assert gap_finder.find() is expected
        embedding_model.cleanup.assert_not_called()
        language_model.cleanup.assert_not_called()

    def test_find_skips_topic_relevance_step_without_domain_description(self, monkeypatch):
        gap_finder = GapFinder(
            NumpyVectorStore(dim=4), "emb-model", "language-model", api_key="-"
        )
        calls = []
        chunks_data = object()
        initial_questions = object()
        generated_topics = object()
        assigned_questions = object()
        topic_model = object()
        ranked_topics_data = object()
        ranked_questions_data = object()
        explained_topics_data = object()
        reported_topics = object()

        def get_questions(*, chunks: object):
            calls.append("questions")
            assert chunks is chunks_data
            return initial_questions

        def create_topics(*, questions: object):
            calls.append("topics")
            assert questions is initial_questions
            return generated_topics, assigned_questions, topic_model

        def rank_topics(*, topics: object, questions_with_topics: object):
            calls.append("rank")
            assert topics is generated_topics
            assert questions_with_topics is assigned_questions
            return ranked_topics_data, ranked_questions_data

        def explain_topics(*, questions_with_topics: object, topics: object):
            calls.append("explain")
            assert questions_with_topics is ranked_questions_data
            assert topics is ranked_topics_data
            return explained_topics_data

        def create_report(*, topics: object):
            calls.append("report")
            assert topics is explained_topics_data
            return reported_topics

        monkeypatch.setattr(
            gap_finder, "_load_chunks_and_embeddings", Mock(return_value=chunks_data)
        )
        monkeypatch.setattr(gap_finder, "_get_or_create_questions", get_questions)
        monkeypatch.setattr(gap_finder, "_create_topics", create_topics)
        monkeypatch.setattr(gap_finder, "_rank_topics", rank_topics)
        monkeypatch.setattr(gap_finder, "_explain_topics", explain_topics)
        monkeypatch.setattr(gap_finder, "_create_report", create_report)

        result_topics = gap_finder.find()

        assert result_topics is reported_topics
        assert gap_finder.topics is reported_topics
        assert gap_finder.questions is ranked_questions_data
        assert gap_finder.topic_model is topic_model
        assert calls == ["questions", "topics", "rank", "explain", "report"]

    def test_find_runs_topic_relevance_step_with_domain_description(self, monkeypatch):
        gap_finder = GapFinder(
            NumpyVectorStore(dim=4),
            "emb-model",
            "language-model",
            api_key="-",
            domain_description="The domain is machine learning research.",
        )
        calls = []
        chunks_data = object()
        initial_questions = object()
        generated_topics = object()
        assigned_questions = object()
        topic_model = object()
        ranked_topics_data = object()
        ranked_questions_data = object()
        explained_topics_data = object()
        relevant_topics_data = object()
        reported_topics = object()

        def get_questions(*, chunks: object):
            calls.append("questions")
            assert chunks is chunks_data
            return initial_questions

        def create_topics(*, questions: object):
            calls.append("topics")
            assert questions is initial_questions
            return generated_topics, assigned_questions, topic_model

        def rank_topics(*, topics: object, questions_with_topics: object):
            calls.append("rank")
            assert topics is generated_topics
            assert questions_with_topics is assigned_questions
            return ranked_topics_data, ranked_questions_data

        def explain_topics(*, questions_with_topics: object, topics: object):
            calls.append("explain")
            assert questions_with_topics is ranked_questions_data
            assert topics is ranked_topics_data
            return explained_topics_data

        def score_topic_relevance(*, topics: object):
            calls.append("relevance")
            assert topics is explained_topics_data
            return relevant_topics_data

        def create_report(*, topics: object):
            calls.append("report")
            assert topics is relevant_topics_data
            return reported_topics

        monkeypatch.setattr(
            gap_finder, "_load_chunks_and_embeddings", Mock(return_value=chunks_data)
        )
        monkeypatch.setattr(gap_finder, "_get_or_create_questions", get_questions)
        monkeypatch.setattr(gap_finder, "_create_topics", create_topics)
        monkeypatch.setattr(gap_finder, "_rank_topics", rank_topics)
        monkeypatch.setattr(gap_finder, "_explain_topics", explain_topics)
        monkeypatch.setattr(gap_finder, "_score_topic_relevance", score_topic_relevance)
        monkeypatch.setattr(gap_finder, "_create_report", create_report)

        result_topics = gap_finder.find()

        assert result_topics is reported_topics
        assert gap_finder.topics is reported_topics
        assert gap_finder.questions is ranked_questions_data
        assert gap_finder.topic_model is topic_model
        assert calls == ["questions", "topics", "rank", "explain", "relevance", "report"]

    def test_from_pdf_warns_when_folder_has_no_pdfs(
        self, tmp_path, mock_embedding_model, mock_language_model
    ):
        # Mirrors SentenceTransformer.encode([]): a 1-D empty array,
        # not (0, dim) - that's what makes embeddings.shape[1] fail.
        mock_embedding_model.encode.return_value = np.empty((0,), dtype=np.float32)

        with pytest.warns(UserWarning, match="No PDFs found"):
            with pytest.raises(IndexError):
                # embeddings.shape[1] on the empty array raises - documents the
                # pre-existing empty-chunks edge case; the warning is the fix.
                GapFinder.from_pdf(
                    folder_path=str(tmp_path),
                    embedding_model=mock_embedding_model,
                    language_model=mock_language_model,
                )

    @pytest.mark.parametrize("vector_store_cls", [NumpyVectorStore, FaissVectorStore])
    def test_save_vector_store_and_reload_via_init(self, vector_store_cls, tmp_path):
        rng = np.random.default_rng(0)
        embeddings = rng.random((3, 4), dtype=np.float64).astype(np.float32)
        chunks = ["a", "b", "c"]

        store = vector_store_cls(dim=4)
        store.add(embeddings, chunks)

        gap_finder = GapFinder(store, "emb-model", "language-a", api_key="-")
        gap_finder.save_vector_store(tmp_path)

        loaded_store = vector_store_cls.load_pdf_vector_store(tmp_path, "emb-model")
        restored = GapFinder(
            loaded_store, "emb-model", "language-b", api_key="-"
        )

        assert restored.vector_store.get_all()[1] == chunks
        assert restored.language_model.client.model_name == "language-b"

        _, _, _, retrieved_chunks = restored.vector_store.search(embeddings[1:2].copy(), k=1)
        assert retrieved_chunks[0] == "b"
