import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from gap_finder import persistence
from gap_finder.embedding_model.client import OpenAIEmbeddingClient
from gap_finder.embedding_model.embedding_model import EmbeddingModel
from gap_finder.language_model.local_language_model import LocalLanguageModel
from gap_finder.main import GapFinder
from gap_finder.vector_store.faiss_vector_store import FaissVectorStore
from gap_finder.vector_store.numpy_vector_store import NumpyVectorStore


def make_store(vector_store_class):
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    store = vector_store_class(dim=3)
    store.add(embeddings, ["first chunk", "second chunk"])
    return store, embeddings


@pytest.mark.parametrize("vector_store_class", [NumpyVectorStore, FaissVectorStore])
def test_complete_snapshot_round_trip(vector_store_class, tmp_path):
    store, embeddings = make_store(vector_store_class)
    gap_finder = GapFinder(
        store,
        embedding_model="embedding-model",
        language_model="language-model",
        topic_embedding_model="topic-embedding-model",
        api_key="-",
        k_questions=7,
        embedding_batch_size=13,
        question_generation_batch_size=4,
        summarize_chunks=False,
        language="german",
        domain_specific_questions=True,
        domain_description="Machine learning",
        explain_top_k_topics=5,
        topic_explanation_sample_size=6,
        cleanup_models_after_usage=True,
        save_intermediate_results=True,
        base_path=tmp_path,
    )
    gap_finder.questions = pd.DataFrame(
        {
            "id": [0],
            "question": ["Was fehlt?"],
            "embedding": [[0.25, 0.75]],
            "topic_id": [2],
        }
    )
    gap_finder.topics = pd.DataFrame(
        {
            "id": [2],
            "topic_words": [["gap", "evidence"]],
            "avg_similarity": [0.25],
            "report": ["## Gap\n"],
        }
    )

    snapshot_path = gap_finder.save_model()
    restored = GapFinder.load_model(snapshot_path, api_key="-")

    assert snapshot_path.parent == tmp_path
    assert snapshot_path.name.startswith("gap_finder_model_")
    assert restored.base_path == tmp_path
    assert restored.intermediate_results_path is None
    assert isinstance(restored.vector_store, vector_store_class)
    restored_vectors, restored_chunks = restored.vector_store.get_all()
    original_vectors, original_chunks = store.get_all()
    np.testing.assert_allclose(restored_vectors, original_vectors)
    assert restored_chunks == original_chunks
    _, _, _, matches = restored.vector_store.search(embeddings[1], k=1)
    assert matches == ["second chunk"]
    pd.testing.assert_frame_equal(restored.questions, gap_finder.questions)
    pd.testing.assert_frame_equal(restored.topics, gap_finder.topics)
    assert not hasattr(restored, "topic_model")
    assert restored.k_questions == 7
    assert restored.embedding_batch_size == 13
    assert restored.question_generation_batch_size == 4
    assert restored.summarize_chunks is False
    assert restored.language == "german"
    assert restored.domain_specific_questions is True
    assert restored.domain_description == "Machine learning"
    assert restored.explain_top_k_topics == 5
    assert restored.topic_explanation_sample_size == 6
    assert restored.cleanup_models_after_usage is True
    assert restored.save_intermediate_results is True


def test_snapshot_preserves_none_empty_and_absent_attributes(tmp_path):
    store, _ = make_store(NumpyVectorStore)
    gap_finder = GapFinder(
        store,
        "embedding-model",
        "language-model",
        api_key="-",
        questions=pd.DataFrame(columns=["id", "question"]),
        base_path=tmp_path,
    )
    gap_finder.topics = None

    restored = GapFinder.load_model(gap_finder.save_model(), api_key="-")

    assert isinstance(restored.questions, pd.DataFrame)
    assert restored.questions.empty
    assert list(restored.questions.columns) == ["id", "question"]
    assert restored.topics is None
    assert not hasattr(restored, "topic_model")


def test_snapshot_preserves_none_questions(tmp_path):
    store, _ = make_store(NumpyVectorStore)
    gap_finder = GapFinder(
        store,
        "embedding-model",
        "language-model",
        api_key="-",
        base_path=tmp_path,
    )

    restored = GapFinder.load_model(gap_finder.save_model(), api_key="-")

    assert restored.questions is None
    assert not hasattr(restored, "topics")


def test_save_model_uses_default_base_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    store, _ = make_store(NumpyVectorStore)
    gap_finder = GapFinder(
        store, "embedding-model", "language-model", api_key="-"
    )

    first_path = gap_finder.save_model()
    second_path = gap_finder.save_model()

    assert first_path.parent == tmp_path / "gap_finder_models"
    assert second_path.parent == first_path.parent
    assert first_path != second_path


def test_api_credentials_are_not_saved_and_load_keys_are_used(tmp_path):
    from openai import OpenAI

    store, _ = make_store(NumpyVectorStore)
    embedding_client = OpenAIEmbeddingClient(
        client=OpenAI(
            api_key="embedding-save-secret",
            base_url="http://embedding.test/v1",
        ),
        model_name="embedding-api-model",
    )
    gap_finder = GapFinder(
        store,
        embedding_model=embedding_client,
        language_model="language-api-model",
        base_url="http://language.test/v1",
        api_key="language-save-secret",
        base_path=tmp_path,
    )

    snapshot_path = gap_finder.save_model()
    manifest_text = (snapshot_path / persistence.MANIFEST_FILE).read_text()
    restored = GapFinder.load_model(
        snapshot_path,
        api_key="language-load-secret",
        embedding_api_key="embedding-load-secret",
    )

    assert "embedding-save-secret" not in manifest_text
    assert "language-save-secret" not in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["models"]["embedding"]["base_url"] == (
        "http://embedding.test/v1/"
    )
    assert manifest["models"]["language"]["base_url"] == (
        "http://language.test/v1/"
    )
    assert restored.embedding_model.client.api_key == "embedding-load-secret"
    assert restored.language_model.client.client.api_key == "language-load-secret"


def test_local_language_model_descriptor_round_trip(monkeypatch):
    client = Mock()
    client.task = "text-generation"
    client.model.name_or_path = "local-language-model"
    client.tokenizer.name_or_path = "local-tokenizer"
    language_model = LocalLanguageModel(client=client)
    restored_client = Mock(task="text-generation")
    pipeline_factory = Mock(return_value=restored_client)
    monkeypatch.setattr("transformers.pipeline", pipeline_factory)

    descriptor = persistence.describe_language_model(language_model)
    restored = persistence.restore_language_model(
        descriptor,
        api_key=None,
        pipeline_kwargs={
            "device_map": "auto",
            "model_kwargs": {"dtype": "auto"},
        },
    )

    assert descriptor == {
        "kind": "transformers_pipeline",
        "task": "text-generation",
        "model_name": "local-language-model",
        "tokenizer_name": "local-tokenizer",
    }
    assert isinstance(restored, LocalLanguageModel)
    pipeline_factory.assert_called_once_with(
        "text-generation",
        model="local-language-model",
        tokenizer="local-tokenizer",
        device_map="auto",
        model_kwargs={"dtype": "auto"},
    )


def test_local_language_model_rejects_identifier_overrides(monkeypatch):
    descriptor = {
        "kind": "transformers_pipeline",
        "task": "text-generation",
        "model_name": "local-language-model",
        "tokenizer_name": "local-tokenizer",
    }
    pipeline_factory = Mock()
    monkeypatch.setattr("transformers.pipeline", pipeline_factory)

    with pytest.raises(ValueError, match="cannot replace"):
        persistence.restore_language_model(
            descriptor,
            api_key=None,
            pipeline_kwargs={"model": "different-model"},
        )

    pipeline_factory.assert_not_called()


def test_local_language_model_rejects_unsupported_task(monkeypatch):
    descriptor = {
        "kind": "transformers_pipeline",
        "task": "text-classification",
        "model_name": "local-language-model",
        "tokenizer_name": "local-tokenizer",
    }
    pipeline_factory = Mock()
    monkeypatch.setattr("transformers.pipeline", pipeline_factory)

    with pytest.raises(ValueError, match="must use the 'text-generation' task"):
        persistence.restore_language_model(descriptor, api_key=None)

    pipeline_factory.assert_not_called()


def test_topic_model_is_saved_and_restored(monkeypatch, tmp_path):
    store, _ = make_store(NumpyVectorStore)
    gap_finder = GapFinder(
        store,
        "embedding-model",
        "language-model",
        api_key="-",
        topic_embedding_model="topic-embedding-model",
        base_path=tmp_path,
    )
    gap_finder.topic_model = object()
    restored_topic_model = object()
    save_topic_model = Mock(
        side_effect=lambda _, snapshot_path, **__: (
            snapshot_path / persistence.TOPIC_MODEL_DIR
        ).mkdir()
    )
    restore_topic_model = Mock(return_value=restored_topic_model)
    monkeypatch.setattr(persistence, "save_topic_model", save_topic_model)
    monkeypatch.setattr(persistence, "restore_topic_model", restore_topic_model)

    snapshot_path = gap_finder.save_model()
    restored = GapFinder.load_model(snapshot_path, api_key="-")

    assert restored.topic_model is restored_topic_model
    save_topic_model.assert_called_once_with(
        gap_finder.topic_model,
        snapshot_path.parent / f".{snapshot_path.name}.tmp",
        embedding_model_name="topic-embedding-model",
    )
    restore_topic_model.assert_called_once_with(
        snapshot_path,
        embedding_model="topic-embedding-model",
    )


def test_topic_model_helpers_use_safe_bertopic_format(monkeypatch, tmp_path):
    from bertopic import BERTopic

    topic_model = Mock(spec=BERTopic)
    persistence.save_topic_model(
        topic_model,
        tmp_path,
        embedding_model_name="topic-embedding-model",
    )

    topic_model.save.assert_called_once_with(
        tmp_path / persistence.TOPIC_MODEL_DIR,
        serialization="safetensors",
        save_embedding_model="topic-embedding-model",
        save_ctfidf=True,
    )

    (tmp_path / persistence.TOPIC_MODEL_DIR).mkdir()
    restored_topic_model = object()
    loader = Mock(return_value=restored_topic_model)
    monkeypatch.setattr(BERTopic, "load", loader)

    restored = persistence.restore_topic_model(
        tmp_path,
        embedding_model="topic-embedding-model",
    )

    assert restored is restored_topic_model
    loader.assert_called_once_with(
        str(tmp_path / persistence.TOPIC_MODEL_DIR),
        embedding_model="topic-embedding-model",
    )


def test_unsupported_custom_model_fails_before_creating_snapshot(tmp_path):
    class CustomEmbeddingModel(EmbeddingModel):
        def encode(self, texts, show_progress_bar=False, batch_size=32):
            return np.ones((len(texts), 3), dtype=np.float32)

    store, _ = make_store(NumpyVectorStore)
    gap_finder = GapFinder(
        store,
        embedding_model=CustomEmbeddingModel(),
        language_model="language-model",
        api_key="-",
        base_path=tmp_path,
    )

    with pytest.raises(ValueError, match="corpus embedding model"):
        gap_finder.save_model()

    assert list(tmp_path.iterdir()) == []


def test_failed_save_removes_temporary_snapshot(monkeypatch, tmp_path):
    store, _ = make_store(NumpyVectorStore)
    gap_finder = GapFinder(
        store,
        "embedding-model",
        "language-model",
        api_key="-",
        base_path=tmp_path,
    )
    monkeypatch.setattr(
        persistence,
        "save_vector_store",
        Mock(side_effect=RuntimeError("save failed")),
    )

    with pytest.raises(RuntimeError, match="save failed"):
        gap_finder.save_model()

    assert list(tmp_path.iterdir()) == []


def test_load_rejects_unsupported_snapshot_version(tmp_path):
    snapshot_path = tmp_path / "snapshot"
    snapshot_path.mkdir()
    (snapshot_path / persistence.MANIFEST_FILE).write_text(
        json.dumps({"format_version": persistence.SNAPSHOT_FORMAT_VERSION + 1}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported GapFinder snapshot"):
        GapFinder.load_model(snapshot_path)


def test_load_warns_about_dependency_version_difference(tmp_path):
    store, _ = make_store(NumpyVectorStore)
    gap_finder = GapFinder(
        store,
        "embedding-model",
        "language-model",
        api_key="-",
        base_path=tmp_path,
    )
    snapshot_path = gap_finder.save_model()
    manifest_path = snapshot_path / persistence.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["environment"]["packages"]["pandas"] = "0.0.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="pandas 0.0.0"):
        restored = GapFinder.load_model(snapshot_path, api_key="-")

    assert isinstance(restored, GapFinder)


def test_load_rejects_unknown_vector_store_kind(tmp_path):
    store, _ = make_store(NumpyVectorStore)
    gap_finder = GapFinder(
        store,
        "embedding-model",
        "language-model",
        api_key="-",
        base_path=tmp_path,
    )
    snapshot_path = gap_finder.save_model()
    manifest_path = snapshot_path / persistence.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["vector_store"]["kind"] = "unknown"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported vector store kind"):
        GapFinder.load_model(snapshot_path, api_key="-")


def test_load_reports_missing_dataframe(tmp_path):
    store, _ = make_store(NumpyVectorStore)
    gap_finder = GapFinder(
        store,
        "embedding-model",
        "language-model",
        api_key="-",
        questions=pd.DataFrame({"question": ["What is missing?"]}),
        base_path=tmp_path,
    )
    snapshot_path = gap_finder.save_model()
    (snapshot_path / persistence.QUESTIONS_FILE).unlink()

    with pytest.raises(FileNotFoundError, match="DataFrame is missing"):
        GapFinder.load_model(snapshot_path, api_key="-")
