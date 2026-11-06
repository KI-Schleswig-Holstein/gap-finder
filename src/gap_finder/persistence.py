"""Versioned, safe persistence helpers for :class:`GapFinder`."""

from __future__ import annotations

import json
import platform
import warnings
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from gap_finder.embedding_model.embedding_model import EmbeddingModel
from gap_finder.language_model.language_model import LanguageModel
from gap_finder.vector_store.vector_store import VectorStore

SNAPSHOT_FORMAT_VERSION = 1
SNAPSHOT_PREFIX = "gap_finder_model"
DEFAULT_SNAPSHOT_BASE_PATH = Path("gap_finder_models")
MANIFEST_FILE = "manifest.json"
QUESTIONS_FILE = Path("data/questions.json")
TOPICS_FILE = Path("data/topics.json")
VECTOR_STORE_DIR = Path("vector_store")
VECTOR_STORE_NAME = "store"
TOPIC_MODEL_DIR = Path("topic_model")

_VERSIONED_PACKAGES = (
    "gap-finder",
    "bertopic",
    "faiss-cpu",
    "numpy",
    "pandas",
    "safetensors",
    "sentence-transformers",
    "transformers",
)


def create_snapshot_paths(base_path: Path) -> tuple[Path, Path]:
    """Create a temporary snapshot directory and return it with its final path."""
    base_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot_id = f"{SNAPSHOT_PREFIX}_{timestamp}_{uuid4().hex[:8]}"
    final_path = base_path / snapshot_id
    temporary_path = base_path / f".{snapshot_id}.tmp"
    temporary_path.mkdir(parents=False, exist_ok=False)
    return temporary_path, final_path


def save_dataframe(dataframe: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame in a self-describing, non-pickle format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_json(
        path,
        orient="table",
        force_ascii=False,
        indent=2,
        double_precision=15,
        date_format="iso",
    )


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a DataFrame saved by :func:`save_dataframe`."""
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot DataFrame is missing: {path}")
    try:
        return pd.read_json(path, orient="table")
    except (TypeError, ValueError) as error:
        raise ValueError(f"Snapshot DataFrame is invalid: {path}") from error


def installed_versions() -> dict[str, str]:
    """Return versions useful for diagnosing snapshot compatibility."""
    versions = {}
    for package in _VERSIONED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def environment_metadata() -> dict[str, Any]:
    """Describe the runtime that created a snapshot."""
    return {
        "python": platform.python_version(),
        "packages": installed_versions(),
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Write the manifest last so its presence marks a complete temporary save."""
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_manifest(snapshot_path: Path) -> dict[str, Any]:
    """Load and validate a versioned GapFinder snapshot manifest."""
    if not snapshot_path.is_dir():
        raise FileNotFoundError(f"GapFinder snapshot directory not found: {snapshot_path}")
    manifest_path = snapshot_path / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"GapFinder snapshot manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"GapFinder snapshot manifest is invalid: {manifest_path}") from error

    if not isinstance(manifest, dict):
        raise ValueError("GapFinder snapshot manifest must contain a JSON object.")
    format_version = manifest.get("format_version")
    if format_version != SNAPSHOT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported GapFinder snapshot format version: "
            f"{format_version!r}; expected {SNAPSHOT_FORMAT_VERSION}."
        )
    for key in ("configuration", "artifacts", "models", "vector_store"):
        if not isinstance(manifest.get(key), dict):
            raise ValueError(f"GapFinder snapshot manifest is missing '{key}'.")

    _warn_about_environment(manifest.get("environment"))
    return manifest


def _warn_about_environment(saved_environment: Any) -> None:
    if not isinstance(saved_environment, dict):
        warnings.warn(
            "GapFinder snapshot has no environment metadata.",
            RuntimeWarning,
            stacklevel=3,
        )
        return

    saved_python = saved_environment.get("python")
    current_python = platform.python_version()
    if isinstance(saved_python, str) and saved_python != current_python:
        warnings.warn(
            f"Snapshot was created with Python {saved_python}; current Python is "
            f"{current_python}.",
            RuntimeWarning,
            stacklevel=3,
        )

    saved_packages = saved_environment.get("packages")
    if not isinstance(saved_packages, dict):
        return
    current_packages = installed_versions()
    mismatches = [
        f"{package} {saved_version} -> {current_packages[package]}"
        for package, saved_version in saved_packages.items()
        if package in current_packages and current_packages[package] != saved_version
    ]
    if mismatches:
        warnings.warn(
            "Snapshot dependency versions differ from the current environment: "
            + ", ".join(mismatches),
            RuntimeWarning,
            stacklevel=3,
        )


def describe_embedding_model(model: EmbeddingModel) -> dict[str, Any]:
    """Create a non-secret descriptor for a supported corpus embedding model."""
    from gap_finder.embedding_model.local_embedding_model import LocalEmbeddingModel
    from gap_finder.embedding_model.openai_embedding_model import OpenAIEmbeddingModel

    if isinstance(model, LocalEmbeddingModel):
        return {
            "kind": "sentence_transformer",
            "model_name": _require_identifier(model.model_name, "embedding model"),
        }
    if isinstance(model, OpenAIEmbeddingModel):
        return {
            "kind": "openai",
            "model_name": _require_identifier(model.model_name, "embedding model"),
            "base_url": _base_url(model.client),
        }
    raise ValueError(
        "Cannot save the corpus embedding model. Only LocalEmbeddingModel and "
        "OpenAIEmbeddingModel instances with stable identifiers are supported."
    )


def restore_embedding_model(
    descriptor: dict[str, Any],
    *,
    api_key: str | None,
) -> Any:
    """Reconstruct an embedding-model input from a saved descriptor."""
    kind = descriptor.get("kind")
    model_name = _descriptor_identifier(descriptor, "embedding model")
    if kind == "sentence_transformer":
        return model_name
    if kind == "openai":
        from gap_finder.embedding_model.client import OpenAIEmbeddingClient

        base_url = descriptor.get("base_url")
        if base_url is not None and not isinstance(base_url, str):
            raise ValueError("Embedding model base_url must be a string or null.")
        return OpenAIEmbeddingClient(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
        )
    raise ValueError(f"Unsupported embedding model kind in snapshot: {kind!r}.")


def describe_language_model(model: LanguageModel) -> dict[str, Any]:
    """Create a non-secret descriptor for a supported language model."""
    from gap_finder.language_model.local_language_model import LocalLanguageModel
    from gap_finder.language_model.openai_language_model import OpenAILanguageModel

    if isinstance(model, OpenAILanguageModel):
        return {
            "kind": "openai",
            "model_name": _require_identifier(
                model.client.model_name,
                "language model",
            ),
            "base_url": _base_url(model.client.client),
        }
    if isinstance(model, LocalLanguageModel):
        pipeline = model.client
        model_name = _first_identifier(
            getattr(pipeline.model, "name_or_path", None),
            getattr(getattr(pipeline.model, "config", None), "_name_or_path", None),
        )
        tokenizer_name = _first_identifier(
            getattr(pipeline.tokenizer, "name_or_path", None),
            model_name,
        )
        return {
            "kind": "transformers_pipeline",
            "task": _require_identifier(pipeline.task, "language model task"),
            "model_name": _require_identifier(model_name, "language model"),
            "tokenizer_name": _require_identifier(tokenizer_name, "tokenizer"),
        }
    raise ValueError(
        "Cannot save the language model. Only LocalLanguageModel and "
        "OpenAILanguageModel instances with stable identifiers are supported."
    )


def restore_language_model(
    descriptor: dict[str, Any],
    *,
    api_key: str | None,
    pipeline_kwargs: dict[str, Any] | None = None,
) -> LanguageModel:
    """Reconstruct a language-model wrapper from a saved descriptor."""
    kind = descriptor.get("kind")
    model_name = _descriptor_identifier(descriptor, "language model")
    if kind == "openai":
        from gap_finder.language_model.client import OpenAIClient
        from gap_finder.language_model.openai_language_model import OpenAILanguageModel

        base_url = descriptor.get("base_url")
        if base_url is not None and not isinstance(base_url, str):
            raise ValueError("Language model base_url in snapshot must be a string or null.")
        return OpenAILanguageModel(
            client=OpenAIClient(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
            )
        )
    if kind == "transformers_pipeline":
        from transformers import pipeline

        from gap_finder.language_model.local_language_model import LocalLanguageModel

        task = _required_descriptor_string(descriptor, "task", "language model task")
        if task != "text-generation":
            raise ValueError(
                "Local language-model snapshots must use the "
                "'text-generation' task."
            )
        tokenizer_name = _required_descriptor_string(
            descriptor,
            "tokenizer_name",
            "tokenizer",
        )
        runtime_kwargs = dict(pipeline_kwargs or {})
        reserved_kwargs = {"task", "model", "tokenizer"} & runtime_kwargs.keys()
        if reserved_kwargs:
            raise ValueError(
                "Local language-model pipeline overrides cannot replace: "
                f"{sorted(reserved_kwargs)}."
            )
        client = pipeline(
            task,
            model=model_name,
            tokenizer=tokenizer_name,
            **runtime_kwargs,
        )
        return LocalLanguageModel(client=client)
    raise ValueError(f"Unsupported language model kind in snapshot: {kind!r}.")


def describe_topic_embedding_model(model: Any) -> dict[str, str]:
    """Create a pointer descriptor for BERTopic's embedding model."""
    if isinstance(model, str):
        model_name = model
    else:
        from sentence_transformers import SentenceTransformer

        if not isinstance(model, SentenceTransformer):
            raise ValueError(
                "Cannot save the topic embedding model. Use a model name or a "
                "SentenceTransformer with a stable model_name_or_path."
            )
        model_name = getattr(model, "model_name_or_path", None)
    return {
        "kind": "sentence_transformer",
        "model_name": _require_identifier(model_name, "topic embedding model"),
    }


def restore_topic_embedding_model(descriptor: dict[str, Any]) -> str:
    """Restore the model pointer used by TopicGenerator and BERTopic."""
    if descriptor.get("kind") != "sentence_transformer":
        raise ValueError(
            "Unsupported topic embedding model kind in snapshot: "
            f"{descriptor.get('kind')!r}."
        )
    return _descriptor_identifier(descriptor, "topic embedding model")


def describe_vector_store(vector_store: VectorStore) -> dict[str, str]:
    """Return the stable registry key for a supported vector store."""
    from gap_finder.vector_store.numpy_vector_store import NumpyVectorStore

    if isinstance(vector_store, NumpyVectorStore):
        return {"kind": "numpy", "file_name": VECTOR_STORE_NAME}

    from gap_finder.vector_store.faiss_vector_store import FaissVectorStore

    if isinstance(vector_store, FaissVectorStore):
        return {"kind": "faiss", "file_name": VECTOR_STORE_NAME}
    raise ValueError(
        "Cannot save the vector store. Only NumpyVectorStore and "
        "FaissVectorStore are supported."
    )


def save_vector_store(vector_store: VectorStore, snapshot_path: Path) -> dict[str, str]:
    """Save a supported vector store and return its manifest descriptor."""
    descriptor = describe_vector_store(vector_store)
    vector_store.save_to(
        snapshot_path / VECTOR_STORE_DIR,
        descriptor["file_name"],
    )
    return descriptor


def restore_vector_store(descriptor: dict[str, Any], snapshot_path: Path) -> VectorStore:
    """Restore a vector store through the explicit implementation registry."""
    kind = descriptor.get("kind")
    file_name = _required_descriptor_string(
        descriptor,
        "file_name",
        "vector store file name",
    )
    if file_name != VECTOR_STORE_NAME:
        raise ValueError(f"Unsupported vector store file name: {file_name!r}.")
    if kind == "numpy":
        from gap_finder.vector_store.numpy_vector_store import NumpyVectorStore

        vector_store_class = NumpyVectorStore
    elif kind == "faiss":
        from gap_finder.vector_store.faiss_vector_store import FaissVectorStore

        vector_store_class = FaissVectorStore
    else:
        raise ValueError(f"Unsupported vector store kind in snapshot: {kind!r}.")
    try:
        return vector_store_class.load_from(
            snapshot_path / VECTOR_STORE_DIR,
            file_name,
        )
    except FileNotFoundError as error:
        raise FileNotFoundError("GapFinder snapshot vector-store files are missing.") from error


def save_topic_model(
    topic_model: Any,
    snapshot_path: Path,
    *,
    embedding_model_name: str,
) -> None:
    """Save fitted BERTopic state using its compact safe serialization."""
    from bertopic import BERTopic

    if not isinstance(topic_model, BERTopic):
        raise ValueError("Cannot save topic_model because it is not a BERTopic instance.")
    topic_model.save(
        snapshot_path / TOPIC_MODEL_DIR,
        serialization="safetensors",
        save_embedding_model=embedding_model_name,
        save_ctfidf=True,
    )


def restore_topic_model(snapshot_path: Path, *, embedding_model: Any) -> Any:
    """Load BERTopic state and attach its reconstructed embedding backend."""
    from bertopic import BERTopic

    topic_model_path = snapshot_path / TOPIC_MODEL_DIR
    if not topic_model_path.is_dir():
        raise FileNotFoundError(
            f"GapFinder snapshot topic model is missing: {topic_model_path}"
        )
    return BERTopic.load(str(topic_model_path), embedding_model=embedding_model)


def _descriptor_identifier(descriptor: dict[str, Any], label: str) -> str:
    return _required_descriptor_string(descriptor, "model_name", label)


def _required_descriptor_string(
    descriptor: dict[str, Any],
    key: str,
    label: str,
) -> str:
    return _require_identifier(descriptor.get(key), label)


def _require_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Cannot save or restore {label} without a stable identifier.")
    return value


def _first_identifier(*values: Any) -> str | None:
    return next(
        (value for value in values if isinstance(value, str) and value.strip()),
        None,
    )


def _base_url(client: Any) -> str | None:
    base_url = getattr(client, "base_url", None)
    return str(base_url) if base_url is not None else None
