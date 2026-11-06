"""Shared runtime configuration for integration tests."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
from transformers import pipeline

from gap_finder.language_model.language_model import LanguageModel
from gap_finder.language_model import OpenAIClient, create_language_model
from gap_finder.embedding_model import (
    EmbeddingModel,
    OpenAIEmbeddingClient,
    create_embedding_model,
)

ENV_PREFIX = "GAP_FINDER_"


def env(name: str, default: str) -> str:
    """Read an integration setting with GAP_FINDER_ and legacy fallbacks."""
    return os.getenv(f"{ENV_PREFIX}{name}", os.getenv(name, default))


def env_int(name: str, default: int) -> int:
    return int(env(name, str(default)))


def _normalize_backend(backend: str) -> str:
    normalized = backend.strip().lower().replace("-", "_")
    aliases = {
        "both": "all",
    }
    return aliases.get(normalized, normalized)


@dataclass(frozen=True)
class LlmIntegrationSettings:
    run_llm_integration: bool
    backend: str
    local_model_name: str
    device: str
    api_model_name: str
    api_base_url: str

    @property
    def selected_backends(self) -> list[str]:
        if self.backend == "all":
            return ["local", "api"]
        if self.backend in {"local", "api"}:
            return [self.backend]
        raise ValueError(
            "GAP_FINDER_LLM_INTEGRATION_BACKEND must be one of "
            "all, api, local, or both"
        )


def llm_settings() -> LlmIntegrationSettings:
    return LlmIntegrationSettings(
        run_llm_integration=env("RUN_LLM_INTEGRATION", "0") == "1",
        backend=_normalize_backend(env("LLM_INTEGRATION_BACKEND", "all")),
        local_model_name=env(
            "LLM_INTEGRATION_LOCAL_MODEL_NAME",
            "Qwen/Qwen3-0.6B",
        ),
        device=env("LLM_INTEGRATION_DEVICE", "cuda"),
        api_model_name=env(
            "LLM_INTEGRATION_API_MODEL_NAME",
            "gpt-oss-20b",
        ),
        api_base_url=env(
            "LLM_INTEGRATION_API_BASE_URL",
            "http://localhost:11434/v1",
        ),
    )


@dataclass(frozen=True)
class TopicGenerationIntegrationSettings:
    embedding_model_name: str
    language: str


def topic_generation_settings() -> TopicGenerationIntegrationSettings:
    return TopicGenerationIntegrationSettings(
        embedding_model_name=env(
            "TOPIC_GENERATION_EMBEDDING_MODEL",
            "paraphrase-multilingual-MiniLM-L12-v2",
        ),
        language=env("TOPIC_GENERATION_LANGUAGE", "german"),
    )


def pdf_embedding_model_name() -> str:
    return env(
        "PDF_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )


@dataclass(frozen=True)
class EmbeddingIntegrationSettings:
    run_embedding_integration: bool
    backend: str
    local_model_name: str
    api_model_name: str
    api_base_url: str

    @property
    def selected_backends(self) -> list[str]:
        if self.backend == "all":
            return ["local", "api"]
        if self.backend in {"local", "api"}:
            return [self.backend]
        raise ValueError(
            "GAP_FINDER_EMBEDDING_INTEGRATION_BACKEND must be one of "
            "all, api, local, or both"
        )


def embedding_settings() -> EmbeddingIntegrationSettings:
    return EmbeddingIntegrationSettings(
        run_embedding_integration=env("RUN_EMBEDDING_INTEGRATION", "0") == "1",
        backend=_normalize_backend(env("EMBEDDING_INTEGRATION_BACKEND", "all")),
        local_model_name=env(
            "EMBEDDING_INTEGRATION_LOCAL_MODEL_NAME",
            pdf_embedding_model_name(),
        ),
        api_model_name=env(
            "EMBEDDING_INTEGRATION_API_MODEL_NAME",
            "nomic-embed-text",
        ),
        api_base_url=env(
            "EMBEDDING_INTEGRATION_API_BASE_URL",
            "http://localhost:11434/v1",
        ),
    )


def llm_pytestmark(reason: str) -> list[pytest.MarkDecorator]:
    settings = llm_settings()
    return [
        pytest.mark.llm,
        pytest.mark.skipif(
            not settings.run_llm_integration,
            reason=reason,
        ),
    ]


def embedding_pytestmark(reason: str) -> list[pytest.MarkDecorator]:
    settings = embedding_settings()
    return [
        pytest.mark.embedding,
        pytest.mark.skipif(
            not settings.run_embedding_integration,
            reason=reason,
        ),
    ]


def language_backend_params() -> list[pytest.ParameterSet]:
    return [
        pytest.param(backend, id=backend)
        for backend in llm_settings().selected_backends
    ]


def embedding_backend_params() -> list[pytest.ParameterSet]:
    return [
        pytest.param(backend, id=backend)
        for backend in embedding_settings().selected_backends
    ]


def embedding_model_name_for_backend(backend: str, *, topic: bool = False) -> str:
    settings = embedding_settings()
    if backend == "local":
        if topic:
            return topic_generation_settings().embedding_model_name
        return settings.local_model_name
    if backend == "api":
        return settings.api_model_name
    raise ValueError(f"Unsupported embedding integration backend: {backend!r}")


def build_embedding_model(backend: str, *, topic: bool = False) -> EmbeddingModel:
    settings = embedding_settings()
    model_name = embedding_model_name_for_backend(backend, topic=topic)
    if backend == "local":
        return create_embedding_model(model_name)
    if backend == "api":
        return create_embedding_model(
            OpenAIEmbeddingClient(
                model_name=model_name,
                base_url=settings.api_base_url,
            )
        )
    raise ValueError(f"Unsupported embedding integration backend: {backend!r}")


def language_model_name_for_backend(backend: str) -> str:
    settings = llm_settings()
    if backend == "local":
        return settings.local_model_name
    if backend == "api":
        return settings.api_model_name
    raise ValueError(f"Unsupported LLM integration backend: {backend!r}")


def build_language_model_or_skip(backend: str) -> LanguageModel:
    settings = llm_settings()
    try:
        if backend == "local":
            return create_language_model(pipeline(
                "text-generation",
                model=settings.local_model_name,
                device=settings.device,
            ))
        if backend == "api":
            return create_language_model(OpenAIClient(
                model_name=settings.api_model_name,
                base_url=settings.api_base_url,
            ))
    except RuntimeError as error:
        if backend == "local" and "Found no NVIDIA driver" in str(error):
            pytest.skip(
                "Hugging Face chat model attempted to initialize CUDA although "
                f"the integration test selected device {settings.device!r}. "
                "Run with GAP_FINDER_LLM_INTEGRATION_BACKEND=api to use only "
                "the API backend, or check the installed torch/transformers "
                "configuration."
            )
        raise

    raise ValueError(f"Unsupported LLM integration backend: {backend!r}")
