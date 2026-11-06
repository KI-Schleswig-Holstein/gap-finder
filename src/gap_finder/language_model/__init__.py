from importlib import import_module
from typing import TYPE_CHECKING

from gap_finder._auth import MissingAPIKeyError
from gap_finder.language_model.client import ClientError, OpenAIClient
from gap_finder.language_model.language_model import LanguageModel

if TYPE_CHECKING:
    from gap_finder.language_model.factory import create_language_model
    from gap_finder.language_model.local_language_model import LocalLanguageModel
    from gap_finder.language_model.openai_language_model import OpenAILanguageModel

__all__ = [
    "ClientError",
    "LanguageModel",
    "LocalLanguageModel",
    "MissingAPIKeyError",
    "OpenAIClient",
    "OpenAILanguageModel",
    "create_language_model",
]


def __getattr__(name: str):
    if name == "create_language_model":
        from gap_finder.language_model.factory import create_language_model

        return create_language_model
    if name == "LocalLanguageModel":
        from gap_finder.language_model.local_language_model import LocalLanguageModel

        return LocalLanguageModel
    if name == "OpenAILanguageModel":
        from gap_finder.language_model.openai_language_model import OpenAILanguageModel

        return OpenAILanguageModel
    if name == "factory":
        return import_module("gap_finder.language_model.factory")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
