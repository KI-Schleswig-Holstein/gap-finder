from abc import ABC, abstractmethod


class LanguageModel(ABC):
    """Abstract interface for prompt-based language models."""

    @abstractmethod
    def invoke(self, prompt: str, system_prompt: str | None = None) -> str:
        """Send a prompt and an optional system prompt to the model."""
        raise NotImplementedError

    @abstractmethod
    def invoke_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
    ) -> list[str]:
        """Send multiple prompts and an optional system prompt to the model."""
        raise NotImplementedError

    def activate(self) -> None:
        """Make internally owned resources ready for use."""
        return

    def deactivate(self) -> None:
        """Optionally offload internally owned resources between steps."""
        return

    def cleanup(self) -> None:
        """Release internally owned resources after the pipeline finishes."""
        return
