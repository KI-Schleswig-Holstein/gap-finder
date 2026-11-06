from openai import OpenAI

from gap_finder._auth import resolve_api_key


class OpenAIEmbeddingClient:
    """An OpenAI client paired with the embedding model it should call."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.model_name = model_name
        self.client = (
            client
            if client is not None
            else OpenAI(
                base_url=base_url,
                api_key=resolve_api_key(api_key),
            )
        )
