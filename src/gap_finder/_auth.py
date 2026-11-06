import os


class MissingAPIKeyError(RuntimeError):
    """Raised when no API key was provided or configured."""


def resolve_api_key(
    explicit: str | None = None,
    env_var: str = "GAP_FINDER_API_KEY",
) -> str:
    key = explicit or os.getenv(env_var)
    if not key:
        raise MissingAPIKeyError(
            f"No API key found. Pass api_key=... or set the environment "
            f"variable {env_var}."
        )
    return key
