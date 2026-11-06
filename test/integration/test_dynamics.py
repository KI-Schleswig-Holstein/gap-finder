from dynamics import llm_settings


def test_llm_model_names_do_not_share_a_generic_fallback(monkeypatch):
    monkeypatch.setenv(
        "GAP_FINDER_LLM_INTEGRATION_MODEL_NAME",
        "qwen3.8:27b",
    )
    monkeypatch.delenv("GAP_FINDER_LLM_INTEGRATION_LOCAL_MODEL_NAME", raising=False)
    monkeypatch.delenv("LLM_INTEGRATION_LOCAL_MODEL_NAME", raising=False)
    monkeypatch.delenv("GAP_FINDER_LLM_INTEGRATION_API_MODEL_NAME", raising=False)
    monkeypatch.delenv("LLM_INTEGRATION_API_MODEL_NAME", raising=False)

    settings = llm_settings()

    assert settings.local_model_name == "Qwen/Qwen3-0.6B"
    assert settings.api_model_name == "gpt-oss-20b"


def test_llm_model_names_use_backend_specific_settings(monkeypatch):
    monkeypatch.setenv(
        "GAP_FINDER_LLM_INTEGRATION_LOCAL_MODEL_NAME",
        "Qwen/local-model",
    )
    monkeypatch.setenv(
        "GAP_FINDER_LLM_INTEGRATION_API_MODEL_NAME",
        "qwen3.8:27b",
    )

    settings = llm_settings()

    assert settings.local_model_name == "Qwen/local-model"
    assert settings.api_model_name == "qwen3.8:27b"
