# Integration Tests

These tests exercise the heavier parts of Gap Finder, including real embedding
models, BERTopic, local Hugging Face chat models, and OpenAI-compatible API
models.

The PDF fixtures under `test/data/pdf/` are original synthetic documents.
Regenerate them with `uv run scripts/generate_fixture_pdfs.py`.

## Run

LLM integration tests are skipped unless explicitly enabled:

```bash
GAP_FINDER_RUN_LLM_INTEGRATION=1 uv run pytest -s test/integration
```

Run only one chat backend:

```bash
GAP_FINDER_RUN_LLM_INTEGRATION=1 \
GAP_FINDER_LLM_INTEGRATION_BACKEND=api \
uv run pytest -s test/integration
```

```bash
GAP_FINDER_RUN_LLM_INTEGRATION=1 \
GAP_FINDER_LLM_INTEGRATION_BACKEND=local \
uv run pytest -s test/integration
```

## Configuration

Copy `.env.example` to `.env` for the full list of settings.
Integration tests load this file automatically before test modules are imported,
so settings also apply to parametrized tests during collection. Shell environment
variables take precedence over values from `.env`.

Common options:

```bash
GAP_FINDER_RUN_LLM_INTEGRATION=1
GAP_FINDER_LLM_INTEGRATION_BACKEND=all
GAP_FINDER_LLM_INTEGRATION_LOCAL_MODEL_NAME=Qwen/Qwen3-0.6B
GAP_FINDER_LLM_INTEGRATION_API_MODEL_NAME=gpt-oss-20b
GAP_FINDER_LLM_INTEGRATION_DEVICE=cuda
GAP_FINDER_LLM_INTEGRATION_API_BASE_URL=http://localhost:11434/v1
GAP_FINDER_API_KEY=-
GAP_FINDER_PDF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
GAP_FINDER_TOPIC_GENERATION_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
```

`GAP_FINDER_LLM_INTEGRATION_BACKEND` accepts `all`, `api`, or `local`.
Alias `both` is also accepted.

Use `GAP_FINDER_LLM_INTEGRATION_API_MODEL_NAME` for OpenAI-compatible API tests
and `GAP_FINDER_LLM_INTEGRATION_LOCAL_MODEL_NAME` for local Hugging Face
tests.

## Notes

The `-s` flag is useful because the tests print selected backends and generated
outputs. Hugging Face tests may download models and need a working CPU/GPU
runtime. API tests require an OpenAI-compatible server at the configured base
URL.

## Embedding backends

Embedding integrations are controlled separately from chat-model tests:

```bash
GAP_FINDER_RUN_EMBEDDING_INTEGRATION=1 \
GAP_FINDER_EMBEDDING_INTEGRATION_BACKEND=all \
uv run pytest -s -m embedding test/integration
```

Run only the API embedding backend with:

```bash
GAP_FINDER_RUN_EMBEDDING_INTEGRATION=1 \
GAP_FINDER_EMBEDDING_INTEGRATION_BACKEND=api \
GAP_FINDER_EMBEDDING_INTEGRATION_API_MODEL_NAME=nomic-embed-text \
GAP_FINDER_EMBEDDING_INTEGRATION_API_BASE_URL=http://localhost:11434/v1 \
GAP_FINDER_API_KEY=- \
uv run pytest -s -m embedding test/integration
```

`GAP_FINDER_EMBEDDING_INTEGRATION_BACKEND` accepts `all`, `api`, or
`local`. Alias `both` is also accepted. The suite
covers batched embedding requests, vector search/ranking, and BERTopic.
