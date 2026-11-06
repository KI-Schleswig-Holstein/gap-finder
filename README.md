# Gap Finder

Gap Finder identifies potential knowledge gaps in a vector store, a collection of PDF documents,
or text chunks. It generates questions that the source material does not answer,
groups them into topics with BERTopic, and ranks those topics by how well the
indexed corpus covers them.

The project was created during a feasibility study at KI.SH in
Schleswig-Holstein. Its pipeline is based on `GapView`, described in
[Vanessa Klotzmann's dissertation](https://escholarship.org/uc/item/2zs4w991).

## Installation

Install Gap Finder from PyPI:

```bash
python -m pip install gap-finder
```

For development, clone the repository and install the project with
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/KI-Schleswig-Holstein/gap-finder.git
cd gap-finder
uv sync
```

## Quick start

The following example indexes all PDFs below `documents/` and uses an
OpenAI-compatible chat API for question generation and topic explanations:

```bash
export GAP_FINDER_API_KEY="your-api-key"
```

```python
from gap_finder import GapFinder

gap_finder = GapFinder.from_pdf(
    folder_path="documents",
    language_model="your-model-name",
    base_url="https://your-provider.example/v1",
)

topics = gap_finder.find()
print(topics[["title", "avg_similarity", "explanation"]])
```

For a local OpenAI-compatible server such as Ollama, set its `/v1` endpoint as
`base_url`. If the server does not require authentication, pass a placeholder
such as `api_key="-"`.

The default corpus and topic embedding models run locally and may be downloaded
on first use. Question generation is normally the most computationally
expensive part of the pipeline. Using an API offloads that work, but embedding
the corpus and fitting BERTopic still require local CPU or GPU time and memory.

### Use a local language model

Pass a Hugging Face text-generation pipeline to run the language model on your
machine. A GPU is recommended for all but very small models.

```python
from transformers import pipeline

from gap_finder import GapFinder

language_model = pipeline(
    "text-generation",
    model="Qwen/Qwen3-0.6B",
    device="cuda",
    max_new_tokens=1024,
)

gap_finder = GapFinder.from_pdf(
    folder_path="documents",
    language_model=language_model,
    question_generation_batch_size=8,
)
topics = gap_finder.find()
```

### Generate or provide questions separately

Questions can be generated before the rest of the pipeline. Because
`create_questions()` stores its result on the instance, the following call to
`find()` reuses them instead of generating them again:

```python
questions = gap_finder.create_questions()
# Inspect or persist questions here.
topics = gap_finder.find()
```

To skip automatic question generation, provide existing questions when the
instance is created or set them later:

```python
gap_finder = GapFinder.from_pdf(
    folder_path="documents",
    language_model="your-model-name",
    base_url="https://your-provider.example/v1",
    questions=[
        "Which populations are not represented in the study?",
        "How does the method behave outside the tested conditions?",
    ],
)

# Equivalent after construction:
gap_finder.set_questions(["Which limitations remain unexamined?"])
```

A questions DataFrame must contain a `question` column. Additional columns are
preserved as the pipeline adds topic and ranking information.

## Configuration

Use `GapFinder.from_pdf()` for a folder of PDFs, `GapFinder.from_chunks()` for
pre-split text, or construct `GapFinder` directly with a populated vector store.

## Vector stores

Two vector-store implementations are included:

- `NumpyVectorStore` is the default and performs exact cosine-similarity search
  with NumPy.
- `FaissVectorStore` performs exact cosine-similarity search with a FAISS
  `IndexFlatIP` index.

Select FAISS while building the index:

```python
from gap_finder import GapFinder
from gap_finder.vector_store import FaissVectorStore

gap_finder = GapFinder.from_pdf(
    folder_path="documents",
    vector_store_cls=FaissVectorStore,
    language_model="your-model-name",
    base_url="https://your-provider.example/v1",
)
```

Custom backends can subclass `VectorStore`.

## Pipeline

`find()` runs the following steps:

1. Load chunks and embeddings from the vector store.
2. Reuse configured questions or generate questions with the language model.
3. Group questions into topics with BERTopic.
4. Compare questions with the corpus and aggregate coverage scores per topic.
5. Add titles and explanations to the lowest-coverage topics.
6. Score those topics for domain relevance when `domain_description` is set.
7. Create a Markdown report section for each explained topic.

Lower coverage means the source corpus covers a topic less well and therefore
indicates a stronger potential knowledge gap.

## Results

`find()` returns the final topics DataFrame and stores the same object in
`gap_finder.topics`. Its columns are:

| Column | Description |
| --- | --- |
| `id` | BERTopic topic identifier; `-1` represents outliers. |
| `topic_words` | Representative `(word, score)` pairs. |
| `question_count` | Number of questions assigned to the topic. |
| `avg_similarity` | Mean question-to-corpus similarity and primary coverage score. |
| `std_similarity` | Standard deviation of question similarity. |
| `min_similarity`, `max_similarity` | Similarity range within the topic. |
| `title`, `explanation` | LLM-generated description for the configured lowest-coverage topics. |
| `relevance_score`, `relevance_reason` | Optional domain-relevance result, present when `domain_description` is set. |
| `report` | Markdown summary for an explained topic. |

After `find()`, `gap_finder.questions` contains the question records enriched
with `topic_id`, an optional `topic_prob`, an `embedding`, and a
`similarity_score`. Automatically generated questions also contain `id` and
`chunk_id`; caller-provided columns are retained.

`gap_finder.topic_model` is the fitted `BERTopic` object, not a table. Use its
standard API for deeper inspection, for example:

```python
topic_info = gap_finder.topic_model.get_topic_info()
```

## Development

Install the project and development dependencies with:

```bash
uv sync
```

Run the test suite:

```bash
uv run pytest
```

The repository also has opt-in integration tests for real embedding models,
BERTopic, local language models, and OpenAI-compatible APIs. See the
[integration test guide](test/integration/README.md) for configuration and
commands. These tests may download models and can require a GPU or a configured
API endpoint.

To contribute, open an issue/ticket or submit a pull/merge request.

## License

Gap Finder is available under the [MIT License](LICENSE).
