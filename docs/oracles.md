# Oracle integration

The scheduler does not decide whether records match. It selects current entity
components and asks an `Oracle` for one **complete partition**. Local truth,
test doubles, OpenAI, and OpenRouter all implement the same synchronous method:

```python
partition(entities: Sequence[EntityView]) -> OracleResult
```

## Ground truth

Use `GroundTruthOracle` for reproducible experiments and tests:

```python
from perbacco import GroundTruthOracle

oracle = GroundTruthOracle(
    {"record-a": "entity-1", "record-b": "entity-1", "record-c": "entity-2"}
)
```

Every member record needs a truth label. If a current component contains
conflicting labels, the oracle raises `OracleProtocolError`. It performs no I/O
and reports no tokens.

## Custom protocol

No base class or registration is required. Return arbitrary hashable labels
aligned with the input entities and optional usage counters:

```python
--8<-- "examples/python/custom_oracle.py"
```

The engine separately verifies the label count and validates the partition
against known state before mutation.

## OpenAI Responses API

Set the key in the process environment, then construct the provider preset:

```bash
export OPENAI_API_KEY='...'
```

```python
from perbacco import OpenAICompatibleOracle

oracle = OpenAICompatibleOracle.openai(
    model="MODEL_ID",
    max_retries=3,
    journal_path="artifacts/oracle.jsonl",
)
```

The adapter posts to `https://api.openai.com/v1/responses` using the Python
standard library. It supplies a strict JSON Schema through `text.format`, the
location used by the Responses API. OpenAI documents Structured Outputs as
schema adherence rather than merely valid JSON; model support still depends on
the selected model. See the official [Structured Outputs
guide](https://developers.openai.com/api/docs/guides/structured-outputs).

## OpenRouter

Set `OPENROUTER_API_KEY` and select a model slug that supports structured
outputs:

```bash
export OPENROUTER_API_KEY='...'
```

```python
oracle = OpenAICompatibleOracle.openrouter(
    model="provider/model",
    extra_headers={
        "HTTP-Referer": "https://example.org",
        "X-OpenRouter-Title": "My ER job",
    },
)
```

This preset uses `https://openrouter.ai/api/v1/chat/completions` and sends the
schema in `response_format`. OpenRouter's [Structured Outputs
guide](https://openrouter.ai/docs/guides/features/structured-outputs) explains
model compatibility, and its [quickstart](https://openrouter.ai/docs/quickstart)
documents the optional attribution headers shown above.

## Structured-output contract

The wire response must be equivalent to:

```json
{
  "clusters": [
    {"entity_ids": ["e0", "e2"]},
    {"entity_ids": ["e1"]}
  ]
}
```

For each request, the adapter replaces current external entity IDs with aliases
`e0`, `e1`, and so on. Validation requires:

- a nonempty `clusters` array;
- a nonempty `entity_ids` array in every cluster;
- every supplied alias exactly once;
- no unknown or repeated aliases;
- no additional schema properties.

Member record IDs and attributes remain in the prompt because the oracle needs
them for comparison. The model is instructed never to return member IDs.

## Retries and failures

The first request plus `max_retries` attempts use exponential delays beginning
at `retry_base_seconds`. Network errors, timeouts, HTTP 429/5xx responses,
invalid JSON, malformed response envelopes, refusals, and incomplete partitions
are retryable. Other HTTP 4xx responses stop immediately. Final failure raises
`OracleProtocolError` and never submits a partial answer to the engine.

Retries can still incur provider cost. Token usage from every decoded response,
including a malformed response that is retried, is accumulated into the
successful `OracleResult.usage`. Transport failures with no decodable usage
object cannot be counted.

## Journaling and token accounting

When `journal_path` is set, one JSON object is appended per attempt. Entries
contain the endpoint path, model, request body, decoded response or error,
per-attempt usage, and cumulative usage. Normalized keys are `input_tokens`,
`output_tokens`, and `total_tokens`; Chat Completions prompt/completion names are
mapped to the same fields.

!!! warning "Journals contain record data"

    Journals include prompts, record attributes, and raw model output. They may
    contain personal or confidential information. Store them under access
    control, define a retention policy, and do not commit them. API keys are
    never written to the journal.

## API-key safety

Never place a key in source code, examples, notebooks, shell history, committed
configuration, or a JSONL journal. Prefer environment variables locally and a
secret manager in deployment. OpenAI's official [production best-practices
guide](https://developers.openai.com/api/docs/guides/production-best-practices)
also recommends keeping keys out of code and public repositories.

The repository's tests patch the HTTP transport, and documentation acceptance
tests additionally block socket creation. Running `mkdocs build`, `make test`,
or any canonical example cannot make a paid API request. Only calling
`OpenAICompatibleOracle.partition()` with a configured key initiates network
I/O.
