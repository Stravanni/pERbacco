from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class EntityView:
    """Current entity component presented to an oracle.

    Attributes:
        entity_id: Stable external ID of the component representative.
        members: External record IDs currently belonging to the component.
        records: Attribute mappings aligned one-to-one with ``members``.
    """

    entity_id: Hashable
    members: tuple[Hashable, ...]
    records: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class OracleResult:
    """A complete oracle answer and optional accounting metadata.

    Attributes:
        labels: One arbitrary hashable cluster label per input entity. Equality
            means that two current entities should be merged.
        usage: Normalized token counts. Supported keys are ``input_tokens``,
            ``output_tokens``, and ``total_tokens``.
        raw_response: Optional provider response for diagnostics. Callers should
            treat it as potentially sensitive.
    """

    labels: tuple[Hashable, ...]
    usage: Mapping[str, int]
    raw_response: Mapping[str, object] | None = None


class Oracle(Protocol):
    """Structural protocol for synchronous entity-resolution oracles."""

    def partition(self, entities: Sequence[EntityView]) -> OracleResult:
        """Partition every supplied current entity exactly once.

        Args:
            entities: Ordered entity views from one engine batch.

        Returns:
            Labels aligned with ``entities`` and optional usage metadata.

        Raises:
            OracleProtocolError: If a transport or answer is invalid.
        """
        ...


class OracleProtocolError(RuntimeError):
    """An oracle transport failed or returned an invalid complete partition."""


class GroundTruthOracle:
    """Deterministic local oracle backed by record-level ground-truth labels.

    This adapter performs no I/O and is the recommended oracle for tests and
    paper experiments.
    """

    def __init__(self, labels: Mapping[Hashable, Hashable]):
        """Store ground-truth labels keyed by every graph record ID.

        Args:
            labels: Mapping whose equal values denote the same real entity.
        """
        self.labels = labels

    def partition(self, entities: Sequence[EntityView]) -> OracleResult:
        """Return the ground-truth partition of current entities.

        Args:
            entities: Current components whose members all require labels.

        Returns:
            One ground-truth label per entity and an empty usage mapping.

        Raises:
            KeyError: If a member has no ground-truth label.
            OracleProtocolError: If one current component already contains
                conflicting ground-truth labels.
        """
        result: list[Hashable] = []
        for entity in entities:
            labels = {self.labels[member] for member in entity.members}
            if len(labels) != 1:
                raise OracleProtocolError(
                    f"entity {entity.entity_id!r} contains conflicting ground-truth labels"
                )
            result.append(next(iter(labels)))
        return OracleResult(tuple(result), {})


PARTITION_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "clusters": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    }
                },
                "required": ["entity_ids"],
            },
        }
    },
    "required": ["clusters"],
}


DEFAULT_INSTRUCTIONS = """You are an entity-resolution oracle.
Partition the supplied entities by exact real-world identity. Compare all non-empty fields and all
member records jointly. Formatting, abbreviation, punctuation, and missing fields can be noise;
conflicting identifiers or core attributes are negative evidence. Every supplied entity_id must
appear exactly once. Return only the required JSON object and never return member record IDs."""


def _response_text(payload: Mapping[str, object], transport: str) -> str:
    if transport == "chat":
        try:
            message = payload["choices"][0]["message"]  # type: ignore[index]
            if message.get("refusal"):
                raise OracleProtocolError(f"model refused: {message['refusal']}")
            content = message["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    str(item.get("text", "")) for item in content if isinstance(item, dict)
                )
        except (KeyError, IndexError, TypeError) as exc:
            raise OracleProtocolError("malformed Chat Completions response") from exc
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text
    output = payload.get("output")
    if isinstance(output, list):
        pieces: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "refusal":
                    raise OracleProtocolError(f"model refused: {part.get('refusal', '')}")
                if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    pieces.append(part["text"])
        if pieces:
            return "".join(pieces)
    raise OracleProtocolError("malformed Responses API response")


def _usage(payload: Mapping[str, object]) -> dict[str, int]:
    raw = payload.get("usage")
    if not isinstance(raw, dict):
        return {}
    aliases = {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "total_tokens": ("total_tokens",),
    }
    result: dict[str, int] = {}
    for target, names in aliases.items():
        for name in names:
            value = raw.get(name)
            if isinstance(value, int):
                result[target] = value
                break
    if "total_tokens" not in result and result:
        result["total_tokens"] = result.get("input_tokens", 0) + result.get("output_tokens", 0)
    return result


class OpenAICompatibleOracle:
    """Standard-library client for structured-output OpenAI-compatible APIs.

    Use `openai` or `openrouter` for provider defaults. The client performs
    synchronous HTTP requests only when `partition` is called,
    validates that each anonymous entity alias appears exactly once, retries
    transient or malformed responses with exponential backoff, normalizes token
    usage, and can append a JSONL audit journal.

    The journal contains input record attributes and raw model responses. Store
    it as sensitive data. API keys are sent in the Authorization header and are
    never written to the journal.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        transport: str = "responses",
        instructions: str = DEFAULT_INSTRUCTIONS,
        timeout_seconds: float = 90.0,
        max_retries: int = 3,
        retry_base_seconds: float = 1.0,
        journal_path: str | Path | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        """Configure an OpenAI-compatible oracle transport.

        Args:
            base_url: API root ending before ``/responses`` or
                ``/chat/completions``.
            model: Provider model identifier.
            api_key: Secret bearer token. Prefer an environment variable over a
                source-code literal.
            transport: ``"responses"`` or ``"chat"``.
            instructions: System instructions sent with each batch.
            timeout_seconds: Per-request network timeout.
            max_retries: Number of retries after the first attempt.
            retry_base_seconds: Initial exponential-backoff delay.
            journal_path: Optional append-only JSONL request/response journal.
            extra_headers: Provider-specific HTTP headers. Do not place secrets
                here if the surrounding process logs request headers.

        Raises:
            ValueError: If the transport, key, or retry count is invalid.
        """
        if transport not in {"responses", "chat"}:
            raise ValueError("transport must be 'responses' or 'chat'")
        if not api_key:
            raise ValueError("an API key is required")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.transport = transport
        self.instructions = instructions
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.journal_path = Path(journal_path) if journal_path else None
        self.extra_headers = dict(extra_headers or {})

    @classmethod
    def openai(
        cls, *, model: str, api_key: str | None = None, **kwargs: object
    ) -> OpenAICompatibleOracle:
        """Create a Responses API oracle for OpenAI.

        Args:
            model: OpenAI model identifier supporting Structured Outputs.
            api_key: Explicit key, or ``OPENAI_API_KEY`` when omitted.
            **kwargs: Additional constructor options except provider defaults.

        Returns:
            An oracle using ``https://api.openai.com/v1/responses``.

        Raises:
            ValueError: If no API key is available or options are invalid.
        """
        return cls(
            base_url="https://api.openai.com/v1",
            model=model,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            transport="responses",
            **kwargs,
        )

    @classmethod
    def openrouter(
        cls, *, model: str, api_key: str | None = None, **kwargs: object
    ) -> OpenAICompatibleOracle:
        """Create a Chat Completions oracle for OpenRouter.

        Args:
            model: OpenRouter model identifier, usually ``provider/model``.
            api_key: Explicit key, or ``OPENROUTER_API_KEY`` when omitted.
            **kwargs: Additional constructor options except provider defaults.

        Returns:
            An oracle using OpenRouter's OpenAI-compatible chat endpoint.

        Raises:
            ValueError: If no API key is available or options are invalid.
        """
        return cls(
            base_url="https://openrouter.ai/api/v1",
            model=model,
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY", ""),
            transport="chat",
            **kwargs,
        )

    def _request_body(self, entities: Sequence[EntityView]) -> tuple[str, dict[str, object]]:
        serialized = []
        for index, entity in enumerate(entities):
            records = []
            for member, record in zip(entity.members, entity.records, strict=True):
                records.append({"record_id": str(member), **dict(record)})
            serialized.append(
                {"entity_id": f"e{index}", "entity_size": len(entity.members), "records": records}
            )
        user_content = json.dumps({"entities": serialized}, ensure_ascii=False, default=str)
        if self.transport == "responses":
            return "/responses", {
                "model": self.model,
                "input": [
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": user_content},
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "entity_resolution_partition",
                        "strict": True,
                        "schema": PARTITION_SCHEMA,
                    }
                },
            }
        return "/chat/completions", {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": user_content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "entity_resolution_partition",
                    "strict": True,
                    "schema": PARTITION_SCHEMA,
                },
            },
        }

    def _journal(self, value: Mapping[str, object]) -> None:
        if self.journal_path is None:
            return
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, default=str, sort_keys=True) + "\n")

    def _post(self, path: str, body: Mapping[str, object]) -> dict[str, object]:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body, ensure_ascii=False, default=str).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "perbacco/0.1.0",
                **self.extra_headers,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            error = OracleProtocolError(f"HTTP {exc.code}: {detail}")
            error.retryable = exc.code == 429 or exc.code >= 500  # type: ignore[attr-defined]
            raise error from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            error = OracleProtocolError(f"oracle transport failed: {exc}")
            error.retryable = True  # type: ignore[attr-defined]
            raise error from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            error = OracleProtocolError("oracle returned a non-JSON HTTP response")
            error.retryable = True  # type: ignore[attr-defined]
            raise error from exc
        if not isinstance(decoded, dict):
            raise OracleProtocolError("oracle response must be a JSON object")
        return decoded

    @staticmethod
    def _labels_from_output(output: object, entity_count: int) -> tuple[str, ...]:
        if not isinstance(output, dict) or not isinstance(output.get("clusters"), list):
            raise OracleProtocolError("structured output has no clusters array")
        expected = {f"e{index}" for index in range(entity_count)}
        seen: set[str] = set()
        labels: list[str | None] = [None] * entity_count
        for cluster_index, cluster in enumerate(output["clusters"]):
            if not isinstance(cluster, dict) or not isinstance(cluster.get("entity_ids"), list):
                raise OracleProtocolError("each cluster must contain entity_ids")
            if not cluster["entity_ids"]:
                raise OracleProtocolError("clusters cannot be empty")
            for alias in cluster["entity_ids"]:
                if not isinstance(alias, str) or alias not in expected or alias in seen:
                    raise OracleProtocolError(f"invalid or repeated entity id: {alias!r}")
                seen.add(alias)
                labels[int(alias[1:])] = f"c{cluster_index}"
        if seen != expected:
            missing = sorted(expected - seen)
            raise OracleProtocolError(f"partition omitted entity ids: {missing}")
        return tuple(label for label in labels if label is not None)

    def partition(self, entities: Sequence[EntityView]) -> OracleResult:
        """Request and validate a complete structured-output partition.

        Input entity IDs are replaced with per-request aliases before transport;
        member record IDs and attributes remain in the prompt. A successful
        result always contains one label per input entity.

        Args:
            entities: At least two current entity views.

        Returns:
            Validated cluster labels, normalized usage, and the raw response.

        Raises:
            ValueError: If fewer than two entities are supplied.
            OracleProtocolError: If all attempts fail, the model refuses, or the
                response is not a complete, non-overlapping partition.
        """
        if len(entities) < 2:
            raise ValueError("an oracle batch needs at least two entities")
        path, body = self._request_body(entities)
        last_error: Exception | None = None
        cumulative_usage: dict[str, int] = {}
        for attempt in range(self.max_retries + 1):
            payload: dict[str, object] | None = None
            try:
                payload = self._post(path, body)
                attempt_usage = _usage(payload)
                for name, count in attempt_usage.items():
                    cumulative_usage[name] = cumulative_usage.get(name, 0) + count
                text = _response_text(payload, self.transport)
                parsed = json.loads(text)
                labels = self._labels_from_output(parsed, len(entities))
                self._journal(
                    {
                        "attempt": attempt,
                        "endpoint": path,
                        "model": self.model,
                        "request": body,
                        "response": payload,
                        "usage": attempt_usage,
                        "cumulative_usage": cumulative_usage,
                    }
                )
                return OracleResult(labels, dict(cumulative_usage), payload)
            except (OracleProtocolError, json.JSONDecodeError) as exc:
                last_error = exc
                retryable = getattr(exc, "retryable", True)
                journal: dict[str, object] = {
                    "attempt": attempt,
                    "endpoint": path,
                    "model": self.model,
                    "request": body,
                    "error": str(exc),
                }
                if payload is not None:
                    journal["response"] = payload
                    journal["usage"] = _usage(payload)
                    journal["cumulative_usage"] = cumulative_usage
                self._journal(journal)
                if not retryable or attempt >= self.max_retries:
                    break
                time.sleep(self.retry_base_seconds * (2**attempt))
        raise OracleProtocolError(
            f"oracle failed after {self.max_retries + 1} attempt(s): {last_error}"
        ) from last_error
