from __future__ import annotations

import json
import math
import random
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import llm_config


SCHEMA_NAME = "entity_resolution_clusters"
OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "cluster_id": {"type": "string"},
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                },
                "required": ["cluster_id", "entity_ids"],
            },
            "minItems": 1,
        }
    },
    "required": ["clusters"],
}

ZERO_SHOT_INSTRUCTIONS = """You are an entity-resolution oracle for noisy bibliographic citations.

Goal:
- Partition the input entities by exact real-world paper identity.
- Two entities match only if they are citations of the same paper, not merely related papers.
- Every input entity must appear exactly once in the output, including singleton clusters.

How to decide:
- Compare title, author set, year, venue, pages, publisher, editor, institution, note, month, and volume jointly.
- Each entity also includes helper summaries such as `normalized_title_variants`, `author_last_names_union`, `year_signatures`, `page_signatures`, and `venue_signatures`.
- Treat abbreviations, punctuation changes, reordered authors, venue aliases, and missing fields as normal citation noise.
- Strong positive evidence usually requires near-equivalent title plus compatible authors and compatible year/pages/venue.
- If normalized title tokens are almost identical and author last names overlap strongly, prefer merging even when venue text differs or page formatting is slightly inconsistent.
- If an entity already has `entity_size > 1`, treat its summary fields as evidence accumulated from previously merged citations.
- Strong negative evidence includes different paper titles, incompatible author sets, incompatible page ranges, or conflicting years when the citation otherwise looks complete.
- Same venue and same year do not imply a match if normalized titles differ materially.
- Do not merge based only on topic similarity, venue overlap, author overlap, or same year.
- If evidence is mixed or incomplete, choose the safer option and keep the entities separate.

Critical output constraints:
- Use only the provided input entity ids in `entity_ids`.
- Never output member record ids, invented ids, or text explanations.
- Return valid JSON only, matching the exact schema.
"""

def _pair_example(name: str, left: dict[str, str], right: dict[str, str], is_match: bool) -> dict[str, Any]:
    answer = {
        "clusters": [{"cluster_id": "c1", "entity_ids": ["A", "B"]}]
        if is_match
        else [
            {"cluster_id": "c1", "entity_ids": ["A"]},
            {"cluster_id": "c2", "entity_ids": ["B"]},
        ]
    }
    return {
        "name": name,
        "entities": [
            {"entity_id": "A", "entity_size": 1, "records": [{"id": "A1", **left}]},
            {"entity_id": "B", "entity_size": 1, "records": [{"id": "B1", **right}]},
        ],
        "answer": answer,
    }


FEW_SHOT_EXAMPLES = [
    _pair_example(
        "Positive 1",
        {
            "title": "cryptographic primitives based on hard learning problems.",
            "author": "a. blum, m. furst, m. j. kearns, and richard j. lipton.",
            "year": "1993",
            "venue": "in pre-proceedings of crypto '93",
            "pages": "24.1-24.10",
        },
        {
            "title": "cryptographic primitives based on hard learning problems.",
            "author": "avrim blum, merrick furst, michael kearns, and richard j. lipton.",
            "year": "1993",
            "venue": "crypto 93",
            "pages": "24.1-24.10",
        },
        True,
    ),
    _pair_example(
        "Positive 2",
        {
            "title": "a general lower bound on the number of examples needed for learning.",
            "author": "a. ehrenfeucht, d. haussler, m. kearns, and l. valiant.",
            "year": "1989",
            "venue": "information and computation",
            "pages": "247-261",
        },
        {
            "title": "a general lower bound on the number of examples needed for learning.",
            "author": "andrzej ehrenfeucht, david haussler, michael kearns, and leslie valiant.",
            "year": "1989",
            "venue": "inf. and computation",
            "pages": "247-266",
        },
        True,
    ),
    _pair_example(
        "Positive 3",
        {
            "title": "learning sparse multivariate polynomials over a field with queries and counterexamples.",
            "author": "r. e. schapire and l. m. sellie.",
            "year": "1996",
            "venue": "journal of computer and system sciences",
            "pages": "201-213",
        },
        {
            "title": "learning sparse multivariate polynomials over a field with queries and counterexamples.",
            "author": "schapire, r. e.; sellie, l. m.",
            "year": "1996",
            "venue": "j. of computer and system sciences",
            "pages": "201-213",
        },
        True,
    ),
    _pair_example(
        "Positive 4",
        {
            "title": "a polynomial-time algorithm for learning k-variable pattern languages from examples.",
            "author": "m. kearns and l. pitt.",
            "year": "1989",
            "venue": "workshop on computational learning theory",
            "pages": "57-71",
        },
        {
            "title": "a polynomial-time algorithm for learning k-variable pattern languages from examples.",
            "author": "michael kearns and leslie pitt",
            "year": "1989",
            "venue": "computational learning theory workshop",
            "pages": "57-71",
        },
        True,
    ),
    _pair_example(
        "Positive 5",
        {
            "title": "gambling in a rigged casino: the adversarial multi-armed bandit problem.",
            "author": "p. auer, n. cesa-bianchi, y. freund, and r. e. schapire.",
            "year": "1995",
            "venue": "36th annual symposium on foundations of computer science",
            "pages": "322-331",
        },
        {
            "title": "gambling in a rigged casino: the adversarial multi-armed bandit problem",
            "author": "peter auer, nicolo cesa-bianchi, yoav freund, robert e. schapire",
            "year": "1995",
            "venue": "FOCS",
            "pages": "322-331",
        },
        True,
    ),
    _pair_example(
        "Negative 1",
        {
            "title": "a general lower bound on the number of examples needed for learning.",
            "author": "a. ehrenfeucht, d. haussler, m. kearns, and l. valiant.",
            "year": "1989",
            "venue": "information and computation",
            "pages": "247-261",
        },
        {
            "title": "a polynomial-time algorithm for learning k-variable pattern languages from examples.",
            "author": "m. kearns and l. pitt.",
            "year": "1989",
            "venue": "workshop on computational learning theory",
            "pages": "57-71",
        },
        False,
    ),
    _pair_example(
        "Negative 2",
        {
            "title": "cryptographic primitives based on hard learning problems.",
            "author": "a. blum, m. furst, m. j. kearns, and richard j. lipton.",
            "year": "1993",
            "venue": "crypto 93",
            "pages": "24.1-24.10",
        },
        {
            "title": "a general lower bound on the number of examples needed for learning.",
            "author": "a. ehrenfeucht, d. haussler, m. kearns, and l. valiant.",
            "year": "1989",
            "venue": "information and computation",
            "pages": "247-261",
        },
        False,
    ),
    _pair_example(
        "Negative 3",
        {
            "title": "learning sparse multivariate polynomials over a field with queries and counterexamples.",
            "author": "r. e. schapire and l. m. sellie.",
            "year": "1996",
            "venue": "journal of computer and system sciences",
            "pages": "201-213",
        },
        {
            "title": "gambling in a rigged casino: the adversarial multi-armed bandit problem.",
            "author": "p. auer, n. cesa-bianchi, y. freund, and r. e. schapire.",
            "year": "1995",
            "venue": "foundations of computer science",
            "pages": "322-331",
        },
        False,
    ),
    _pair_example(
        "Negative 4",
        {
            "title": "a general lower bound on the number of examples needed for learning.",
            "author": "a. ehrenfeucht, d. haussler, m. kearns, and l. valiant.",
            "year": "1989",
            "venue": "information and computation",
            "pages": "247-261",
        },
        {
            "title": "a general lower bound on the number of examples needed for learning.",
            "author": "a. ehrenfeucht, d. haussler, m. kearns, and l. valiant.",
            "year": "1989",
            "venue": "information and computation",
            "pages": "267-284",
        },
        False,
    ),
    _pair_example(
        "Negative 5",
        {
            "title": "a polynomial-time algorithm for learning k-variable pattern languages from examples.",
            "author": "m. kearns and l. pitt.",
            "year": "1989",
            "venue": "computational learning theory",
            "pages": "57-71",
        },
        {
            "title": "a polynomial-time algorithm for learning k-variable pattern languages from examples.",
            "author": "m. kearns and l. pitt.",
            "year": "1990",
            "venue": "computational learning theory",
            "pages": "101-112",
        },
        False,
    ),
]


def _sanitize_record(record: dict[str, Any]) -> dict[str, str]:
    sanitized = {}
    for key, value in record.items():
        if value is None:
            continue
        text = str(value).strip()
        if text:
            sanitized[key] = text
    return sanitized


def _serialize_entities(entities: list[dict[str, Any]]) -> str:
    return json.dumps(entities, indent=2, ensure_ascii=True)


def build_zero_shot_prompt(entities: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        [
            "Input entities:",
            _serialize_entities(entities),
            "Return the partition now.",
        ]
    )


def build_few_shot_prompt(entities: list[dict[str, Any]]) -> str:
    sections = [ZERO_SHOT_INSTRUCTIONS.strip()]
    for example in FEW_SHOT_EXAMPLES:
        sections.append(f"{example['name']} input:")
        sections.append(_serialize_entities(example["entities"]))
        sections.append(f"{example['name']} output:")
        sections.append(json.dumps(example["answer"], indent=2, ensure_ascii=True))
    sections.append("Input entities:")
    sections.append(_serialize_entities(entities))
    return "\n\n".join(sections)


def clusters_to_pairwise_matches(clusters: list[dict[str, Any]]) -> set[frozenset[str]]:
    match_pairs: set[frozenset[str]] = set()
    for cluster in clusters:
        entity_ids = [str(entity_id) for entity_id in cluster["entity_ids"]]
        for left, right in combinations(entity_ids, 2):
            match_pairs.add(frozenset((left, right)))
    return match_pairs


def _estimate_tokens_from_text(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _extract_output_text(response_payload: dict[str, Any]) -> str:
    outputs = response_payload.get("output", [])
    for output in outputs:
        for content in output.get("content", []):
            if content.get("type") == "output_text":
                return content.get("text", "")
    raise RuntimeError("OpenAI response did not include output_text content.")


def _extract_balanced_json(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise RuntimeError(f"OpenAI response did not include JSON content: {text!r}")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise RuntimeError(f"OpenAI response JSON appears truncated: {text!r}")


def _extract_usage(response_payload: dict[str, Any]) -> dict[str, int]:
    usage = response_payload.get("usage") or {}
    return {
        "llm_input_tokens": int(usage.get("input_tokens", 0) or 0),
        "llm_output_tokens": int(usage.get("output_tokens", 0) or 0),
        "llm_total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def _normalize_clusters(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for cluster in clusters:
        entity_ids = [str(entity_id) for entity_id in cluster["entity_ids"]]
        normalized.append(
            {
                "cluster_id": str(cluster["cluster_id"]),
                "entity_ids": sorted(entity_ids),
            }
        )
    normalized.sort(key=lambda item: item["entity_ids"][0])
    return normalized


@dataclass
class TokenEstimate:
    label: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class OpenAIEntityOracle:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        prompt_mode: str | None = None,
        timeout_seconds: int | None = None,
        max_retries: int | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = (api_key or llm_config.get_openai_api_key()).strip()
        self.model = llm_config.resolve_model(model)
        self.prompt_mode = llm_config.resolve_prompt_mode(prompt_mode)
        self.timeout_seconds = timeout_seconds or llm_config.OPENAI_TIMEOUT_SECONDS
        self.max_retries = max_retries or llm_config.OPENAI_MAX_RETRIES
        self.max_output_tokens = max_output_tokens or llm_config.OPENAI_MAX_OUTPUT_TOKENS
        self.retry_base_seconds = llm_config.OPENAI_RETRY_BASE_SECONDS
        self.retry_max_seconds = llm_config.OPENAI_RETRY_MAX_SECONDS
        self.reasoning_effort = reasoning_effort or llm_config.OPENAI_REASONING_EFFORT
        self.base_url = (base_url or llm_config.OPENAI_BASE_URL).rstrip("/")

    def _supports_reasoning_effort(self, model: str) -> bool:
        return model.startswith("gpt-5")

    def build_prompt(self, entities: list[dict[str, Any]], prompt_mode: str | None = None) -> str:
        chosen_mode = llm_config.resolve_prompt_mode(prompt_mode or self.prompt_mode)
        if chosen_mode == "few-shot":
            return build_few_shot_prompt(entities)
        return build_zero_shot_prompt(entities)

    def _build_response_payload(self, prompt: str, model: str | None = None) -> dict[str, Any]:
        resolved_model = model or self.model
        payload = {
            "model": resolved_model,
            "input": prompt,
            "instructions": ZERO_SHOT_INSTRUCTIONS.strip(),
            "max_output_tokens": self.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": SCHEMA_NAME,
                    "strict": True,
                    "schema": OUTPUT_SCHEMA,
                }
            },
        }
        if self._supports_reasoning_effort(resolved_model):
            payload["reasoning"] = {"effort": self.reasoning_effort}
        return payload

    def _build_response_payload_with_limit(
        self,
        prompt: str,
        max_output_tokens: int,
        model: str | None = None,
    ) -> dict[str, Any]:
        payload = self._build_response_payload(prompt, model=model)
        payload["max_output_tokens"] = max_output_tokens
        return payload

    def _estimate_initial_output_limit(self, entities: list[dict[str, Any]], model: str | None = None) -> int:
        resolved_model = model or self.model
        entity_count = max(1, len(entities))

        # Worst case, every entity is a singleton cluster. Budget both the visible
        # JSON and extra headroom for reasoning-capable models, where reasoning
        # tokens can consume the same output budget.
        singleton_json = {
            "clusters": [
                {
                    "cluster_id": f"c{index + 1}",
                    "entity_ids": [str(entity.get("entity_id", index + 1))],
                }
                for index, entity in enumerate(entities)
            ]
        }
        visible_output_tokens = _estimate_tokens_from_text(json.dumps(singleton_json, ensure_ascii=True))
        reasoning_buffer = max(256, 64 * entity_count) if self._supports_reasoning_effort(resolved_model) else 128

        return max(self.max_output_tokens, visible_output_tokens + reasoning_buffer)

    def _request_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError(
                "OpenAI API key missing. Set OPENAI_API_KEY in llm_config_local.py or the environment."
            )

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API request failed ({exc.code}): {details}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI API request failed: {exc.reason}") from exc

        return json.loads(response_body)

    def _compute_retry_delay(self, attempt: int) -> float:
        base_delay = self.retry_base_seconds * (2 ** max(0, attempt - 1))
        jitter = random.uniform(0.75, 1.25)
        return min(self.retry_max_seconds, base_delay * jitter)

    def _validate_clusters(
        self,
        raw_clusters: list[dict[str, Any]],
        expected_ids: set[str],
        alias_map: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        normalized = _normalize_clusters(raw_clusters)
        seen_ids: set[str] = set()
        alias_map = alias_map or {}
        for cluster in normalized:
            canonical_ids = []
            for entity_id in cluster["entity_ids"]:
                canonical_ids.append(alias_map.get(entity_id, entity_id))
            cluster["entity_ids"] = sorted(set(canonical_ids))
            cluster_ids = set(cluster["entity_ids"])
            if len(cluster_ids) != len(cluster["entity_ids"]):
                raise RuntimeError(f"Duplicate entity id inside cluster: {cluster}")
            overlap = seen_ids & cluster_ids
            if overlap:
                raise RuntimeError(f"Entity ids repeated across clusters: {sorted(overlap)}")
            seen_ids.update(cluster_ids)

        if seen_ids != expected_ids:
            missing = sorted(expected_ids - seen_ids)
            extra = sorted(seen_ids - expected_ids)
            raise RuntimeError(
                f"OpenAI clustering did not cover the batch exactly. Missing={missing}, extra={extra}"
            )

        return normalized

    def resolve_batch(self, entities: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = self.build_prompt(entities)
        expected_ids = {str(entity["entity_id"]) for entity in entities}
        alias_map = {}
        for entity in entities:
            canonical_id = str(entity["entity_id"])
            alias_map[canonical_id] = canonical_id
            for record in entity.get("records", []):
                alias_map[str(record.get("id"))] = canonical_id
        last_error: Exception | None = None
        initial_output_limit = self._estimate_initial_output_limit(entities)

        for attempt in range(1, self.max_retries + 1):
            try:
                output_limit = initial_output_limit * (2 ** (attempt - 1))
                response_payload = self._request_json(
                    "/responses",
                    self._build_response_payload_with_limit(prompt, max_output_tokens=output_limit),
                )
                output_text = _extract_output_text(response_payload)
                try:
                    parsed = json.loads(output_text)
                except json.JSONDecodeError:
                    parsed = json.loads(_extract_balanced_json(output_text))
                clusters = self._validate_clusters(parsed["clusters"], expected_ids, alias_map=alias_map)
                usage = _extract_usage(response_payload)
                return {
                    "clusters": clusters,
                    "usage": usage,
                    "response_id": response_payload.get("id", ""),
                }
            except Exception as exc:  # noqa: BLE001
                last_error = RuntimeError(
                    f"{exc} (attempt={attempt}/{self.max_retries}, max_output_tokens={output_limit})"
                )
                if attempt < self.max_retries:
                    time.sleep(self._compute_retry_delay(attempt))

        raise RuntimeError(f"OpenAI entity resolution failed after {self.max_retries} attempts: {last_error}")

    def estimate_batch_tokens(self, entities: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = self.build_prompt(entities)
        heuristic_input = _estimate_tokens_from_text(prompt)
        heuristic_output = self._estimate_initial_output_limit(entities)

        estimate = {
            "source": "heuristic",
            "prompt_chars": len(prompt),
            "input_tokens": heuristic_input,
            "output_tokens": heuristic_output,
            "total_tokens": heuristic_input + heuristic_output,
        }

        if not self.api_key:
            return estimate

        try:
            response_payload = self._request_json(
                "/responses/input_tokens",
                self._build_response_payload(prompt),
            )
        except RuntimeError:
            return estimate

        usage = response_payload.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", heuristic_input) or heuristic_input)
        estimate.update(
            {
                "source": "openai_input_tokens",
                "input_tokens": input_tokens,
                "output_tokens": heuristic_output,
                "total_tokens": input_tokens + heuristic_output,
            }
        )
        return estimate

    def estimate_experiment_tokens(
        self,
        expected_calls: int,
        batch_estimates: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        scenario_totals = {}
        for label, estimate in batch_estimates.items():
            scenario_totals[label] = {
                "input_tokens": int(estimate["input_tokens"]) * expected_calls,
                "output_tokens": int(estimate["output_tokens"]) * expected_calls,
            }
            scenario_totals[label]["total_tokens"] = (
                scenario_totals[label]["input_tokens"] + scenario_totals[label]["output_tokens"]
            )

        input_values = [values["input_tokens"] for values in scenario_totals.values()]
        output_values = [values["output_tokens"] for values in scenario_totals.values()]
        total_values = [values["total_tokens"] for values in scenario_totals.values()]

        return {
            "expected_calls": expected_calls,
            "scenarios": scenario_totals,
            "input_tokens_median": int(statistics.median(input_values)),
            "output_tokens_median": int(statistics.median(output_values)),
            "total_tokens_median": int(statistics.median(total_values)),
        }
