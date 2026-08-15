from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Self
from unittest.mock import patch

from perbacco.oracle import EntityView, OpenAICompatibleOracle, OracleProtocolError


class _FakeResponse:
    def __init__(self, value: dict[str, object]):
        self.value = value

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.value).encode()


class OracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.requests: list[tuple[str, dict[str, object]]] = []
        self.malformed_once = False
        self.entities = [
            EntityView(index, (index,), ({"name": f"record {index}"},)) for index in range(3)
        ]
        self.patcher = patch("perbacco.oracle.urllib.request.urlopen", side_effect=self.urlopen)
        self.patcher.start()

    def tearDown(self) -> None:
        self.patcher.stop()

    def urlopen(self, request, timeout: float):
        self.assertEqual(timeout, 90.0)
        body = json.loads(request.data)
        self.requests.append((request.full_url, body))
        if self.malformed_once:
            self.malformed_once = False
            return _FakeResponse(
                {
                    "output_text": '{"clusters":[]}',
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            )
        answer = json.dumps({"clusters": [{"entity_ids": ["e0", "e1"]}, {"entity_ids": ["e2"]}]})
        if request.full_url.endswith("/responses"):
            return _FakeResponse(
                {
                    "output_text": answer,
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            )
        return _FakeResponse(
            {
                "choices": [{"message": {"content": answer}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )

    def test_responses_contract_and_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "oracle.jsonl"
            oracle = OpenAICompatibleOracle(
                base_url="https://example.test/v1",
                model="test-model",
                api_key="test-key",
                transport="responses",
                journal_path=journal,
                retry_base_seconds=0,
            )
            answer = oracle.partition(self.entities)
            self.assertEqual(answer.labels, ("c0", "c0", "c1"))
            self.assertEqual(answer.usage["total_tokens"], 15)
            url, request = self.requests[-1]
            self.assertEqual(url, "https://example.test/v1/responses")
            self.assertEqual(request["text"]["format"]["type"], "json_schema")
            self.assertTrue(journal.is_file())

    def test_chat_contract(self) -> None:
        oracle = OpenAICompatibleOracle(
            base_url="https://example.test/v1",
            model="test-model",
            api_key="test-key",
            transport="chat",
            retry_base_seconds=0,
        )
        answer = oracle.partition(self.entities)
        self.assertEqual(answer.labels, ("c0", "c0", "c1"))
        url, request = self.requests[-1]
        self.assertEqual(url, "https://example.test/v1/chat/completions")
        self.assertEqual(request["response_format"]["type"], "json_schema")

    def test_malformed_partition_is_retried(self) -> None:
        self.malformed_once = True
        oracle = OpenAICompatibleOracle(
            base_url="https://example.test/v1",
            model="test-model",
            api_key="test-key",
            max_retries=1,
            retry_base_seconds=0,
        )
        answer = oracle.partition(self.entities)
        self.assertEqual(answer.labels, ("c0", "c0", "c1"))
        self.assertEqual(answer.usage["total_tokens"], 30)
        self.assertEqual(len(self.requests), 2)

    def test_partition_validator_rejects_omissions(self) -> None:
        with self.assertRaises(OracleProtocolError):
            OpenAICompatibleOracle._labels_from_output({"clusters": [{"entity_ids": ["e0"]}]}, 2)


if __name__ == "__main__":
    unittest.main()
