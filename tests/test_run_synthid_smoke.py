import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

import run_synthid_smoke as smoke


class _Usage:
    def __init__(self, cost: str):
        self.cost = Decimal(cost)


class _Completion:
    def __init__(self, cost: str = "0.004"):
        self.content = "rewritten"
        self.provider = "Test"
        self.usage = _Usage(cost)

    def to_dict(self):
        return {
            "content": "rewritten",
            "finishReason": "stop",
            "id": "response-1",
            "model": "test/model",
            "openrouterMetadata": None,
            "provider": "Test",
            "systemFingerprint": None,
            "usage": {
                "completionTokens": 5,
                "promptTokens": 7,
                "providerCostCredits": str(self.usage.cost),
                "totalTokens": 12,
            },
        }


class _Delegate:
    max_tokens = 20

    def __init__(self):
        self.calls = 0

    def complete(self, request, *, model, max_tokens=None, response_format=None):
        self.calls += 1
        return _Completion()


class RunSynthIDSmokeTests(unittest.TestCase):
    def test_methodology_records_dynamic_document_count_and_threshold(self):
        text = smoke.smoke_methodology(document_count=10, threshold=0.5095383054287164)

        self.assertIn("10 marked documents", text)
        self.assertIn("0.509538305429", text)
        self.assertNotIn("pre-existing 0.5067", text)

    def test_non_zdr_adapter_is_local_and_keeps_data_collection_denied(self):
        captured = {}

        def downstream(url, headers, body, timeout):
            captured["body"] = __import__("json").loads(body)
            return {}

        adapter = smoke.ExplicitNonZdrTransport(delegate=downstream)
        adapter(
            "https://guard.invalid/chat/completions",
            {"Authorization": "Bearer fake"},
            b'{"provider":{"data_collection":"deny","zdr":true}}',
            10,
        )

        self.assertFalse(captured["body"]["provider"]["zdr"])
        self.assertEqual(
            captured["body"]["provider"]["data_collection"],
            "deny",
        )

    def test_call_cost_ceiling_uses_utf8_bytes_as_prompt_token_upper_bound(self):
        result = smoke.call_cost_ceiling(
            request_bytes=1_000,
            max_tokens=2_000,
            prompt_usd_per_token=Decimal("0.00000032"),
            completion_usd_per_token=Decimal("0.00000128"),
        )

        self.assertEqual(result, Decimal("0.00288000"))

    def test_checkpointed_client_reuses_a_durable_response_without_second_call(self):
        delegate = _Delegate()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            state = smoke.new_checkpoint_state(
                corpus_sha256="a" * 64,
                model="test/model",
                budget_usd=Decimal("0.10"),
            )
            client = smoke.CheckpointedClient(
                delegate=delegate,
                checkpoint_path=checkpoint,
                state=state,
                budget_usd=Decimal("0.10"),
                prompt_usd_per_token=Decimal("0.00000032"),
                completion_usd_per_token=Decimal("0.00000128"),
            )
            client.begin_transform("doc-01", "synonyms", ("synonyms",))
            first = client.complete("hello", model="test/model")
            client.begin_transform("doc-01", "synonyms", ("synonyms",))
            second = client.complete("hello", model="test/model")

        self.assertEqual(delegate.calls, 1)
        self.assertEqual(first.content, second.content)
        self.assertEqual(smoke.checkpoint_spend(state), Decimal("0.004"))
        self.assertIsNone(state["inFlight"])

    def test_checkpointed_client_stops_before_a_call_that_exceeds_budget(self):
        delegate = _Delegate()
        with tempfile.TemporaryDirectory() as directory:
            state = smoke.new_checkpoint_state(
                corpus_sha256="b" * 64,
                model="test/model",
                budget_usd=Decimal("0.000001"),
            )
            client = smoke.CheckpointedClient(
                delegate=delegate,
                checkpoint_path=Path(directory) / "checkpoint.json",
                state=state,
                budget_usd=Decimal("0.000001"),
                prompt_usd_per_token=Decimal("0.00000032"),
                completion_usd_per_token=Decimal("0.00000128"),
            )
            client.begin_transform("doc-01", "paraphrase", ("paraphrase",))

            with self.assertRaises(smoke.BudgetExceeded):
                client.complete("hello", model="test/model")

        self.assertEqual(delegate.calls, 0)
        self.assertIsNone(state["inFlight"])


if __name__ == "__main__":
    unittest.main()
