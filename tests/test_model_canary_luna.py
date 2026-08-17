from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_model_canary_luna import (
    CACHE_READ_PRICE,
    CanaryError,
    COMPLETION_PRICE,
    EXPECTED_MODELS,
    MAX_COMPLETION_TOKENS,
    MODEL,
    PROMPT_PRICE,
    PROVIDER,
    PROVIDER_TAG,
    build_v4_draft_request,
    conservative_reserve,
    expected_cost,
    expected_payload,
    initial_checkpoint,
    load_json,
    load_sources,
    run_live,
    score_outputs,
    score_sources,
    validate_checkpoint,
    validate_completion,
    validate_v9,
)
from unmark import ChatCompletion, CompletionUsage, protect_tokens


class LunaCanaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = load_sources()
        cls.source_score = score_sources(cls.sources)
        cls.v9 = validate_v9()
        source = cls.sources["doc-11"]
        cls.request = build_v4_draft_request(protect_tokens(source).masked)

    def completion(self) -> ChatCompletion:
        usage = CompletionUsage(
            prompt_tokens=1_000,
            completion_tokens=700,
            total_tokens=1_700,
            cost=(Decimal(1_000) * PROMPT_PRICE + Decimal(700) * COMPLETION_PRICE)
            / Decimal(1_000_000),
        )
        return ChatCompletion(
            content="A valid response.",
            finish_reason="stop",
            model=EXPECTED_MODELS[0],
            openrouter_metadata={
                "attempt": 1,
                "endpoints": {
                    "available": [
                        {
                            "model": EXPECTED_MODELS[1],
                            "provider": PROVIDER,
                            "selected": True,
                        }
                    ]
                },
                "is_byok": False,
                "requested": MODEL,
                "strategy": "direct",
            },
            provider=PROVIDER,
            response_id="test-response",
            system_fingerprint=None,
            usage=usage,
        )

    @staticmethod
    def route() -> dict[str, object]:
        return {
            "catalogUrl": "https://openrouter.ai/api/v1/endpoints/zdr",
            "endpointName": "Azure | openai/gpt-5.6-luna-20260709",
            "provider": PROVIDER,
            "status": 0,
            "tag": PROVIDER_TAG,
            "uptimeLast5m": 100,
        }

    def fake_client_factory(self, transport: object) -> object:
        outer = self

        class FakeClient:
            def complete(self, request: object, *, model: str) -> ChatCompletion:
                self_payload = expected_payload(request)
                transport.last_request = {
                    "body": self_payload,
                    "endpoint": "https://guard.invalid/openrouter/api/v1/chat/completions",
                    "headers": {
                        "Content-Type": "application/json",
                        "X-OpenRouter-Metadata": "enabled",
                    },
                    "timeoutSeconds": 180,
                }
                completion = outer.completion()
                return ChatCompletion(
                    **{
                        **completion.__dict__,
                        "content": json.loads(request.user_json)["sourceText"],
                    }
                )

        return FakeClient()

    def test_payload_is_one_exact_azure_zdr_route(self) -> None:
        payload = expected_payload(self.request)

        self.assertEqual(payload["model"], MODEL)
        self.assertEqual(payload["max_completion_tokens"], MAX_COMPLETION_TOKENS)
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["reasoning"], {"effort": "medium"})
        self.assertEqual(
            payload["provider"],
            {
                "allow_fallbacks": False,
                "data_collection": "deny",
                "max_price": {
                    "completion": float(COMPLETION_PRICE),
                    "prompt": float(PROMPT_PRICE),
                },
                "order": [PROVIDER_TAG],
                "require_parameters": True,
                "zdr": True,
            },
        )
        self.assertNotIn("Authorization", json.dumps(payload))

    def test_cost_and_route_validation_accept_the_frozen_contract(self) -> None:
        completion = self.completion()

        self.assertEqual(expected_cost(completion), completion.usage.cost)
        validate_completion(completion, self.request)

    def test_cache_read_price_is_recomputed_separately(self) -> None:
        usage = CompletionUsage(
            prompt_tokens=1_000,
            completion_tokens=100,
            total_tokens=1_100,
            cached_prompt_tokens=250,
            cost=(
                Decimal(750) * PROMPT_PRICE
                + Decimal(250) * CACHE_READ_PRICE
                + Decimal(100) * COMPLETION_PRICE
            )
            / Decimal(1_000_000),
        )
        completion = ChatCompletion(**{**self.completion().__dict__, "usage": usage})

        self.assertEqual(expected_cost(completion), usage.cost)

    def test_checkpoint_refuses_redispatch_after_an_uncertain_call(self) -> None:
        state = initial_checkpoint(self.sources, self.source_score, self.v9)
        state["inFlight"] = {"callId": "luna:doc-11"}

        with self.assertRaisesRegex(CanaryError, "no-redispatch tombstone"):
            validate_checkpoint(state, self.sources, self.source_score, self.v9)

    def test_pooled_detector_rejects_unchanged_marked_outputs(self) -> None:
        score = score_outputs(self.sources)

        self.assertEqual(score["status"], "detected")
        self.assertEqual(score["hits"], 33)
        self.assertEqual(score["activePositions"], 33)

    def test_live_checkpoint_is_bound_and_resumable_after_one_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            with (
                patch("run_model_canary_luna.fetch_catalog", return_value=self.route()),
                patch(
                    "run_model_canary_luna.build_client",
                    side_effect=self.fake_client_factory,
                ),
            ):
                result = run_live(checkpoint, Decimal("0.01"), 1)

            self.assertEqual(result["newCalls"], 1)
            state = load_json(checkpoint, "test checkpoint")
            validate_checkpoint(state, self.sources, self.source_score, self.v9)
            self.assertEqual(len(state["calls"]), 1)
            self.assertIsNone(state["inFlight"])

    def test_paid_response_is_preserved_when_analysis_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            with (
                patch("run_model_canary_luna.fetch_catalog", return_value=self.route()),
                patch(
                    "run_model_canary_luna.build_client",
                    side_effect=self.fake_client_factory,
                ),
                patch(
                    "run_model_canary_luna.analyze_output",
                    side_effect=RuntimeError("local analysis failed"),
                ),
            ):
                with self.assertRaisesRegex(CanaryError, "will not be retried"):
                    run_live(checkpoint, Decimal("0.01"), 1)

            state = load_json(checkpoint, "test checkpoint")
            self.assertEqual(state["calls"], [])
            self.assertIn("receivedResponse", state["inFlight"])
            self.assertIn("receivedResponse", state["terminalFailure"])
            self.assertEqual(
                state["terminalFailure"]["receivedResponse"]["id"],
                "test-response",
            )

    def test_six_call_reserve_stays_below_five_cents(self) -> None:
        reserve = sum(
            (
                conservative_reserve(
                    build_v4_draft_request(protect_tokens(source).masked)
                )
                for source in self.sources.values()
            ),
            Decimal(0),
        )

        self.assertLess(reserve, Decimal("0.05"))


if __name__ == "__main__":
    unittest.main()
