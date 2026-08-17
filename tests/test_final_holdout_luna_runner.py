from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_final_holdout_v9_luna import (
    CANDIDATE,
    CanaryError,
    atomic_write,
    automated_aggregate,
    build_blind_packet,
    finalize_review,
    initial_checkpoint,
    load_json,
    load_protocol,
    run_live,
    validate_checkpoint,
)
from run_model_canary_luna import (
    COMPLETION_PRICE,
    EXPECTED_MODELS,
    PROMPT_PRICE,
    PROVIDER,
)
from unmark import ChatCompletion, CompletionUsage


class FinalHoldoutLunaRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol()

    @staticmethod
    def route() -> dict[str, object]:
        return {
            "catalogUrl": "https://openrouter.ai/api/v1/endpoints/zdr",
            "endpointName": "Azure | openai/gpt-5.6-luna-20260709",
            "provider": PROVIDER,
            "status": 0,
            "tag": CANDIDATE.route,
            "uptimeLast5m": 100,
        }

    @staticmethod
    def completion(content: str) -> ChatCompletion:
        usage = CompletionUsage(
            prompt_tokens=1_000,
            completion_tokens=700,
            total_tokens=1_700,
            cost=(Decimal(1_000) * PROMPT_PRICE + Decimal(700) * COMPLETION_PRICE)
            / Decimal(1_000_000),
        )
        return ChatCompletion(
            content=content,
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
                "requested": CANDIDATE.model,
                "strategy": "direct",
            },
            provider=PROVIDER,
            response_id="test-final-response",
            system_fingerprint=None,
            usage=usage,
        )

    def candidate_with_fake_client(self):
        outer = self

        def factory(transport):
            class FakeClient:
                def complete(self, request, *, model):
                    payload = CANDIDATE.expected_payload(request)
                    transport.last_request = {
                        "body": payload,
                        "endpoint": "https://guard.invalid/openrouter/api/v1/chat/completions",
                        "headers": {
                            "Content-Type": "application/json",
                            "X-OpenRouter-Metadata": "enabled",
                        },
                        "timeoutSeconds": 180,
                    }
                    source_masked = json.loads(request.user_json)["sourceText"]
                    return outer.completion(source_masked)

            return FakeClient()

        return replace(
            CANDIDATE,
            build_client=factory,
            fetch_catalog=self.route,
        )

    def test_v9_sources_recompute_to_frozen_38_of_42_baseline(self) -> None:
        source_score = self.protocol["sourceScore"]

        self.assertEqual(len(self.protocol["sources"]), 20)
        self.assertEqual(source_score["status"], "detected")
        self.assertEqual(source_score["hits"], 38)
        self.assertEqual(source_score["activePositions"], 42)

    def test_checkpoint_refuses_any_in_flight_redispatch(self) -> None:
        state = initial_checkpoint(self.protocol)
        state["inFlight"] = {"callId": "luna-final:holdout-01"}

        with self.assertRaisesRegex(CanaryError, "no-redispatch tombstone"):
            validate_checkpoint(state, self.protocol)

    def test_one_mocked_call_is_checkpointed_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            with patch(
                "run_final_holdout_v9_luna.CANDIDATE",
                self.candidate_with_fake_client(),
            ):
                result = run_live(checkpoint, Decimal("0.01"), 1)

            self.assertEqual(result["newCalls"], 1)
            state = load_json(checkpoint, "test checkpoint")
            validate_checkpoint(state, self.protocol)
            self.assertEqual(len(state["calls"]), 1)
            self.assertIsNone(state["inFlight"])

    def test_paid_response_survives_local_analysis_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            with (
                patch(
                    "run_final_holdout_v9_luna.CANDIDATE",
                    self.candidate_with_fake_client(),
                ),
                patch(
                    "run_final_holdout_v9_luna.analyze_output",
                    side_effect=RuntimeError("local analysis failed"),
                ),
            ):
                with self.assertRaisesRegex(CanaryError, "never be retried"):
                    run_live(checkpoint, Decimal("0.01"), 1)

            state = load_json(checkpoint, "failed checkpoint")
            self.assertEqual(state["calls"], [])
            self.assertIn("receivedResponse", state["inFlight"])
            self.assertIn("receivedResponse", state["terminalFailure"])

    def test_aggregate_and_blind_review_are_separate_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.json"
            packet_path = root / "packet.json"
            review_path = root / "review.json"
            final_path = root / "final.json"
            with patch(
                "run_final_holdout_v9_luna.CANDIDATE",
                self.candidate_with_fake_client(),
            ):
                result = run_live(checkpoint, Decimal("0.20"), 20)
            self.assertEqual(result["completedCalls"], 20)

            aggregate = automated_aggregate(checkpoint)
            self.assertFalse(aggregate["automatedGate"]["passed"])
            self.assertEqual(
                aggregate["pooledOutputDetector"]["status"],
                "detected",
            )

            packet = build_blind_packet(checkpoint, packet_path)
            serialized = json.dumps(packet).lower()
            self.assertNotIn("openai/gpt", serialized)
            self.assertNotIn("azure/eu", serialized)
            self.assertNotIn("providercost", serialized)
            review = {
                "packetSha256": packet["packetSha256"],
                "reviews": [
                    {"findings": [], "pairId": pair["pairId"], "verdict": "pass"}
                    for pair in packet["pairs"]
                ],
            }
            atomic_write(review_path, review)
            with self.assertRaises(CanaryError):
                finalize_review(checkpoint, packet_path, review_path, final_path)
            final = finalize_review(
                checkpoint,
                packet_path,
                review_path,
                final_path,
                enforce_commit=False,
            )
            self.assertTrue(final["manualReviewGate"]["passed"])
            self.assertFalse(final["finalConfirmationPassed"])
            self.assertFalse(final["commitGateEnforced"])


if __name__ == "__main__":
    unittest.main()
