from __future__ import annotations

import copy
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from corpus_contract import canonical_json_bytes
from unmark import ChatCompletion, CompletionUsage

from run_verified_paraphrase import (
    VerifiedCallLimitReached,
    VerifiedResponseContractError,
    build_verified_dry_run,
    expected_verified_call_ids,
    load_verified_paraphrase_config,
    run_verified_live,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "fixtures" / "verified-paraphrase-config-v2.json"
CONFIG_V3_PATH = ROOT / "fixtures" / "verified-paraphrase-config-v3.json"


class FakeVerifiedClient:
    def __init__(self, *, provider: str = "DeepInfra") -> None:
        self.calls: list[dict[str, str]] = []
        self.provider = provider

    def complete(self, prompt: str, *, model: str) -> ChatCompletion:
        self.calls.append({"model": model, "prompt": prompt})
        if "--- BEGIN AUTHORITATIVE SOURCE ---\n" in prompt:
            content = prompt.split(
                "--- BEGIN AUTHORITATIVE SOURCE ---\n", 1
            )[1].split("\n--- END AUTHORITATIVE SOURCE ---", 1)[0]
        else:
            content = prompt.split("--- BEGIN TEXT ---\n", 1)[1].split(
                "\n--- END TEXT ---", 1
            )[0]
        content += " Rewritten."
        index = len(self.calls)
        return ChatCompletion(
            content=content,
            finish_reason="stop",
            model="qwen/qwen3.6-35b-a3b-20260415",
            openrouter_metadata={
                "attempt": 1,
                "endpoints": {
                    "available": [
                        {
                            "model": "qwen/qwen3.6-35b-a3b-20260415",
                            "provider": self.provider,
                            "selected": True,
                        }
                    ]
                },
                "strategy": "direct",
            },
            provider=self.provider,
            response_id=f"verified-{index}",
            system_fingerprint=None,
            usage=CompletionUsage(
                prompt_tokens=100,
                completion_tokens=100,
                total_tokens=200,
                cost=Decimal("0.0001"),
            ),
        )


class FakeVerifiedV3Client(FakeVerifiedClient):
    def complete(self, prompt: str, *, model: str) -> ChatCompletion:
        stage = len(self.calls) % 3
        if stage == 1:
            content = '{"corrections":[]}'
        elif stage == 2:
            content = prompt.split("--- BEGIN DRAFT TO EDIT ---\n", 1)[1].split(
                "\n--- END DRAFT TO EDIT ---", 1
            )[0]
        else:
            content = prompt.split("--- BEGIN TEXT ---\n", 1)[1].split(
                "\n--- END TEXT ---", 1
            )[0]
            content += " Rewritten."
        self.calls.append({"model": model, "prompt": prompt})
        index = len(self.calls)
        return ChatCompletion(
            content=content,
            finish_reason="stop",
            model="qwen/qwen3.6-35b-a3b-20260415",
            openrouter_metadata={
                "attempt": 1,
                "endpoints": {
                    "available": [
                        {
                            "model": "qwen/qwen3.6-35b-a3b-20260415",
                            "provider": "DeepInfra",
                            "selected": True,
                        }
                    ]
                },
                "strategy": "direct",
            },
            provider="DeepInfra",
            response_id=f"verified-v3-{index}",
            system_fingerprint=None,
            usage=CompletionUsage(
                prompt_tokens=100,
                completion_tokens=100,
                total_tokens=200,
                cost=Decimal("0.0001"),
            ),
        )


class VerifiedParaphraseConfigTests(unittest.TestCase):
    def test_frozen_v2_config_has_two_calls_one_route_and_separate_audit(self) -> None:
        config = load_verified_paraphrase_config(CONFIG_PATH)

        self.assertEqual(config.method_id, "paraphrase-verified")
        self.assertEqual(
            config.call_graph,
            ("paraphrase-draft", "fidelity-repair"),
        )
        self.assertTrue(config.always_run_repair)
        self.assertEqual(config.model, "qwen/qwen3.6-35b-a3b")
        self.assertEqual(config.provider_order, ("deepinfra/fp8",))
        self.assertEqual(config.expected_response_providers, ("DeepInfra",))
        self.assertFalse(config.allow_fallbacks)
        self.assertTrue(config.zdr)
        self.assertEqual(config.data_collection, "deny")
        self.assertIsNone(config.success_target)
        self.assertFalse(config.retuning_after_results)
        self.assertTrue(config.separate_final_audit)
        self.assertEqual(config.final_audit_model, "google/gemini-3.7-flash")

    def test_config_rejects_optional_repair_route_or_success_target(self) -> None:
        raw = load_verified_paraphrase_config(CONFIG_PATH).raw
        cases = []

        optional_repair = copy.deepcopy(raw)
        optional_repair["transform"]["alwaysRunRepair"] = False
        cases.append(optional_repair)

        target = copy.deepcopy(raw)
        target["decisionPolicy"]["successTarget"] = "zero failures"
        cases.append(target)

        retuning = copy.deepcopy(raw)
        retuning["decisionPolicy"]["retuningAfterResults"] = True
        cases.append(retuning)

        fallback = copy.deepcopy(raw)
        fallback["provider"]["allowFallbacks"] = True
        cases.append(fallback)

        for index, broken in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "broken.json"
                path.write_bytes(canonical_json_bytes(broken))
                with self.assertRaises(ValueError):
                    load_verified_paraphrase_config(path, root=ROOT)

    def test_dry_run_is_exactly_two_calls_per_document_without_network(self) -> None:
        config = load_verified_paraphrase_config(CONFIG_PATH)

        dry_run = build_verified_dry_run(config)

        self.assertEqual(dry_run["documentCount"], 20)
        self.assertEqual(dry_run["callCount"], 40)
        self.assertEqual(
            dry_run["callsByStage"],
            {"fidelity-repair": 20, "paraphrase-draft": 20},
        )
        self.assertGreater(dry_run["tokenEstimate"]["promptTokensPlanningEstimate"], 0)
        self.assertGreater(
            float(dry_run["routingCostPlanningEstimateUsd"]),
            0,
        )

    def test_checkpoint_order_alternates_draft_and_repair_for_each_document(self) -> None:
        config = load_verified_paraphrase_config(CONFIG_PATH)

        call_ids = expected_verified_call_ids(config)

        self.assertEqual(len(call_ids), 40)
        self.assertEqual(
            call_ids[:4],
            (
                "doc-01:paraphrase-verified:paraphrase-draft",
                "doc-01:paraphrase-verified:fidelity-repair",
                "doc-02:paraphrase-verified:paraphrase-draft",
                "doc-02:paraphrase-verified:fidelity-repair",
            ),
        )

    def test_v3_freezes_three_calls_and_hides_source_from_final_repair(self) -> None:
        config = load_verified_paraphrase_config(CONFIG_V3_PATH)

        self.assertEqual(config.method_id, "paraphrase-verified-v3")
        self.assertEqual(
            config.call_graph,
            ("paraphrase-draft", "fidelity-audit", "fidelity-repair"),
        )
        self.assertTrue(config.always_run_audit)
        self.assertTrue(config.always_run_repair)
        dry_run = build_verified_dry_run(config)
        self.assertEqual(dry_run["callCount"], 60)
        self.assertEqual(
            dry_run["callsByStage"],
            {
                "fidelity-audit": 20,
                "fidelity-repair": 20,
                "paraphrase-draft": 20,
            },
        )
        self.assertEqual(
            expected_verified_call_ids(config)[:6],
            (
                "doc-01:paraphrase-verified-v3:paraphrase-draft",
                "doc-01:paraphrase-verified-v3:fidelity-audit",
                "doc-01:paraphrase-verified-v3:fidelity-repair",
                "doc-02:paraphrase-verified-v3:paraphrase-draft",
                "doc-02:paraphrase-verified-v3:fidelity-audit",
                "doc-02:paraphrase-verified-v3:fidelity-repair",
            ),
        )

    def test_v3_live_matrix_runs_three_calls_per_document(self) -> None:
        config = load_verified_paraphrase_config(CONFIG_V3_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "verified-v3-checkpoint.json"
            client = FakeVerifiedV3Client()

            artifact = run_verified_live(
                config,
                client=client,
                max_provider_cost_credits=Decimal("1"),
                checkpoint_path=checkpoint,
            )

        self.assertEqual(len(client.calls), 60)
        self.assertEqual(artifact["usage"]["callCount"], 60)
        method = artifact["methods"][0]
        self.assertEqual(method["methodId"], "paraphrase-verified-v3")
        self.assertTrue(
            all(len(document["calls"]) == 3 for document in method["documents"])
        )

    def test_live_checkpoint_resumes_without_repeating_paid_calls(self) -> None:
        config = load_verified_paraphrase_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "verified-checkpoint.json"
            first_client = FakeVerifiedClient()
            with self.assertRaises(VerifiedCallLimitReached) as paused:
                run_verified_live(
                    config,
                    client=first_client,
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                    max_new_calls=2,
                )
            self.assertEqual(paused.exception.completed_calls, 2)
            self.assertEqual(len(first_client.calls), 2)

            second_client = FakeVerifiedClient()
            artifact = run_verified_live(
                config,
                client=second_client,
                max_provider_cost_credits=Decimal("1"),
                checkpoint_path=checkpoint,
            )

            self.assertEqual(len(second_client.calls), 38)
            self.assertEqual(artifact["usage"]["callCount"], 40)
            self.assertEqual(len(artifact["methods"][0]["documents"]), 20)

    def test_route_mismatch_is_checkpointed_and_never_retried(self) -> None:
        config = load_verified_paraphrase_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "verified-checkpoint.json"
            with self.assertRaisesRegex(
                VerifiedResponseContractError, "provider"
            ):
                run_verified_live(
                    config,
                    client=FakeVerifiedClient(provider="Other Provider"),
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                    max_new_calls=1,
                )
            state = json.loads(checkpoint.read_text())
            self.assertEqual(len(state["calls"]), 1)
            self.assertIsNone(state["inFlightCall"])


if __name__ == "__main__":
    unittest.main()
