from __future__ import annotations

import copy
import codecs
from decimal import Decimal
import json
from pathlib import Path
import re
import tempfile
import unittest

from corpus_contract import canonical_json_bytes
from unmark import (
    ChatCompletion,
    CompletionUsage,
    StageRequest,
    build_semantic_audit_request,
    build_semantic_repair_request,
    build_v4_draft_request,
    semantic_audit_response_format,
)

from run_verified_paraphrase import (
    VerifiedCanaryGateError,
    VerifiedCallLimitReached,
    VerifiedResponseContractError,
    build_v4_canary_gate,
    build_verified_dry_run,
    expected_verified_call_ids,
    load_verified_paraphrase_config,
    run_verified_live,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "fixtures" / "verified-paraphrase-config-v2.json"
CONFIG_V3_PATH = ROOT / "fixtures" / "verified-paraphrase-config-v3.json"
CONFIG_V4_PATH = ROOT / "fixtures" / "verified-paraphrase-config-v4.json"


class FakeVerifiedClient:
    def __init__(
        self,
        *,
        provider: str = "DeepInfra",
        cost: Decimal = Decimal("0.000105"),
    ) -> None:
        self.calls: list[dict[str, str]] = []
        self.provider = provider
        self.cost = cost

    def complete(self, prompt: str, *, model: str) -> ChatCompletion:
        self.calls.append({"model": model, "prompt": prompt})
        if "--- BEGIN AUTHORITATIVE SOURCE ---\n" in prompt:
            content = prompt.split("--- BEGIN AUTHORITATIVE SOURCE ---\n", 1)[1].split(
                "\n--- END AUTHORITATIVE SOURCE ---", 1
            )[0]
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
                cost=self.cost,
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
                cost=Decimal("0.000105"),
            ),
        )


class FakeVerifiedV4Client(FakeVerifiedClient):
    def __init__(
        self,
        *,
        malformed_audit: bool = False,
        strong_rewrite: bool = True,
    ) -> None:
        super().__init__()
        self.malformed_audit = malformed_audit
        self.strong_rewrite = strong_rewrite
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        request: str | StageRequest,
        *,
        model: str,
        max_tokens: int | None = None,
        response_format: dict[str, object] | None = None,
    ) -> ChatCompletion:
        if not isinstance(request, StageRequest):
            raise AssertionError("v4 must use StageRequest")
        payload = json.loads(request.user_json)
        if request.stage == "semantic-audit":
            content = "not json" if self.malformed_audit else '{"corrections":[]}'
        elif request.stage == "fidelity-repair":
            content = payload["draftText"]
        else:
            content = payload["sourceText"]
            if self.strong_rewrite:
                content = _rot13_preserving_placeholders(content)
            else:
                content += " Rewritten."
        self.calls.append(
            {
                "maxTokens": max_tokens,
                "messages": list(request.to_messages()),
                "model": model,
                "responseFormat": response_format,
                "stage": request.stage,
            }
        )
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
            response_id=f"verified-v4-{index}",
            system_fingerprint=None,
            usage=CompletionUsage(
                prompt_tokens=100,
                completion_tokens=100,
                total_tokens=200,
                cost=Decimal("0.000105"),
            ),
        )


def _rot13_preserving_placeholders(value: str) -> str:
    return "".join(
        part if re.fullmatch(r"⟦T[1-9][0-9]*⟧", part) else codecs.decode(part, "rot_13")
        for part in re.split(r"(⟦T[1-9][0-9]*⟧)", value)
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

    def test_checkpoint_order_alternates_draft_and_repair_for_each_document(
        self,
    ) -> None:
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

    def test_v4_freezes_strict_semantic_audit_and_prior_v3_pilot(self) -> None:
        config = load_verified_paraphrase_config(CONFIG_V4_PATH)

        self.assertEqual(config.method_id, "paraphrase-verified-v4")
        self.assertEqual(
            config.call_graph,
            ("paraphrase-draft", "semantic-audit", "fidelity-repair"),
        )
        self.assertEqual(config.stage_max_tokens["semantic-audit"], 1536)
        self.assertLess(
            config.stage_max_tokens["semantic-audit"],
            config.stage_max_tokens["paraphrase-draft"],
        )
        self.assertEqual(config.audit_max_corrections, 12)
        self.assertEqual(config.development_canary_exact_call_count, 3)
        self.assertEqual(config.canary_min_final_normalized_word_distance, 0.15)
        self.assertEqual(config.canary_min_final_to_draft_word_distance_ratio, 0.60)
        self.assertEqual(config.article_demo_maximum_total_final_failures, 1)
        self.assertEqual(config.article_demo_maximum_pipeline_defects, 0)
        self.assertEqual(
            config.article_demo_minimum_mean_normalized_word_distance, 0.15
        )
        self.assertEqual(config.development_document_ids, ("doc-01",))
        self.assertEqual(
            config.holdout_document_ids,
            tuple(f"doc-{index:02d}" for index in range(2, 21)),
        )
        self.assertEqual(
            expected_verified_call_ids(config)[:3],
            (
                "doc-01:paraphrase-verified-v4:paraphrase-draft",
                "doc-01:paraphrase-verified-v4:semantic-audit",
                "doc-01:paraphrase-verified-v4:fidelity-repair",
            ),
        )
        self.assertEqual(build_verified_dry_run(config)["callCount"], 60)

    def test_v4_rejects_changed_message_boundary_or_decision_gates(self) -> None:
        raw = load_verified_paraphrase_config(CONFIG_V4_PATH).raw
        cases = []

        user_only = copy.deepcopy(raw)
        user_only["transform"]["requestBoundary"]["instructionRole"] = "user"
        cases.append(user_only)

        weak_canary = copy.deepcopy(raw)
        weak_canary["developmentCanaryGate"]["minFinalNormalizedWordDistance"] = 0.05
        cases.append(weak_canary)

        permissive_demo = copy.deepcopy(raw)
        permissive_demo["decisionPolicy"]["articleDemoGate"]["enableDemoOnlyIf"][
            "maximumPipelineDefects"
        ] = 1
        cases.append(permissive_demo)

        for index, broken in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "broken.json"
                path.write_bytes(canonical_json_bytes(broken))
                with self.assertRaises(ValueError):
                    load_verified_paraphrase_config(path, root=ROOT)

    def test_v4_parity_fixture_is_byte_exact_with_python_stage_builders(self) -> None:
        config = load_verified_paraphrase_config(CONFIG_V4_PATH)
        fixture = json.loads(config.parity_fixture_path.read_text())
        sample = fixture["sample"]
        requests = (
            (
                build_v4_draft_request(sample["sourceMaskedText"]),
                config.stage_max_tokens["paraphrase-draft"],
                None,
            ),
            (
                build_semantic_audit_request(
                    sample["sourceMaskedText"],
                    sample["draftMaskedText"],
                ),
                config.stage_max_tokens["semantic-audit"],
                semantic_audit_response_format(),
            ),
            (
                build_semantic_repair_request(
                    sample["draftMaskedText"],
                    sample["canonicalSemanticAudit"],
                ),
                config.stage_max_tokens["fidelity-repair"],
                None,
            ),
        )

        self.assertEqual(
            fixture["stages"],
            [
                {
                    "maxTokens": max_tokens,
                    "messages": list(request.to_messages()),
                    "responseFormat": response_format,
                    "stage": request.stage,
                }
                for request, max_tokens, response_format in requests
            ],
        )

    def test_v4_artifact_separates_development_from_primary_holdout_metrics(
        self,
    ) -> None:
        config = load_verified_paraphrase_config(CONFIG_V4_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "verified-v4-checkpoint.json"
            with self.assertRaises(VerifiedCallLimitReached):
                run_verified_live(
                    config,
                    client=FakeVerifiedV4Client(),
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                    max_new_calls=3,
                )
            gate = build_v4_canary_gate(config, checkpoint_path=checkpoint)
            self.assertEqual(gate["status"], "go")
            artifact = run_verified_live(
                config,
                client=FakeVerifiedV4Client(),
                max_provider_cost_credits=Decimal("1"),
                checkpoint_path=checkpoint,
            )

        cohorts = artifact["analysisCohorts"]
        self.assertEqual(cohorts["development"]["documentIds"], ["doc-01"])
        self.assertEqual(
            cohorts["holdoutPrimary"]["documentIds"],
            [f"doc-{index:02d}" for index in range(2, 21)],
        )
        self.assertEqual(
            cohorts["holdoutPrimary"]["aggregate"]["detector"]["documentCount"],
            19,
        )
        self.assertEqual(
            cohorts["holdoutPrimary"]["aggregate"]["usage"]["callCount"],
            57,
        )
        self.assertEqual(
            cohorts["development"]["aggregate"]["usage"]["callCount"],
            3,
        )
        self.assertEqual(artifact["developmentCanaryGate"]["status"], "go")
        self.assertEqual(
            artifact["decisionPolicy"]["articleDemoGate"]["status"],
            "pending_final_audit",
        )

    def test_v4_forwards_strict_schema_and_only_sanitized_audit_to_repair(self) -> None:
        config = load_verified_paraphrase_config(CONFIG_V4_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "verified-v4-checkpoint.json"
            client = FakeVerifiedV4Client()
            with self.assertRaises(VerifiedCallLimitReached):
                run_verified_live(
                    config,
                    client=client,
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                    max_new_calls=3,
                )
            saved = json.loads(checkpoint.read_text())

        self.assertEqual(len(client.calls), 3)
        self.assertEqual(
            [call["maxTokens"] for call in client.calls],
            [4096, 1536, 4096],
        )
        audit_format = client.calls[1]["responseFormat"]
        self.assertIsInstance(audit_format, dict)
        self.assertTrue(audit_format["json_schema"]["strict"])
        self.assertIsNone(client.calls[0]["responseFormat"])
        self.assertIsNone(client.calls[2]["responseFormat"])
        repair_payload = json.loads(client.calls[2]["messages"][1]["content"])
        self.assertEqual(repair_payload["validatedCorrections"], [])
        self.assertNotIn("authoritativeSourceText", repair_payload)
        with self.assertRaises(KeyError):
            _ = client.calls[2]["prompt"]
        self.assertEqual(
            [message["role"] for message in saved["calls"][0]["messages"]],
            ["system", "user"],
        )
        self.assertNotIn("prompt", saved["calls"][0])
        self.assertRegex(saved["calls"][0]["requestSha256"], r"^[0-9a-f]{64}$")

    def test_v4_malformed_audit_still_runs_repair_with_empty_sanitized_list(
        self,
    ) -> None:
        config = load_verified_paraphrase_config(CONFIG_V4_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "verified-v4-checkpoint.json"
            client = FakeVerifiedV4Client(malformed_audit=True)
            with self.assertRaises(VerifiedCallLimitReached):
                run_verified_live(
                    config,
                    client=client,
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                    max_new_calls=3,
                )
            saved = json.loads(checkpoint.read_text())

        self.assertEqual(len(client.calls), 3)
        self.assertEqual(len(saved["calls"]), 3)
        repair_payload = json.loads(client.calls[2]["messages"][1]["content"])
        self.assertEqual(repair_payload["validatedCorrections"], [])
        self.assertNotIn("not json", client.calls[2]["messages"][1]["content"])

    def test_v4_canary_gate_blocks_holdout_after_source_copy_like_output(self) -> None:
        config = load_verified_paraphrase_config(CONFIG_V4_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "verified-v4-checkpoint.json"
            with self.assertRaises(VerifiedCallLimitReached):
                run_verified_live(
                    config,
                    client=FakeVerifiedV4Client(strong_rewrite=False),
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                    max_new_calls=3,
                )

            gate = build_v4_canary_gate(config, checkpoint_path=checkpoint)
            self.assertEqual(gate["status"], "no_go")
            self.assertFalse(gate["checks"]["minimumFinalWordDistance"]["passed"])
            resumed = FakeVerifiedV4Client()
            with self.assertRaises(VerifiedCanaryGateError):
                run_verified_live(
                    config,
                    client=resumed,
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                )
            self.assertEqual(resumed.calls, [])

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
            with self.assertRaisesRegex(VerifiedResponseContractError, "provider"):
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
            self.assertEqual(
                state["calls"][0]["recordStatus"],
                "response_contract_failure",
            )

    def test_provider_cost_must_match_the_frozen_endpoint_prices(self) -> None:
        config = load_verified_paraphrase_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "verified-checkpoint.json"
            with self.assertRaisesRegex(VerifiedResponseContractError, "cost"):
                run_verified_live(
                    config,
                    client=FakeVerifiedClient(cost=Decimal("0.0001")),
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                    max_new_calls=1,
                )
            state = json.loads(checkpoint.read_text())

        self.assertEqual(
            state["calls"][0]["recordStatus"],
            "response_contract_failure",
        )


if __name__ == "__main__":
    unittest.main()
