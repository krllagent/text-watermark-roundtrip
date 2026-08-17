from __future__ import annotations

import codecs
import copy
from decimal import Decimal
import json
from pathlib import Path
import re
import tempfile
import unittest

from corpus_contract import canonical_json_bytes
from run_model_canary_v6 import (
    ModelCanaryCheckpointError,
    ModelCanaryResponseContractError,
    build_blind_review_packet,
    build_model_canary_dry_run,
    build_model_canary_parity,
    expected_model_canary_call_ids,
    finalize_model_canary_review,
    load_model_canary_config,
    run_model_canary_live,
)
from unmark import ChatCompletion, CompletionUsage, StageRequest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "fixtures" / "model-canary-config-v6.json"
PARITY_PATH = ROOT / "fixtures" / "model-canary-parity-v6.json"
LUNA = "openai/gpt-5.6-luna"
TERRA = "openai/gpt-5.6-terra"


def _rot13_preserving_placeholders(value: str) -> str:
    return "".join(
        part if re.fullmatch(r"⟦T[1-9][0-9]*⟧", part) else codecs.decode(part, "rot_13")
        for part in re.split(r"(⟦T[1-9][0-9]*⟧)", value)
    )


class FakeCandidateClient:
    def __init__(
        self,
        model: str,
        *,
        provider: str = "OpenAI",
        fail_unknown: bool = False,
    ) -> None:
        self.model = model
        self.provider = provider
        self.fail_unknown = fail_unknown
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        request: str | StageRequest,
        *,
        model: str,
        max_tokens: int | None = None,
        response_format: dict[str, object] | None = None,
    ) -> ChatCompletion:
        if self.fail_unknown:
            self.fail_unknown = False
            raise RuntimeError("connection outcome unknown")
        if not isinstance(request, StageRequest):
            raise AssertionError("v6 must use the hardened StageRequest")
        if model != self.model:
            raise AssertionError("client/model mismatch")
        payload = json.loads(request.user_json)
        content = _rot13_preserving_placeholders(payload["sourceText"])
        self.calls.append(
            {
                "maxTokens": max_tokens,
                "messages": list(request.to_messages()),
                "model": model,
                "responseFormat": response_format,
                "stage": request.stage,
            }
        )
        dated = f"{model}-20260709"
        prompt_tokens = 100
        completion_tokens = 100
        if model == LUNA:
            cost = Decimal("0.00007")
        elif model == TERRA:
            cost = Decimal("0.0007")
        else:
            raise AssertionError("unexpected fake model")
        return ChatCompletion(
            content=content,
            finish_reason="stop",
            model=dated,
            openrouter_metadata={
                "attempt": 1,
                "attempts": [
                    {"model": dated, "provider": self.provider, "status": 200}
                ],
                "endpoints": {
                    "available": [
                        {
                            "model": dated,
                            "provider": self.provider,
                            "selected": True,
                        }
                    ]
                },
                "pipeline": [],
                "strategy": "direct",
            },
            provider=self.provider,
            response_id=f"fake-{model.rsplit('/', 1)[1]}-{len(self.calls):02d}",
            system_fingerprint=None,
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost=cost,
            ),
        )


def _clock() -> float:
    _clock.value += 0.125
    return _clock.value


_clock.value = 0.0


class ModelCanaryV6Tests(unittest.TestCase):
    def setUp(self) -> None:
        _clock.value = 0.0

    def test_config_freezes_development_cases_routes_and_strict_gate(self) -> None:
        config = load_model_canary_config(CONFIG_PATH)

        self.assertEqual(
            config.document_ids,
            ("doc-11", "doc-12", "doc-15", "doc-20", "doc-03", "doc-19"),
        )
        self.assertEqual(config.candidate_models, (LUNA, TERRA))
        self.assertEqual(config.calls_per_candidate, 6)
        self.assertEqual(config.total_provider_calls, 12)
        self.assertEqual(config.provider_order, ("openai",))
        self.assertEqual(config.expected_response_providers, ("OpenAI",))
        self.assertFalse(config.allow_fallbacks)
        self.assertEqual(config.data_collection, "deny")
        self.assertTrue(config.zdr)
        self.assertTrue(config.require_parameters)
        self.assertEqual(config.reasoning_effort, "medium")
        self.assertEqual(config.seed, 20260817)
        self.assertEqual(config.max_tokens, 4096)
        self.assertFalse(config.temperature_present)
        self.assertFalse(config.confirmatory_claim_allowed)
        self.assertEqual(config.minimum_mean_word_distance, 0.15)
        self.assertEqual(config.maximum_major_findings, 0)
        self.assertEqual(config.maximum_minor_findings, 0)
        self.assertEqual(config.maximum_pipeline_failures, 0)
        self.assertEqual(config.qwen_reference_provider_calls, 0)

        self.assertEqual(
            config.codex_plan_sha256,
            "217fdd8e394580497b4436a2e320892f0305b2f26296b4ccc6d9b4c31d9988ea",
        )
        self.assertEqual(
            config.codex_result_sha256,
            "13e241d222c63a720896bc879b53443b42224e1ebe9ac255ed4e20429ff390db",
        )

    def test_exact_candidate_major_call_order_and_request_parity_are_frozen(
        self,
    ) -> None:
        config = load_model_canary_config(CONFIG_PATH)
        call_ids = expected_model_canary_call_ids(config)

        self.assertEqual(len(call_ids), 12)
        self.assertEqual(
            call_ids[:7],
            (
                "openai/gpt-5.6-luna:doc-11:paraphrase-draft",
                "openai/gpt-5.6-luna:doc-12:paraphrase-draft",
                "openai/gpt-5.6-luna:doc-15:paraphrase-draft",
                "openai/gpt-5.6-luna:doc-20:paraphrase-draft",
                "openai/gpt-5.6-luna:doc-03:paraphrase-draft",
                "openai/gpt-5.6-luna:doc-19:paraphrase-draft",
                "openai/gpt-5.6-terra:doc-11:paraphrase-draft",
            ),
        )

        rebuilt = canonical_json_bytes(build_model_canary_parity(config))
        self.assertEqual(rebuilt, PARITY_PATH.read_bytes())
        fixture = json.loads(rebuilt)
        self.assertEqual(len(fixture["calls"]), 12)
        for call in fixture["calls"]:
            self.assertEqual(call["stage"], "paraphrase-draft")
            self.assertEqual(call["request"]["reasoning"]["effort"], "medium")
            self.assertNotIn("temperature", call["request"])
            self.assertEqual(call["request"]["max_tokens"], 4096)
            self.assertEqual(call["request"]["seed"], 20260817)
            self.assertEqual(call["request"]["provider"]["order"], ["openai"])
            self.assertFalse(call["request"]["provider"]["allow_fallbacks"])
            self.assertEqual(call["request"]["provider"]["data_collection"], "deny")
            self.assertTrue(call["request"]["provider"]["require_parameters"])
            self.assertTrue(call["request"]["provider"]["zdr"])
            self.assertNotIn("response_format", call["request"])
            self.assertRegex(call["requestSha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(len(call["request"]["messages"]), 2)
            self.assertEqual(call["request"]["messages"][0]["role"], "system")
            self.assertEqual(call["request"]["messages"][1]["role"], "user")

    def test_dry_run_is_local_only_and_reserves_all_twelve_calls(self) -> None:
        config = load_model_canary_config(CONFIG_PATH)

        first = build_model_canary_dry_run(config)
        second = build_model_canary_dry_run(config)

        self.assertEqual(first, second)
        self.assertEqual(first["callCount"], 12)
        self.assertEqual(first["callsByCandidate"], {LUNA: 6, TERRA: 6})
        self.assertEqual(first["qwenReference"]["providerCalls"], 0)
        self.assertGreater(Decimal(first["maximumRoutingCostReserveCredits"]), 0)
        self.assertEqual(
            first["requestSha256s"],
            json.loads(PARITY_PATH.read_text())["requestSha256s"],
        )

    def test_live_fake_run_checkpoints_all_outputs_metrics_and_qwen_reference(
        self,
    ) -> None:
        config = load_model_canary_config(CONFIG_PATH)
        clients = {
            LUNA: FakeCandidateClient(LUNA),
            TERRA: FakeCandidateClient(TERRA),
        }
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.json"
            artifact = run_model_canary_live(
                config,
                clients=clients,
                max_provider_cost_credits=Decimal("1"),
                checkpoint_path=checkpoint,
                clock=_clock,
            )

        self.assertEqual(len(clients[LUNA].calls), 6)
        self.assertEqual(len(clients[TERRA].calls), 6)
        self.assertEqual(artifact["usage"]["callCount"], 12)
        self.assertEqual(len(artifact["documents"]), 12)
        self.assertEqual(artifact["qwenReference"]["providerCallsMade"], 0)
        self.assertEqual(len(artifact["qwenReference"]["documents"]), 6)
        self.assertEqual(
            artifact["selectionGate"]["status"],
            "pending_blind_manual_review",
        )
        self.assertFalse(artifact["confirmatoryClaimAllowed"])
        for row in artifact["documents"]:
            self.assertTrue(row["pipeline"]["passed"])
            self.assertTrue(row["fidelity"]["protectedTokens"]["exactlyRestored"])
            self.assertGreaterEqual(
                row["fidelity"]["wordLevenshtein"]["normalizedDistance"],
                0.15,
            )
            self.assertIn("rawMaskedText", row)
            self.assertIn("outputText", row)
            self.assertIn("detector", row)
            self.assertIn("actualCostCredits", row)
            self.assertIn("latencyMs", row)

    def test_blind_packet_hides_candidate_document_cost_route_and_detector(
        self,
    ) -> None:
        config = load_model_canary_config(CONFIG_PATH)
        clients = {
            LUNA: FakeCandidateClient(LUNA),
            TERRA: FakeCandidateClient(TERRA),
        }
        with tempfile.TemporaryDirectory() as temporary:
            artifact = run_model_canary_live(
                config,
                clients=clients,
                max_provider_cost_credits=Decimal("1"),
                checkpoint_path=Path(temporary) / "checkpoint.json",
                clock=_clock,
            )
        packet = build_blind_review_packet(config, artifact)

        self.assertEqual(len(packet["pairs"]), 12)
        encoded = canonical_json_bytes(packet).decode()
        for forbidden in (LUNA, TERRA, '"documentId":', '"candidateModel":'):
            self.assertNotIn(forbidden, encoded)
        for pair in packet["pairs"]:
            self.assertEqual(set(pair), {"candidateText", "opaquePairId", "sourceText"})

    def test_manual_gate_selects_cheapest_passing_candidate_and_stops_if_none(
        self,
    ) -> None:
        config = load_model_canary_config(CONFIG_PATH)
        clients = {
            LUNA: FakeCandidateClient(LUNA),
            TERRA: FakeCandidateClient(TERRA),
        }
        with tempfile.TemporaryDirectory() as temporary:
            artifact = run_model_canary_live(
                config,
                clients=clients,
                max_provider_cost_credits=Decimal("1"),
                checkpoint_path=Path(temporary) / "checkpoint.json",
                clock=_clock,
            )
        packet = build_blind_review_packet(config, artifact)
        all_pass = {
            "schemaVersion": "model-canary-manual-review-v6/1.0",
            "artifactSha256": artifact["artifactSha256"],
            "manualPlanSha256": config.manual_plan_sha256,
            "reviews": [
                {"opaquePairId": pair["opaquePairId"], "verdict": "pass"}
                for pair in packet["pairs"]
            ],
        }

        selected = finalize_model_canary_review(config, artifact, all_pass)
        self.assertEqual(selected["selectionGate"]["status"], "selected")
        self.assertEqual(selected["selectionGate"]["selectedCandidate"], LUNA)

        luna_pair = next(
            row["opaquePairId"]
            for row in artifact["documents"]
            if row["candidateModel"] == LUNA
        )
        one_luna_minor = copy.deepcopy(all_pass)
        next(
            review
            for review in one_luna_minor["reviews"]
            if review["opaquePairId"] == luna_pair
        )["verdict"] = "minor"
        selected_terra = finalize_model_canary_review(config, artifact, one_luna_minor)
        self.assertEqual(selected_terra["selectionGate"]["selectedCandidate"], TERRA)

        one_each_minor = copy.deepcopy(one_luna_minor)
        terra_pair = next(
            row["opaquePairId"]
            for row in artifact["documents"]
            if row["candidateModel"] == TERRA
        )
        next(
            review
            for review in one_each_minor["reviews"]
            if review["opaquePairId"] == terra_pair
        )["verdict"] = "minor"
        stopped = finalize_model_canary_review(config, artifact, one_each_minor)
        self.assertEqual(stopped["selectionGate"]["status"], "stop_no_candidate_passed")
        self.assertIsNone(stopped["selectionGate"]["selectedCandidate"])

    def test_unknown_charge_tombstone_requires_explicit_resolution(self) -> None:
        config = load_model_canary_config(CONFIG_PATH)
        failing = FakeCandidateClient(LUNA, fail_unknown=True)
        clients = {LUNA: failing, TERRA: FakeCandidateClient(TERRA)}
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.json"
            with self.assertRaisesRegex(RuntimeError, "unknown"):
                run_model_canary_live(
                    config,
                    clients=clients,
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                    clock=_clock,
                )
            state = json.loads(checkpoint.read_text())
            call_id = state["inFlightCall"]["callId"]
            with self.assertRaises(ModelCanaryCheckpointError):
                run_model_canary_live(
                    config,
                    clients=clients,
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                    clock=_clock,
                )
            artifact = run_model_canary_live(
                config,
                clients=clients,
                max_provider_cost_credits=Decimal("1"),
                checkpoint_path=checkpoint,
                confirm_not_charged_call_id=call_id,
                clock=_clock,
            )
        self.assertEqual(artifact["usage"]["callCount"], 12)

    def test_bad_route_is_preserved_and_never_reissued(self) -> None:
        config = load_model_canary_config(CONFIG_PATH)
        bad_luna = FakeCandidateClient(LUNA, provider="Azure")
        clients = {LUNA: bad_luna, TERRA: FakeCandidateClient(TERRA)}
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.json"
            with self.assertRaises(ModelCanaryResponseContractError):
                run_model_canary_live(
                    config,
                    clients=clients,
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                    clock=_clock,
                )
            self.assertEqual(len(bad_luna.calls), 1)
            state = json.loads(checkpoint.read_text())
            self.assertEqual(
                state["calls"][0]["recordStatus"], "response_contract_failure"
            )
            with self.assertRaises(ModelCanaryResponseContractError):
                run_model_canary_live(
                    config,
                    clients={
                        LUNA: FakeCandidateClient(LUNA),
                        TERRA: FakeCandidateClient(TERRA),
                    },
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                    clock=_clock,
                )

    def test_config_rejects_temperature_fallbacks_or_changed_development_set(
        self,
    ) -> None:
        config = load_model_canary_config(CONFIG_PATH)
        cases: list[dict[str, object]] = []

        temperature = copy.deepcopy(config.raw)
        temperature["request"]["temperature"] = 0
        cases.append(temperature)
        fallback = copy.deepcopy(config.raw)
        fallback["routing"]["allowFallbacks"] = True
        cases.append(fallback)
        docs = copy.deepcopy(config.raw)
        docs["developmentCases"]["documentIds"][-1] = "doc-18"
        cases.append(docs)

        for index, raw in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "config.json"
                path.write_bytes(canonical_json_bytes(raw))
                with self.assertRaises(ValueError):
                    load_model_canary_config(path, root=ROOT)


if __name__ == "__main__":
    unittest.main()
