from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from unmark import ChatCompletion, CompletionUsage

from run_semantic_audit import (
    AuditBatchLimitReached,
    AuditResponseContractError,
    build_audit_batches,
    build_audit_prompt,
    build_blinded_pairs,
    load_audit_config,
    load_audit_source,
    parse_audit_response,
    protected_token_failure,
    run_audit,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "fixtures" / "semantic-audit-config-v1.json"


class FakeAuditClient:
    def __init__(self, *, provider: str = "Google") -> None:
        self.calls: list[dict[str, str]] = []
        self.provider = provider

    def complete(self, prompt: str, *, model: str) -> ChatCompletion:
        self.calls.append({"model": model, "prompt": prompt})
        pairs = json.loads(prompt.split("PAIRS_JSON\n", 1)[1])
        reviews = []
        for pair in pairs:
            reviews.append(
                {
                    "addedClaims": False,
                    "caveatDrift": False,
                    "certaintyDrift": False,
                    "changedClaims": False,
                    "evidenceNotes": [],
                    "lostClaims": False,
                    "lostOrChangedExamples": False,
                    "pairId": pair["pairId"],
                    "paragraphRoleOrOrderDrift": False,
                    "stanceDrift": False,
                    "voiceDrift": "none",
                }
            )
        content = json.dumps({"reviews": reviews}, separators=(",", ":"), sort_keys=True)
        suffix = len(self.calls)
        return ChatCompletion(
            content=content,
            finish_reason="stop",
            model="google/gemini-3.7-flash",
            openrouter_metadata={
                "attempt": 1,
                "endpoints": {
                    "available": [
                        {
                            "model": "google/gemini-3.7-flash-20260813",
                            "provider": self.provider,
                            "selected": True,
                        }
                    ]
                },
                "strategy": "direct",
            },
            provider=self.provider,
            response_id=f"audit-{suffix}",
            system_fingerprint=None,
            usage=CompletionUsage(
                prompt_tokens=100,
                completion_tokens=100,
                total_tokens=200,
                cost=Decimal("0.0001"),
            ),
        )


class SemanticAuditTests(unittest.TestCase):
    def test_blinded_pairs_are_complete_deterministic_and_hide_method_metadata(self) -> None:
        config = load_audit_config(CONFIG_PATH)
        source = load_audit_source(config)
        first = build_blinded_pairs(config, source)
        second = build_blinded_pairs(config, source)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 80)
        self.assertEqual(len({pair.pair_id for pair in first}), 80)
        batches = build_audit_batches(first, config.batch_size)
        self.assertEqual([len(batch) for batch in batches], [4] * 20)
        prompt = build_audit_prompt(batches[0])
        for hidden in ("methodId", "detector", "latency", "costUsd"):
            self.assertNotIn(hidden, prompt)
        self.assertIn("sourceText", prompt)
        self.assertIn("candidateText", prompt)

    def test_follow_up_audit_can_freeze_one_method_and_twenty_pairs(self) -> None:
        base_config = load_audit_config(CONFIG_PATH)
        source = load_audit_source(base_config)
        original_method = next(
            method
            for method in source["methods"]
            if method["methodId"] == "paraphrase"
        )
        follow_up_source = {
            "methods": [
                {
                    **original_method,
                    "methodId": "paraphrase-verified",
                }
            ]
        }
        follow_up_config = replace(
            base_config,
            method_ids=("paraphrase-verified",),
            pair_order_seed="exp002-semantic-audit-v2",
            structured_pair_count=20,
        )

        pairs = build_blinded_pairs(follow_up_config, follow_up_source)

        self.assertEqual(len(pairs), 20)
        self.assertEqual({pair.method_id for pair in pairs}, {"paraphrase-verified"})

    def test_parse_response_requires_exact_pair_ids_and_fields(self) -> None:
        pair_ids = ("pair-a", "pair-b")
        review = {
            "addedClaims": False,
            "caveatDrift": False,
            "certaintyDrift": False,
            "changedClaims": False,
            "evidenceNotes": ["A concise note."],
            "lostClaims": False,
            "lostOrChangedExamples": False,
            "paragraphRoleOrOrderDrift": False,
            "stanceDrift": False,
            "voiceDrift": "minor",
        }
        content = json.dumps(
            {
                "reviews": [
                    {"pairId": pair_id, **review}
                    for pair_id in reversed(pair_ids)
                ]
            }
        )
        parsed = parse_audit_response(content, pair_ids)
        self.assertEqual([item["pairId"] for item in parsed], list(pair_ids))
        broken = json.dumps({"reviews": [{"pairId": "pair-a", **review}]})
        with self.assertRaisesRegex(AuditResponseContractError, "pair IDs"):
            parse_audit_response(broken, pair_ids)

    def test_protected_token_check_ignores_new_quotes_but_catches_damage(self) -> None:
        source = 'Keep `CODE-1`, send to a@example.com, and spend $18, today.'
        added_quote = (
            'Keep `CODE-1`, call it "exactly-once", send to a@example.com, '
            'and spend $18, today.'
        )
        self.assertFalse(protected_token_failure(source, added_quote)["failed"])
        doubled = 'Keep `CODE-1`, send to a@example.com, and spend $18,, today.'
        self.assertFalse(protected_token_failure(source, doubled)["failed"])
        missing = 'Keep `CODE-1`, and spend $18, today.'
        self.assertTrue(protected_token_failure(source, missing)["failed"])

    def test_protected_token_check_does_not_cascade_after_first_missing_token(self) -> None:
        source = "Spend $18, email a@example.com, and keep 15%."
        candidate = "Spend $19, email a@example.com, and keep 15%."

        evidence = protected_token_failure(source, candidate)

        self.assertTrue(evidence["failed"])
        self.assertEqual(evidence["missing"], ["$18"])
        self.assertEqual(evidence["expectedCount"], 3)
        self.assertEqual(evidence["observedCount"], 3)

    def test_checkpointed_batches_resume_without_repeating_paid_work(self) -> None:
        config = load_audit_config(CONFIG_PATH)
        source = load_audit_source(config)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "audit-checkpoint.json"
            first_client = FakeAuditClient()
            with self.assertRaises(AuditBatchLimitReached) as paused:
                run_audit(
                    config,
                    source,
                    client=first_client,
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                    max_new_batches=1,
                )
            self.assertEqual(paused.exception.completed_batches, 1)
            self.assertEqual(len(first_client.calls), 1)
            second_client = FakeAuditClient()
            artifact = run_audit(
                config,
                source,
                client=second_client,
                max_provider_cost_credits=Decimal("1"),
                checkpoint_path=checkpoint,
            )
            self.assertEqual(len(second_client.calls), 19)
            self.assertEqual(artifact["usage"]["callCount"], 20)
            self.assertEqual(len(artifact["reviews"]), 80)
            self.assertEqual(len(artifact["opaqueMapping"]), 80)

    def test_route_mismatch_is_preserved_and_stops(self) -> None:
        config = load_audit_config(CONFIG_PATH)
        source = load_audit_source(config)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "audit-checkpoint.json"
            with self.assertRaisesRegex(AuditResponseContractError, "provider"):
                run_audit(
                    config,
                    source,
                    client=FakeAuditClient(provider="Other Provider"),
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                    max_new_batches=1,
                )
            state = json.loads(checkpoint.read_text())
            self.assertEqual(len(state["calls"]), 1)
            self.assertIsNone(state["inFlightCall"])


if __name__ == "__main__":
    unittest.main()
