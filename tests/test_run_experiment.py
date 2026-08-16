from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from corpus_contract import canonical_json_bytes
from text_contract import WORD_RE, analyze_text
from unmark import ChatCompletion, CompletionUsage, ProviderError, ProviderResponseError
from watermark_toy import SynonymLexicon, score_text as real_score_text

from run_experiment import (
    BudgetError,
    CallLimitReached,
    CheckpointError,
    ControlGateError,
    ResponseContractError,
    build_dry_run,
    load_experiment_config,
    load_reviewed_corpus,
    paired_bootstrap,
    protected_restoration_metrics,
    run_live as _run_live,
    verify_prepaid_controls,
    word_levenshtein_metrics,
)


class FakeClient:
    def __init__(self, *, fail_on_backward_de: bool = False) -> None:
        self.calls: list[dict[str, str]] = []
        self.fail_on_backward_de = fail_on_backward_de

    def complete(self, prompt: str, *, model: str) -> ChatCompletion:
        self.calls.append({"model": model, "prompt": prompt})
        if self.fail_on_backward_de and "complete German text" in prompt:
            raise ProviderError("scripted failure")
        source = _prompt_text(prompt)
        if "into German" in prompt:
            content = _prefix_paragraphs(source, "Das ist und die der ")
        elif "into Simplified Chinese" in prompt:
            content = _prefix_paragraphs(
                source,
                "这是一个完整的中文段落它保留所有事实例子限制条件和原始顺序。",
            )
        elif "complete German text" in prompt:
            content = _strip_paragraph_prefix(source, "Das ist und die der ")
            content = _rewrite(content)
        elif "complete Simplified Chinese text" in prompt:
            content = _strip_paragraph_prefix(
                source,
                "这是一个完整的中文段落它保留所有事实例子限制条件和原始顺序。",
            )
            content = _rewrite(content)
        else:
            content = _rewrite(source)
        suffix = str(len(self.calls))
        return ChatCompletion(
            content=content,
            finish_reason="stop",
            model=model,
            openrouter_metadata={
                "attempt": 1,
                "attempts": [
                    {
                        "model": "qwen/qwen3.5-9b",
                        "provider": "DeepInfra",
                        "status": 200,
                    }
                ],
                "endpoints": {
                    "available": [
                        {
                            "model": "qwen/qwen3.5-9b",
                            "provider": "DeepInfra",
                            "selected": True,
                        }
                    ]
                },
                "pipeline": [],
                "strategy": "direct",
            },
            provider="DeepInfra",
            response_id=f"fake-{suffix}",
            system_fingerprint="fake-fingerprint-v1",
            usage=CompletionUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost=Decimal("0.00001"),
            ),
        )


class CurrentMetadataClient(FakeClient):
    """Mirror the compact routing metadata returned by OpenRouter in production."""

    def complete(self, prompt: str, *, model: str) -> ChatCompletion:
        good = super().complete(prompt, model=model)
        metadata = dict(good.openrouter_metadata)
        metadata.pop("attempts")
        metadata.update(
            {
                "is_byok": False,
                "region": "LHR",
                "requested": model,
                "summary": "available=1, selected=DeepInfra",
            }
        )
        return ChatCompletion(
            content=good.content,
            finish_reason=good.finish_reason,
            model=good.model,
            openrouter_metadata=metadata,
            provider=good.provider,
            response_id=good.response_id,
            system_fingerprint=good.system_fingerprint,
            usage=good.usage,
        )


class ForbiddenClient:
    def complete(self, prompt: str, *, model: str) -> ChatCompletion:
        raise AssertionError("provider must not be called")


class BadMetadataClient(FakeClient):
    def complete(self, prompt: str, *, model: str) -> ChatCompletion:
        good = super().complete(prompt, model=model)
        return ChatCompletion(
            content=good.content,
            finish_reason=good.finish_reason,
            model=good.model,
            openrouter_metadata={"attempt": 2, "pipeline": ["fallback"], "strategy": "fallback"},
            provider=good.provider,
            response_id=good.response_id,
            system_fingerprint=good.system_fingerprint,
            usage=good.usage,
        )


class BadAttemptProviderClient(FakeClient):
    def complete(self, prompt: str, *, model: str) -> ChatCompletion:
        good = super().complete(prompt, model=model)
        metadata = dict(good.openrouter_metadata)
        metadata["attempts"] = [
            {
                "model": "qwen/qwen3.5-9b",
                "provider": "OtherProvider",
                "status": 200,
            }
        ]
        return ChatCompletion(
            content=good.content,
            finish_reason=good.finish_reason,
            model=good.model,
            openrouter_metadata=metadata,
            provider=good.provider,
            response_id=good.response_id,
            system_fingerprint=good.system_fingerprint,
            usage=good.usage,
        )


class BadAttemptModelClient(FakeClient):
    def complete(self, prompt: str, *, model: str) -> ChatCompletion:
        good = super().complete(prompt, model=model)
        metadata = dict(good.openrouter_metadata)
        metadata["attempts"] = [
            {
                "model": "other/model",
                "provider": "DeepInfra",
                "status": 200,
            }
        ]
        return ChatCompletion(
            content=good.content,
            finish_reason=good.finish_reason,
            model=good.model,
            openrouter_metadata=metadata,
            provider=good.provider,
            response_id=good.response_id,
            system_fingerprint=good.system_fingerprint,
            usage=good.usage,
        )


class BadAttemptStatusClient(FakeClient):
    def complete(self, prompt: str, *, model: str) -> ChatCompletion:
        good = super().complete(prompt, model=model)
        metadata = dict(good.openrouter_metadata)
        metadata["attempts"] = [
            {
                "model": "qwen/qwen3.5-9b",
                "provider": "DeepInfra",
                "status": 500,
            }
        ]
        return ChatCompletion(
            content=good.content,
            finish_reason=good.finish_reason,
            model=good.model,
            openrouter_metadata=metadata,
            provider=good.provider,
            response_id=good.response_id,
            system_fingerprint=good.system_fingerprint,
            usage=good.usage,
        )


class IdentityClient(FakeClient):
    def complete(self, prompt: str, *, model: str) -> ChatCompletion:
        good = super().complete(prompt, model=model)
        return ChatCompletion(
            content=_prompt_text(prompt),
            finish_reason="length",
            model=good.model,
            openrouter_metadata=good.openrouter_metadata,
            provider=good.provider,
            response_id=good.response_id,
            system_fingerprint=good.system_fingerprint,
            usage=good.usage,
        )


class MissingPlaceholderClient(FakeClient):
    def complete(self, prompt: str, *, model: str) -> ChatCompletion:
        good = super().complete(prompt, model=model)
        content = good.content
        if "limited number of content words" in prompt:
            content = content.replace("⟦T1⟧", "[missing protected token]", 1)
        return ChatCompletion(
            content=content,
            finish_reason=good.finish_reason,
            model=good.model,
            openrouter_metadata=good.openrouter_metadata,
            provider=good.provider,
            response_id=good.response_id,
            system_fingerprint=good.system_fingerprint,
            usage=good.usage,
        )


class BlankSynonymClient(FakeClient):
    def complete(self, prompt: str, *, model: str) -> ChatCompletion:
        good = super().complete(prompt, model=model)
        return ChatCompletion(
            content=(
                " " if "limited number of content words" in prompt else good.content
            ),
            finish_reason=good.finish_reason,
            model=good.model,
            openrouter_metadata=good.openrouter_metadata,
            provider=good.provider,
            response_id=good.response_id,
            system_fingerprint=good.system_fingerprint,
            usage=good.usage,
        )


class DecimalMetadataClient(FakeClient):
    def complete(self, prompt: str, *, model: str) -> ChatCompletion:
        good = super().complete(prompt, model=model)
        metadata = dict(good.openrouter_metadata)
        metadata["numeric"] = {"latencySeconds": Decimal("0.123")}
        return ChatCompletion(
            content=good.content,
            finish_reason=good.finish_reason,
            model=good.model,
            openrouter_metadata=metadata,
            provider=good.provider,
            response_id=good.response_id,
            system_fingerprint=good.system_fingerprint,
            usage=good.usage,
        )


class SimulatedProcessCrash(BaseException):
    pass


class CrashClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, *, model: str) -> ChatCompletion:
        self.calls += 1
        raise SimulatedProcessCrash("simulated process death after request dispatch")


class InvalidProviderResponseClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, *, model: str) -> ChatCompletion:
        self.calls += 1
        raise ProviderResponseError(
            "OpenRouter response is missing usage.cost",
            raw_response={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Paid partial output."},
                    }
                ],
                "id": "paid-invalid-1",
                "model": model,
                "openrouter_metadata": {
                    "attempt": 1,
                    "numeric": Decimal("0.123"),
                },
                "provider": "DeepInfra",
                "usage": {
                    "completion_tokens": 20,
                    "prompt_tokens": 100,
                    "total_tokens": 120,
                },
            },
        )


def _prompt_text(prompt: str) -> str:
    start = "--- BEGIN TEXT ---\n"
    end = "\n--- END TEXT ---"
    return prompt.split(start, 1)[1].rsplit(end, 1)[0]


def _prefix_paragraphs(text: str, prefix: str) -> str:
    return "\n\n".join(prefix + paragraph for paragraph in text.split("\n\n"))


def _strip_paragraph_prefix(text: str, prefix: str) -> str:
    return "\n\n".join(
        paragraph[len(prefix) :] if paragraph.startswith(prefix) else paragraph
        for paragraph in text.split("\n\n")
    )


def _rewrite(text: str) -> str:
    for old, new in (("plan", "scheme"), ("Plan", "Scheme"), ("test", "trial")):
        if old in text:
            return text.replace(old, new, 1)
    return text + " Revised."


_TEST_GATE_RESULT = {
    "acceptancePassed": True,
    "artifactSha256": "f" * 64,
    "outputsMatch": True,
}


def run_live(*args, **kwargs):
    """Keep fake matrix tests cheap while retaining a dedicated real gate test."""
    with patch(
        "run_experiment.verify_prepaid_controls",
        return_value=dict(_TEST_GATE_RESULT),
    ):
        return _run_live(*args, **kwargs)


class ExperimentFixture:
    def __init__(self, root: Path, *, bootstrap_replicates: int = 40) -> None:
        self.root = root
        self.lexicon = SynonymLexicon.from_pairs(
            [["big", "large"], ["small", "little"], ["begin", "start"]]
        )
        self.lexicon_path = root / "fixtures" / "synonym_pairs-v1.json"
        self.lexicon_path.parent.mkdir(parents=True)
        lexicon_raw = {
            "lexiconVersion": "test-lexicon-v1",
            "manualContextReviewRequired": True,
            "methodology": "Small deterministic test lexicon.",
            "pairs": [["big", "large"], ["small", "little"], ["begin", "start"]],
            "schemaVersion": 1,
            "sources": [{"title": "Test", "url": "https://example.com/source"}],
            "verifiedAt": "2026-08-16",
        }
        self.lexicon_path.write_bytes(canonical_json_bytes(lexicon_raw))

        source_dir = root / "corpus" / "original"
        source_dir.mkdir(parents=True)
        documents: list[dict[str, object]] = []
        inventory_documents: list[dict[str, object]] = []
        for index in range(1, 21):
            document_id = f"doc-{index:02d}"
            relative = f"corpus/original/{document_id}.md"
            text = (
                f"# Document {index}\n\n"
                "A big plan can start with a small test beside "
                f"https://example.com/{index}.\n\n"
                "A large team can begin with a little trial and record the outcome.\n"
            )
            path = root / relative
            path.write_text(text, encoding="utf-8", newline="\n")
            raw = path.read_bytes()
            analysis = analyze_text(text)
            eligible = [
                token
                for token in analysis.context_tokens
                if not token.protected
                and token.text is not None
                and token.normalized in self.lexicon.token_to_pair
            ]
            sha256 = hashlib.sha256(raw).hexdigest()
            documents.append(
                {
                    "documentId": document_id,
                    "eligiblePositions": len(eligible),
                    "genre": "test",
                    "path": relative,
                    "protectedSpanCount": len(analysis.protected_spans),
                    "sha256": sha256,
                    "title": f"Document {index}",
                    "wordCount": len(WORD_RE.findall(text)),
                }
            )
            inventory_documents.append(
                {
                    "documentId": document_id,
                    "eligibleOccurrences": len(eligible),
                    "occurrences": [],
                    "reviewStatus": "pending_manual_context_review",
                    "sha256": sha256,
                }
            )

        evidence = {
            "methodology": "Deterministic test evidence.",
            "schemaVersion": 1,
            "sources": [{"title": "Test", "url": "https://example.com/source"}],
            "verifiedAt": "2026-08-16",
        }
        manifest = {
            **evidence,
            "corpusVersion": "test-corpus-v1",
            "documentCount": 20,
            "documents": documents,
            "eligiblePositions": sum(int(item["eligiblePositions"]) for item in documents),
            "lexiconSha256": self.lexicon.sha256,
            "planSha256": "a" * 64,
            "wordCount": sum(int(item["wordCount"]) for item in documents),
        }
        inventory = {
            **evidence,
            "corpusVersion": "test-corpus-v1",
            "documents": inventory_documents,
            "lexiconSha256": self.lexicon.sha256,
        }
        manifest_path = root / "corpus" / "manifest-v1.json"
        inventory_path = root / "corpus" / "context-inventory-v1.json"
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        inventory_path.write_bytes(canonical_json_bytes(inventory))

        inventory_sha256 = hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()
        review = {
            **evidence,
            "corpusVersion": "test-corpus-v1",
            "documents": [
                {
                    "decision": "approved",
                    "documentId": item["documentId"],
                    "documentSha256": item["sha256"],
                    "eligibleOccurrences": item["eligibleOccurrences"],
                    "findings": [],
                    "reviewedOccurrences": item["eligibleOccurrences"],
                }
                for item in inventory_documents
            ],
            "inventorySha256": inventory_sha256,
            "lexiconSha256": self.lexicon.sha256,
            "reviewVersion": "test-review-v1",
            "reviewer": "test-reviewer",
        }
        review_path = root / "corpus" / "reviews" / "review-v1.json"
        review_path.parent.mkdir(parents=True)
        review_path.write_bytes(canonical_json_bytes(review))

        endpoint_snapshot = {
            **evidence,
            "catalogModelId": "qwen/qwen3.5-9b",
            "endpoint": {
                "contextLength": 262144,
                "maxCompletionTokens": 81920,
                "name": "DeepInfra test endpoint",
                "pricingUsdPerToken": {
                    "completion": "0.00000015",
                    "prompt": "0.0000001",
                },
                "providerName": "DeepInfra",
                "quantization": "bf16",
                "supportedParameters": ["max_tokens", "seed", "temperature"],
                "tag": "deepinfra/bf16",
            },
            "requestedModelId": "qwen/qwen3.5-9b",
            "snapshotVersion": "test-endpoint-v1",
        }
        endpoint_path = root / "fixtures" / "openrouter-endpoint-snapshot-v1.json"
        endpoint_path.write_bytes(canonical_json_bytes(endpoint_snapshot))

        semantic_audit_plan = {
            **evidence,
            "auditVersion": "test-semantic-audit-v1",
            "closeReadingSample": {"documentsPerMethod": 3},
            "scope": {
                "documentCountPerMethod": 20,
                "methods": [
                    "synonyms",
                    "roundtrip-de",
                    "roundtrip-zh",
                    "paraphrase",
                ],
                "structuredPairCount": 80,
            },
        }
        semantic_audit_path = root / "fixtures" / "semantic-audit-plan-v1.json"
        semantic_audit_path.write_bytes(canonical_json_bytes(semantic_audit_plan))

        config = {
            **evidence,
            "analysis": {
                "bootstrapReplicates": bootstrap_replicates,
                "bootstrapSeed": 20260816,
                "primaryScoringUnit": "pooled_corpus",
                "qualityMetric": "normalized_word_levenshtein",
                "resamplingUnit": "document_id",
                "semanticAuditPlanPath": "fixtures/semantic-audit-plan-v1.json",
                "semanticAuditPlanSha256": hashlib.sha256(
                    semantic_audit_path.read_bytes()
                ).hexdigest(),
            },
            "billing": {
                "creditBaseCurrency": "USD",
                "creditUsdBaseUnit": "1",
                "inferencePricingMarkupPercent": 0,
                "promptTokenOverheadReserve": 2048,
                "purchaseFeeExcluded": True,
            },
            "corpus": {
                "documentCount": 20,
                "inventoryPath": "corpus/context-inventory-v1.json",
                "inventorySha256": hashlib.sha256(
                    inventory_path.read_bytes()
                ).hexdigest(),
                "manifestPath": "corpus/manifest-v1.json",
                "manifestSha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
                "planPath": "fixtures/corpus-plan-v1.json",
                "reviews": [
                    {
                        "path": "corpus/reviews/review-v1.json",
                        "sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
                    }
                ],
                "reviewsDirectory": "corpus/reviews",
            },
            "experimentVersion": "test-experiment-v1",
            "lexicon": {
                "path": "fixtures/synonym_pairs-v1.json",
                "sha256": hashlib.sha256(self.lexicon_path.read_bytes()).hexdigest(),
            },
            "marker": {
                "contextWidth": 4,
                "densitiesBps": [500, 1000, 2000],
                "keyHex": "11" * 32,
                "mainDensityBps": 1000,
                "minActivePositions": 1,
                "wrongKeyCount": 10,
                "wrongKeySeedHex": "22" * 32,
            },
            "transforms": {
                "allowFallbacks": False,
                "dataCollection": "deny",
                "endpointSnapshotPath": "fixtures/openrouter-endpoint-snapshot-v1.json",
                "endpointSnapshotSha256": hashlib.sha256(
                    endpoint_path.read_bytes()
                ).hexdigest(),
                "expectedResponseModels": [
                    "qwen/qwen3.5-9b",
                    "qwen/qwen3.5-9b-20260310",
                ],
                "expectedResponseProviders": ["DeepInfra"],
                "maxPriceUsdPerMillionTokens": {
                    "completion": 0.15,
                    "prompt": 0.10,
                },
                "maxTokens": 512,
                "methods": [
                    {"id": "none", "pivot": None},
                    {"id": "synonyms", "pivot": None},
                    {"id": "roundtrip-de", "method": "roundtrip", "pivot": "de"},
                    {"id": "roundtrip-zh", "method": "roundtrip", "pivot": "zh"},
                    {"id": "paraphrase", "pivot": None},
                ],
                "modelBackward": "qwen/qwen3.5-9b",
                "modelForward": "qwen/qwen3.5-9b",
                "pricingUsdPerMillionTokens": {
                    "completion": "0.15",
                    "prompt": "0.10",
                },
                "providerOrder": ["deepinfra/bf16"],
                "reasoningEffort": "none",
                "requireParameters": True,
                "seed": 20260816,
                "temperature": 0,
                "timeoutSeconds": 30,
                "zdr": True,
            },
        }
        self.config_path = root / "fixtures" / "experiment-config-v1.json"
        self.config_path.write_bytes(canonical_json_bytes(config))

    def load(self):
        config = load_experiment_config(self.config_path, root=self.root)
        return config, load_reviewed_corpus(config)


class DryRunTests(unittest.TestCase):
    def test_dry_run_needs_no_environment_key_or_client_and_counts_exact_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ExperimentFixture(Path(temporary))
            config, corpus = fixture.load()
            report = build_dry_run(config, corpus)

        self.assertEqual(report["documentCount"], 20)
        self.assertEqual(report["callCount"], 120)
        self.assertEqual(
            report["callsByMethod"],
            {
                "none": 0,
                "paraphrase": 20,
                "roundtrip-de": 40,
                "roundtrip-zh": 40,
                "synonyms": 20,
            },
        )
        estimate = report["tokenEstimate"]
        self.assertGreater(estimate["promptTokensPlanningEstimate"], 0)
        self.assertEqual(
            estimate["completionTokensAtConfiguredMaximum"],
            120 * 512,
        )
        self.assertGreater(
            estimate["totalTokensPlanningEstimate"],
            estimate["completionTokensAtConfiguredMaximum"],
        )
        self.assertGreater(
            Decimal(report["routingCostPlanningEstimateUsd"]),
            Decimal("0"),
        )
        self.assertIn("not a mathematical upper bound", report["methodology"])
        self.assertEqual(
            report["semanticAuditPlanSha256"],
            config.semantic_audit_plan_sha256,
        )

    def test_loader_rejects_manifest_drift_and_provider_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExperimentFixture(root)
            config = json.loads(fixture.config_path.read_text(encoding="utf-8"))
            config["transforms"]["allowFallbacks"] = True
            fixture.config_path.write_bytes(canonical_json_bytes(config))
            with self.assertRaisesRegex(ValueError, "allowFallbacks"):
                load_experiment_config(fixture.config_path, root=root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExperimentFixture(root)
            (root / "corpus" / "original" / "doc-01.md").write_text(
                "# Drifted\n\nChanged bytes.\n",
                encoding="utf-8",
            )
            config = load_experiment_config(fixture.config_path, root=root)
            with self.assertRaisesRegex(ValueError, "document hash"):
                load_reviewed_corpus(config)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExperimentFixture(root)
            manifest_path = root / "corpus" / "manifest-v1.json"
            manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "frozen config binding"):
                load_experiment_config(fixture.config_path, root=root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExperimentFixture(root)
            audit_path = root / "fixtures" / "semantic-audit-plan-v1.json"
            audit_path.write_bytes(audit_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "semantic audit plan"):
                load_experiment_config(fixture.config_path, root=root)


class ControlGateTests(unittest.TestCase):
    def test_gate_rebuilds_in_memory_and_requires_acceptance_and_freshness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ExperimentFixture(Path(temporary))
            config, corpus = fixture.load()
            outputs = SimpleNamespace(
                artifact={"acceptance": {"passed": True}},
                files={},
            )
            with (
                patch(
                    "run_corpus_controls.control_spec_from_config",
                    return_value=object(),
                ) as spec,
                patch(
                    "run_corpus_controls.build_corpus_controls",
                    return_value=outputs,
                ) as build,
                patch(
                    "run_corpus_controls.check_control_outputs",
                    return_value={
                        "files": {"results/corpus-controls-v1.json": True},
                        "passed": True,
                        "unexpectedFiles": [],
                    },
                ) as freshness,
            ):
                report = verify_prepaid_controls(config, corpus)

            spec.assert_called_once_with(config)
            build.assert_called_once_with(config, corpus, spec=spec.return_value)
            freshness.assert_called_once_with(config.root, outputs)
            self.assertTrue(report["acceptancePassed"])
            self.assertTrue(report["outputsMatch"])
            self.assertRegex(report["artifactSha256"], r"^[0-9a-f]{64}$")

            with (
                patch(
                    "run_corpus_controls.control_spec_from_config",
                    return_value=object(),
                ),
                patch(
                    "run_corpus_controls.build_corpus_controls",
                    return_value=outputs,
                ),
                patch(
                    "run_corpus_controls.check_control_outputs",
                    return_value={
                        "files": {"results/corpus-controls-v1.json": False},
                        "passed": False,
                        "unexpectedFiles": [],
                    },
                ),
            ):
                with self.assertRaisesRegex(ControlGateError, "byte-exact"):
                    verify_prepaid_controls(config, corpus)

    def test_failed_gate_blocks_before_first_client_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExperimentFixture(root)
            config, corpus = fixture.load()
            with patch(
                "run_experiment.verify_prepaid_controls",
                side_effect=ControlGateError("control acceptance failed"),
            ):
                with self.assertRaisesRegex(ControlGateError, "acceptance"):
                    _run_live(
                        config,
                        corpus,
                        client=ForbiddenClient(),
                        max_provider_cost_credits=Decimal("1"),
                        checkpoint_path=root / "checkpoint.json",
                    )


class LiveExperimentTests(unittest.TestCase):
    def test_full_fake_run_has_raw_calls_metrics_pooling_and_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExperimentFixture(root)
            config, corpus = fixture.load()
            client = FakeClient()
            checkpoint = root / "results" / "checkpoint.json"
            with patch("run_experiment.score_text", wraps=real_score_text) as scorer:
                artifact = run_live(
                    config,
                    corpus,
                    client=client,
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                    clock=_StepClock(),
                )

            self.assertEqual(len(client.calls), 120)
            self.assertEqual(scorer.call_count, 100)
            self.assertEqual(artifact["documentCount"], 20)
            self.assertEqual([item["methodId"] for item in artifact["methods"]], [
                "none", "synonyms", "roundtrip-de", "roundtrip-zh", "paraphrase"
            ])
            self.assertEqual(artifact["usage"]["callCount"], 120)
            self.assertEqual(artifact["usage"]["providerCostCredits"], "0.00120")
            self.assertEqual(artifact["usage"]["providerCostBudgetCredits"], "1")
            self.assertEqual(artifact["prepaidControlGate"], _TEST_GATE_RESULT)
            self.assertEqual(
                artifact["semanticAuditPlanSha256"],
                config.semantic_audit_plan_sha256,
            )
            self.assertEqual(artifact["pairedBootstrap"]["sampleSize"], 20)
            self.assertEqual(len(artifact["pairedBootstrap"]["documentIds"]), 20)
            first = artifact["methods"][2]["documents"][0]
            self.assertEqual([call["stage"] for call in first["calls"]], [
                "forward-de", "backward-de"
            ])
            for call in first["calls"]:
                self.assertIn("prompt", call)
                self.assertIn("inputText", call)
                self.assertIn("outputText", call)
                self.assertEqual(call["response"]["provider"], "DeepInfra")
                self.assertEqual(call["response"]["systemFingerprint"], "fake-fingerprint-v1")
                self.assertEqual(call["latencyMs"], 10.0)
            self.assertTrue(first["fidelity"]["protectedTokens"]["exactlyRestored"])
            self.assertIn("survivingActive", first["fingerprints"])
            synonym_aggregate = artifact["methods"][1]["aggregate"]
            self.assertEqual(len(synonym_aggregate["detector"]["documents"]), 20)
            self.assertIn("insufficientDocumentCount", synonym_aggregate["detector"])
            self.assertIn("insufficientDocumentRate", synonym_aggregate["detector"])
            self.assertEqual(
                synonym_aggregate["usage"]["medianPerDocumentLatencyMs"],
                10.0,
            )
            self.assertEqual(
                synonym_aggregate["usage"]["medianPerDocumentProviderCostCredits"],
                "0.00001",
            )
            self.assertEqual(
                synonym_aggregate["usage"]["meanPerDocumentProviderCostCredits"],
                "0.00001",
            )
            self.assertEqual(
                synonym_aggregate["usage"]["providerCostCreditsPer1000Documents"],
                "0.01000",
            )
            self.assertGreater(synonym_aggregate["usage"]["totalInputWordCount"], 0)
            self.assertIsNotNone(
                synonym_aggregate["usage"][
                    "providerCostCreditsPer1000MarkedInputWords"
                ]
            )
            for method in artifact["methods"]:
                controls = method["wrongKeyControls"]
                self.assertEqual(controls["count"], 10)
                self.assertEqual(len(controls["scores"]), 10)
                self.assertNotIn("keySha256", json.dumps(controls, sort_keys=True))
                self.assertEqual(method["aggregate"]["fidelityFailureCount"], 0)
                self.assertEqual(method["aggregate"]["fidelityFailureRate"], 0)
            canonical_json_bytes(artifact)

    def test_stage1_validation_failures_are_rows_not_retry_or_abort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExperimentFixture(root, bootstrap_replicates=10)
            config, corpus = fixture.load()
            client = IdentityClient()
            artifact = run_live(
                config,
                corpus,
                client=client,
                max_provider_cost_credits=Decimal("1"),
                checkpoint_path=root / "checkpoint.json",
            )

            self.assertEqual(len(client.calls), 120)
            self.assertEqual(artifact["usage"]["callCount"], 120)
            methods = {item["methodId"]: item for item in artifact["methods"]}
            self.assertEqual(methods["none"]["aggregate"]["fidelityFailureCount"], 0)
            for method_id in (
                "synonyms",
                "roundtrip-de",
                "roundtrip-zh",
                "paraphrase",
            ):
                method = methods[method_id]
                self.assertEqual(len(method["documents"]), 20)
                self.assertEqual(method["aggregate"]["fidelityFailureCount"], 20)
                self.assertEqual(method["aggregate"]["fidelityFailureRate"], 1)
                first = method["documents"][0]
                self.assertEqual(
                    first["transformationOutcome"]["status"],
                    "validation_failure",
                )
                self.assertIn(
                    "unchanged_output",
                    {
                        issue["code"]
                        for issue in first["transformationOutcome"]["issues"]
                    },
                )
                self.assertIn(
                    "finish_reason_contract",
                    {
                        issue["code"]
                        for issue in first["transformationOutcome"]["issues"]
                    },
                )
                self.assertEqual(
                    first["calls"][-1]["response"]["finishReason"],
                    "length",
                )
                self.assertEqual(first["outputText"], first["markedInputText"])

    def test_missing_placeholder_is_visible_best_effort_fidelity_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExperimentFixture(root, bootstrap_replicates=10)
            config, corpus = fixture.load()
            artifact = run_live(
                config,
                corpus,
                client=MissingPlaceholderClient(),
                max_provider_cost_credits=Decimal("1"),
                checkpoint_path=root / "checkpoint.json",
            )

            synonyms = next(
                item for item in artifact["methods"] if item["methodId"] == "synonyms"
            )
            self.assertEqual(synonyms["aggregate"]["fidelityFailureCount"], 20)
            first = synonyms["documents"][0]
            self.assertEqual(
                first["transformationOutcome"]["restorationMode"],
                "best_effort",
            )
            self.assertIn("[missing protected token]", first["outputText"])
            self.assertFalse(first["fidelity"]["protectedTokens"]["exactlyRestored"])
            self.assertEqual(len(first["calls"]), 1)
            self.assertIn("⟦T1⟧", first["calls"][0]["inputText"])
            self.assertIn("[missing protected token]", first["calls"][0]["outputText"])

    def test_paid_blank_outputs_remain_in_all_twenty_denominators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExperimentFixture(root, bootstrap_replicates=10)
            config, corpus = fixture.load()
            artifact = run_live(
                config,
                corpus,
                client=BlankSynonymClient(),
                max_provider_cost_credits=Decimal("1"),
                checkpoint_path=root / "checkpoint.json",
            )

            synonyms = next(
                item for item in artifact["methods"] if item["methodId"] == "synonyms"
            )
            self.assertEqual(len(synonyms["documents"]), 20)
            self.assertEqual(synonyms["aggregate"]["fidelityFailureCount"], 20)
            self.assertEqual(
                synonyms["aggregate"]["detector"]["documentCount"],
                20,
            )
            first = synonyms["documents"][0]
            self.assertEqual(first["outputText"], " ")
            self.assertIn(
                "empty_output",
                {
                    issue["code"]
                    for issue in first["transformationOutcome"]["issues"]
                },
            )

    def test_forward_checkpoint_is_reused_after_backward_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExperimentFixture(root, bootstrap_replicates=10)
            config, corpus = fixture.load()
            checkpoint = root / "checkpoint.json"
            failing = FakeClient(fail_on_backward_de=True)
            with self.assertRaisesRegex(ProviderError, "scripted failure"):
                run_live(
                    config,
                    corpus,
                    client=failing,
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                )
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(len(state["calls"]), 2)
            self.assertEqual(state["calls"][-1]["stage"], "forward-de")
            in_flight_call_id = "doc-01:roundtrip-de:backward-de"
            self.assertEqual(state["inFlightCall"]["callId"], in_flight_call_id)

            with self.assertRaisesRegex(CheckpointError, "charge is unknown"):
                run_live(
                    config,
                    corpus,
                    client=ForbiddenClient(),
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                )

            resumed = FakeClient()
            artifact = run_live(
                config,
                corpus,
                client=resumed,
                max_provider_cost_credits=Decimal("1"),
                checkpoint_path=checkpoint,
                confirm_not_charged_call_id=in_flight_call_id,
            )
            self.assertEqual(len(resumed.calls), 118)
            self.assertIn("complete German text", resumed.calls[0]["prompt"])
            self.assertEqual(artifact["usage"]["callCount"], 120)
            final_state = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(len(final_state["calls"]), 120)
            self.assertIsNone(final_state["inFlightCall"])
            self.assertEqual(len({call["callId"] for call in final_state["calls"]}), 120)

    def test_crash_leaves_durable_in_flight_and_requires_not_charged_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExperimentFixture(root, bootstrap_replicates=10)
            config, corpus = fixture.load()
            checkpoint = root / "crash.json"
            crashing = CrashClient()
            with self.assertRaises(SimulatedProcessCrash):
                run_live(
                    config,
                    corpus,
                    client=crashing,
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                )
            self.assertEqual(crashing.calls, 1)
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(state["calls"], [])
            call_id = "doc-01:synonyms:synonyms"
            self.assertEqual(state["inFlightCall"]["callId"], call_id)
            self.assertIn("startedAtUnixMs", state["inFlightCall"])

            with self.assertRaisesRegex(CheckpointError, "charge is unknown"):
                run_live(
                    config,
                    corpus,
                    client=ForbiddenClient(),
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                )

            resumed = FakeClient()
            with self.assertRaises(CallLimitReached):
                run_live(
                    config,
                    corpus,
                    client=resumed,
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                    confirm_not_charged_call_id=call_id,
                    max_new_calls=1,
                )
            self.assertEqual(len(resumed.calls), 1)
            resolved = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertIsNone(resolved["inFlightCall"])
            self.assertEqual(len(resolved["calls"]), 1)

    def test_obtained_invalid_provider_response_is_saved_and_never_reissued(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExperimentFixture(root, bootstrap_replicates=10)
            config, corpus = fixture.load()
            checkpoint = root / "invalid-response.json"
            invalid = InvalidProviderResponseClient()
            with self.assertRaisesRegex(ResponseContractError, "preserved"):
                run_live(
                    config,
                    corpus,
                    client=invalid,
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                    max_new_calls=1,
                )
            self.assertEqual(invalid.calls, 1)
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertIsNone(state["inFlightCall"])
            self.assertEqual(len(state["calls"]), 1)
            record = state["calls"][0]
            self.assertEqual(record["recordStatus"], "provider_response_invalid")
            self.assertEqual(
                record["rawResponse"]["openrouter_metadata"]["numeric"],
                "0.123",
            )
            self.assertEqual(
                record["chargeAccounting"]["status"],
                "conservative_reserve_unknown",
            )

            with self.assertRaisesRegex(ResponseContractError, "preserved"):
                run_live(
                    config,
                    corpus,
                    client=ForbiddenClient(),
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                )

    def test_numeric_metadata_is_checkpointed_as_json_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExperimentFixture(root, bootstrap_replicates=10)
            config, corpus = fixture.load()
            checkpoint = root / "decimal-metadata.json"
            with self.assertRaises(CallLimitReached):
                run_live(
                    config,
                    corpus,
                    client=DecimalMetadataClient(),
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                    max_new_calls=1,
                )
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(
                state["calls"][0]["response"]["openrouterMetadata"]["numeric"][
                    "latencySeconds"
                ],
                "0.123",
            )

    def test_one_call_canary_pauses_and_route_mismatch_stops_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExperimentFixture(root, bootstrap_replicates=10)
            config, corpus = fixture.load()
            checkpoint = root / "canary.json"
            canary = FakeClient()
            with self.assertRaises(CallLimitReached) as paused:
                run_live(
                    config,
                    corpus,
                    client=canary,
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                    max_new_calls=1,
                )
            self.assertEqual(paused.exception.completed_calls, 1)
            self.assertEqual(len(canary.calls), 1)
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(state["calls"][0]["stage"], "synonyms")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExperimentFixture(root, bootstrap_replicates=10)
            config, corpus = fixture.load()
            checkpoint = root / "bad-route.json"
            bad = BadMetadataClient()
            with self.assertRaisesRegex(ResponseContractError, "strategy"):
                run_live(
                    config,
                    corpus,
                    client=bad,
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                    max_new_calls=1,
                )
            self.assertEqual(len(bad.calls), 1)
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(len(state["calls"]), 1)

        attempt_cases = (
            (BadAttemptProviderClient, "attempt provider"),
            (BadAttemptModelClient, "attempt model"),
            (BadAttemptStatusClient, "attempt status"),
        )
        for client_type, message in attempt_cases:
            with self.subTest(client=client_type.__name__):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    fixture = ExperimentFixture(root, bootstrap_replicates=10)
                    config, corpus = fixture.load()
                    checkpoint = root / "bad-attempt.json"
                    with self.assertRaisesRegex(ResponseContractError, message):
                        run_live(
                            config,
                            corpus,
                            client=client_type(),
                            max_provider_cost_credits=Decimal("1"),
                            checkpoint_path=checkpoint,
                            max_new_calls=1,
                        )
                    state = json.loads(checkpoint.read_text(encoding="utf-8"))
                    self.assertEqual(len(state["calls"]), 1)

    def test_current_compact_openrouter_metadata_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExperimentFixture(root, bootstrap_replicates=10)
            config, corpus = fixture.load()
            checkpoint = root / "current-metadata.json"
            client = CurrentMetadataClient()
            with self.assertRaises(CallLimitReached) as paused:
                run_live(
                    config,
                    corpus,
                    client=client,
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                    max_new_calls=1,
                )
            self.assertEqual(paused.exception.completed_calls, 1)
            self.assertEqual(len(client.calls), 1)

    def test_budget_is_required_before_calls_and_checkpoint_is_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExperimentFixture(root)
            config, corpus = fixture.load()
            checkpoint = root / "checkpoint.json"
            with self.assertRaisesRegex(BudgetError, "next-call reserve"):
                run_live(
                    config,
                    corpus,
                    client=ForbiddenClient(),
                    max_provider_cost_credits=Decimal("0.00001"),
                    checkpoint_path=checkpoint,
                )

            client = FakeClient(fail_on_backward_de=True)
            with self.assertRaises(ProviderError):
                run_live(
                    config,
                    corpus,
                    client=client,
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                )
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            semantic_drift = dict(state)
            semantic_drift["semanticAuditPlanSha256"] = "0" * 64
            checkpoint.write_bytes(canonical_json_bytes(semantic_drift))
            with self.assertRaisesRegex(CheckpointError, "semantic audit plan"):
                run_live(
                    config,
                    corpus,
                    client=ForbiddenClient(),
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                )

            state["keySha256"] = "0" * 64
            checkpoint.write_bytes(canonical_json_bytes(state))
            with self.assertRaisesRegex(CheckpointError, "watermark key"):
                run_live(
                    config,
                    corpus,
                    client=ForbiddenClient(),
                    max_provider_cost_credits=Decimal("1"),
                    checkpoint_path=checkpoint,
                )


class MetricTests(unittest.TestCase):
    def test_word_distance_and_protected_restoration_are_exact(self) -> None:
        metric = word_levenshtein_metrics("One small test", "One little test")
        self.assertEqual(metric["distance"], 1)
        self.assertEqual(metric["normalizationDenominator"], 3)
        self.assertAlmostEqual(metric["normalizedDistance"], 1 / 3)

        original = "Keep https://example.com/a and 15% exact."
        self.assertTrue(
            protected_restoration_metrics(original, original)["exactlyRestored"]
        )
        self.assertFalse(
            protected_restoration_metrics(
                original,
                "Keep https://example.com/b and 15% exact.",
            )["exactlyRestored"]
        )

    def test_bootstrap_requires_exactly_twenty_paired_unique_ids(self) -> None:
        row = {
            "detector": {"activePositions": 2, "hits": 1},
            "documentId": "",
            "fidelity": {"wordLevenshtein": {"normalizedDistance": 0.1}},
            "fingerprints": {"baselineActive": 2, "survivingActive": 1},
        }
        left = [{**row, "documentId": f"doc-{index:02d}"} for index in range(1, 21)]
        right = [{**item} for item in left]
        first = paired_bootstrap(left, right, replicates=20, seed=7)
        second = paired_bootstrap(left, right, replicates=20, seed=7)
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "exactly 20"):
            paired_bootstrap(left[:-1], right[:-1], replicates=20, seed=7)
        duplicate = list(left)
        duplicate[-1] = {**duplicate[-1], "documentId": "doc-01"}
        with self.assertRaisesRegex(ValueError, "unique"):
            paired_bootstrap(duplicate, right, replicates=20, seed=7)


class _StepClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        current = self.value
        self.value += 0.01
        return current


if __name__ == "__main__":
    unittest.main()
