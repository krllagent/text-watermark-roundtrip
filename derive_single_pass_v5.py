"""Derive the post-hoc single-pass v5 result from immutable v4 evidence.

This module performs no provider calls. It keeps only the already-paid v4
paraphrase-draft stage, proves that every v4 repair returned that draft byte for
byte, recomputes local metrics, and transfers the independent blind review only
because the reviewed candidate text is identical.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Mapping, Sequence

from corpus_contract import canonical_json_bytes
from run_experiment import (
    EXPECTED_DOCUMENT_IDS,
    aggregate_call_usage,
    fidelity_metrics,
    load_experiment_config,
    load_reviewed_corpus,
)
from run_verified_paraphrase import load_verified_paraphrase_config
from unmark import (
    build_v4_draft_request,
    canonicalize_placeholders,
    protect_tokens,
    request_messages,
    result_validation_issues,
    restore_tokens,
)
from watermark_toy import Document, score_corpus, score_text


CONFIG_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
METHOD_ID = "paraphrase-v5-single-pass"
CALL_GRAPH = ("paraphrase-draft",)
V4_METHOD_ID = "paraphrase-verified-v4"


@dataclass(frozen=True)
class SinglePassV5Config:
    path: Path
    root: Path
    raw: dict[str, object]
    sha256: str
    experiment_version: str
    verified_at: str
    methodology: str
    sources: tuple[dict[str, str], ...]
    method_id: str
    call_graph: tuple[str, ...]
    post_hoc_simplification: bool
    pre_registered_holdout: bool
    holdout_interpretation: str
    v4_config_path: Path
    v4_config_sha256: str
    v4_checkpoint_path: Path
    v4_checkpoint_sha256: str
    v4_result_path: Path
    v4_result_sha256: str
    audit_config_path: Path
    audit_config_sha256: str
    audit_result_path: Path
    audit_result_sha256: str
    independent_audit_commit: str
    parity_fixture_path: Path
    parity_fixture_sha256: str
    model: str
    provider_order: tuple[str, ...]
    expected_response_models: tuple[str, ...]
    expected_response_providers: tuple[str, ...]
    allow_fallbacks: bool
    data_collection: str
    zdr: bool
    require_parameters: bool
    reasoning_effort: str
    temperature: float
    max_tokens: int
    seed: int
    max_prompt_price: Decimal
    max_completion_price: Decimal
    development_document_ids: tuple[str, ...]
    holdout_document_ids: tuple[str, ...]
    minimum_mean_word_distance: float
    maximum_total_failures: int
    maximum_pipeline_failures: int


def load_single_pass_v5_config(
    path: str | Path,
    *,
    root: str | Path | None = None,
) -> SinglePassV5Config:
    """Load and strictly validate the frozen post-hoc derivation contract."""
    config_path = Path(path).resolve()
    root_path = (
        Path(root).resolve() if root is not None else Path(__file__).resolve().parent
    )
    raw_bytes = config_path.read_bytes()
    raw = _json_object(raw_bytes, "single-pass v5 config")
    if raw.get("schemaVersion") != CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported single-pass v5 config schemaVersion")
    _validate_evidence(raw, "single-pass v5 config")

    source_artifacts = _mapping(raw.get("sourceArtifacts"), "sourceArtifacts")
    bindings: dict[str, tuple[Path, str]] = {}
    for name in (
        "v4Config",
        "v4Checkpoint",
        "v4Result",
        "auditConfig",
        "auditResult",
    ):
        binding = _mapping(source_artifacts.get(name), f"sourceArtifacts.{name}")
        source_path = _safe_path(
            root_path,
            binding.get("path"),
            f"sourceArtifacts.{name}.path",
        )
        source_sha = _sha256(
            binding.get("sha256"),
            f"sourceArtifacts.{name}.sha256",
        )
        _require_sha(source_path, source_sha, f"sourceArtifacts.{name}")
        bindings[name] = (source_path, source_sha)
    independent_audit_commit = _sha1(
        source_artifacts.get("independentAuditCommit"),
        "sourceArtifacts.independentAuditCommit",
    )
    if independent_audit_commit != "032fd1691e895acef071a9c4e1b19fbefa447b93":
        raise ValueError("single-pass v5 must bind the independent v4 audit commit")

    v4_config = load_verified_paraphrase_config(
        bindings["v4Config"][0],
        root=root_path,
    )
    if v4_config.sha256 != bindings["v4Config"][1]:
        raise ValueError("v4 config binding differs after validation")

    derivation = _mapping(raw.get("derivation"), "derivation")
    expected_derivation = {
        "callGraph": ["paraphrase-draft"],
        "draftSource": "v4.documents[].transformationOutcome.rawDraftMaskedText",
        "exactDraftCallSource": ("v4Checkpoint.calls[stage=paraphrase-draft]"),
        "holdoutInterpretation": "descriptive_only",
        "methodId": METHOD_ID,
        "postHocSimplification": True,
        "preRegisteredHoldout": False,
        "proofRequirements": [
            "draft_call_output_equals_raw_draft_masked_text",
            "raw_draft_masked_text_equals_v4_raw_final_masked_text",
            "canonicalized_restored_draft_equals_v4_output_text",
            "blind_audit_source_artifact_equals_exact_v4_result",
        ],
        "providerCalls": 0,
    }
    if dict(derivation) != expected_derivation:
        raise ValueError("single-pass v5 derivation contract differs")

    provider = _mapping(raw.get("provider"), "provider")
    expected_provider = {
        "allowFallbacks": False,
        "dataCollection": "deny",
        "expectedResponseModels": list(v4_config.expected_response_models),
        "expectedResponseProviders": list(v4_config.expected_response_providers),
        "maxPriceUsdPerMillionTokens": {
            "completion": _decimal_text(v4_config.completion_price_usd_per_million),
            "prompt": _decimal_text(v4_config.prompt_price_usd_per_million),
        },
        "maxTokens": v4_config.stage_max_tokens["paraphrase-draft"],
        "model": v4_config.model,
        "providerOrder": list(v4_config.provider_order),
        "reasoningEffort": v4_config.reasoning_effort,
        "requireParameters": True,
        "seed": v4_config.seed,
        "temperature": 0,
        "transport": "openrouter",
        "zdr": True,
    }
    if dict(provider) != expected_provider:
        raise ValueError("single-pass v5 provider contract differs from v4 draft")

    parity = _mapping(raw.get("parityFixture"), "parityFixture")
    parity_path = _safe_path(root_path, parity.get("path"), "parityFixture.path")
    parity_sha = _sha256(parity.get("sha256"), "parityFixture.sha256")
    _require_sha(parity_path, parity_sha, "single-pass v5 parity fixture")
    _validate_parity_fixture(
        _json_object(parity_path.read_bytes(), "single-pass v5 parity fixture"),
        provider=provider,
    )

    analysis = _mapping(raw.get("analysis"), "analysis")
    development_ids = _string_tuple(
        analysis.get("developmentDocumentIds"),
        "analysis.developmentDocumentIds",
    )
    holdout_ids = _string_tuple(
        analysis.get("holdoutDocumentIds"),
        "analysis.holdoutDocumentIds",
    )
    if development_ids != ("doc-01",) or holdout_ids != EXPECTED_DOCUMENT_IDS[1:]:
        raise ValueError("single-pass v5 cohorts differ from the frozen v4 partition")
    if (
        analysis.get("holdoutInterpretation") != "descriptive_only"
        or analysis.get("latencyP95Method") != "nearest_rank_ceiling_0.95"
        or analysis.get("wordDistanceMetric") != "normalized_word_levenshtein"
        or analysis.get("costScope") != "paraphrase_draft_calls_only"
    ):
        raise ValueError("single-pass v5 analysis contract differs")

    decision = _mapping(raw.get("decisionPolicy"), "decisionPolicy")
    expected_decision = {
        "classification": "exploratory_post_hoc",
        "confirmatoryHoldoutClaimAllowed": False,
        "demoGate": {
            "maximumPipelineFailures": 0,
            "maximumTotalFailures": 1,
            "minimumMeanNormalizedWordDistance": 0.15,
            "pooledHoldoutDetectorStatus": "not_detected",
            "requireAllDraftFinalIdentityProofs": True,
            "requireAllProtectedTokensRestored": True,
            "requireTransferredBlindReviewCount": 20,
        },
        "onFail": "keep_demo_off_and_publish_negative_or_mixed_result",
        "onPass": "single_best_method_demo_may_use_exact_single_stage_contract",
        "publishCostAndLatencyRegardless": True,
        "publishEveryDerivedDocument": True,
    }
    if dict(decision) != expected_decision:
        raise ValueError("single-pass v5 decision policy differs")
    demo_gate = _mapping(decision.get("demoGate"), "decisionPolicy.demoGate")

    return SinglePassV5Config(
        path=config_path,
        root=root_path,
        raw=raw,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        experiment_version=_text(raw.get("experimentVersion"), "experimentVersion"),
        verified_at=_text(raw.get("verifiedAt"), "verifiedAt"),
        methodology=_text(raw.get("methodology"), "methodology"),
        sources=tuple(_sources(raw.get("sources"))),
        method_id=METHOD_ID,
        call_graph=CALL_GRAPH,
        post_hoc_simplification=True,
        pre_registered_holdout=False,
        holdout_interpretation="descriptive_only",
        v4_config_path=bindings["v4Config"][0],
        v4_config_sha256=bindings["v4Config"][1],
        v4_checkpoint_path=bindings["v4Checkpoint"][0],
        v4_checkpoint_sha256=bindings["v4Checkpoint"][1],
        v4_result_path=bindings["v4Result"][0],
        v4_result_sha256=bindings["v4Result"][1],
        audit_config_path=bindings["auditConfig"][0],
        audit_config_sha256=bindings["auditConfig"][1],
        audit_result_path=bindings["auditResult"][0],
        audit_result_sha256=bindings["auditResult"][1],
        independent_audit_commit=independent_audit_commit,
        parity_fixture_path=parity_path,
        parity_fixture_sha256=parity_sha,
        model=v4_config.model,
        provider_order=v4_config.provider_order,
        expected_response_models=v4_config.expected_response_models,
        expected_response_providers=v4_config.expected_response_providers,
        allow_fallbacks=False,
        data_collection="deny",
        zdr=True,
        require_parameters=True,
        reasoning_effort=v4_config.reasoning_effort,
        temperature=0.0,
        max_tokens=v4_config.stage_max_tokens["paraphrase-draft"],
        seed=v4_config.seed,
        max_prompt_price=v4_config.prompt_price_usd_per_million,
        max_completion_price=v4_config.completion_price_usd_per_million,
        development_document_ids=development_ids,
        holdout_document_ids=holdout_ids,
        minimum_mean_word_distance=float(
            demo_gate["minimumMeanNormalizedWordDistance"]
        ),
        maximum_total_failures=int(demo_gate["maximumTotalFailures"]),
        maximum_pipeline_failures=int(demo_gate["maximumPipelineFailures"]),
    )


def build_single_pass_v5_result(
    config: SinglePassV5Config,
    *,
    v4_result: Mapping[str, object] | None = None,
    v4_checkpoint: Mapping[str, object] | None = None,
    audit_result: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the complete derived artifact without making a provider call."""
    if v4_result is None:
        _require_sha(config.v4_result_path, config.v4_result_sha256, "v4 result")
    if v4_checkpoint is None:
        _require_sha(
            config.v4_checkpoint_path,
            config.v4_checkpoint_sha256,
            "v4 checkpoint",
        )
    if audit_result is None:
        _require_sha(
            config.audit_result_path,
            config.audit_result_sha256,
            "v4 audit result",
        )
    v4_config = load_verified_paraphrase_config(
        config.v4_config_path,
        root=config.root,
    )
    base_config = load_experiment_config(v4_config.base_config_path, root=config.root)
    corpus = load_reviewed_corpus(base_config)
    raw = (
        _json_object(config.v4_result_path.read_bytes(), "v4 result")
        if v4_result is None
        else dict(v4_result)
    )
    checkpoint = (
        _json_object(config.v4_checkpoint_path.read_bytes(), "v4 checkpoint")
        if v4_checkpoint is None
        else dict(v4_checkpoint)
    )
    audit = (
        _json_object(config.audit_result_path.read_bytes(), "v4 audit result")
        if audit_result is None
        else dict(audit_result)
    )
    _validate_source_artifact_bindings(config, raw, checkpoint, audit)

    methods = _mapping_list(raw.get("methods"), "v4 methods")
    if len(methods) != 1 or methods[0].get("methodId") != V4_METHOD_ID:
        raise ValueError("v4 result must contain exactly the verified v4 method")
    v4_documents = _mapping_list(methods[0].get("documents"), "v4 documents")
    by_document = _ordered_by_document(v4_documents, "v4 documents")

    calls = _mapping_list(checkpoint.get("calls"), "v4 checkpoint calls")
    if len(calls) != 60 or checkpoint.get("inFlightCall") is not None:
        raise ValueError("v4 checkpoint must contain 60 completed calls")
    draft_calls = [call for call in calls if call.get("stage") == "paraphrase-draft"]
    draft_calls_by_document = _ordered_by_document(draft_calls, "v4 draft calls")

    audit_reviews_by_document = _audit_reviews_by_document(audit)
    documents: list[dict[str, object]] = []
    for corpus_item in corpus.documents:
        document_id = corpus_item.document_id
        v4_row = by_document[document_id]
        if v4_row.get("sourceSha256") != corpus_item.sha256:
            raise ValueError(
                f"v4 source hash differs from frozen corpus: {document_id}"
            )
        outcome = _mapping(
            v4_row.get("transformationOutcome"),
            f"{document_id} transformationOutcome",
        )
        raw_draft = _text(
            outcome.get("rawDraftMaskedText"),
            f"{document_id} rawDraftMaskedText",
        )
        raw_final = _text(
            outcome.get("rawFinalMaskedText"),
            f"{document_id} rawFinalMaskedText",
        )
        if raw_draft != raw_final:
            raise ValueError(f"draft/final identity failed for {document_id}")

        draft_call = draft_calls_by_document[document_id]
        if draft_call.get("recordStatus") != "accepted_response":
            raise ValueError(f"draft call was not accepted for {document_id}")
        if draft_call.get("outputText") != raw_draft:
            raise ValueError(
                f"draft call output differs from v4 raw draft: {document_id}"
            )

        marked_input = _text(v4_row.get("markedInputText"), "v4 markedInputText")
        protected = protect_tokens(marked_input)
        expected_request = build_v4_draft_request(protected.masked)
        if draft_call.get("messages") != list(request_messages(expected_request)):
            raise ValueError(
                f"draft call messages differ from frozen request: {document_id}"
            )
        request_options = _mapping(draft_call.get("request"), "draft call request")
        if request_options != {"maxTokens": config.max_tokens, "model": config.model}:
            raise ValueError(
                f"draft call options differ from v5 contract: {document_id}"
            )

        canonical_draft = canonicalize_placeholders(raw_draft, protected.tokens)
        observable_issues = list(
            result_validation_issues(protected.masked, canonical_draft, None)
        )
        restored_output = restore_tokens(canonical_draft, protected.tokens)
        v4_output = _text(v4_row.get("outputText"), "v4 outputText")
        if restored_output != v4_output:
            raise ValueError(f"restored draft differs from v4 output: {document_id}")

        detector = score_text(
            restored_output,
            key=base_config.key,
            document_id=document_id,
            density_bps=base_config.density_bps,
            lexicon=corpus.lexicon,
            context_width=base_config.context_width,
            min_active_positions=base_config.min_active_positions,
        )
        fidelity = fidelity_metrics(marked_input, restored_output)
        protected_failure = (
            _mapping(fidelity.get("protectedTokens"), "protected tokens").get(
                "exactlyRestored"
            )
            is not True
        )
        pipeline_failure = bool(observable_issues)
        audit_mapping, review = audit_reviews_by_document[document_id]
        semantic_failure = review.get("semanticFidelityFailure") is True
        total_failure = protected_failure or pipeline_failure or semantic_failure

        documents.append(
            {
                "auditTransfer": {
                    "blindReview": dict(review),
                    "blindReviewSha256": _object_sha256(review),
                    "candidateOutputSha256": _text_sha256(restored_output),
                    "pairId": review.get("pairId"),
                    "sourceArtifactSha256": config.v4_result_sha256,
                    "transferredFromMethodId": audit_mapping.get("methodId"),
                    "transferBasis": "byte_identical_candidate_text",
                },
                "detector": detector.to_dict(),
                "documentId": document_id,
                "draftCall": dict(draft_call),
                "draftCallSha256": _object_sha256(draft_call),
                "failures": {
                    "pipelineFailure": pipeline_failure,
                    "pipelineIssues": observable_issues,
                    "protectedTokenFailure": protected_failure,
                    "semanticFidelityFailure": semantic_failure,
                    "totalFailure": total_failure,
                },
                "fidelity": fidelity,
                "genre": v4_row.get("genre"),
                "identityProof": {
                    "draftCallOutputEqualsRawDraftMaskedText": True,
                    "draftEqualsV4FinalMaskedText": True,
                    "rawDraftMaskedSha256": _text_sha256(raw_draft),
                    "restoredOutputEqualsV4OutputText": True,
                    "restoredOutputSha256": _text_sha256(restored_output),
                    "v4FinalMaskedSha256": _text_sha256(raw_final),
                    "v4OutputSha256": _text_sha256(v4_output),
                },
                "markedInputText": marked_input,
                "methodId": config.method_id,
                "outputText": restored_output,
                "rawDraftMaskedText": raw_draft,
                "sourceSha256": v4_row.get("sourceSha256"),
            }
        )

    development = _build_cohort(
        documents,
        config.development_document_ids,
        draft_calls_by_document,
        base_config=base_config,
        lexicon=corpus.lexicon,
    )
    holdout = _build_cohort(
        documents,
        config.holdout_document_ids,
        draft_calls_by_document,
        base_config=base_config,
        lexicon=corpus.lexicon,
    )
    all_documents = _build_cohort(
        documents,
        EXPECTED_DOCUMENT_IDS,
        draft_calls_by_document,
        base_config=base_config,
        lexicon=corpus.lexicon,
    )
    proof = {
        "allBlindAuditCandidatesByteIdentical": all(
            document["auditTransfer"]["candidateOutputSha256"]
            == document["identityProof"]["v4OutputSha256"]
            for document in documents
        ),
        "allDraftCallsAccepted": all(
            document["draftCall"]["recordStatus"] == "accepted_response"
            for document in documents
        ),
        "allDraftsByteIdenticalToV4FinalMaskedText": all(
            document["identityProof"]["draftEqualsV4FinalMaskedText"]
            for document in documents
        ),
        "allRestoredOutputsByteIdenticalToV4OutputText": all(
            document["identityProof"]["restoredOutputEqualsV4OutputText"]
            for document in documents
        ),
        "documentCount": len(documents),
        "transferredBlindReviewCount": len(audit_reviews_by_document),
    }
    gate = _build_decision_gate(config, holdout=holdout, proof=proof)
    source_bindings = {
        "auditConfigSha256": config.audit_config_sha256,
        "auditResultSha256": config.audit_result_sha256,
        "independentAuditCommit": config.independent_audit_commit,
        "v4CheckpointSha256": config.v4_checkpoint_sha256,
        "v4ConfigSha256": config.v4_config_sha256,
        "v4ResultSha256": config.v4_result_sha256,
    }
    artifact = {
        "cohorts": {
            "allDocumentsSecondary": all_documents,
            "development": development,
            "holdoutDescriptive": holdout,
        },
        "configSha256": config.sha256,
        "decisionGate": gate,
        "derivationProof": proof,
        "documentCount": len(documents),
        "documents": documents,
        "draftOnlyActualUsage": all_documents["draftOnlyActualUsage"],
        "experimentVersion": config.experiment_version,
        "holdoutInterpretation": config.holdout_interpretation,
        "methodId": config.method_id,
        "methodology": config.methodology,
        "postHocSimplification": True,
        "preRegisteredHoldout": False,
        "providerCallsMadeForDerivation": 0,
        "schemaVersion": RESULT_SCHEMA_VERSION,
        "sourceBindings": source_bindings,
        "sources": list(config.sources),
        "verifiedAt": config.verified_at,
    }
    canonical_json_bytes(artifact)
    return artifact


def _validate_source_artifact_bindings(
    config: SinglePassV5Config,
    raw: Mapping[str, object],
    checkpoint: Mapping[str, object],
    audit: Mapping[str, object],
) -> None:
    if raw.get("configSha256") != config.v4_config_sha256:
        raise ValueError("v4 result config binding mismatch")
    if checkpoint.get("configSha256") != config.v4_config_sha256:
        raise ValueError("v4 checkpoint config binding mismatch")
    if audit.get("sourceArtifactSha256") != config.v4_result_sha256:
        raise ValueError("audit source artifact is not the exact v4 result")
    if audit.get("auditConfigSha256") != config.audit_config_sha256:
        raise ValueError("audit config binding mismatch")


def _audit_reviews_by_document(
    audit: Mapping[str, object],
) -> dict[str, tuple[Mapping[str, object], Mapping[str, object]]]:
    raw_mapping = _mapping_list(audit.get("opaqueMapping"), "audit opaqueMapping")
    reviews = _mapping_list(audit.get("reviews"), "audit reviews")
    if len(raw_mapping) != 20 or len(reviews) != 20:
        raise ValueError("blind audit must contain exactly 20 mapped reviews")
    review_by_pair: dict[str, Mapping[str, object]] = {}
    for review in reviews:
        pair_id = _text(review.get("pairId"), "audit review pairId")
        if pair_id in review_by_pair:
            raise ValueError("blind audit reviews contain duplicate pair IDs")
        review_by_pair[pair_id] = review
    output: dict[str, tuple[Mapping[str, object], Mapping[str, object]]] = {}
    for mapping in raw_mapping:
        document_id = _text(mapping.get("documentId"), "audit mapping documentId")
        pair_id = _text(mapping.get("pairId"), "audit mapping pairId")
        if mapping.get("methodId") != V4_METHOD_ID:
            raise ValueError("blind audit mapping method differs from v4")
        if document_id in output or pair_id not in review_by_pair:
            raise ValueError("blind audit mapping is incomplete or duplicated")
        output[document_id] = (mapping, review_by_pair[pair_id])
    if tuple(sorted(output)) != tuple(sorted(EXPECTED_DOCUMENT_IDS)):
        raise ValueError("blind audit mapping does not cover all documents")
    return output


def _build_cohort(
    documents: Sequence[Mapping[str, object]],
    document_ids: Sequence[str],
    draft_calls_by_document: Mapping[str, Mapping[str, object]],
    *,
    base_config: object,
    lexicon: object,
) -> dict[str, object]:
    by_id = {
        _text(document.get("documentId"), "derived documentId"): document
        for document in documents
    }
    selected = [by_id[document_id] for document_id in document_ids]
    detector = score_corpus(
        tuple(
            Document(
                document_id=document_id,
                text=_text(by_id[document_id].get("outputText"), "derived outputText"),
            )
            for document_id in document_ids
        ),
        key=base_config.key,  # type: ignore[attr-defined]
        density_bps=base_config.density_bps,  # type: ignore[attr-defined]
        lexicon=lexicon,  # type: ignore[arg-type]
        context_width=base_config.context_width,  # type: ignore[attr-defined]
        min_active_positions=base_config.min_active_positions,  # type: ignore[attr-defined]
    ).to_dict()
    distances = [
        float(
            _mapping(
                _mapping(document.get("fidelity"), "derived fidelity").get(
                    "wordLevenshtein"
                ),
                "derived wordLevenshtein",
            )["normalizedDistance"]
        )
        for document in selected
    ]
    failures = [
        _mapping(document.get("failures"), "derived failures") for document in selected
    ]
    calls = [draft_calls_by_document[document_id] for document_id in document_ids]
    return {
        "detector": detector,
        "documentCount": len(selected),
        "documentIds": list(document_ids),
        "draftOnlyActualUsage": _draft_usage(calls),
        "meanNormalizedWordDistance": sum(distances) / len(distances),
        "pipelineFailureCount": sum(
            failure.get("pipelineFailure") is True for failure in failures
        ),
        "protectedTokenFailureCount": sum(
            failure.get("protectedTokenFailure") is True for failure in failures
        ),
        "semanticFidelityFailureCount": sum(
            failure.get("semanticFidelityFailure") is True for failure in failures
        ),
        "totalFailureCount": sum(
            failure.get("totalFailure") is True for failure in failures
        ),
    }


def _draft_usage(calls: Sequence[Mapping[str, object]]) -> dict[str, object]:
    usage = aggregate_call_usage(calls)
    cost = _decimal(usage.get("providerCostCredits"), "draft provider cost")
    latencies = sorted(float(call["latencyMs"]) for call in calls)
    p95_index = math.ceil(0.95 * len(latencies)) - 1
    return {
        **usage,
        "latencyMsMedian": statistics.median(latencies),
        "latencyMsP95Method": "nearest_rank_ceiling_0.95",
        "latencyMsP95NearestRank": latencies[p95_index],
        "providerCostCreditsPer1000Documents": _decimal_text(
            cost * Decimal(1000) / len(calls)
        ),
    }


def _build_decision_gate(
    config: SinglePassV5Config,
    *,
    holdout: Mapping[str, object],
    proof: Mapping[str, object],
) -> dict[str, object]:
    detector = _mapping(holdout.get("detector"), "holdout detector")
    checks = {
        "allDraftFinalIdentityProofs": {
            "observed": proof.get("allDraftsByteIdenticalToV4FinalMaskedText"),
            "passed": proof.get("allDraftsByteIdenticalToV4FinalMaskedText") is True,
            "required": True,
        },
        "allProtectedTokensRestored": {
            "observedFailureCount": holdout.get("protectedTokenFailureCount"),
            "passed": holdout.get("protectedTokenFailureCount") == 0,
            "requiredFailureCount": 0,
        },
        "maximumPipelineFailures": {
            "observed": holdout.get("pipelineFailureCount"),
            "passed": int(holdout["pipelineFailureCount"])
            <= config.maximum_pipeline_failures,
            "requiredMaximum": config.maximum_pipeline_failures,
        },
        "maximumTotalFailures": {
            "observed": holdout.get("totalFailureCount"),
            "passed": int(holdout["totalFailureCount"])
            <= config.maximum_total_failures,
            "requiredMaximum": config.maximum_total_failures,
        },
        "minimumMeanNormalizedWordDistance": {
            "observed": holdout.get("meanNormalizedWordDistance"),
            "passed": float(holdout["meanNormalizedWordDistance"])
            >= config.minimum_mean_word_distance,
            "requiredMinimum": config.minimum_mean_word_distance,
        },
        "pooledHoldoutDetectorStatus": {
            "observed": detector.get("status"),
            "passed": detector.get("status") == "not_detected",
            "required": "not_detected",
        },
        "transferredBlindReviewCount": {
            "observed": proof.get("transferredBlindReviewCount"),
            "passed": proof.get("transferredBlindReviewCount") == 20,
            "required": 20,
        },
    }
    passed = all(check["passed"] is True for check in checks.values())
    return {
        "checks": checks,
        "classification": "exploratory_post_hoc",
        "confirmatoryHoldoutClaimAllowed": False,
        "passed": passed,
        "status": (
            "pass_exploratory_post_hoc" if passed else "fail_exploratory_post_hoc"
        ),
    }


def _validate_parity_fixture(
    fixture: Mapping[str, object],
    *,
    provider: Mapping[str, object],
) -> None:
    if fixture.get("schemaVersion") != 1:
        raise ValueError("single-pass v5 parity fixture schemaVersion differs")
    _validate_evidence(fixture, "single-pass v5 parity fixture")
    sample = _mapping(fixture.get("sample"), "parity sample")
    source = _text(sample.get("sourceMaskedText"), "parity sourceMaskedText")
    request = build_v4_draft_request(source)
    expected = {
        "maxTokens": provider.get("maxTokens"),
        "messages": list(request.to_messages()),
        "model": provider.get("model"),
        "provider": {
            "allowFallbacks": provider.get("allowFallbacks"),
            "dataCollection": provider.get("dataCollection"),
            "maxPriceUsdPerMillionTokens": provider.get("maxPriceUsdPerMillionTokens"),
            "order": provider.get("providerOrder"),
            "requireParameters": provider.get("requireParameters"),
            "zdr": provider.get("zdr"),
        },
        "reasoningEffort": provider.get("reasoningEffort"),
        "responseFormat": None,
        "seed": provider.get("seed"),
        "stage": "paraphrase-draft",
        "temperature": provider.get("temperature"),
    }
    if fixture.get("request") != expected:
        raise ValueError("single-pass v5 parity request differs from frozen contract")


def _ordered_by_document(
    rows: Sequence[Mapping[str, object]],
    label: str,
) -> dict[str, Mapping[str, object]]:
    output: dict[str, Mapping[str, object]] = {}
    for row in rows:
        document_id = _text(row.get("documentId"), f"{label} documentId")
        if document_id in output:
            raise ValueError(f"{label} contains duplicate document IDs")
        output[document_id] = row
    if tuple(output) != EXPECTED_DOCUMENT_IDS:
        raise ValueError(f"{label} must be ordered doc-01 through doc-20")
    return output


def _object_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_object(content: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _mapping_list(value: object, label: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or any(
        not isinstance(item, Mapping) for item in value
    ):
        raise ValueError(f"{label} must be a list of objects")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{label} must be a nonempty string list")
    return tuple(value)


def _sources(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("sources must be a nonempty list")
    output: list[dict[str, str]] = []
    for raw in value:
        item = _mapping(raw, "source")
        if set(item) != {"title", "url"}:
            raise ValueError("source fields must be title and url")
        output.append(
            {
                "title": _text(item.get("title"), "source.title"),
                "url": _text(item.get("url"), "source.url"),
            }
        )
    return output


def _validate_evidence(value: Mapping[str, object], label: str) -> None:
    _text(value.get("verifiedAt"), f"{label}.verifiedAt")
    if len(_text(value.get("methodology"), f"{label}.methodology")) < 20:
        raise ValueError(f"{label}.methodology is too short")
    _sources(value.get("sources"))


def _safe_path(root: Path, value: object, label: str) -> Path:
    relative = Path(_text(value, label))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a safe relative path")
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} escapes the repository root")
    return resolved


def _require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} file is missing")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch")


def _sha256(value: object, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _sha1(value: object, label: str) -> str:
    text = _text(value, label)
    if len(text) != 40 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be a lowercase Git SHA-1")
    return text


def _decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a nonnegative decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{label} must be a decimal") from error
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"{label} must be a nonnegative finite decimal")
    return parsed


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            root / "fixtures" / "verified-paraphrase-config-v5-single-pass.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            root / "results" / "verified-paraphrase-derived-v5-single-pass.json"
        ),
    )
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_single_pass_v5_config(args.config)
    content = canonical_json_bytes(build_single_pass_v5_result(config))
    output = Path(args.output)
    if args.check:
        if not output.is_file() or output.read_bytes() != content:
            raise SystemExit("derived v5 artifact is missing or stale")
        print(
            json.dumps(
                {
                    "artifactSha256": hashlib.sha256(content).hexdigest(),
                    "output": str(output.resolve()),
                    "status": "fresh",
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    output.write_bytes(content)
    print(
        json.dumps(
            {
                "artifactSha256": hashlib.sha256(content).hexdigest(),
                "output": str(output.resolve()),
                "providerCalls": 0,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
