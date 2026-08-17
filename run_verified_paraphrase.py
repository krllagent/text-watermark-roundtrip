"""Run the frozen verified-paraphrase follow-up experiments.

Dry-run reads only frozen local evidence. Live mode follows the exact frozen
call graph on one pinned OpenRouter endpoint, checkpoints every paid response,
and never retries a request whose charge status is unknown.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Callable, Mapping, Protocol, Sequence

from corpus_contract import canonical_json_bytes
from run_experiment import (
    EXPECTED_DOCUMENT_IDS,
    _aggregate_method,
    aggregate_call_usage,
    compare_active_fingerprints,
    fidelity_metrics,
    load_experiment_config,
    load_reviewed_corpus,
)
from unmark import (
    ChatCompletion,
    CompletionUsage,
    OpenRouterClient,
    PlaceholderError,
    ProviderResponseError,
    SEMANTIC_AUDIT_CATEGORIES,
    SEMANTIC_AUDIT_DRAFT_QUOTE_MAX_CHARS,
    SEMANTIC_AUDIT_DRAFT_QUOTE_MIN_CHARS,
    SEMANTIC_AUDIT_MAX_CANONICAL_CHARS,
    SEMANTIC_AUDIT_MAX_CORRECTIONS,
    SEMANTIC_AUDIT_REQUIRED_CHANGE_MAX_CHARS,
    SEMANTIC_AUDIT_RESPONSE_FORMAT_NAME,
    SEMANTIC_AUDIT_SOURCE_NGRAM_WORDS,
    SemanticAuditContractError,
    StageRequest,
    V4_STAGE_PAYLOAD_FIELDS,
    V4_SYSTEM_INSTRUCTIONS,
    build_audit_guided_repair_prompt,
    build_fidelity_audit_prompt,
    build_fidelity_repair_prompt,
    build_paraphrase_prompt,
    build_semantic_audit_request,
    build_semantic_repair_request,
    build_v4_draft_request,
    canonicalize_placeholders,
    json_safe_value,
    parse_semantic_audit,
    protect_tokens,
    result_validation_issues,
    request_messages,
    request_utf8_size,
    restore_tokens,
    semantic_audit_repair_issues,
    semantic_audit_response_format,
)
from watermark_toy import Document, encode_text, score_corpus, score_text


CONFIG_SCHEMA_VERSION = 2
CHECKPOINT_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 2
V2_METHOD_ID = "paraphrase-verified"
V2_CALL_GRAPH = ("paraphrase-draft", "fidelity-repair")
V3_METHOD_ID = "paraphrase-verified-v3"
V3_CALL_GRAPH = (
    "paraphrase-draft",
    "fidelity-audit",
    "fidelity-repair",
)
V4_METHOD_ID = "paraphrase-verified-v4"
V4_CALL_GRAPH = (
    "paraphrase-draft",
    "semantic-audit",
    "fidelity-repair",
)


class VerifiedExperimentError(Exception):
    """Base class for expected follow-up failures."""


class VerifiedBudgetError(VerifiedExperimentError):
    """Raised before a paid call when the explicit budget is insufficient."""


class VerifiedCheckpointError(VerifiedExperimentError):
    """Raised when a checkpoint cannot be resumed safely."""


class VerifiedResponseContractError(VerifiedExperimentError):
    """Raised after preserving a paid response that violates the frozen route."""


class VerifiedCanaryGateError(VerifiedExperimentError):
    """Raised before holdout dispatch when the v4 development canary fails."""


class VerifiedCallLimitReached(VerifiedExperimentError):
    """Intentional pause after a requested number of new calls."""

    def __init__(self, *, completed_calls: int, new_calls: int) -> None:
        super().__init__(
            f"paused after {new_calls} new call(s); "
            f"{completed_calls} total call(s) checkpointed"
        )
        self.completed_calls = completed_calls
        self.new_calls = new_calls


class CompletionClient(Protocol):
    def complete(
        self,
        request: str | StageRequest,
        *,
        model: str,
        max_tokens: int | None = None,
        response_format: Mapping[str, object] | None = None,
    ) -> ChatCompletion: ...


@dataclass(frozen=True)
class VerifiedParaphraseConfig:
    path: Path
    root: Path
    raw: dict[str, object]
    sha256: str
    experiment_version: str
    verified_at: str
    methodology: str
    sources: tuple[dict[str, str], ...]
    base_config_path: Path
    base_config_sha256: str
    base_result_path: Path
    base_result_sha256: str
    document_count: int
    method_id: str
    call_graph: tuple[str, ...]
    always_run_audit: bool
    always_run_repair: bool
    model: str
    provider_order: tuple[str, ...]
    endpoint_snapshot_path: Path
    endpoint_snapshot_sha256: str
    expected_response_models: tuple[str, ...]
    expected_response_providers: tuple[str, ...]
    allow_fallbacks: bool
    data_collection: str
    zdr: bool
    require_parameters: bool
    reasoning_effort: str
    temperature: float
    max_tokens: int
    stage_max_tokens: Mapping[str, int]
    audit_max_corrections: int
    parity_fixture_path: Path | None
    parity_fixture_sha256: str | None
    seed: int
    timeout_seconds: float
    prompt_price_usd_per_million: Decimal
    completion_price_usd_per_million: Decimal
    prompt_token_overhead_reserve: int
    bootstrap_replicates: int
    bootstrap_seed: int
    development_document_ids: tuple[str, ...]
    holdout_document_ids: tuple[str, ...]
    development_canary_exact_call_count: int
    canary_min_final_normalized_word_distance: float
    canary_min_final_to_draft_word_distance_ratio: float
    article_demo_maximum_total_final_failures: int
    article_demo_maximum_pipeline_defects: int
    article_demo_minimum_mean_normalized_word_distance: float
    separate_final_audit: bool
    final_audit_model: str
    success_target: object
    retuning_after_results: bool


def load_verified_paraphrase_config(
    path: str | Path,
    *,
    root: str | Path | None = None,
) -> VerifiedParaphraseConfig:
    """Load and strictly validate the frozen v2 call graph and route."""
    config_path = Path(path).resolve()
    root_path = (
        Path(root).resolve() if root is not None else Path(__file__).resolve().parent
    )
    raw_bytes = config_path.read_bytes()
    raw = _json_object(raw_bytes, "verified paraphrase config")
    if raw.get("schemaVersion") != CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported verified paraphrase config schemaVersion")
    _validate_evidence(raw, "verified paraphrase config")

    base = _mapping(raw.get("baseExperiment"), "baseExperiment")
    base_config_path = _safe_path(root_path, base.get("configPath"), "base config")
    base_config_sha = _sha256(base.get("configSha256"), "base config SHA-256")
    _require_sha(base_config_path, base_config_sha, "base config")
    base_result_path = _safe_path(root_path, base.get("resultPath"), "base result")
    base_result_sha = _sha256(base.get("resultSha256"), "base result SHA-256")
    _require_sha(base_result_path, base_result_sha, "base result")
    document_count = _positive_int(base.get("documentCount"), "documentCount")
    if document_count != 20:
        raise ValueError("base experiment must contain exactly 20 documents")
    if base.get("sourceSelection") != "baseline.documents[].markedText":
        raise ValueError("base source selection must be the frozen marked baseline")

    transform = _mapping(raw.get("transform"), "transform")
    method_id = _text(transform.get("id"), "transform.id")
    call_graph = _string_tuple(transform.get("callGraph"), "transform.callGraph")
    always_run_audit = transform.get("alwaysRunAudit") is True
    always_run_repair = transform.get("alwaysRunRepair")
    is_v2 = (
        method_id == V2_METHOD_ID
        and transform.get("method") == V2_METHOD_ID
        and call_graph == V2_CALL_GRAPH
        and transform.get("repairPolicy") == "always_once"
        and always_run_repair is True
        and transform.get("repairGrounding") == ["masked_source", "draft"]
        and not always_run_audit
    )
    is_v3 = (
        method_id == V3_METHOD_ID
        and transform.get("method") == V3_METHOD_ID
        and call_graph == V3_CALL_GRAPH
        and transform.get("auditPolicy") == "always_once"
        and transform.get("repairPolicy") == "always_once"
        and always_run_audit
        and always_run_repair is True
        and transform.get("auditGrounding") == ["masked_source", "draft"]
        and transform.get("repairGrounding") == ["draft", "fidelity_audit"]
        and transform.get("finalRepairSeesSource") is False
    )
    is_v4 = (
        method_id == V4_METHOD_ID
        and transform.get("method") == V4_METHOD_ID
        and call_graph == V4_CALL_GRAPH
        and transform.get("auditPolicy") == "always_once"
        and transform.get("repairPolicy") == "always_once"
        and always_run_audit
        and always_run_repair is True
        and transform.get("auditGrounding") == ["masked_source", "draft"]
        and transform.get("repairGrounding")
        == ["draft", "validated_canonical_semantic_audit"]
        and transform.get("finalRepairReceivesRawSource") is False
        and transform.get("finalRepairReceivesRawAudit") is False
        and transform.get("invalidAuditRepairInput") == "canonical_empty_corrections"
        and transform.get("postRepairValidation")
        == {
            "eachAcceptedDraftQuoteMustBeChangedOrAbsent": True,
            "failureCode": "semantic_audit_correction_unapplied",
        }
    )
    if not (is_v2 or is_v3 or is_v4):
        raise ValueError("verified paraphrase must freeze a supported exact call graph")
    audit_max_corrections = 0
    parity_fixture_path: Path | None = None
    parity_fixture_sha256: str | None = None
    if is_v4:
        audit_contract = _mapping(
            transform.get("auditContract"), "transform.auditContract"
        )
        expected_audit_contract = {
            "categories": list(SEMANTIC_AUDIT_CATEGORIES),
            "draftQuoteMaxChars": SEMANTIC_AUDIT_DRAFT_QUOTE_MAX_CHARS,
            "draftQuoteMinChars": SEMANTIC_AUDIT_DRAFT_QUOTE_MIN_CHARS,
            "maxCanonicalChars": SEMANTIC_AUDIT_MAX_CANONICAL_CHARS,
            "maxCorrections": SEMANTIC_AUDIT_MAX_CORRECTIONS,
            "requiredChangeMaxChars": SEMANTIC_AUDIT_REQUIRED_CHANGE_MAX_CHARS,
            "responseFormatName": SEMANTIC_AUDIT_RESPONSE_FORMAT_NAME,
            "sourceNgramWords": SEMANTIC_AUDIT_SOURCE_NGRAM_WORDS,
            "strictJsonSchema": True,
        }
        if dict(audit_contract) != expected_audit_contract:
            raise ValueError("v4 semantic audit contract differs from frozen code")
        audit_max_corrections = SEMANTIC_AUDIT_MAX_CORRECTIONS
        request_boundary = _mapping(
            transform.get("requestBoundary"), "transform.requestBoundary"
        )
        expected_request_boundary = {
            "instructionRole": "system",
            "messageCount": 2,
            "stagePayloadFields": {
                stage: list(fields) for stage, fields in V4_STAGE_PAYLOAD_FIELDS.items()
            },
            "systemInstructionSha256": {
                stage: hashlib.sha256(instruction.encode("utf-8")).hexdigest()
                for stage, instruction in V4_SYSTEM_INSTRUCTIONS.items()
            },
            "untrustedPayloadRole": "user",
            "userPayloadEncoding": "canonical_json_utf8",
        }
        boundary_without_parity = {
            key: value
            for key, value in request_boundary.items()
            if key not in {"parityFixturePath", "parityFixtureSha256"}
        }
        if boundary_without_parity != expected_request_boundary:
            raise ValueError("v4 request message boundary differs from frozen code")
        parity_fixture_path = _safe_path(
            root_path,
            request_boundary.get("parityFixturePath"),
            "transform.requestBoundary.parityFixturePath",
        )
        parity_fixture_sha256 = _sha256(
            request_boundary.get("parityFixtureSha256"),
            "transform.requestBoundary.parityFixtureSha256",
        )
        _require_sha(
            parity_fixture_path,
            parity_fixture_sha256,
            "v4 request parity fixture",
        )
        parity_fixture = _json_object(
            parity_fixture_path.read_bytes(), "v4 request parity fixture"
        )
        if parity_fixture.get("schemaVersion") != 1:
            raise ValueError("v4 request parity fixture schemaVersion differs")
        _validate_evidence(parity_fixture, "v4 request parity fixture")

    if is_v3 or is_v4:
        pilot = _mapping(raw.get("priorPilot"), "priorPilot")
        pilot_config_path = _safe_path(
            root_path, pilot.get("configPath"), "prior pilot config"
        )
        pilot_config_sha = _sha256(
            pilot.get("configSha256"), "prior pilot config SHA-256"
        )
        _require_sha(pilot_config_path, pilot_config_sha, "prior pilot config")
        pilot_checkpoint_path = _safe_path(
            root_path, pilot.get("checkpointPath"), "prior pilot checkpoint"
        )
        pilot_checkpoint_sha = _sha256(
            pilot.get("checkpointSha256"), "prior pilot checkpoint SHA-256"
        )
        _require_sha(
            pilot_checkpoint_path,
            pilot_checkpoint_sha,
            "prior pilot checkpoint",
        )
        if is_v3:
            if (
                pilot.get("completedCalls") != 2
                or pilot.get("decision")
                != "abort_before_full_matrix_source_copy_collapse"
            ):
                raise ValueError("v3 must bind the exact aborted v2 pilot")
        elif (
            pilot.get("completedCalls") != 3
            or pilot.get("decision") != "abort_source_copy_via_audit"
        ):
            raise ValueError("v4 must bind the exact aborted v3 pilot")

    provider = _mapping(raw.get("provider"), "provider")
    if provider.get("transport") != "openrouter":
        raise ValueError("verified paraphrase transport must be OpenRouter")
    model = _text(provider.get("model"), "provider.model")
    provider_order = _string_tuple(
        provider.get("providerOrder"), "provider.providerOrder"
    )
    expected_models = _string_tuple(
        provider.get("expectedResponseModels"), "provider.expectedResponseModels"
    )
    expected_providers = _string_tuple(
        provider.get("expectedResponseProviders"),
        "provider.expectedResponseProviders",
    )
    allow_fallbacks = provider.get("allowFallbacks")
    data_collection = provider.get("dataCollection")
    zdr = provider.get("zdr")
    require_parameters = provider.get("requireParameters")
    if allow_fallbacks is not False:
        raise ValueError("verified paraphrase fallbacks must be disabled")
    if data_collection != "deny" or zdr is not True:
        raise ValueError(
            "verified paraphrase must deny data collection and require ZDR"
        )
    if require_parameters is not True:
        raise ValueError("verified paraphrase must require endpoint parameters")
    reasoning_effort = _text(
        provider.get("reasoningEffort"), "provider.reasoningEffort"
    )
    if reasoning_effort != "none":
        raise ValueError("verified paraphrase reasoning effort must be none")
    temperature_value = provider.get("temperature")
    if temperature_value != 0 or isinstance(temperature_value, bool):
        raise ValueError("verified paraphrase temperature must be zero")
    max_tokens = _positive_int(provider.get("maxTokens"), "provider.maxTokens")
    if is_v4:
        raw_stage_max_tokens = _mapping(
            provider.get("stageMaxTokens"), "provider.stageMaxTokens"
        )
        if set(raw_stage_max_tokens) != set(V4_CALL_GRAPH):
            raise ValueError("v4 stageMaxTokens must match the exact call graph")
        stage_max_tokens = {
            stage: _positive_int(
                raw_stage_max_tokens.get(stage), f"provider.stageMaxTokens.{stage}"
            )
            for stage in V4_CALL_GRAPH
        }
        if any(value > max_tokens for value in stage_max_tokens.values()):
            raise ValueError("v4 stage output cap exceeds provider.maxTokens")
        if not (
            stage_max_tokens["semantic-audit"] < stage_max_tokens["paraphrase-draft"]
            and stage_max_tokens["semantic-audit"] < stage_max_tokens["fidelity-repair"]
        ):
            raise ValueError("v4 semantic-audit output cap must be the smallest")
    else:
        if "stageMaxTokens" in provider:
            raise ValueError("stageMaxTokens is only valid for v4")
        stage_max_tokens = {stage: max_tokens for stage in call_graph}
    seed = _nonnegative_int(provider.get("seed"), "provider.seed")
    timeout_seconds = _positive_number(
        provider.get("timeoutSeconds"), "provider.timeoutSeconds"
    )
    prices = _mapping(
        provider.get("pricingUsdPerMillionTokens"),
        "provider.pricingUsdPerMillionTokens",
    )
    prompt_price = _decimal(prices.get("prompt"), "provider prompt price")
    completion_price = _decimal(prices.get("completion"), "provider completion price")
    max_prices = _mapping(
        provider.get("maxPriceUsdPerMillionTokens"),
        "provider.maxPriceUsdPerMillionTokens",
    )
    if (
        _decimal(max_prices.get("prompt"), "provider maximum prompt price")
        != prompt_price
        or _decimal(max_prices.get("completion"), "provider maximum completion price")
        != completion_price
    ):
        raise ValueError("provider price ceiling must equal the frozen endpoint price")

    endpoint_path = _safe_path(
        root_path,
        provider.get("endpointSnapshotPath"),
        "provider endpoint snapshot",
    )
    endpoint_sha = _sha256(
        provider.get("endpointSnapshotSha256"),
        "provider endpoint snapshot SHA-256",
    )
    _require_sha(endpoint_path, endpoint_sha, "provider endpoint snapshot")
    endpoint_snapshot = _json_object(
        endpoint_path.read_bytes(), "provider endpoint snapshot"
    )
    _validate_evidence(endpoint_snapshot, "provider endpoint snapshot")
    endpoint = _mapping(endpoint_snapshot.get("endpoint"), "provider endpoint")
    endpoint_prices = _mapping(
        endpoint.get("pricingUsdPerToken"), "provider endpoint prices"
    )
    if endpoint_snapshot.get("requestedModelId") != model:
        raise ValueError("provider endpoint model binding mismatch")
    if endpoint.get("tag") not in provider_order:
        raise ValueError("provider endpoint tag binding mismatch")
    if endpoint.get("providerName") not in expected_providers:
        raise ValueError("provider endpoint name binding mismatch")
    if (
        _decimal(endpoint_prices.get("prompt"), "endpoint prompt price")
        * Decimal(1_000_000)
        != prompt_price
        or _decimal(endpoint_prices.get("completion"), "endpoint completion price")
        * Decimal(1_000_000)
        != completion_price
    ):
        raise ValueError("provider endpoint price binding mismatch")
    if (
        _positive_int(endpoint.get("maxCompletionTokens"), "endpoint maximum")
        < max_tokens
    ):
        raise ValueError("provider endpoint maximum is below configured maxTokens")

    billing = _mapping(raw.get("billing"), "billing")
    if (
        billing.get("creditBaseCurrency") != "USD"
        or billing.get("creditUsdBaseUnit") != "1"
        or billing.get("inferencePricingMarkupPercent") != 0
    ):
        raise ValueError("verified paraphrase billing premise is invalid")
    overhead = _positive_int(
        billing.get("promptTokenOverheadReserve"),
        "billing.promptTokenOverheadReserve",
    )

    analysis = _mapping(raw.get("analysis"), "analysis")
    if analysis.get("documentCount") != document_count:
        raise ValueError("analysis document count mismatch")
    if analysis.get("qualityMetric") != "normalized_word_levenshtein":
        raise ValueError("analysis quality metric is not frozen")
    if analysis.get("resamplingUnit") != "document_id":
        raise ValueError("analysis resampling unit is not frozen")
    if is_v4:
        development_document_ids = _string_tuple(
            analysis.get("developmentDocumentIds"),
            "analysis.developmentDocumentIds",
        )
        holdout_document_ids = _string_tuple(
            analysis.get("holdoutDocumentIds"),
            "analysis.holdoutDocumentIds",
        )
        if development_document_ids != ("doc-01",):
            raise ValueError("v4 development cohort must be exactly doc-01")
        if holdout_document_ids != EXPECTED_DOCUMENT_IDS[1:]:
            raise ValueError("v4 holdout cohort must be exactly doc-02 through doc-20")
        if set(development_document_ids) & set(holdout_document_ids):
            raise ValueError("v4 development and holdout cohorts overlap")
        if development_document_ids + holdout_document_ids != EXPECTED_DOCUMENT_IDS:
            raise ValueError("v4 cohorts do not partition the frozen corpus")
        if (
            analysis.get("primaryDocumentCount") != 19
            or analysis.get("primaryScoringUnit") != "holdout_pooled_corpus"
            or analysis.get("secondaryScoringUnit") != "all_20_documents"
            or analysis.get("headlineFidelityCohort") != "holdout"
            or analysis.get("finalAuditScope")
            != "all_20_documents_with_holdout_headline"
        ):
            raise ValueError("v4 primary holdout analysis contract differs")
        canary_gate = _mapping(
            raw.get("developmentCanaryGate"), "developmentCanaryGate"
        )
        expected_canary_gate = {
            "developmentOnlyNotProductSuccessTarget": True,
            "documentId": "doc-01",
            "exactCallCount": 3,
            "minFinalNormalizedWordDistance": 0.15,
            "minFinalToDraftWordDistanceRatio": 0.60,
            "onFail": "abort_v4_before_holdout",
            "onPass": "resume_exact_same_v4_without_changes",
            "requireNoPipelineIssues": True,
            "requireRouteFinishAndUsageContracts": True,
            "requireSemanticAuditValidationStatus": "accepted",
        }
        if dict(canary_gate) != expected_canary_gate:
            raise ValueError("v4 development canary gate differs from frozen contract")
        development_canary_exact_call_count = 3
        canary_min_final_normalized_word_distance = 0.15
        canary_min_final_to_draft_word_distance_ratio = 0.60
    else:
        if analysis.get("primaryScoringUnit") != "pooled_corpus":
            raise ValueError("analysis scoring unit must be pooled_corpus")
        development_document_ids = ()
        holdout_document_ids = EXPECTED_DOCUMENT_IDS
        development_canary_exact_call_count = 0
        canary_min_final_normalized_word_distance = 0.0
        canary_min_final_to_draft_word_distance_ratio = 0.0
    bootstrap_replicates = _positive_int(
        analysis.get("bootstrapReplicates"), "analysis.bootstrapReplicates"
    )
    bootstrap_seed = _nonnegative_int(
        analysis.get("bootstrapSeed"), "analysis.bootstrapSeed"
    )

    final_audit = _mapping(raw.get("finalAudit"), "finalAudit")
    separate_final_audit = final_audit.get("separateFromRepair") is True
    if (
        final_audit.get("required") is not True
        or not separate_final_audit
        or final_audit.get("judgeDoesNotControlRepair") is not True
        or final_audit.get("freezeAfterTransformResult") is not True
        or final_audit.get("structuredPairCount") != 20
    ):
        raise ValueError("final semantic audit must be independent and cover 20 pairs")
    final_audit_model = _text(final_audit.get("model"), "finalAudit.model")
    final_audit_provider_order = _string_tuple(
        final_audit.get("providerOrder"), "finalAudit.providerOrder"
    )
    final_audit_endpoint_path = _safe_path(
        root_path,
        final_audit.get("endpointSnapshotPath"),
        "finalAudit.endpointSnapshotPath",
    )
    final_audit_endpoint_sha = _sha256(
        final_audit.get("endpointSnapshotSha256"),
        "finalAudit.endpointSnapshotSha256",
    )
    _require_sha(
        final_audit_endpoint_path,
        final_audit_endpoint_sha,
        "final audit endpoint snapshot",
    )
    final_audit_endpoint_snapshot = _json_object(
        final_audit_endpoint_path.read_bytes(), "final audit endpoint snapshot"
    )
    final_audit_endpoint = _mapping(
        final_audit_endpoint_snapshot.get("endpoint"), "final audit endpoint"
    )
    if final_audit_endpoint_snapshot.get("requestedModelId") != final_audit_model:
        raise ValueError("final audit endpoint model binding mismatch")
    if final_audit_endpoint.get("tag") not in final_audit_provider_order:
        raise ValueError("final audit endpoint provider binding mismatch")
    if is_v4:
        final_audit_plan_path = _safe_path(
            root_path, final_audit.get("planPath"), "finalAudit.planPath"
        )
        final_audit_plan_sha = _sha256(
            final_audit.get("planSha256"), "finalAudit.planSha256"
        )
        _require_sha(final_audit_plan_path, final_audit_plan_sha, "final audit plan")

    decision = _mapping(raw.get("decisionPolicy"), "decisionPolicy")
    success_target = decision.get("successTarget")
    retuning = decision.get("retuningAfterResults")
    if success_target is not None or retuning is not False:
        raise ValueError("v2 cannot freeze a success target or retune after results")
    if decision.get("publishAllOutputs") is not True:
        raise ValueError("v2 must publish every output")
    if is_v4:
        article_demo_gate = _mapping(
            decision.get("articleDemoGate"), "decisionPolicy.articleDemoGate"
        )
        expected_article_demo_gate = {
            "cohort": "holdout",
            "costAndLatency": "always_publish_without_pass_threshold",
            "documentCount": 19,
            "enableDemoOnlyIf": {
                "maximumPipelineDefects": 0,
                "maximumTotalFinalFailures": 1,
                "minimumMeanNormalizedWordDistance": 0.15,
                "pooledDetectorStatus": "not_detected",
            },
            "onFail": (
                "publish_advanced_result_as_negative_or_mixed_and_keep_demo_off"
            ),
            "onPass": "publish_article_with_best_only_demo_enabled",
        }
        if dict(article_demo_gate) != expected_article_demo_gate:
            raise ValueError(
                "v4 article/demo decision gate differs from frozen contract"
            )
        article_demo_maximum_total_final_failures = 1
        article_demo_maximum_pipeline_defects = 0
        article_demo_minimum_mean_normalized_word_distance = 0.15
    else:
        if "articleDemoGate" in decision:
            raise ValueError("articleDemoGate is only valid for v4")
        article_demo_maximum_total_final_failures = 0
        article_demo_maximum_pipeline_defects = 0
        article_demo_minimum_mean_normalized_word_distance = 0.0

    return VerifiedParaphraseConfig(
        path=config_path,
        root=root_path,
        raw=raw,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        experiment_version=_text(raw.get("experimentVersion"), "experimentVersion"),
        verified_at=_text(raw.get("verifiedAt"), "verifiedAt"),
        methodology=_text(raw.get("methodology"), "methodology"),
        sources=tuple(_sources(raw.get("sources"))),
        base_config_path=base_config_path,
        base_config_sha256=base_config_sha,
        base_result_path=base_result_path,
        base_result_sha256=base_result_sha,
        document_count=document_count,
        method_id=method_id,
        call_graph=call_graph,
        always_run_audit=always_run_audit,
        always_run_repair=always_run_repair,
        model=model,
        provider_order=provider_order,
        endpoint_snapshot_path=endpoint_path,
        endpoint_snapshot_sha256=endpoint_sha,
        expected_response_models=expected_models,
        expected_response_providers=expected_providers,
        allow_fallbacks=allow_fallbacks,
        data_collection=data_collection,
        zdr=zdr,
        require_parameters=require_parameters,
        reasoning_effort=reasoning_effort,
        temperature=0.0,
        max_tokens=max_tokens,
        stage_max_tokens=stage_max_tokens,
        audit_max_corrections=audit_max_corrections,
        parity_fixture_path=parity_fixture_path,
        parity_fixture_sha256=parity_fixture_sha256,
        seed=seed,
        timeout_seconds=timeout_seconds,
        prompt_price_usd_per_million=prompt_price,
        completion_price_usd_per_million=completion_price,
        prompt_token_overhead_reserve=overhead,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        development_document_ids=development_document_ids,
        holdout_document_ids=holdout_document_ids,
        development_canary_exact_call_count=development_canary_exact_call_count,
        canary_min_final_normalized_word_distance=(
            canary_min_final_normalized_word_distance
        ),
        canary_min_final_to_draft_word_distance_ratio=(
            canary_min_final_to_draft_word_distance_ratio
        ),
        article_demo_maximum_total_final_failures=(
            article_demo_maximum_total_final_failures
        ),
        article_demo_maximum_pipeline_defects=(article_demo_maximum_pipeline_defects),
        article_demo_minimum_mean_normalized_word_distance=(
            article_demo_minimum_mean_normalized_word_distance
        ),
        separate_final_audit=separate_final_audit,
        final_audit_model=final_audit_model,
        success_target=success_target,
        retuning_after_results=retuning,
    )


def expected_verified_call_ids(
    config: VerifiedParaphraseConfig,
) -> tuple[str, ...]:
    """Return the exact document-major call matrix for the frozen version."""
    return tuple(
        f"{document_id}:{config.method_id}:{stage}"
        for document_id in EXPECTED_DOCUMENT_IDS
        for stage in config.call_graph
    )


def build_verified_dry_run(
    config: VerifiedParaphraseConfig,
) -> dict[str, object]:
    """Build a conservative local planning estimate without reading credentials."""
    marked = _load_frozen_marked_documents(config)
    prompt_estimate = 0
    draft_completion_estimate = config.stage_max_tokens["paraphrase-draft"] * 4
    for document_id in EXPECTED_DOCUMENT_IDS:
        protected = protect_tokens(marked[document_id])
        draft_prompt: str | StageRequest = (
            build_v4_draft_request(protected.masked)
            if config.method_id == V4_METHOD_ID
            else build_paraphrase_prompt(protected.masked)
        )
        prompt_estimate += request_utf8_size(draft_prompt)
        estimated_draft = "x" * draft_completion_estimate
        if config.method_id == V4_METHOD_ID:
            audit_prompt = build_semantic_audit_request(
                protected.masked,
                estimated_draft,
            )
            repair_prompt = build_semantic_repair_request(
                estimated_draft,
                '{"corrections":[]}',
            )
            prompt_estimate += request_utf8_size(audit_prompt)
            prompt_estimate += request_utf8_size(repair_prompt)
            prompt_estimate += SEMANTIC_AUDIT_MAX_CANONICAL_CHARS
        elif config.always_run_audit:
            audit_prompt = build_fidelity_audit_prompt(
                protected.masked,
                estimated_draft,
            )
            repair_prompt = build_audit_guided_repair_prompt(
                estimated_draft,
                "x" * draft_completion_estimate,
            )
            prompt_estimate += request_utf8_size(audit_prompt)
            prompt_estimate += request_utf8_size(repair_prompt)
        else:
            repair_prompt = build_fidelity_repair_prompt(
                protected.masked,
                estimated_draft,
            )
            prompt_estimate += request_utf8_size(repair_prompt)
    call_count = config.document_count * len(config.call_graph)
    prompt_estimate += call_count * config.prompt_token_overhead_reserve
    completion_estimate = config.document_count * sum(
        config.stage_max_tokens[stage] for stage in config.call_graph
    )
    return {
        "callCount": call_count,
        "callsByStage": {
            stage: config.document_count for stage in sorted(config.call_graph)
        },
        "configSha256": config.sha256,
        "documentCount": config.document_count,
        "developmentDocumentIds": list(config.development_document_ids),
        "holdoutDocumentIds": list(config.holdout_document_ids),
        "experimentVersion": config.experiment_version,
        "methodology": (
            "Local-only conservative planning estimate. UTF-8 prompt bytes stand in "
            "for prompt tokens; the unknown repair input reserves four bytes for every "
            "allowed draft token. Every call also reserves the configured prompt "
            "overhead and the full completion maximum."
        ),
        "routingCostPlanningEstimateUsd": _decimal_text(
            _routing_cost(prompt_estimate, completion_estimate, config)
        ),
        "schemaVersion": ARTIFACT_SCHEMA_VERSION,
        "tokenEstimate": {
            "completionTokensAtConfiguredMaximum": completion_estimate,
            "promptTokensPlanningEstimate": prompt_estimate,
            "totalTokensPlanningEstimate": prompt_estimate + completion_estimate,
        },
        "verifiedAt": config.verified_at,
    }


def run_verified_live(
    config: VerifiedParaphraseConfig,
    *,
    client: CompletionClient,
    max_provider_cost_credits: Decimal,
    checkpoint_path: str | Path,
    max_new_calls: int | None = None,
    confirm_not_charged_call_id: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Run or resume the frozen matrix and return its raw artifact."""
    budget = _required_budget(max_provider_cost_credits)
    if max_new_calls is not None and (
        not isinstance(max_new_calls, int)
        or isinstance(max_new_calls, bool)
        or max_new_calls < 0
    ):
        raise ValueError("max_new_calls must be a nonnegative integer or null")
    build_verified_dry_run(config)

    base_config = load_experiment_config(config.base_config_path, root=config.root)
    corpus = load_reviewed_corpus(base_config)
    frozen_result = _load_base_result(config)
    baseline_documents = _baseline_documents(frozen_result)
    manager = _VerifiedCheckpointManager(
        config=config,
        client=client,
        checkpoint_path=Path(checkpoint_path),
        expected_call_ids=expected_verified_call_ids(config),
        budget=budget,
        max_new_calls=max_new_calls,
        confirm_not_charged_call_id=confirm_not_charged_call_id,
        clock=clock,
    )
    canary_gate_report: dict[str, object] | None = None
    if config.method_id == V4_METHOD_ID:
        completed = len(manager.calls)
        required = config.development_canary_exact_call_count
        if completed < required:
            remaining = required - completed
            if max_new_calls != remaining:
                raise VerifiedCanaryGateError(
                    "v4 must checkpoint exactly the remaining development canary "
                    f"calls before any holdout dispatch; use max_new_calls={remaining}"
                )
        else:
            canary_gate_report = build_v4_canary_gate(
                config,
                checkpoint_path=checkpoint_path,
            )
            if canary_gate_report.get("status") != "go":
                raise VerifiedCanaryGateError(
                    "v4 development canary failed; holdout dispatch is disabled"
                )

    rows: list[dict[str, object]] = []
    scores = []
    for document in corpus.documents:
        baseline = baseline_documents[document.document_id]
        marked_text = _text(baseline.get("markedText"), "baseline markedText")
        recomputed = encode_text(
            document.text,
            key=base_config.key,
            document_id=document.document_id,
            density_bps=base_config.density_bps,
            lexicon=corpus.lexicon,
            context_width=base_config.context_width,
        )
        if recomputed.text != marked_text:
            raise ValueError(f"frozen marked source mismatch: {document.document_id}")
        baseline_score = score_text(
            marked_text,
            key=base_config.key,
            document_id=document.document_id,
            density_bps=base_config.density_bps,
            lexicon=corpus.lexicon,
            context_width=base_config.context_width,
            min_active_positions=base_config.min_active_positions,
        )

        protected = protect_tokens(marked_text)
        issues: list[dict[str, str]] = []
        draft_prompt: str | StageRequest = (
            build_v4_draft_request(protected.masked)
            if config.method_id == V4_METHOD_ID
            else build_paraphrase_prompt(protected.masked)
        )
        draft, draft_record = manager.complete(
            call_id=f"{document.document_id}:{config.method_id}:paraphrase-draft",
            document_id=document.document_id,
            stage="paraphrase-draft",
            input_text=protected.masked,
            prompt=draft_prompt,
            max_tokens=(
                config.stage_max_tokens["paraphrase-draft"]
                if config.method_id == V4_METHOD_ID
                else None
            ),
        )
        if draft.finish_reason != "stop":
            issues.append(
                {
                    "code": "finish_reason_contract",
                    "message": f"draft finish reason was {draft.finish_reason!r}",
                    "stage": "paraphrase-draft",
                }
            )

        fidelity_audit: ChatCompletion | None = None
        audit_record: dict[str, object] | None = None
        validated_semantic_audit: str | None = None
        parsed_semantic_audit = None
        semantic_audit_status: str | None = None
        if config.method_id == V4_METHOD_ID:
            audit_prompt = build_semantic_audit_request(
                protected.masked,
                draft.content,
            )
            fidelity_audit, audit_record = manager.complete(
                call_id=f"{document.document_id}:{config.method_id}:semantic-audit",
                document_id=document.document_id,
                stage="semantic-audit",
                input_text=draft.content,
                prompt=audit_prompt,
                max_tokens=config.stage_max_tokens["semantic-audit"],
                response_format=semantic_audit_response_format(),
            )
            if fidelity_audit.finish_reason != "stop":
                issues.append(
                    {
                        "code": "finish_reason_contract",
                        "message": (
                            "semantic audit finish reason was "
                            f"{fidelity_audit.finish_reason!r}"
                        ),
                        "stage": "semantic-audit",
                    }
                )
            try:
                parsed_audit = parse_semantic_audit(
                    fidelity_audit.content,
                    source_masked=protected.masked,
                    draft_masked=draft.content,
                )
                validated_semantic_audit = parsed_audit.canonical_json
                parsed_semantic_audit = parsed_audit
                semantic_audit_status = "accepted"
            except SemanticAuditContractError as error:
                validated_semantic_audit = '{"corrections":[]}'
                semantic_audit_status = "validation_failure_empty_fallback"
                issues.append(
                    {
                        "code": "semantic_audit_contract",
                        "message": str(error),
                        "stage": "semantic-audit",
                    }
                )
            repair_prompt = build_semantic_repair_request(
                draft.content,
                validated_semantic_audit,
            )
        elif config.always_run_audit:
            audit_prompt = build_fidelity_audit_prompt(
                protected.masked,
                draft.content,
            )
            fidelity_audit, audit_record = manager.complete(
                call_id=f"{document.document_id}:{config.method_id}:fidelity-audit",
                document_id=document.document_id,
                stage="fidelity-audit",
                input_text=draft.content,
                prompt=audit_prompt,
            )
            if fidelity_audit.finish_reason != "stop":
                issues.append(
                    {
                        "code": "finish_reason_contract",
                        "message": (
                            "fidelity audit finish reason was "
                            f"{fidelity_audit.finish_reason!r}"
                        ),
                        "stage": "fidelity-audit",
                    }
                )
            repair_prompt = build_audit_guided_repair_prompt(
                draft.content,
                fidelity_audit.content,
            )
        else:
            repair_prompt = build_fidelity_repair_prompt(
                protected.masked,
                draft.content,
            )
        repair, repair_record = manager.complete(
            call_id=f"{document.document_id}:{config.method_id}:fidelity-repair",
            document_id=document.document_id,
            stage="fidelity-repair",
            input_text=draft.content,
            prompt=repair_prompt,
            max_tokens=(
                config.stage_max_tokens["fidelity-repair"]
                if config.method_id == V4_METHOD_ID
                else None
            ),
        )
        if repair.finish_reason != "stop":
            issues.append(
                {
                    "code": "finish_reason_contract",
                    "message": f"repair finish reason was {repair.finish_reason!r}",
                    "stage": "fidelity-repair",
                }
            )

        final_masked = repair.content
        restoration_mode = "exact"
        try:
            canonical = canonicalize_placeholders(final_masked, protected.tokens)
            for issue in result_validation_issues(protected.masked, canonical, None):
                issues.append({"stage": "final", **issue})
            if parsed_semantic_audit is not None:
                for issue in semantic_audit_repair_issues(
                    parsed_semantic_audit,
                    canonical,
                ):
                    issues.append({"stage": "final", **issue})
            output = restore_tokens(canonical, protected.tokens)
            final_masked = canonical
        except PlaceholderError as error:
            restoration_mode = "best_effort"
            issues.append(
                {
                    "code": "placeholder_contract",
                    "message": str(error),
                    "stage": "final",
                }
            )
            output = _best_effort_restore(final_masked, protected.tokens)

        output_score = score_text(
            output,
            key=base_config.key,
            document_id=document.document_id,
            density_bps=base_config.density_bps,
            lexicon=corpus.lexicon,
            context_width=base_config.context_width,
            min_active_positions=base_config.min_active_positions,
        )
        scores.append(output_score)
        fidelity = fidelity_metrics(marked_text, output)
        fidelity["failure"] = bool(issues)
        fidelity["failureReasons"] = [issue["code"] for issue in issues]
        rows.append(
            {
                "calls": [
                    draft_record,
                    *([audit_record] if audit_record is not None else []),
                    repair_record,
                ],
                "detector": output_score.to_dict(),
                "documentId": document.document_id,
                "fidelity": fidelity,
                "fingerprints": compare_active_fingerprints(
                    baseline_score, output_score
                ).to_dict(),
                "genre": document.genre,
                "markedInputText": marked_text,
                "originalText": document.text,
                "outputText": output,
                "sourceSha256": document.sha256,
                "transformationOutcome": {
                    "issues": issues,
                    "rawDraftMaskedText": draft.content,
                    "rawFidelityAuditText": (
                        fidelity_audit.content if fidelity_audit is not None else None
                    ),
                    "semanticAuditValidationStatus": semantic_audit_status,
                    "validatedCanonicalSemanticAudit": validated_semantic_audit,
                    "rawFinalMaskedText": repair.content,
                    "restorationMode": restoration_mode,
                    "status": "validation_failure" if issues else "accepted",
                },
            }
        )

    method = {
        "aggregate": _aggregate_method(rows, scores),
        "aggregateScope": "all_20_documents_secondary"
        if config.method_id == V4_METHOD_ID
        else "all_20_documents",
        "documents": rows,
        "method": config.method_id,
        "methodId": config.method_id,
        "pivot": None,
    }
    analysis_cohorts: dict[str, object] | None = None
    if config.method_id == V4_METHOD_ID:
        analysis_cohorts = {
            "development": _build_verified_cohort(
                rows,
                config.development_document_ids,
                key=base_config.key,
                density_bps=base_config.density_bps,
                lexicon=corpus.lexicon,
                context_width=base_config.context_width,
                min_active_positions=base_config.min_active_positions,
            ),
            "holdoutPrimary": _build_verified_cohort(
                rows,
                config.holdout_document_ids,
                key=base_config.key,
                density_bps=base_config.density_bps,
                lexicon=corpus.lexicon,
                context_width=base_config.context_width,
                min_active_positions=base_config.min_active_positions,
            ),
        }
    artifact = {
        **(
            {"analysisCohorts": analysis_cohorts}
            if analysis_cohorts is not None
            else {}
        ),
        "baseExperimentConfigSha256": config.base_config_sha256,
        "baseExperimentResultSha256": config.base_result_sha256,
        "configSha256": config.sha256,
        "documentCount": config.document_count,
        **(
            {"developmentCanaryGate": canary_gate_report}
            if canary_gate_report is not None
            else {}
        ),
        "endpointSnapshotSha256": config.endpoint_snapshot_sha256,
        "experimentVersion": config.experiment_version,
        "finalAudit": {
            "developmentDocumentIds": list(config.development_document_ids),
            "headlineFailureCohort": (
                "holdout" if config.method_id == V4_METHOD_ID else "all_documents"
            ),
            "holdoutDocumentIds": list(config.holdout_document_ids),
            "model": config.final_audit_model,
            "required": True,
            "separateFromRepair": config.separate_final_audit,
            "status": "pending_frozen_transform_output",
        },
        "methodology": config.methodology,
        "methods": [method],
        "schemaVersion": ARTIFACT_SCHEMA_VERSION,
        "decisionPolicy": {
            "articleDemoGate": {
                "criteria": config.raw["decisionPolicy"]["articleDemoGate"],
                "status": "pending_final_audit",
            }
        }
        if config.method_id == V4_METHOD_ID
        else {"publishAllOutputs": True},
        "sources": list(config.sources),
        "usage": {
            **aggregate_call_usage(manager.calls),
            "providerCostBudgetCredits": _decimal_text(budget),
        },
        "verifiedAt": config.verified_at,
    }
    canonical_json_bytes(artifact)
    return artifact


def _build_verified_cohort(
    rows: Sequence[Mapping[str, object]],
    document_ids: Sequence[str],
    *,
    key: bytes,
    density_bps: int,
    lexicon: object,
    context_width: int,
    min_active_positions: int,
) -> dict[str, object]:
    """Aggregate one preregistered cohort without folding development into holdout."""
    by_id = {_text(row.get("documentId"), "cohort documentId"): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("cohort source rows contain duplicate document IDs")
    selected: list[Mapping[str, object]] = []
    for document_id in document_ids:
        if document_id not in by_id:
            raise ValueError(f"cohort source row is missing: {document_id}")
        selected.append(by_id[document_id])
    if not selected:
        raise ValueError("verified cohort must not be empty")

    detector = score_corpus(
        tuple(
            Document(
                document_id=_text(row.get("documentId"), "cohort documentId"),
                text=_text(row.get("outputText"), "cohort outputText"),
            )
            for row in selected
        ),
        key=key,
        density_bps=density_bps,
        lexicon=lexicon,  # type: ignore[arg-type]
        context_width=context_width,
        min_active_positions=min_active_positions,
    ).to_dict()
    calls = [
        call
        for row in selected
        for call in _mapping_list(row.get("calls"), "cohort calls")
    ]
    usage = aggregate_call_usage(calls)
    total_cost = _decimal(usage["providerCostCredits"], "cohort provider cost")
    fidelities = [_mapping(row.get("fidelity"), "cohort fidelity") for row in selected]
    word_metrics = [
        _mapping(fidelity.get("wordLevenshtein"), "cohort wordLevenshtein")
        for fidelity in fidelities
    ]
    length_metrics = [
        _mapping(fidelity.get("length"), "cohort length") for fidelity in fidelities
    ]
    paragraph_metrics = [
        _mapping(fidelity.get("paragraphs"), "cohort paragraphs")
        for fidelity in fidelities
    ]
    protected_metrics = [
        _mapping(fidelity.get("protectedTokens"), "cohort protectedTokens")
        for fidelity in fidelities
    ]
    fingerprints = [
        _mapping(row.get("fingerprints"), "cohort fingerprints") for row in selected
    ]
    input_words = sum(
        _nonnegative_int(metric.get("originalWordCount"), "originalWordCount")
        for metric in word_metrics
    )
    transformation_failures = sum(
        fidelity.get("failure") is True for fidelity in fidelities
    )
    audit_contract_failures = sum(
        "semantic_audit_contract"
        in _string_list(fidelity.get("failureReasons"), "failureReasons")
        for fidelity in fidelities
    )
    usage.update(
        {
            "providerCostCreditsPer1000Documents": _decimal_text(
                total_cost * Decimal(1000) / len(selected)
            ),
            "providerCostCreditsPer1000MarkedInputWords": (
                _decimal_text(total_cost * Decimal(1000) / input_words)
                if input_words
                else None
            ),
            "totalInputWordCount": input_words,
        }
    )
    fingerprint_totals = {
        field: sum(_nonnegative_int(item.get(field), field) for item in fingerprints)
        for field in (
            "baselineActive",
            "lostActive",
            "newActive",
            "outputActive",
            "survivingActive",
        )
    }
    return {
        "aggregate": {
            "detector": detector,
            "fidelity": {
                "allProtectedTokensExactlyRestored": all(
                    metric.get("exactlyRestored") is True
                    for metric in protected_metrics
                ),
                "meanLengthRatio": sum(
                    float(metric["outputPerInput"]) for metric in length_metrics
                )
                / len(selected),
                "meanNormalizedWordDistance": sum(
                    float(metric["normalizedDistance"]) for metric in word_metrics
                )
                / len(selected),
                "meanParagraphRatio": sum(
                    float(metric["outputPerInput"]) for metric in paragraph_metrics
                )
                / len(selected),
                "semanticAuditContractFailureCount": audit_contract_failures,
                "transformationValidationFailureCount": transformation_failures,
                "transformationValidationFailureRate": (
                    transformation_failures / len(selected)
                ),
            },
            "fingerprints": fingerprint_totals,
            "usage": usage,
        },
        "documentIds": list(document_ids),
        "documentCount": len(selected),
    }


class _VerifiedCheckpointManager:
    def __init__(
        self,
        *,
        config: VerifiedParaphraseConfig,
        client: CompletionClient,
        checkpoint_path: Path,
        expected_call_ids: tuple[str, ...],
        budget: Decimal,
        max_new_calls: int | None,
        confirm_not_charged_call_id: str | None,
        clock: Callable[[], float],
    ) -> None:
        self.config = config
        self.client = client
        self.path = checkpoint_path
        self.expected_call_ids = expected_call_ids
        self.budget = budget
        self.max_new_calls = max_new_calls
        self.clock = clock
        self.new_calls = 0
        self.state = self._load_or_create()
        raw_calls = self.state.get("calls")
        if not isinstance(raw_calls, list) or any(
            not isinstance(call, dict) for call in raw_calls
        ):
            raise VerifiedCheckpointError("checkpoint calls must be a list of objects")
        self.calls: list[dict[str, object]] = raw_calls
        ids = tuple(call.get("callId") for call in self.calls)
        if ids != expected_call_ids[: len(ids)]:
            raise VerifiedCheckpointError("checkpoint is not an exact call prefix")
        in_flight = self.state.get("inFlightCall")
        if in_flight is None:
            if confirm_not_charged_call_id is not None:
                raise VerifiedCheckpointError("no in-flight call exists to resolve")
        else:
            tombstone = _mapping(in_flight, "inFlightCall")
            if tombstone.get("callId") != confirm_not_charged_call_id:
                raise VerifiedCheckpointError(
                    "in-flight charge is unknown; confirm its exact call ID only after "
                    "checking provider activity"
                )
            if len(self.calls) >= len(expected_call_ids):
                raise VerifiedCheckpointError(
                    "in-flight call follows a complete matrix"
                )
            if tombstone.get("callId") != expected_call_ids[len(self.calls)]:
                raise VerifiedCheckpointError(
                    "in-flight call is not the next matrix call"
                )

    def complete(
        self,
        *,
        call_id: str,
        document_id: str,
        stage: str,
        input_text: str,
        prompt: str | StageRequest,
        max_tokens: int | None = None,
        response_format: Mapping[str, object] | None = None,
    ) -> tuple[ChatCompletion, dict[str, object]]:
        if self.config.method_id == V4_METHOD_ID:
            if not isinstance(prompt, StageRequest) or prompt.stage != stage:
                raise VerifiedCheckpointError(
                    "v4 call must use the exact stage-bound message request"
                )
        elif isinstance(prompt, StageRequest):
            raise VerifiedCheckpointError("stage requests are only valid for v4")
        effective_max_tokens = (
            self.config.max_tokens if max_tokens is None else max_tokens
        )
        if (
            not isinstance(effective_max_tokens, int)
            or isinstance(effective_max_tokens, bool)
            or effective_max_tokens <= 0
            or effective_max_tokens > self.config.max_tokens
        ):
            raise VerifiedCheckpointError(
                "call max_tokens differs from the frozen contract"
            )
        normalized_response_format: dict[str, object] | None = None
        if response_format is not None:
            normalized = json_safe_value(response_format)
            if not isinstance(normalized, dict) or not normalized:
                raise VerifiedCheckpointError("call response_format is invalid")
            normalized_response_format = normalized
        try:
            expected_index = self.expected_call_ids.index(call_id)
        except ValueError as error:
            raise VerifiedCheckpointError(f"unknown call ID: {call_id}") from error
        if expected_index < len(self.calls):
            record = self.calls[expected_index]
            self._validate_saved_request(
                record,
                call_id=call_id,
                document_id=document_id,
                stage=stage,
                input_text=input_text,
                prompt=prompt,
                max_tokens=max_tokens,
                response_format=normalized_response_format,
            )
            if record.get("recordStatus") != "accepted_response":
                raise VerifiedResponseContractError(
                    "saved provider response failed its contract and will not be reissued"
                )
            completion = _completion_from_record(record)
            self._validate_response(
                completion,
                prompt,
                max_tokens=effective_max_tokens,
            )
            return completion, record
        if expected_index != len(self.calls):
            raise VerifiedCheckpointError("attempted to skip an uncheckpointed call")
        if self.max_new_calls is not None and self.new_calls >= self.max_new_calls:
            raise VerifiedCallLimitReached(
                completed_calls=len(self.calls), new_calls=self.new_calls
            )

        spent = _checkpoint_cost(self.calls)
        reserve = _routing_cost(
            request_utf8_size(prompt) + self.config.prompt_token_overhead_reserve,
            effective_max_tokens,
            self.config,
        )
        if spent >= self.budget or self.budget - spent < reserve:
            raise VerifiedBudgetError(
                "remaining provider budget is below the conservative next-call "
                f"reserve: need {reserve}, have {self.budget - spent}"
            )
        request_sha = _request_sha256(
            call_id=call_id,
            document_id=document_id,
            stage=stage,
            input_text=input_text,
            prompt=prompt,
            model=self.config.model,
            max_tokens=max_tokens,
            response_format=normalized_response_format,
        )
        existing_tombstone = self.state.get("inFlightCall")
        if existing_tombstone is not None:
            tombstone = _mapping(existing_tombstone, "inFlightCall")
            if tombstone.get("requestSha256") != request_sha:
                raise VerifiedCheckpointError("in-flight request hash mismatch")
        self.state["inFlightCall"] = {
            "callId": call_id,
            "conservativeCostReserveCredits": _decimal_text(reserve),
            "dispatchResolution": (
                "confirmed_not_charged_redispatch"
                if existing_tombstone is not None
                else "new_dispatch"
            ),
            "requestSha256": request_sha,
            "startedAtUnixMs": int(time.time() * 1000),
        }
        self._save()
        started = self.clock()
        try:
            if max_tokens is None and normalized_response_format is None:
                completion = self.client.complete(prompt, model=self.config.model)
            else:
                completion = self.client.complete(
                    prompt,
                    model=self.config.model,
                    max_tokens=effective_max_tokens,
                    response_format=normalized_response_format,
                )
        except ProviderResponseError as error:
            raw = json_safe_value(error.raw_response)
            if not isinstance(raw, dict):
                raw = {"unparseable": True}
            record = {
                "callId": call_id,
                "conservativeCostReserveCredits": _decimal_text(reserve),
                "documentId": document_id,
                "inputText": input_text,
                "latencyMs": round((self.clock() - started) * 1000, 3),
                "methodId": self.config.method_id,
                "outputText": None,
                **_request_record_fields(prompt),
                "providerError": str(error),
                "rawResponse": raw,
                "recordStatus": "provider_response_invalid",
                "requestSha256": request_sha,
                "request": _checkpoint_request(
                    self.config.model,
                    max_tokens=max_tokens,
                    response_format=normalized_response_format,
                ),
                "stage": stage,
            }
            self.calls.append(record)
            self.state["inFlightCall"] = None
            self._save()
            self.new_calls += 1
            raise VerifiedResponseContractError(
                "provider response was invalid and preserved; it will not be reissued"
            ) from error

        record = {
            "callId": call_id,
            "conservativeCostReserveCredits": _decimal_text(reserve),
            "documentId": document_id,
            "inputText": input_text,
            "latencyMs": round((self.clock() - started) * 1000, 3),
            "methodId": self.config.method_id,
            "outputText": completion.content,
            **_request_record_fields(prompt),
            "recordStatus": "accepted_response",
            "requestSha256": request_sha,
            "request": _checkpoint_request(
                self.config.model,
                max_tokens=max_tokens,
                response_format=normalized_response_format,
            ),
            "response": completion.to_dict(),
            "routingCostEstimateUsd": _decimal_text(
                _routing_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                    self.config,
                )
            ),
            "stage": stage,
        }
        try:
            self._validate_response(
                completion,
                prompt,
                max_tokens=effective_max_tokens,
            )
        except VerifiedResponseContractError:
            record["recordStatus"] = "response_contract_failure"
            self.calls.append(record)
            self.state["inFlightCall"] = None
            self._save()
            self.new_calls += 1
            raise
        self.calls.append(record)
        self.state["inFlightCall"] = None
        self._save()
        self.new_calls += 1
        if _checkpoint_cost(self.calls) > self.budget:
            raise VerifiedBudgetError("checkpointed provider cost exceeded the budget")
        return completion, record

    def _load_or_create(self) -> dict[str, object]:
        expected = {
            "baseExperimentResultSha256": self.config.base_result_sha256,
            "configSha256": self.config.sha256,
            "endpointSnapshotSha256": self.config.endpoint_snapshot_sha256,
            "expectedCallIds": list(self.expected_call_ids),
            "schemaVersion": CHECKPOINT_SCHEMA_VERSION,
        }
        if not self.path.exists():
            return {**expected, "calls": [], "inFlightCall": None}
        state = _json_object(self.path.read_bytes(), "verified checkpoint")
        for field, value in expected.items():
            if state.get(field) != value:
                raise VerifiedCheckpointError(f"checkpoint binding mismatch: {field}")
        return state

    def _validate_saved_request(
        self,
        record: Mapping[str, object],
        *,
        call_id: str,
        document_id: str,
        stage: str,
        input_text: str,
        prompt: str | StageRequest,
        max_tokens: int | None,
        response_format: Mapping[str, object] | None,
    ) -> None:
        expected = {
            "callId": call_id,
            "documentId": document_id,
            "inputText": input_text,
            "methodId": self.config.method_id,
            "stage": stage,
        }
        for field, value in expected.items():
            if record.get(field) != value:
                raise VerifiedCheckpointError(f"saved request mismatch: {field}")
        expected_request_content = _request_record_fields(prompt)
        for field in ("prompt", "messages"):
            if field in expected_request_content:
                if record.get(field) != expected_request_content[field]:
                    raise VerifiedCheckpointError(f"saved request mismatch: {field}")
            elif field in record:
                raise VerifiedCheckpointError(
                    f"saved request has unexpected content field: {field}"
                )
        request = _mapping(record.get("request"), "saved request")
        expected_request = _checkpoint_request(
            self.config.model,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        if dict(request) != expected_request:
            raise VerifiedCheckpointError("saved request options mismatch")
        if self.config.method_id == V4_METHOD_ID:
            expected_sha = _request_sha256(
                call_id=call_id,
                document_id=document_id,
                stage=stage,
                input_text=input_text,
                prompt=prompt,
                model=self.config.model,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            if record.get("requestSha256") != expected_sha:
                raise VerifiedCheckpointError("saved v4 request hash mismatch")

    def _validate_response(
        self,
        completion: ChatCompletion,
        prompt: str | StageRequest,
        *,
        max_tokens: int,
    ) -> None:
        if completion.model not in self.config.expected_response_models:
            raise VerifiedResponseContractError(
                f"unexpected response model: {completion.model}"
            )
        if completion.provider not in self.config.expected_response_providers:
            raise VerifiedResponseContractError(
                f"unexpected selected provider: {completion.provider}"
            )
        metadata = completion.openrouter_metadata
        if not isinstance(metadata, Mapping):
            raise VerifiedResponseContractError(
                "OpenRouter routing metadata is required"
            )
        if metadata.get("strategy") != "direct" or metadata.get("attempt") != 1:
            raise VerifiedResponseContractError(
                "OpenRouter must use one direct attempt"
            )
        if metadata.get("pipeline") not in (None, []):
            raise VerifiedResponseContractError(
                "OpenRouter routing pipeline must be empty"
            )
        endpoints = _mapping(metadata.get("endpoints"), "router endpoints")
        available = endpoints.get("available")
        if not isinstance(available, list) or any(
            not isinstance(item, Mapping) for item in available
        ):
            raise VerifiedResponseContractError("router endpoint metadata is invalid")
        selected = [item for item in available if item.get("selected") is True]
        if len(selected) != 1:
            raise VerifiedResponseContractError(
                "router must select exactly one endpoint"
            )
        provider = _first_present(
            selected[0], "provider", "provider_name", "providerName"
        )
        model = _first_present(selected[0], "model", "model_id", "modelId")
        if provider != completion.provider:
            raise VerifiedResponseContractError("router selected provider mismatch")
        if model not in self.config.expected_response_models:
            raise VerifiedResponseContractError("router selected model mismatch")
        attempts = metadata.get("attempts")
        if attempts is not None:
            if not isinstance(attempts, list) or len(attempts) != 1:
                raise VerifiedResponseContractError(
                    "router attempts must contain one item"
                )
            attempt = _mapping(attempts[0], "router attempt")
            if (
                _first_present(attempt, "provider", "provider_name", "providerName")
                != completion.provider
            ):
                raise VerifiedResponseContractError("router attempt provider mismatch")
            if _first_present(attempt, "model", "model_id", "modelId") not in (
                self.config.expected_response_models
            ):
                raise VerifiedResponseContractError("router attempt model mismatch")
            if attempt.get("status") != 200 or isinstance(attempt.get("status"), bool):
                raise VerifiedResponseContractError("router attempt status must be 200")
        if completion.usage.prompt_tokens > (
            request_utf8_size(prompt) + self.config.prompt_token_overhead_reserve
        ):
            raise VerifiedResponseContractError("prompt usage exceeds frozen reserve")
        if (
            completion.usage.total_tokens
            != completion.usage.prompt_tokens + completion.usage.completion_tokens
        ):
            raise VerifiedResponseContractError(
                "response token totals are inconsistent"
            )
        if completion.usage.completion_tokens > max_tokens:
            raise VerifiedResponseContractError("completion usage exceeds maxTokens")
        expected_cost = _routing_cost(
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
            self.config,
        )
        if completion.usage.cost != expected_cost:
            raise VerifiedResponseContractError(
                "provider cost differs from frozen endpoint pricing"
            )

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.path, canonical_json_bytes(self.state))


class _CanaryNoCallClient:
    def complete(
        self,
        request: str | StageRequest,
        *,
        model: str,
        max_tokens: int | None = None,
        response_format: Mapping[str, object] | None = None,
    ) -> ChatCompletion:
        raise AssertionError("canary gate is local-only and must not call a provider")


def build_v4_canary_gate(
    config: VerifiedParaphraseConfig,
    *,
    checkpoint_path: str | Path,
) -> dict[str, object]:
    """Evaluate the frozen doc-01 gate from checkpointed responses, without network."""
    if config.method_id != V4_METHOD_ID:
        raise ValueError("development canary gate is only defined for v4")
    path = Path(checkpoint_path).resolve()
    manager = _VerifiedCheckpointManager(
        config=config,
        client=_CanaryNoCallClient(),
        checkpoint_path=path,
        expected_call_ids=expected_verified_call_ids(config),
        budget=Decimal("1"),
        max_new_calls=0,
        confirm_not_charged_call_id=None,
        clock=time.perf_counter,
    )
    expected_count = config.development_canary_exact_call_count
    development_calls = [
        call for call in manager.calls if call.get("documentId") == "doc-01"
    ]
    exact_calls_pass = (
        len(development_calls) == expected_count
        and manager.calls[:expected_count] == development_calls
    )
    checks: dict[str, dict[str, object]] = {
        "exactDevelopmentCallCount": {
            "observed": len(development_calls),
            "passed": exact_calls_pass,
            "required": expected_count,
        }
    }
    checkpoint_sha = (
        hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    )
    if not exact_calls_pass:
        report = {
            "checks": checks,
            "checkpointCallCount": len(manager.calls),
            "checkpointSha256": checkpoint_sha,
            "configSha256": config.sha256,
            "documentId": "doc-01",
            "issues": [
                "development canary does not contain exactly three prefix calls"
            ],
            "schemaVersion": 1,
            "status": "incomplete"
            if len(development_calls) < expected_count
            else "no_go",
        }
        canonical_json_bytes(report)
        return report

    marked = _load_frozen_marked_documents(config)["doc-01"]
    protected = protect_tokens(marked)
    issues: list[str] = []
    route_finish_usage_pass = True

    def accepted_completion(
        index: int,
        *,
        stage: str,
        input_text: str,
        request: StageRequest,
        max_tokens: int,
        response_format: Mapping[str, object] | None = None,
    ) -> ChatCompletion:
        nonlocal route_finish_usage_pass
        record = development_calls[index]
        try:
            manager._validate_saved_request(
                record,
                call_id=f"doc-01:{config.method_id}:{stage}",
                document_id="doc-01",
                stage=stage,
                input_text=input_text,
                prompt=request,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            if record.get("recordStatus") != "accepted_response":
                raise VerifiedResponseContractError(
                    f"{stage} checkpoint record is not accepted"
                )
            completion = _completion_from_record(record)
            manager._validate_response(
                completion,
                request,
                max_tokens=max_tokens,
            )
            if completion.finish_reason != "stop":
                raise VerifiedResponseContractError(
                    f"{stage} finish reason is not stop"
                )
            return completion
        except (ValueError, VerifiedExperimentError) as error:
            route_finish_usage_pass = False
            issues.append(f"{stage}: {error}")
            raise

    draft_request = build_v4_draft_request(protected.masked)
    try:
        draft = accepted_completion(
            0,
            stage="paraphrase-draft",
            input_text=protected.masked,
            request=draft_request,
            max_tokens=config.stage_max_tokens["paraphrase-draft"],
        )
    except (ValueError, VerifiedExperimentError):
        draft = _completion_from_record(development_calls[0])

    audit_request = build_semantic_audit_request(protected.masked, draft.content)
    audit_format = semantic_audit_response_format()
    try:
        audit = accepted_completion(
            1,
            stage="semantic-audit",
            input_text=draft.content,
            request=audit_request,
            max_tokens=config.stage_max_tokens["semantic-audit"],
            response_format=audit_format,
        )
    except (ValueError, VerifiedExperimentError):
        audit = _completion_from_record(development_calls[1])

    parsed_audit = None
    semantic_audit_status = "accepted"
    try:
        parsed_audit = parse_semantic_audit(
            audit.content,
            source_masked=protected.masked,
            draft_masked=draft.content,
        )
        canonical_audit = parsed_audit.canonical_json
    except SemanticAuditContractError as error:
        semantic_audit_status = "validation_failure_empty_fallback"
        canonical_audit = '{"corrections":[]}'
        issues.append(f"semantic-audit: {error}")

    repair_request = build_semantic_repair_request(draft.content, canonical_audit)
    try:
        repair = accepted_completion(
            2,
            stage="fidelity-repair",
            input_text=draft.content,
            request=repair_request,
            max_tokens=config.stage_max_tokens["fidelity-repair"],
        )
    except (ValueError, VerifiedExperimentError):
        repair = _completion_from_record(development_calls[2])

    try:
        canonical_draft = canonicalize_placeholders(draft.content, protected.tokens)
        draft_text = restore_tokens(canonical_draft, protected.tokens)
    except PlaceholderError as error:
        issues.append(f"draft placeholder contract: {error}")
        draft_text = _best_effort_restore(draft.content, protected.tokens)

    try:
        canonical_final = canonicalize_placeholders(repair.content, protected.tokens)
        for issue in result_validation_issues(protected.masked, canonical_final, None):
            issues.append(f"final {issue['code']}: {issue['message']}")
        if parsed_audit is not None:
            for issue in semantic_audit_repair_issues(parsed_audit, canonical_final):
                issues.append(f"final {issue['code']}: {issue['message']}")
        final_text = restore_tokens(canonical_final, protected.tokens)
    except PlaceholderError as error:
        issues.append(f"final placeholder contract: {error}")
        final_text = _best_effort_restore(repair.content, protected.tokens)

    draft_distance = float(
        _mapping(
            fidelity_metrics(marked, draft_text).get("wordLevenshtein"),
            "canary draft word distance",
        )["normalizedDistance"]
    )
    final_distance = float(
        _mapping(
            fidelity_metrics(marked, final_text).get("wordLevenshtein"),
            "canary final word distance",
        )["normalizedDistance"]
    )
    distance_ratio = final_distance / draft_distance if draft_distance > 0 else 0.0
    checks.update(
        {
            "routeFinishAndUsageContracts": {
                "passed": route_finish_usage_pass,
                "required": True,
            },
            "semanticAuditAccepted": {
                "observed": semantic_audit_status,
                "passed": semantic_audit_status == "accepted",
                "required": "accepted",
            },
            "noPipelineIssues": {
                "observedIssueCount": len(issues),
                "passed": not issues,
                "requiredIssueCount": 0,
            },
            "minimumFinalWordDistance": {
                "observed": final_distance,
                "passed": (
                    final_distance >= config.canary_min_final_normalized_word_distance
                ),
                "requiredMinimum": config.canary_min_final_normalized_word_distance,
            },
            "minimumFinalToDraftWordDistanceRatio": {
                "observed": distance_ratio,
                "passed": (
                    distance_ratio
                    >= config.canary_min_final_to_draft_word_distance_ratio
                ),
                "requiredMinimum": (
                    config.canary_min_final_to_draft_word_distance_ratio
                ),
            },
        }
    )
    passed = all(check.get("passed") is True for check in checks.values())
    report = {
        "checks": checks,
        "checkpointCallCount": len(manager.calls),
        "checkpointSha256": checkpoint_sha,
        "configSha256": config.sha256,
        "documentId": "doc-01",
        "issues": issues,
        "observed": {
            "draftNormalizedWordDistance": draft_distance,
            "finalNormalizedWordDistance": final_distance,
            "finalToDraftWordDistanceRatio": distance_ratio,
        },
        "schemaVersion": 1,
        "status": "go" if passed else "no_go",
    }
    canonical_json_bytes(report)
    return report


def _load_frozen_marked_documents(
    config: VerifiedParaphraseConfig,
) -> dict[str, str]:
    result = _load_base_result(config)
    return {
        document_id: _text(document.get("markedText"), "baseline markedText")
        for document_id, document in _baseline_documents(result).items()
    }


def _load_base_result(config: VerifiedParaphraseConfig) -> dict[str, object]:
    _require_sha(config.base_result_path, config.base_result_sha256, "base result")
    result = _json_object(config.base_result_path.read_bytes(), "base result")
    if result.get("documentCount") != config.document_count:
        raise ValueError("base result document count mismatch")
    return result


def _baseline_documents(
    result: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    baseline = _mapping(result.get("baseline"), "base result baseline")
    documents = baseline.get("documents")
    if not isinstance(documents, list) or len(documents) != 20:
        raise ValueError("base result baseline must contain 20 documents")
    output: dict[str, Mapping[str, object]] = {}
    for expected_id, raw in zip(EXPECTED_DOCUMENT_IDS, documents, strict=True):
        document = _mapping(raw, "base result baseline document")
        if document.get("documentId") != expected_id:
            raise ValueError("base result document IDs are not frozen in order")
        output[expected_id] = document
    return output


def _best_effort_restore(text: str, tokens: Sequence[object]) -> str:
    restored = text
    for token in tokens:
        placeholder = token.placeholder  # type: ignore[attr-defined]
        original = token.original  # type: ignore[attr-defined]
        number = placeholder.removeprefix("⟦T").removesuffix("⟧")
        for variant in (
            placeholder,
            f"[T{number}]",
            f"[ T{number} ]",
            f"⟦ T{number} ⟧",
        ):
            restored = restored.replace(variant, original)
    return restored


def _completion_from_record(record: Mapping[str, object]) -> ChatCompletion:
    response = _mapping(record.get("response"), "saved response")
    usage = _mapping(response.get("usage"), "saved response usage")
    metadata = response.get("openrouterMetadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise VerifiedCheckpointError("saved routing metadata is invalid")
    return ChatCompletion(
        content=_text(record.get("outputText"), "saved outputText"),
        finish_reason=_text(response.get("finishReason"), "saved finishReason"),
        model=_text(response.get("model"), "saved model"),
        openrouter_metadata=metadata,
        provider=_text(response.get("provider"), "saved provider"),
        response_id=_text(response.get("id"), "saved response ID"),
        system_fingerprint=(
            response.get("systemFingerprint")
            if isinstance(response.get("systemFingerprint"), str)
            else None
        ),
        usage=CompletionUsage(
            prompt_tokens=_nonnegative_int(
                usage.get("promptTokens"), "saved promptTokens"
            ),
            completion_tokens=_nonnegative_int(
                usage.get("completionTokens"), "saved completionTokens"
            ),
            total_tokens=_nonnegative_int(
                usage.get("totalTokens"), "saved totalTokens"
            ),
            cost=_decimal(usage.get("providerCostCredits"), "saved provider cost"),
        ),
    )


def _checkpoint_cost(calls: Sequence[Mapping[str, object]]) -> Decimal:
    total = Decimal(0)
    for call in calls:
        response = call.get("response")
        if isinstance(response, Mapping):
            usage = _mapping(response.get("usage"), "checkpoint usage")
            total += _decimal(usage.get("providerCostCredits"), "provider cost")
        else:
            total += _decimal(
                call.get("conservativeCostReserveCredits"),
                "invalid response reserve",
            )
    return total


def _request_sha256(
    *,
    call_id: str,
    document_id: str,
    stage: str,
    input_text: str,
    prompt: str | StageRequest,
    model: str,
    max_tokens: int | None,
    response_format: Mapping[str, object] | None,
) -> str:
    hashed_content = (
        {
            "messagesSha256": hashlib.sha256(
                canonical_json_bytes(list(request_messages(prompt)))
            ).hexdigest()
        }
        if isinstance(prompt, StageRequest)
        else {"promptSha256": hashlib.sha256(prompt.encode()).hexdigest()}
    )
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "callId": call_id,
                "documentId": document_id,
                **hashed_content,
                "inputSha256": hashlib.sha256(input_text.encode()).hexdigest(),
                "model": model,
                "requestOptions": _checkpoint_request(
                    model,
                    max_tokens=max_tokens,
                    response_format=response_format,
                ),
                "stage": stage,
            }
        )
    ).hexdigest()


def _request_record_fields(
    request: str | StageRequest,
) -> dict[str, object]:
    if isinstance(request, StageRequest):
        return {"messages": list(request_messages(request))}
    return {"prompt": request}


def _checkpoint_request(
    model: str,
    *,
    max_tokens: int | None,
    response_format: Mapping[str, object] | None,
) -> dict[str, object]:
    request: dict[str, object] = {"model": model}
    if max_tokens is not None:
        request["maxTokens"] = max_tokens
    if response_format is not None:
        normalized = json_safe_value(response_format)
        if not isinstance(normalized, dict):
            raise VerifiedCheckpointError("checkpoint response format is invalid")
        request["responseFormat"] = normalized
    return request


def _routing_cost(
    prompt_tokens: int,
    completion_tokens: int,
    config: VerifiedParaphraseConfig,
) -> Decimal:
    return (
        Decimal(prompt_tokens) * config.prompt_price_usd_per_million
        + Decimal(completion_tokens) * config.completion_price_usd_per_million
    ) / Decimal(1_000_000)


def _required_budget(value: Decimal) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
        raise VerifiedBudgetError("live mode requires an explicit positive budget")
    return value


def _validate_evidence(value: Mapping[str, object], label: str) -> None:
    _text(value.get("verifiedAt"), f"{label}.verifiedAt")
    methodology = _text(value.get("methodology"), f"{label}.methodology")
    if len(methodology) < 20:
        raise ValueError(f"{label}.methodology is too short")
    _sources(value.get("sources"))


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


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a list of strings")
    return value


def _json_object(content: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
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


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _positive_number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive number")
    return float(value)


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


def _first_present(value: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        if key in value:
            return value[key]
    return None


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--canary-gate", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument(
        "--config",
        default=str(root / "fixtures" / "verified-paraphrase-config-v2.json"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(root / "results" / "verified-paraphrase-checkpoint-v2.json"),
    )
    parser.add_argument(
        "--output",
        default=str(root / "results" / "verified-paraphrase-raw-v2.json"),
    )
    parser.add_argument("--max-provider-cost-credits")
    parser.add_argument("--max-new-calls", type=int)
    parser.add_argument("--confirm-not-charged-call-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_verified_paraphrase_config(args.config)
    if args.canary_gate:
        if (
            args.max_provider_cost_credits is not None
            or args.max_new_calls is not None
            or args.confirm_not_charged_call_id is not None
        ):
            raise SystemExit("canary gate is local-only and accepts no live controls")
        print(
            canonical_json_bytes(
                build_v4_canary_gate(
                    config,
                    checkpoint_path=args.checkpoint,
                )
            ).decode(),
            end="",
        )
        return 0
    if args.dry_run:
        if (
            args.max_provider_cost_credits is not None
            or args.max_new_calls is not None
            or args.confirm_not_charged_call_id is not None
        ):
            raise SystemExit("budget and checkpoint controls are live-only")
        print(
            canonical_json_bytes(build_verified_dry_run(config)).decode(),
            end="",
        )
        return 0
    if args.max_provider_cost_credits is None:
        raise SystemExit("--live requires --max-provider-cost-credits")
    budget = _decimal(args.max_provider_cost_credits, "provider budget")
    client = OpenRouterClient.from_env(
        timeout=config.timeout_seconds,
        provider_order=config.provider_order,
        allow_fallbacks=False,
        require_parameters=True,
        reasoning_effort=config.reasoning_effort,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        seed=config.seed,
        max_prompt_price=float(config.prompt_price_usd_per_million),
        max_completion_price=float(config.completion_price_usd_per_million),
    )
    try:
        artifact = run_verified_live(
            config,
            client=client,
            max_provider_cost_credits=budget,
            checkpoint_path=args.checkpoint,
            max_new_calls=args.max_new_calls,
            confirm_not_charged_call_id=args.confirm_not_charged_call_id,
        )
    except VerifiedCallLimitReached as pause:
        print(
            json.dumps(
                {
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "completedCalls": pause.completed_calls,
                    "newCalls": pause.new_calls,
                    "status": "paused_at_call_limit",
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    output = Path(args.output)
    content = canonical_json_bytes(artifact)
    _atomic_write(output, content)
    print(
        json.dumps(
            {
                "artifactSha256": hashlib.sha256(content).hexdigest(),
                "callCount": artifact["usage"]["callCount"],
                "output": str(output.resolve()),
                "providerCostCredits": artifact["usage"]["providerCostCredits"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
