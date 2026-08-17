"""Freeze, run, and review the GPT-5.6 development model canary.

The default dry-run is local-only. Live mode performs exactly the frozen
candidate-major matrix, checkpoints every paid response, and never retries a
request whose charge status is unknown.
"""

from __future__ import annotations

import argparse
import copy
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
    aggregate_call_usage,
    fidelity_metrics,
    load_experiment_config,
    load_reviewed_corpus,
)
from run_verified_paraphrase import load_verified_paraphrase_config
from unmark import (
    ChatCompletion,
    CompletionUsage,
    OpenRouterClient,
    PlaceholderError,
    ProviderResponseError,
    StageRequest,
    V4_SYSTEM_INSTRUCTIONS,
    build_v4_draft_request,
    canonicalize_placeholders,
    json_safe_value,
    protect_tokens,
    request_messages,
    request_utf8_size,
    restore_tokens,
    result_validation_issues,
)
from watermark_toy import score_text


CONFIG_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 1
REVIEW_SCHEMA_VERSION = "model-canary-manual-review-v6/1.0"
LUNA_MODEL = "openai/gpt-5.6-luna"
TERRA_MODEL = "openai/gpt-5.6-terra"
EXPECTED_CANDIDATES = (LUNA_MODEL, TERRA_MODEL)
EXPECTED_DOCUMENT_IDS = (
    "doc-11",
    "doc-12",
    "doc-15",
    "doc-20",
    "doc-03",
    "doc-19",
)
EXPECTED_MAJOR_DOCUMENT_IDS = ("doc-11", "doc-12", "doc-15", "doc-20")
EXPECTED_MINOR_DOCUMENT_IDS = ("doc-03", "doc-19")
EXPECTED_PROVIDER_ORDER = ("openai",)
EXPECTED_RESPONSE_PROVIDERS = ("OpenAI",)
EXPECTED_CODEX_PLAN_SHA256 = (
    "217fdd8e394580497b4436a2e320892f0305b2f26296b4ccc6d9b4c31d9988ea"
)
EXPECTED_CODEX_RESULT_SHA256 = (
    "13e241d222c63a720896bc879b53443b42224e1ebe9ac255ed4e20429ff390db"
)
_SHA256_ZERO = "0" * 64


class ModelCanaryError(Exception):
    """Base class for expected v6 canary failures."""


class ModelCanaryBudgetError(ModelCanaryError):
    """Raised before dispatch when the explicit budget cannot cover a call."""


class ModelCanaryCheckpointError(ModelCanaryError):
    """Raised when a checkpoint cannot be resumed safely."""


class ModelCanaryResponseContractError(ModelCanaryError):
    """Raised after preserving a paid response that violates the frozen route."""


class ModelCanaryCallLimitReached(ModelCanaryError):
    """Intentional pause after the requested number of new calls."""

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
class CandidateConfig:
    model: str
    official_workload_classification: str
    endpoint_snapshot_path: Path
    endpoint_snapshot_sha256: str
    expected_response_models: tuple[str, ...]
    prompt_price_usd_per_million: Decimal
    completion_price_usd_per_million: Decimal
    max_prompt_price_usd_per_million: float
    max_completion_price_usd_per_million: float


@dataclass(frozen=True)
class ModelCanaryConfig:
    path: Path
    root: Path
    raw: dict[str, object]
    sha256: str
    experiment_version: str
    verified_at: str
    methodology: str
    sources: tuple[dict[str, str], ...]
    source_commit: str
    reserved_holdout_commit: str
    v4_config_path: Path
    v4_config_sha256: str
    v4_result_path: Path
    v4_result_sha256: str
    v5_config_path: Path
    v5_config_sha256: str
    v5_result_path: Path
    v5_result_sha256: str
    codex_plan_path: Path
    codex_plan_sha256: str
    codex_result_path: Path
    codex_result_sha256: str
    reserved_holdout_plan_path: Path
    reserved_holdout_plan_sha256: str
    reserved_holdout_manifest_path: Path
    reserved_holdout_manifest_sha256: str
    document_ids: tuple[str, ...]
    candidates: tuple[CandidateConfig, ...]
    provider_order: tuple[str, ...]
    expected_response_providers: tuple[str, ...]
    allow_fallbacks: bool
    data_collection: str
    zdr: bool
    require_parameters: bool
    reasoning_effort: str
    max_tokens: int
    seed: int
    temperature_present: bool
    timeout_seconds: float
    prompt_token_overhead_reserve: int
    manual_plan_path: Path
    manual_plan_sha256: str
    parity_fixture_path: Path
    parity_fixture_sha256: str
    confirmatory_claim_allowed: bool
    maximum_major_findings: int
    maximum_minor_findings: int
    maximum_pipeline_failures: int
    minimum_mean_word_distance: float
    required_reviewed_document_count: int
    qwen_reference_provider_calls: int

    @property
    def candidate_models(self) -> tuple[str, ...]:
        return tuple(candidate.model for candidate in self.candidates)

    @property
    def calls_per_candidate(self) -> int:
        return len(self.document_ids)

    @property
    def total_provider_calls(self) -> int:
        return len(self.candidates) * self.calls_per_candidate

    def candidate(self, model: str) -> CandidateConfig:
        for candidate in self.candidates:
            if candidate.model == model:
                return candidate
        raise KeyError(model)


def load_model_canary_config(
    path: str | Path,
    *,
    root: str | Path | None = None,
    validate_parity: bool = True,
) -> ModelCanaryConfig:
    """Load and strictly validate the frozen development-canary contract."""
    config_path = Path(path).resolve()
    root_path = (
        Path(root).resolve() if root is not None else Path(__file__).resolve().parent
    )
    raw_bytes = config_path.read_bytes()
    raw = _json_object(raw_bytes, "model canary config")
    if raw.get("schemaVersion") != CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported model canary config schemaVersion")
    _validate_evidence(raw, "model canary config")
    if (
        raw.get("experimentVersion")
        != "text-watermark-roundtrip-model-canary-v6-development"
    ):
        raise ValueError("unexpected model canary experimentVersion")

    source_artifacts = _mapping(raw.get("sourceArtifacts"), "sourceArtifacts")
    source_commit = _text(source_artifacts.get("sourceCommit"), "sourceCommit")
    if source_commit != "b642f42":
        raise ValueError("sourceCommit must bind the frozen Codex review commit")
    reserved_holdout_commit = _text(
        source_artifacts.get("reservedHoldoutCommit"),
        "reservedHoldoutCommit",
    )
    if reserved_holdout_commit != "19ea51e":
        raise ValueError("reserved holdout commit binding changed")
    bindings: dict[str, tuple[Path, str]] = {}
    for key in (
        "v4Config",
        "v4Result",
        "v5Config",
        "v5Result",
        "codexPlan",
        "codexResult",
        "reservedHoldoutPlan",
        "reservedHoldoutManifest",
    ):
        binding = _mapping(source_artifacts.get(key), f"sourceArtifacts.{key}")
        bound_path = _safe_path(root_path, binding.get("path"), f"{key} path")
        bound_sha = _sha256(binding.get("sha256"), f"{key} SHA-256")
        _require_sha(bound_path, bound_sha, key)
        bindings[key] = (bound_path, bound_sha)
    if bindings["codexPlan"][1] != EXPECTED_CODEX_PLAN_SHA256:
        raise ValueError("Codex plan binding differs from b642f42")
    if bindings["codexResult"][1] != EXPECTED_CODEX_RESULT_SHA256:
        raise ValueError("Codex result binding differs from b642f42")

    development = _mapping(raw.get("developmentCases"), "developmentCases")
    if development.get("classification") != "development_only":
        raise ValueError("v6 cases must be development_only")
    document_ids = _string_tuple(development.get("documentIds"), "documentIds")
    if document_ids != EXPECTED_DOCUMENT_IDS:
        raise ValueError("v6 development document set or order changed")
    if development.get("sourceSelection") != (
        "v4.methods[0].documents[].markedInputText"
    ):
        raise ValueError("v6 marked source selection changed")
    if development.get("qwenReferenceSelection") != "v5.documents[].outputText":
        raise ValueError("v6 Qwen reference selection changed")
    selected = _mapping(
        development.get("selectionFrozenFromCodexResult"),
        "selectionFrozenFromCodexResult",
    )
    if _string_tuple(selected.get("majorDocumentIds"), "majorDocumentIds") != (
        EXPECTED_MAJOR_DOCUMENT_IDS
    ):
        raise ValueError("v6 major development cases changed")
    if (
        _string_tuple(
            selected.get("deterministicMinorDocumentIds"),
            "deterministicMinorDocumentIds",
        )
        != EXPECTED_MINOR_DOCUMENT_IDS
    ):
        raise ValueError("v6 minor development cases changed")
    _validate_codex_selection(bindings["codexResult"][0])

    raw_candidates = _mapping_list(raw.get("candidates"), "candidates")
    if tuple(item.get("model") for item in raw_candidates) != EXPECTED_CANDIDATES:
        raise ValueError("v6 candidate set or order changed")
    candidates = tuple(
        _load_candidate(root_path, item, expected_model)
        for item, expected_model in zip(
            raw_candidates, EXPECTED_CANDIDATES, strict=True
        )
    )

    routing = _mapping(raw.get("routing"), "routing")
    if routing.get("transport") != "openrouter":
        raise ValueError("v6 transport must be OpenRouter")
    provider_order = _string_tuple(routing.get("providerOrder"), "providerOrder")
    response_providers = _string_tuple(
        routing.get("expectedResponseProviders"), "expectedResponseProviders"
    )
    if provider_order != EXPECTED_PROVIDER_ORDER:
        raise ValueError("v6 provider tag must be openai")
    if response_providers != EXPECTED_RESPONSE_PROVIDERS:
        raise ValueError("v6 response provider must be OpenAI")
    required_routing = {
        "allowFallbacks": False,
        "dataCollection": "deny",
        "zdr": True,
        "requireParameters": True,
        "requireDirectStrategy": True,
        "requireSingleAttempt": True,
        "requireEmptyPipeline": True,
    }
    for field, expected in required_routing.items():
        if routing.get(field) != expected:
            raise ValueError(f"v6 routing contract changed: {field}")

    request = _mapping(raw.get("request"), "request")
    if "temperature" in request:
        raise ValueError("temperature must be omitted for GPT-5.6 endpoint parity")
    expected_request = {
        "stage": "paraphrase-draft",
        "callGraph": ["paraphrase-draft"],
        "instructionRole": "system",
        "untrustedPayloadRole": "user",
        "userPayloadEncoding": "canonical_json_utf8",
        "userPayloadFields": ["sourceText"],
        "messageCount": 2,
        "systemInstructionSource": ("unmark.V4_SYSTEM_INSTRUCTIONS[paraphrase-draft]"),
        "systemInstructionSha256": hashlib.sha256(
            V4_SYSTEM_INSTRUCTIONS["paraphrase-draft"].encode("utf-8")
        ).hexdigest(),
        "reasoningEffort": "medium",
        "maxTokens": 4096,
        "seed": 20260817,
        "temperaturePolicy": "omit_unsupported_parameter",
        "responseFormat": None,
    }
    if dict(request) != expected_request:
        raise ValueError("v6 request contract changed")

    execution = _mapping(raw.get("execution"), "execution")
    expected_execution = {
        "order": "candidate_major_then_document_order",
        "callsPerCandidate": 6,
        "totalProviderCalls": 12,
        "retryPolicy": "never",
        "fallbackPolicy": "never",
        "checkpointEveryPaidResponse": True,
        "unknownChargePolicy": (
            "leave_in_flight_tombstone_and_require_explicit_confirm_not_charged"
        ),
        "timeoutSeconds": 180,
        "futureLiveCommand": (
            "python run_model_canary_v6.py --live --max-new-calls 12 "
            "--max-provider-cost-credits 0.30"
        ),
    }
    if dict(execution) != expected_execution:
        raise ValueError("v6 execution contract changed")

    billing = _mapping(raw.get("billing"), "billing")
    if billing.get("creditBaseCurrency") != "USD":
        raise ValueError("v6 credit base currency must be USD")
    if billing.get("creditUsdBaseUnit") != "1":
        raise ValueError("v6 credit base unit changed")
    if billing.get("inferencePricingMarkupPercent") != 0:
        raise ValueError("v6 pricing assumes no inference markup")
    if billing.get("purchaseFeeExcluded") is not True:
        raise ValueError("v6 purchase fee policy changed")
    if billing.get("reserveFullMaxCompletionForEveryNextCall") is not True:
        raise ValueError("v6 must reserve the full next-call completion maximum")
    prompt_overhead = _positive_int(
        billing.get("promptTokenOverheadReserve"),
        "promptTokenOverheadReserve",
    )
    if prompt_overhead != 2048:
        raise ValueError("v6 prompt-token overhead reserve changed")

    qwen = _mapping(raw.get("qwenReference"), "qwenReference")
    if dict(qwen) != {
        "model": "qwen/qwen3.6-35b-a3b",
        "providerCalls": 0,
        "outputSource": "v5.documents[].outputText",
        "manualVerdictSource": "codexResult.summaries.all20.documentsByVerdict",
        "selectable": False,
    }:
        raise ValueError("v6 Qwen reference contract changed")

    manual = _mapping(raw.get("manualSelectionPlan"), "manualSelectionPlan")
    manual_path = _safe_path(root_path, manual.get("path"), "manual plan path")
    manual_sha = _sha256(manual.get("sha256"), "manual plan SHA-256")
    _require_sha(manual_path, manual_sha, "manual selection plan")
    _validate_manual_plan(manual_path)

    parity = _mapping(raw.get("parityFixture"), "parityFixture")
    parity_path = _safe_path(root_path, parity.get("path"), "parity path")
    parity_sha = _sha256(parity.get("sha256"), "parity SHA-256")
    if validate_parity:
        if parity_sha == _SHA256_ZERO:
            raise ValueError("v6 parity fixture is not frozen")
        _require_sha(parity_path, parity_sha, "parity fixture")

    decision = _mapping(raw.get("decisionPolicy"), "decisionPolicy")
    if decision.get("classification") != "development_only":
        raise ValueError("v6 decision classification changed")
    if decision.get("confirmatoryClaimAllowed") is not False:
        raise ValueError("v6 cannot permit a confirmatory claim")
    if decision.get("retuningAfterResults") is not False:
        raise ValueError("v6 cannot retune after results")
    if decision.get("qwenReferenceSelectable") is not False:
        raise ValueError("Qwen reference cannot enter v6 selection")
    if decision.get("ifBothPass") != (
        "lower_actual_provider_cost_then_lower_total_latency"
    ):
        raise ValueError("v6 both-pass selection changed")
    if decision.get("ifOnePasses") != "select_passing_candidate":
        raise ValueError("v6 one-pass selection changed")
    if decision.get("ifNeitherPasses") != "stop_without_demo_candidate":
        raise ValueError("v6 no-pass selection changed")
    gate = _mapping(decision.get("candidatePassesOnlyIf"), "candidate gate")
    expected_gate = {
        "maximumMajorMaterialFindings": 0,
        "maximumMinorMaterialFindings": 0,
        "maximumPipelineFailures": 0,
        "minimumMeanNormalizedWordDistance": 0.15,
        "requiredReviewedDocumentCount": 6,
    }
    if dict(gate) != expected_gate:
        raise ValueError("v6 candidate gate changed")

    config = ModelCanaryConfig(
        path=config_path,
        root=root_path,
        raw=raw,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        experiment_version=_text(raw.get("experimentVersion"), "experimentVersion"),
        verified_at=_text(raw.get("verifiedAt"), "verifiedAt"),
        methodology=_text(raw.get("methodology"), "methodology"),
        sources=tuple(_sources(raw.get("sources"))),
        source_commit=source_commit,
        reserved_holdout_commit=reserved_holdout_commit,
        v4_config_path=bindings["v4Config"][0],
        v4_config_sha256=bindings["v4Config"][1],
        v4_result_path=bindings["v4Result"][0],
        v4_result_sha256=bindings["v4Result"][1],
        v5_config_path=bindings["v5Config"][0],
        v5_config_sha256=bindings["v5Config"][1],
        v5_result_path=bindings["v5Result"][0],
        v5_result_sha256=bindings["v5Result"][1],
        codex_plan_path=bindings["codexPlan"][0],
        codex_plan_sha256=bindings["codexPlan"][1],
        codex_result_path=bindings["codexResult"][0],
        codex_result_sha256=bindings["codexResult"][1],
        reserved_holdout_plan_path=bindings["reservedHoldoutPlan"][0],
        reserved_holdout_plan_sha256=bindings["reservedHoldoutPlan"][1],
        reserved_holdout_manifest_path=bindings["reservedHoldoutManifest"][0],
        reserved_holdout_manifest_sha256=bindings["reservedHoldoutManifest"][1],
        document_ids=document_ids,
        candidates=candidates,
        provider_order=provider_order,
        expected_response_providers=response_providers,
        allow_fallbacks=False,
        data_collection="deny",
        zdr=True,
        require_parameters=True,
        reasoning_effort="medium",
        max_tokens=4096,
        seed=20260817,
        temperature_present=False,
        timeout_seconds=180.0,
        prompt_token_overhead_reserve=prompt_overhead,
        manual_plan_path=manual_path,
        manual_plan_sha256=manual_sha,
        parity_fixture_path=parity_path,
        parity_fixture_sha256=parity_sha,
        confirmatory_claim_allowed=False,
        maximum_major_findings=0,
        maximum_minor_findings=0,
        maximum_pipeline_failures=0,
        minimum_mean_word_distance=0.15,
        required_reviewed_document_count=6,
        qwen_reference_provider_calls=0,
    )
    if validate_parity:
        frozen_parity = _json_object(parity_path.read_bytes(), "parity fixture")
        rebuilt = build_model_canary_parity(config)
        if frozen_parity != rebuilt:
            raise ValueError("v6 parity fixture does not match frozen requests")
    return config


def expected_model_canary_call_ids(config: ModelCanaryConfig) -> tuple[str, ...]:
    """Return the exact candidate-major one-pass call order."""
    return tuple(
        f"{model}:{document_id}:paraphrase-draft"
        for model in config.candidate_models
        for document_id in config.document_ids
    )


def build_model_canary_parity(config: ModelCanaryConfig) -> dict[str, object]:
    """Build the byte-exact twelve-request parity artifact locally."""
    sources = _load_development_sources(config)
    calls: list[dict[str, object]] = []
    request_shas: dict[str, str] = {}
    for model in config.candidate_models:
        candidate = config.candidate(model)
        for document_id in config.document_ids:
            source = sources[document_id]
            marked_text = _nonempty_string(
                source.get("markedInputText"), "markedInputText"
            )
            protected = protect_tokens(marked_text)
            request = build_v4_draft_request(protected.masked)
            call_id = f"{model}:{document_id}:paraphrase-draft"
            request_record = _frozen_request_record(config, candidate, request)
            request_sha = _request_sha256(
                call_id=call_id,
                document_id=document_id,
                input_text=protected.masked,
                request=request_record,
            )
            request_shas[call_id] = request_sha
            calls.append(
                {
                    "callId": call_id,
                    "candidateModel": model,
                    "documentId": document_id,
                    "inputTextSha256": _text_sha256(protected.masked),
                    "markedInputTextSha256": _text_sha256(marked_text),
                    "request": request_record,
                    "requestSha256": request_sha,
                    "stage": "paraphrase-draft",
                }
            )
    return {
        "schemaVersion": 1,
        "verifiedAt": config.verified_at,
        "methodology": (
            "Byte-exact parity for the twelve future v6 development calls. Each "
            "request reuses the hardened v4 system instruction and canonical JSON "
            "sourceText payload, pins the OpenAI endpoint tag, omits temperature, "
            "and binds the full messages and routing options into requestSha256."
        ),
        "sources": [
            {
                "title": "Frozen marked-source and Qwen-reference commit",
                "url": (
                    "https://github.com/krllagent/text-watermark-roundtrip/commit/"
                    f"{config.source_commit}"
                ),
            },
            {
                "title": "OpenAI GPT-5.6 model guidance",
                "url": (
                    "https://developers.openai.com/api/docs/guides/latest-model"
                    "#update-api-and-model-parameters"
                ),
            },
        ],
        "callOrder": "candidate_major_then_document_order",
        "calls": calls,
        "manualSelectionPlanSha256": config.manual_plan_sha256,
        "requestSha256s": request_shas,
        "temperatureOmitted": True,
        "totalProviderCalls": 12,
    }


def build_model_canary_dry_run(config: ModelCanaryConfig) -> dict[str, object]:
    """Return a deterministic local plan without reading credentials."""
    parity = build_model_canary_parity(config)
    calls_by_candidate: dict[str, int] = {}
    reserves: dict[str, str] = {}
    total_reserve = Decimal(0)
    for candidate in config.candidates:
        candidate_reserve = Decimal(0)
        for call in parity["calls"]:
            if call["candidateModel"] != candidate.model:
                continue
            request = _mapping(call.get("request"), "parity request")
            messages = _mapping_list(request.get("messages"), "request messages")
            prompt_bytes = sum(
                len(
                    _nonempty_string(message.get("content"), "message content").encode(
                        "utf-8"
                    )
                )
                for message in messages
            )
            candidate_reserve += _routing_cost(
                prompt_bytes + config.prompt_token_overhead_reserve,
                config.max_tokens,
                candidate,
            )
        calls_by_candidate[candidate.model] = config.calls_per_candidate
        reserves[candidate.model] = _decimal_text(candidate_reserve)
        total_reserve += candidate_reserve
    return {
        "schemaVersion": ARTIFACT_SCHEMA_VERSION,
        "verifiedAt": config.verified_at,
        "experimentVersion": config.experiment_version,
        "methodology": (
            "Local-only maximum routing-cost reserve. UTF-8 message bytes plus the "
            "frozen overhead stand in for prompt tokens and every call reserves the "
            "full 4,096-token completion maximum. No credential or provider is read."
        ),
        "classification": "development_only",
        "confirmatoryClaimAllowed": False,
        "configSha256": config.sha256,
        "callCount": config.total_provider_calls,
        "callsByCandidate": calls_by_candidate,
        "documentIds": list(config.document_ids),
        "maximumRoutingCostReserveByCandidateCredits": reserves,
        "maximumRoutingCostReserveCredits": _decimal_text(total_reserve),
        "parityFixtureSha256": config.parity_fixture_sha256,
        "qwenReference": {"providerCalls": 0, "selectable": False},
        "requestSha256s": parity["requestSha256s"],
    }


def run_model_canary_live(
    config: ModelCanaryConfig,
    *,
    clients: Mapping[str, CompletionClient],
    max_provider_cost_credits: Decimal,
    checkpoint_path: str | Path,
    max_new_calls: int | None = None,
    confirm_not_charged_call_id: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Run or resume the frozen twelve-call matrix."""
    budget = _required_budget(max_provider_cost_credits)
    if max_new_calls is not None and (
        not isinstance(max_new_calls, int)
        or isinstance(max_new_calls, bool)
        or max_new_calls < 0
    ):
        raise ValueError("max_new_calls must be a nonnegative integer or null")
    if set(clients) != set(config.candidate_models):
        raise ValueError("clients must contain exactly the two frozen candidate models")
    build_model_canary_dry_run(config)
    sources = _load_development_sources(config)
    detector_config, corpus = _load_detector(config)
    manager = _CheckpointManager(
        config=config,
        clients=clients,
        checkpoint_path=Path(checkpoint_path),
        expected_call_ids=expected_model_canary_call_ids(config),
        budget=budget,
        max_new_calls=max_new_calls,
        confirm_not_charged_call_id=confirm_not_charged_call_id,
        clock=clock,
    )
    opaque_ids = _opaque_pair_ids(config)
    rows: list[dict[str, object]] = []
    for candidate in config.candidates:
        for document_id in config.document_ids:
            source = sources[document_id]
            marked_text = _nonempty_string(
                source.get("markedInputText"), "markedInputText"
            )
            protected = protect_tokens(marked_text)
            request = build_v4_draft_request(protected.masked)
            call_id = f"{candidate.model}:{document_id}:paraphrase-draft"
            completion, call = manager.complete(
                call_id=call_id,
                candidate=candidate,
                document_id=document_id,
                input_text=protected.masked,
                request=request,
            )
            issues: list[dict[str, str]] = []
            canonical_masked: str | None = None
            try:
                canonical_masked = canonicalize_placeholders(
                    completion.content,
                    protected.tokens,
                )
                output_text = restore_tokens(canonical_masked, protected.tokens)
            except PlaceholderError as error:
                issues.append(
                    {
                        "code": "placeholder_contract",
                        "message": str(error),
                        "stage": "paraphrase-draft",
                    }
                )
                output_text = _best_effort_restore(
                    completion.content,
                    protected.tokens,
                )
            validation_input = (
                canonical_masked if canonical_masked is not None else completion.content
            )
            for issue in result_validation_issues(
                protected.masked,
                validation_input,
                None,
            ):
                issues.append({**issue, "stage": "paraphrase-draft"})
            output_score = score_text(
                output_text,
                key=detector_config.key,
                document_id=document_id,
                density_bps=detector_config.density_bps,
                lexicon=corpus.lexicon,
                context_width=detector_config.context_width,
                min_active_positions=detector_config.min_active_positions,
            )
            usage = _mapping(
                _mapping(call.get("response"), "call response").get("usage"),
                "call usage",
            )
            rows.append(
                {
                    "actualCostCredits": _text(
                        usage.get("providerCostCredits"), "provider cost"
                    ),
                    "opaquePairId": opaque_ids[(candidate.model, document_id)],
                    "call": call,
                    "candidateModel": candidate.model,
                    "canonicalMaskedText": canonical_masked,
                    "detector": output_score.to_dict(),
                    "documentId": document_id,
                    "fidelity": fidelity_metrics(marked_text, output_text),
                    "latencyMs": call["latencyMs"],
                    "markedInputText": marked_text,
                    "outputText": output_text,
                    "pipeline": {
                        "failureCount": len(issues),
                        "issues": issues,
                        "passed": not issues,
                    },
                    "rawMaskedText": completion.content,
                    "sourceSha256": _text_sha256(marked_text),
                }
            )
    artifact_without_hash: dict[str, object] = {
        "schemaVersion": ARTIFACT_SCHEMA_VERSION,
        "verifiedAt": config.verified_at,
        "experimentVersion": config.experiment_version,
        "methodology": config.methodology,
        "sources": list(config.sources),
        "classification": "development_only",
        "confirmatoryClaimAllowed": False,
        "configSha256": config.sha256,
        "manualSelectionPlanSha256": config.manual_plan_sha256,
        "parityFixtureSha256": config.parity_fixture_sha256,
        "sourceBindings": {
            "codexPlanSha256": config.codex_plan_sha256,
            "codexResultSha256": config.codex_result_sha256,
            "v4ResultSha256": config.v4_result_sha256,
            "v5ResultSha256": config.v5_result_sha256,
            "reservedHoldoutPlanSha256": config.reserved_holdout_plan_sha256,
            "reservedHoldoutManifestSha256": (config.reserved_holdout_manifest_sha256),
        },
        "documentIds": list(config.document_ids),
        "documents": rows,
        "candidateSummaries": _candidate_summaries(config, rows),
        "qwenReference": _qwen_reference(config, sources),
        "selectionGate": {
            "status": "pending_blind_manual_review",
            "selectedCandidate": None,
            "policy": _selection_policy(config),
        },
        "usage": aggregate_call_usage(manager.calls),
    }
    artifact_sha = _object_sha256(artifact_without_hash)
    return {**artifact_without_hash, "artifactSha256": artifact_sha}


def build_blind_review_packet(
    config: ModelCanaryConfig,
    artifact: Mapping[str, object],
) -> dict[str, object]:
    """Remove every condition-revealing field from the manual review packet."""
    _validate_artifact_binding(config, artifact)
    rows = _mapping_list(artifact.get("documents"), "artifact documents")
    if len(rows) != config.total_provider_calls:
        raise ValueError("artifact must contain twelve candidate rows")
    pairs = sorted(
        (
            {
                "candidateText": _nonempty_string(row.get("outputText"), "outputText"),
                "opaquePairId": _text(row.get("opaquePairId"), "opaquePairId"),
                "sourceText": _nonempty_string(
                    row.get("markedInputText"), "markedInputText"
                ),
            }
            for row in rows
        ),
        key=lambda pair: pair["opaquePairId"],
    )
    return {
        "schemaVersion": "model-canary-blind-packet-v6/1.0",
        "artifactSha256": _text(artifact.get("artifactSha256"), "artifactSha256"),
        "manualPlanSha256": config.manual_plan_sha256,
        "methodology": (
            "Review all opaque pairs and commit pass, minor, or major before reveal."
        ),
        "pairs": pairs,
        "status": "ready_for_blind_review",
    }


def finalize_model_canary_review(
    config: ModelCanaryConfig,
    artifact: Mapping[str, object],
    manual_review: Mapping[str, object],
) -> dict[str, object]:
    """Apply the frozen zero-finding gate and deterministic winner rule locally."""
    _validate_artifact_binding(config, artifact)
    if manual_review.get("schemaVersion") != REVIEW_SCHEMA_VERSION:
        raise ValueError("unsupported manual review schemaVersion")
    if manual_review.get("artifactSha256") != artifact.get("artifactSha256"):
        raise ValueError("manual review artifact binding mismatch")
    if manual_review.get("manualPlanSha256") != config.manual_plan_sha256:
        raise ValueError("manual review plan binding mismatch")
    reviews = _mapping_list(manual_review.get("reviews"), "manual reviews")
    rows = _mapping_list(artifact.get("documents"), "artifact documents")
    expected_pair_ids = {_text(row.get("opaquePairId"), "opaquePairId") for row in rows}
    if len(reviews) != len(expected_pair_ids):
        raise ValueError("manual review must cover all twelve opaque pairs")
    by_pair: dict[str, str] = {}
    for review in reviews:
        pair_id = _text(review.get("opaquePairId"), "opaquePairId")
        verdict = _text(review.get("verdict"), "verdict")
        if pair_id in by_pair or pair_id not in expected_pair_ids:
            raise ValueError("manual review pair IDs are invalid")
        if verdict not in {"pass", "minor", "major"}:
            raise ValueError("manual review verdict must be pass, minor, or major")
        by_pair[pair_id] = verdict

    candidate_summaries = {
        _text(item.get("candidateModel"), "candidateModel"): item
        for item in _mapping_list(
            artifact.get("candidateSummaries"), "candidateSummaries"
        )
    }
    evaluations: list[dict[str, object]] = []
    passing: list[str] = []
    for model in config.candidate_models:
        candidate_rows = [row for row in rows if row.get("candidateModel") == model]
        verdicts = [
            by_pair[_text(row.get("opaquePairId"), "opaquePairId")]
            for row in candidate_rows
        ]
        major_count = verdicts.count("major")
        minor_count = verdicts.count("minor")
        summary = _mapping(candidate_summaries.get(model), "candidate summary")
        pipeline_failures = _nonnegative_int(
            summary.get("pipelineFailureCount"), "pipelineFailureCount"
        )
        mean_distance = _nonnegative_float(
            summary.get("meanNormalizedWordDistance"),
            "meanNormalizedWordDistance",
        )
        checks = {
            "majorMaterialFindingCount": {
                "actual": major_count,
                "maximum": config.maximum_major_findings,
                "passed": major_count <= config.maximum_major_findings,
            },
            "minorMaterialFindingCount": {
                "actual": minor_count,
                "maximum": config.maximum_minor_findings,
                "passed": minor_count <= config.maximum_minor_findings,
            },
            "pipelineFailureCount": {
                "actual": pipeline_failures,
                "maximum": config.maximum_pipeline_failures,
                "passed": pipeline_failures <= config.maximum_pipeline_failures,
            },
            "meanNormalizedWordDistance": {
                "actual": mean_distance,
                "minimum": config.minimum_mean_word_distance,
                "passed": mean_distance >= config.minimum_mean_word_distance,
            },
            "reviewedDocumentCount": {
                "actual": len(verdicts),
                "required": config.required_reviewed_document_count,
                "passed": len(verdicts) == config.required_reviewed_document_count,
            },
        }
        passed = all(bool(check["passed"]) for check in checks.values())
        if passed:
            passing.append(model)
        evaluations.append(
            {
                "candidateModel": model,
                "checks": checks,
                "passed": passed,
                "verdictCounts": {
                    "major": major_count,
                    "minor": minor_count,
                    "pass": verdicts.count("pass"),
                },
            }
        )

    selected: str | None
    if not passing:
        status = "stop_no_candidate_passed"
        selected = None
    elif len(passing) == 1:
        status = "selected"
        selected = passing[0]
    else:
        status = "selected"
        selected = min(
            passing,
            key=lambda model: (
                _decimal(
                    _mapping(candidate_summaries[model], "candidate summary").get(
                        "actualCostCredits"
                    ),
                    "actualCostCredits",
                ),
                _nonnegative_float(
                    _mapping(candidate_summaries[model], "candidate summary").get(
                        "totalLatencyMs"
                    ),
                    "totalLatencyMs",
                ),
                config.candidate_models.index(model),
            ),
        )
    finalized = copy.deepcopy(dict(artifact))
    finalized["manualReview"] = {
        "reviews": [dict(review) for review in reviews],
        "schemaVersion": REVIEW_SCHEMA_VERSION,
    }
    finalized["selectionGate"] = {
        "candidateEvaluations": evaluations,
        "policy": _selection_policy(config),
        "selectedCandidate": selected,
        "status": status,
    }
    finalized_without_hash = {
        key: value
        for key, value in finalized.items()
        if key != "finalizedArtifactSha256"
    }
    finalized["finalizedArtifactSha256"] = _object_sha256(finalized_without_hash)
    return finalized


class _CheckpointManager:
    def __init__(
        self,
        *,
        config: ModelCanaryConfig,
        clients: Mapping[str, CompletionClient],
        checkpoint_path: Path,
        expected_call_ids: tuple[str, ...],
        budget: Decimal,
        max_new_calls: int | None,
        confirm_not_charged_call_id: str | None,
        clock: Callable[[], float],
    ) -> None:
        self.config = config
        self.clients = clients
        self.path = checkpoint_path
        self.expected_call_ids = expected_call_ids
        self.budget = budget
        self.max_new_calls = max_new_calls
        self.clock = clock
        self.new_calls = 0
        self.state = self._load_or_create()
        self.calls = _mapping_list(self.state.get("calls"), "checkpoint calls")
        self.state["calls"] = self.calls
        saved_call_ids = tuple(call.get("callId") for call in self.calls)
        if saved_call_ids != expected_call_ids[: len(saved_call_ids)]:
            raise ModelCanaryCheckpointError("checkpoint is not an exact call prefix")
        in_flight = self.state.get("inFlightCall")
        if in_flight is not None:
            tombstone = _mapping(in_flight, "inFlightCall")
            if confirm_not_charged_call_id is None:
                raise ModelCanaryCheckpointError(
                    "an in-flight call has unknown charge status; verify it was not "
                    "charged, then pass --confirm-not-charged-call-id"
                )
            if tombstone.get("callId") != confirm_not_charged_call_id:
                raise ModelCanaryCheckpointError(
                    "confirmed call ID does not match the in-flight tombstone"
                )
            if len(self.calls) >= len(expected_call_ids) or (
                tombstone.get("callId") != expected_call_ids[len(self.calls)]
            ):
                raise ModelCanaryCheckpointError(
                    "in-flight tombstone is not the next frozen call"
                )
        elif confirm_not_charged_call_id is not None:
            raise ModelCanaryCheckpointError("no in-flight call exists to resolve")

    def complete(
        self,
        *,
        call_id: str,
        candidate: CandidateConfig,
        document_id: str,
        input_text: str,
        request: StageRequest,
    ) -> tuple[ChatCompletion, dict[str, object]]:
        try:
            expected_index = self.expected_call_ids.index(call_id)
        except ValueError as error:
            raise ModelCanaryCheckpointError(f"unknown call ID: {call_id}") from error
        request_record = _frozen_request_record(self.config, candidate, request)
        request_sha = _request_sha256(
            call_id=call_id,
            document_id=document_id,
            input_text=input_text,
            request=request_record,
        )
        if expected_index < len(self.calls):
            record = self.calls[expected_index]
            self._validate_saved_request(
                record,
                call_id=call_id,
                candidate=candidate,
                document_id=document_id,
                input_text=input_text,
                request_record=request_record,
                request_sha=request_sha,
            )
            if record.get("recordStatus") != "accepted_response":
                raise ModelCanaryResponseContractError(
                    "saved paid response failed its contract and will not be reissued"
                )
            completion = _completion_from_record(record)
            self._validate_response(completion, request, candidate)
            return completion, record
        if expected_index != len(self.calls):
            raise ModelCanaryCheckpointError("attempted to skip an uncheckpointed call")
        if self.max_new_calls is not None and self.new_calls >= self.max_new_calls:
            raise ModelCanaryCallLimitReached(
                completed_calls=len(self.calls), new_calls=self.new_calls
            )

        reserve = _routing_cost(
            request_utf8_size(request) + self.config.prompt_token_overhead_reserve,
            self.config.max_tokens,
            candidate,
        )
        spent = _checkpoint_cost(self.calls)
        if spent >= self.budget or self.budget - spent < reserve:
            raise ModelCanaryBudgetError(
                "remaining provider budget is below the conservative next-call "
                f"reserve: need {reserve}, have {self.budget - spent}"
            )
        existing_tombstone = self.state.get("inFlightCall")
        if existing_tombstone is not None:
            tombstone = _mapping(existing_tombstone, "inFlightCall")
            if tombstone.get("requestSha256") != request_sha:
                raise ModelCanaryCheckpointError("in-flight request hash mismatch")
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
            completion = self.clients[candidate.model].complete(
                request,
                model=candidate.model,
                max_tokens=self.config.max_tokens,
                response_format=None,
            )
        except ProviderResponseError as error:
            raw = json_safe_value(error.raw_response)
            if not isinstance(raw, dict):
                raw = {"unparseable": True}
            record = {
                "callId": call_id,
                "candidateModel": candidate.model,
                "conservativeCostReserveCredits": _decimal_text(reserve),
                "documentId": document_id,
                "inputText": input_text,
                "latencyMs": round((self.clock() - started) * 1000, 3),
                "messages": list(request_messages(request)),
                "outputText": None,
                "providerError": str(error),
                "rawResponse": raw,
                "recordStatus": "provider_response_invalid",
                "request": request_record,
                "requestSha256": request_sha,
                "stage": "paraphrase-draft",
            }
            self.calls.append(record)
            self.state["inFlightCall"] = None
            self._save()
            self.new_calls += 1
            raise ModelCanaryResponseContractError(
                "provider response was invalid and preserved; it will not be reissued"
            ) from error

        record = {
            "callId": call_id,
            "candidateModel": candidate.model,
            "conservativeCostReserveCredits": _decimal_text(reserve),
            "documentId": document_id,
            "inputText": input_text,
            "latencyMs": round((self.clock() - started) * 1000, 3),
            "messages": list(request_messages(request)),
            "outputText": completion.content,
            "recordStatus": "accepted_response",
            "request": request_record,
            "requestSha256": request_sha,
            "response": completion.to_dict(),
            "routingCostEstimateCredits": _decimal_text(
                _routing_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                    candidate,
                )
            ),
            "stage": "paraphrase-draft",
        }
        try:
            self._validate_response(completion, request, candidate)
        except ModelCanaryResponseContractError:
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
            raise ModelCanaryBudgetError("checkpointed provider cost exceeded budget")
        return completion, record

    def _load_or_create(self) -> dict[str, object]:
        expected = {
            "configSha256": self.config.sha256,
            "endpointSnapshotSha256s": {
                candidate.model: candidate.endpoint_snapshot_sha256
                for candidate in self.config.candidates
            },
            "expectedCallIds": list(self.expected_call_ids),
            "manualSelectionPlanSha256": self.config.manual_plan_sha256,
            "parityFixtureSha256": self.config.parity_fixture_sha256,
            "schemaVersion": CHECKPOINT_SCHEMA_VERSION,
            "sourceArtifactSha256s": {
                "codexResult": self.config.codex_result_sha256,
                "reservedHoldoutManifest": (
                    self.config.reserved_holdout_manifest_sha256
                ),
                "reservedHoldoutPlan": self.config.reserved_holdout_plan_sha256,
                "v4Result": self.config.v4_result_sha256,
                "v5Result": self.config.v5_result_sha256,
            },
        }
        if not self.path.exists():
            return {**expected, "calls": [], "inFlightCall": None}
        state = _json_object(self.path.read_bytes(), "model canary checkpoint")
        for field, value in expected.items():
            if state.get(field) != value:
                raise ModelCanaryCheckpointError(
                    f"checkpoint binding mismatch: {field}"
                )
        return state

    def _validate_saved_request(
        self,
        record: Mapping[str, object],
        *,
        call_id: str,
        candidate: CandidateConfig,
        document_id: str,
        input_text: str,
        request_record: Mapping[str, object],
        request_sha: str,
    ) -> None:
        expected = {
            "callId": call_id,
            "candidateModel": candidate.model,
            "documentId": document_id,
            "inputText": input_text,
            "request": dict(request_record),
            "requestSha256": request_sha,
            "stage": "paraphrase-draft",
        }
        for field, value in expected.items():
            if record.get(field) != value:
                raise ModelCanaryCheckpointError(f"saved request mismatch: {field}")

    def _validate_response(
        self,
        completion: ChatCompletion,
        request: StageRequest,
        candidate: CandidateConfig,
    ) -> None:
        if completion.finish_reason != "stop":
            raise ModelCanaryResponseContractError(
                f"finish reason must be stop, got {completion.finish_reason!r}"
            )
        if completion.model not in candidate.expected_response_models:
            raise ModelCanaryResponseContractError(
                f"unexpected response model: {completion.model}"
            )
        if completion.provider not in self.config.expected_response_providers:
            raise ModelCanaryResponseContractError(
                f"unexpected selected provider: {completion.provider}"
            )
        metadata = completion.openrouter_metadata
        if not isinstance(metadata, Mapping):
            raise ModelCanaryResponseContractError(
                "OpenRouter routing metadata is required"
            )
        if metadata.get("strategy") != "direct" or metadata.get("attempt") != 1:
            raise ModelCanaryResponseContractError(
                "OpenRouter must use one direct attempt"
            )
        if metadata.get("pipeline") not in (None, []):
            raise ModelCanaryResponseContractError(
                "OpenRouter routing pipeline must be empty"
            )
        endpoints = _mapping(metadata.get("endpoints"), "router endpoints")
        available = endpoints.get("available")
        if not isinstance(available, list) or any(
            not isinstance(item, Mapping) for item in available
        ):
            raise ModelCanaryResponseContractError(
                "router endpoint metadata is invalid"
            )
        selected = [item for item in available if item.get("selected") is True]
        if len(selected) != 1:
            raise ModelCanaryResponseContractError(
                "router must select exactly one endpoint"
            )
        if (
            _first_present(selected[0], "provider", "provider_name", "providerName")
            != completion.provider
        ):
            raise ModelCanaryResponseContractError("router selected provider mismatch")
        if (
            _first_present(selected[0], "model", "model_id", "modelId")
            not in candidate.expected_response_models
        ):
            raise ModelCanaryResponseContractError("router selected model mismatch")
        attempts = metadata.get("attempts")
        if attempts is not None:
            if not isinstance(attempts, list) or len(attempts) != 1:
                raise ModelCanaryResponseContractError(
                    "router attempts must contain one item"
                )
            attempt = _mapping(attempts[0], "router attempt")
            if (
                _first_present(attempt, "provider", "provider_name", "providerName")
                != completion.provider
            ):
                raise ModelCanaryResponseContractError(
                    "router attempt provider mismatch"
                )
            if (
                _first_present(attempt, "model", "model_id", "modelId")
                not in candidate.expected_response_models
            ):
                raise ModelCanaryResponseContractError("router attempt model mismatch")
            status = attempt.get("status")
            if status != 200 or isinstance(status, bool):
                raise ModelCanaryResponseContractError(
                    "router attempt status must be 200"
                )
        if completion.usage.prompt_tokens > (
            request_utf8_size(request) + self.config.prompt_token_overhead_reserve
        ):
            raise ModelCanaryResponseContractError(
                "prompt usage exceeds frozen reserve"
            )
        if completion.usage.total_tokens != (
            completion.usage.prompt_tokens + completion.usage.completion_tokens
        ):
            raise ModelCanaryResponseContractError(
                "response token totals are inconsistent"
            )
        if completion.usage.completion_tokens > self.config.max_tokens:
            raise ModelCanaryResponseContractError("completion usage exceeds maxTokens")
        expected_cost = _routing_cost(
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
            candidate,
        )
        if completion.usage.cost != expected_cost:
            raise ModelCanaryResponseContractError(
                "provider cost differs from frozen endpoint pricing"
            )

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.path, canonical_json_bytes(self.state))


def _load_candidate(
    root: Path,
    raw: Mapping[str, object],
    expected_model: str,
) -> CandidateConfig:
    model = _text(raw.get("model"), "candidate model")
    if model != expected_model:
        raise ValueError("candidate model order changed")
    expected_classification = (
        "efficient, high-volume workloads"
        if model == LUNA_MODEL
        else "balance of intelligence and cost"
    )
    classification = _text(
        raw.get("officialWorkloadClassification"),
        "officialWorkloadClassification",
    )
    if classification != expected_classification:
        raise ValueError("official workload classification changed")
    snapshot_binding = _mapping(raw.get("endpointSnapshot"), "endpointSnapshot")
    snapshot_path = _safe_path(root, snapshot_binding.get("path"), "snapshot path")
    snapshot_sha = _sha256(snapshot_binding.get("sha256"), "snapshot SHA-256")
    _require_sha(snapshot_path, snapshot_sha, "endpoint snapshot")
    expected_models = _string_tuple(
        raw.get("expectedResponseModels"), "expectedResponseModels"
    )
    dated = f"{model}-20260709"
    if expected_models != (model, dated):
        raise ValueError("candidate response-model allowlist changed")
    pricing = _mapping(
        raw.get("pricingUsdPerMillionTokens"),
        "pricingUsdPerMillionTokens",
    )
    max_price = _mapping(
        raw.get("maxPriceUsdPerMillionTokens"),
        "maxPriceUsdPerMillionTokens",
    )
    prompt_price = _decimal(pricing.get("prompt"), "prompt price")
    completion_price = _decimal(pricing.get("completion"), "completion price")
    expected_prompt = Decimal("0.10") if model == LUNA_MODEL else Decimal("1.00")
    expected_completion = Decimal("0.60") if model == LUNA_MODEL else Decimal("6.00")
    if prompt_price != expected_prompt or completion_price != expected_completion:
        raise ValueError("candidate pricing changed")
    if _decimal(max_price.get("prompt"), "maximum prompt price") != expected_prompt:
        raise ValueError("candidate maximum prompt price changed")
    if (
        _decimal(max_price.get("completion"), "maximum completion price")
        != expected_completion
    ):
        raise ValueError("candidate maximum completion price changed")
    snapshot = _json_object(snapshot_path.read_bytes(), "endpoint snapshot")
    _validate_evidence(snapshot, "endpoint snapshot")
    if snapshot.get("requestedModelId") != model:
        raise ValueError("endpoint snapshot model mismatch")
    if snapshot.get("officialWorkloadClassification") != classification:
        raise ValueError("endpoint snapshot workload classification mismatch")
    endpoint = _mapping(snapshot.get("endpoint"), "endpoint")
    if endpoint.get("tag") != "openai" or endpoint.get("providerName") != "OpenAI":
        raise ValueError("endpoint snapshot route mismatch")
    supported = _string_tuple(
        endpoint.get("supportedParameters"), "supportedParameters"
    )
    for required in ("reasoning", "reasoning_effort", "seed", "max_tokens"):
        if required not in supported:
            raise ValueError(f"endpoint lacks required parameter: {required}")
    if "temperature" in supported:
        raise ValueError("frozen GPT-5.6 endpoint unexpectedly supports temperature")
    snapshot_pricing = _mapping(
        endpoint.get("pricingUsdPerToken"), "pricingUsdPerToken"
    )
    if (
        _decimal(snapshot_pricing.get("prompt"), "snapshot prompt price")
        * Decimal(1_000_000)
        != expected_prompt
    ):
        raise ValueError("snapshot prompt pricing mismatch")
    if (
        _decimal(snapshot_pricing.get("completion"), "snapshot completion price")
        * Decimal(1_000_000)
        != expected_completion
    ):
        raise ValueError("snapshot completion pricing mismatch")
    return CandidateConfig(
        model=model,
        official_workload_classification=classification,
        endpoint_snapshot_path=snapshot_path,
        endpoint_snapshot_sha256=snapshot_sha,
        expected_response_models=expected_models,
        prompt_price_usd_per_million=prompt_price,
        completion_price_usd_per_million=completion_price,
        max_prompt_price_usd_per_million=float(expected_prompt),
        max_completion_price_usd_per_million=float(expected_completion),
    )


def _validate_codex_selection(path: Path) -> None:
    result = _json_object(path.read_bytes(), "Codex close-reading result")
    summaries = _mapping(result.get("summaries"), "Codex summaries")
    all20 = _mapping(summaries.get("all20"), "Codex all20 summary")
    by_verdict = _mapping(all20.get("documentsByVerdict"), "documentsByVerdict")
    majors = _string_tuple(by_verdict.get("major"), "Codex major documents")
    minors = _string_tuple(by_verdict.get("minor"), "Codex minor documents")
    if majors != EXPECTED_MAJOR_DOCUMENT_IDS:
        raise ValueError("Codex major-document summary changed")
    if not set(EXPECTED_MINOR_DOCUMENT_IDS).issubset(minors):
        raise ValueError("Codex deterministic minor cases are absent")


def _validate_manual_plan(path: Path) -> None:
    plan = _json_object(path.read_bytes(), "manual selection plan")
    _validate_evidence(plan, "manual selection plan")
    if plan.get("schemaVersion") != "model-canary-manual-selection-plan-v6/1.0":
        raise ValueError("manual selection plan schema changed")
    if plan.get("status") != "locked-before-provider-calls":
        raise ValueError("manual selection plan was not frozen before calls")
    scope = _mapping(plan.get("scope"), "manual plan scope")
    if scope.get("classification") != "development_only":
        raise ValueError("manual plan must be development only")
    if scope.get("confirmatoryClaimAllowed") is not False:
        raise ValueError("manual plan cannot allow confirmatory claims")
    if _string_tuple(scope.get("candidateModels"), "manual candidates") != (
        EXPECTED_CANDIDATES
    ):
        raise ValueError("manual plan candidate set changed")
    if (
        _string_tuple(scope.get("documentIdsInExecutionOrder"), "manual documents")
        != EXPECTED_DOCUMENT_IDS
    ):
        raise ValueError("manual plan document set changed")


def _load_development_sources(
    config: ModelCanaryConfig,
) -> dict[str, dict[str, object]]:
    v4 = _json_object(config.v4_result_path.read_bytes(), "v4 result")
    methods = _mapping_list(v4.get("methods"), "v4 methods")
    if len(methods) != 1:
        raise ValueError("v4 result must contain exactly one method")
    v4_documents = _mapping_list(methods[0].get("documents"), "v4 documents")
    v4_by_id = {
        _text(row.get("documentId"), "v4 documentId"): row for row in v4_documents
    }
    v5 = _json_object(config.v5_result_path.read_bytes(), "v5 result")
    v5_documents = _mapping_list(v5.get("documents"), "v5 documents")
    v5_by_id = {
        _text(row.get("documentId"), "v5 documentId"): row for row in v5_documents
    }
    codex = _json_object(config.codex_result_path.read_bytes(), "Codex result")
    summaries = _mapping(codex.get("summaries"), "Codex summaries")
    all20 = _mapping(summaries.get("all20"), "Codex all20")
    by_verdict = _mapping(all20.get("documentsByVerdict"), "documentsByVerdict")
    verdict_by_id: dict[str, str] = {}
    for verdict in ("pass", "minor", "major"):
        for document_id in _string_tuple(
            by_verdict.get(verdict), f"Codex {verdict} documents"
        ):
            verdict_by_id[document_id] = verdict
    output: dict[str, dict[str, object]] = {}
    for document_id in config.document_ids:
        if document_id not in v4_by_id or document_id not in v5_by_id:
            raise ValueError(f"development source is missing: {document_id}")
        v4_row = v4_by_id[document_id]
        v5_row = v5_by_id[document_id]
        marked = _nonempty_string(v4_row.get("markedInputText"), "v4 markedInputText")
        if (
            _nonempty_string(v5_row.get("markedInputText"), "v5 markedInputText")
            != marked
        ):
            raise ValueError(f"v4/v5 marked source mismatch: {document_id}")
        output[document_id] = {
            "documentId": document_id,
            "markedInputText": marked,
            "qwenOutputText": _nonempty_string(
                v5_row.get("outputText"), "v5 outputText"
            ),
            "qwenDetector": v5_row.get("detector"),
            "qwenFidelity": v5_row.get("fidelity"),
            "qwenFailures": v5_row.get("failures"),
            "qwenDraftCall": v5_row.get("draftCall"),
            "qwenManualVerdict": verdict_by_id.get(document_id),
            "sourceSha256": _text_sha256(marked),
        }
    return output


def _load_detector(config: ModelCanaryConfig) -> tuple[object, object]:
    v4_config = load_verified_paraphrase_config(
        config.v4_config_path,
        root=config.root,
    )
    base_config = load_experiment_config(v4_config.base_config_path, root=config.root)
    corpus = load_reviewed_corpus(base_config)
    return base_config, corpus


def _frozen_request_record(
    config: ModelCanaryConfig,
    candidate: CandidateConfig,
    request: StageRequest,
) -> dict[str, object]:
    return {
        "max_tokens": config.max_tokens,
        "messages": list(request_messages(request)),
        "model": candidate.model,
        "provider": {
            "allow_fallbacks": False,
            "data_collection": "deny",
            "max_price": {
                "completion": candidate.max_completion_price_usd_per_million,
                "prompt": candidate.max_prompt_price_usd_per_million,
            },
            "order": list(config.provider_order),
            "require_parameters": True,
            "zdr": True,
        },
        "reasoning": {"effort": config.reasoning_effort},
        "seed": config.seed,
        "stream": False,
    }


def _request_sha256(
    *,
    call_id: str,
    document_id: str,
    input_text: str,
    request: Mapping[str, object],
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "callId": call_id,
                "documentId": document_id,
                "inputText": input_text,
                "request": dict(request),
            }
        )
    ).hexdigest()


def _opaque_pair_ids(config: ModelCanaryConfig) -> dict[tuple[str, str], str]:
    pairs = [
        (candidate.model, document_id)
        for candidate in config.candidates
        for document_id in config.document_ids
    ]
    ordered = sorted(
        pairs,
        key=lambda pair: hashlib.sha256(
            f"model-canary-v6|{pair[0]}|{pair[1]}".encode()
        ).hexdigest(),
    )
    return {pair: f"MCV6-{index:02d}" for index, pair in enumerate(ordered, 1)}


def _candidate_summaries(
    config: ModelCanaryConfig,
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for candidate in config.candidates:
        candidate_rows = [
            row for row in rows if row.get("candidateModel") == candidate.model
        ]
        calls = [_mapping(row.get("call"), "candidate call") for row in candidate_rows]
        usage = aggregate_call_usage(calls)
        distances = [
            _nonnegative_float(
                _mapping(
                    _mapping(row.get("fidelity"), "fidelity").get("wordLevenshtein"),
                    "wordLevenshtein",
                ).get("normalizedDistance"),
                "normalizedDistance",
            )
            for row in candidate_rows
        ]
        pipeline_failures = sum(
            _nonnegative_int(
                _mapping(row.get("pipeline"), "pipeline").get("failureCount"),
                "pipeline failure count",
            )
            for row in candidate_rows
        )
        total_cost = _decimal(usage.get("providerCostCredits"), "provider cost")
        total_latency = _nonnegative_float(usage.get("latencyMs"), "latencyMs")
        output.append(
            {
                "actualCostCredits": _decimal_text(total_cost),
                "actualCostCreditsPer1000Documents": _decimal_text(
                    total_cost / Decimal(len(candidate_rows)) * Decimal(1000)
                ),
                "candidateModel": candidate.model,
                "documentCount": len(candidate_rows),
                "manualReviewStatus": "pending",
                "meanLatencyMs": round(total_latency / len(candidate_rows), 3),
                "meanNormalizedWordDistance": sum(distances) / len(distances),
                "pipelineFailureCount": pipeline_failures,
                "totalLatencyMs": total_latency,
                "usage": usage,
            }
        )
    return output


def _qwen_reference(
    config: ModelCanaryConfig,
    sources: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    documents: list[dict[str, object]] = []
    calls: list[Mapping[str, object]] = []
    for document_id in config.document_ids:
        source = sources[document_id]
        call = _mapping(source.get("qwenDraftCall"), "Qwen draft call")
        calls.append(call)
        usage = _mapping(
            _mapping(call.get("response"), "Qwen response").get("usage"),
            "Qwen usage",
        )
        documents.append(
            {
                "actualCostCredits": _text(
                    usage.get("providerCostCredits"), "Qwen provider cost"
                ),
                "detector": source.get("qwenDetector"),
                "documentId": document_id,
                "fidelity": source.get("qwenFidelity"),
                "frozenManualVerdict": source.get("qwenManualVerdict"),
                "latencyMs": call.get("latencyMs"),
                "markedInputText": source.get("markedInputText"),
                "outputText": source.get("qwenOutputText"),
                "pipelineFailures": source.get("qwenFailures"),
            }
        )
    return {
        "documents": documents,
        "model": "qwen/qwen3.6-35b-a3b",
        "providerCallsMade": 0,
        "selectable": False,
        "sourceResultSha256": config.v5_result_sha256,
        "usageFromFrozenCalls": aggregate_call_usage(calls),
    }


def _selection_policy(config: ModelCanaryConfig) -> dict[str, object]:
    return {
        "candidatePassesOnlyIf": {
            "maximumMajorMaterialFindings": config.maximum_major_findings,
            "maximumMinorMaterialFindings": config.maximum_minor_findings,
            "maximumPipelineFailures": config.maximum_pipeline_failures,
            "minimumMeanNormalizedWordDistance": (config.minimum_mean_word_distance),
            "requiredReviewedDocumentCount": (config.required_reviewed_document_count),
        },
        "ifBothPass": "lower_actual_provider_cost_then_lower_total_latency",
        "ifNeitherPasses": "stop_without_demo_candidate",
        "qwenReferenceSelectable": False,
    }


def _completion_from_record(record: Mapping[str, object]) -> ChatCompletion:
    response = _mapping(record.get("response"), "checkpoint response")
    usage = _mapping(response.get("usage"), "checkpoint usage")
    metadata = response.get("openrouterMetadata")
    if not isinstance(metadata, Mapping):
        raise ModelCanaryCheckpointError("saved routing metadata is invalid")
    return ChatCompletion(
        content=_nonempty_string(response.get("content"), "saved content"),
        finish_reason=_text(response.get("finishReason"), "saved finishReason"),
        model=_text(response.get("model"), "saved model"),
        openrouter_metadata=metadata,
        provider=_text(response.get("provider"), "saved provider"),
        response_id=_text(response.get("id"), "saved response id"),
        system_fingerprint=(
            None
            if response.get("systemFingerprint") is None
            else _text(response.get("systemFingerprint"), "systemFingerprint")
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


def _routing_cost(
    prompt_tokens: int,
    completion_tokens: int,
    candidate: CandidateConfig,
) -> Decimal:
    return (
        Decimal(prompt_tokens) * candidate.prompt_price_usd_per_million
        + Decimal(completion_tokens) * candidate.completion_price_usd_per_million
    ) / Decimal(1_000_000)


def _best_effort_restore(text: str, tokens: Sequence[object]) -> str:
    restored = text
    for token in tokens:
        placeholder = getattr(token, "placeholder")
        original = getattr(token, "original")
        restored = restored.replace(placeholder, original)
    return restored


def _validate_artifact_binding(
    config: ModelCanaryConfig,
    artifact: Mapping[str, object],
) -> None:
    if artifact.get("configSha256") != config.sha256:
        raise ValueError("artifact config binding mismatch")
    if artifact.get("parityFixtureSha256") != config.parity_fixture_sha256:
        raise ValueError("artifact parity binding mismatch")
    if artifact.get("manualSelectionPlanSha256") != config.manual_plan_sha256:
        raise ValueError("artifact manual-plan binding mismatch")
    expected_hash = _object_sha256(
        {key: value for key, value in artifact.items() if key != "artifactSha256"}
    )
    if artifact.get("artifactSha256") != expected_hash:
        raise ValueError("artifact self-hash mismatch")


def _validate_evidence(value: Mapping[str, object], label: str) -> None:
    _text(value.get("verifiedAt"), f"{label}.verifiedAt")
    methodology = value.get("methodology")
    if isinstance(methodology, Mapping):
        if not methodology:
            raise ValueError(f"{label}.methodology must be nonempty")
    else:
        _text(methodology, f"{label}.methodology")
    _sources(value.get("sources"))


def _sources(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("sources must be a nonempty list")
    output: list[dict[str, str]] = []
    for item in value:
        source = _mapping(item, "source")
        title = _text(source.get("title"), "source title")
        url = _text(source.get("url"), "source URL")
        output.append({"title": title, "url": url})
    return output


def _safe_path(root: Path, value: object, label: str) -> Path:
    relative = Path(_text(value, label))
    if relative.is_absolute():
        raise ValueError(f"{label} must be relative")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes repository root") from error
    return resolved


def _require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} file is missing: {path}")
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
    return list(value)


def _json_object(content: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty string list")
    output = tuple(_text(item, label) for item in value)
    if len(output) != len(set(output)):
        raise ValueError(f"{label} contains duplicates")
    return output


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _nonnegative_float(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a nonnegative number")
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


def _required_budget(value: Decimal) -> Decimal:
    parsed = _decimal(value, "provider budget")
    if parsed <= 0:
        raise ValueError("provider budget must be positive")
    return parsed


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _first_present(value: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        if key in value:
            return value[key]
    return None


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _object_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


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
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--print-parity", action="store_true")
    mode.add_argument("--blind-packet", action="store_true")
    mode.add_argument("--finalize-review", action="store_true")
    parser.add_argument(
        "--config",
        default=str(root / "fixtures" / "model-canary-config-v6.json"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(root / "results" / "model-canary-checkpoint-v6.json"),
    )
    parser.add_argument(
        "--input",
        default=str(root / "results" / "model-canary-raw-v6.json"),
    )
    parser.add_argument(
        "--output",
        default=str(root / "results" / "model-canary-raw-v6.json"),
    )
    parser.add_argument("--review")
    parser.add_argument("--max-provider-cost-credits")
    parser.add_argument("--max-new-calls", type=int)
    parser.add_argument("--confirm-not-charged-call-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_model_canary_config(
        args.config,
        validate_parity=not args.print_parity,
    )
    if args.print_parity:
        print(canonical_json_bytes(build_model_canary_parity(config)).decode(), end="")
        return 0
    if args.dry_run:
        if any(
            value is not None
            for value in (
                args.max_provider_cost_credits,
                args.max_new_calls,
                args.confirm_not_charged_call_id,
            )
        ):
            raise SystemExit("budget and checkpoint controls are live-only")
        print(canonical_json_bytes(build_model_canary_dry_run(config)).decode(), end="")
        return 0
    if args.blind_packet:
        artifact = _json_object(Path(args.input).read_bytes(), "canary artifact")
        packet = build_blind_review_packet(config, artifact)
        _atomic_write(Path(args.output), canonical_json_bytes(packet))
        return 0
    if args.finalize_review:
        if args.review is None:
            raise SystemExit("--finalize-review requires --review")
        artifact = _json_object(Path(args.input).read_bytes(), "canary artifact")
        review = _json_object(Path(args.review).read_bytes(), "manual review")
        finalized = finalize_model_canary_review(config, artifact, review)
        _atomic_write(Path(args.output), canonical_json_bytes(finalized))
        return 0
    if args.max_provider_cost_credits is None:
        raise SystemExit("--live requires --max-provider-cost-credits")
    clients = {
        candidate.model: OpenRouterClient.from_env(
            timeout=config.timeout_seconds,
            provider_order=config.provider_order,
            allow_fallbacks=False,
            require_parameters=True,
            reasoning_effort=config.reasoning_effort,
            temperature=None,
            max_tokens=config.max_tokens,
            seed=config.seed,
            max_prompt_price=candidate.max_prompt_price_usd_per_million,
            max_completion_price=candidate.max_completion_price_usd_per_million,
        )
        for candidate in config.candidates
    }
    try:
        artifact = run_model_canary_live(
            config,
            clients=clients,
            max_provider_cost_credits=_decimal(
                args.max_provider_cost_credits, "provider budget"
            ),
            checkpoint_path=args.checkpoint,
            max_new_calls=args.max_new_calls,
            confirm_not_charged_call_id=args.confirm_not_charged_call_id,
        )
    except ModelCanaryCallLimitReached as pause:
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
    content = canonical_json_bytes(artifact)
    _atomic_write(Path(args.output), content)
    print(
        json.dumps(
            {
                "artifactSha256": hashlib.sha256(content).hexdigest(),
                "callCount": artifact["usage"]["callCount"],
                "output": str(Path(args.output).resolve()),
                "providerCostCredits": artifact["usage"]["providerCostCredits"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
