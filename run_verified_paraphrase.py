"""Run the frozen two-pass verified-paraphrase follow-up experiment.

Dry-run reads only frozen local evidence. Live mode makes exactly two calls per
document on one pinned OpenRouter endpoint, checkpoints every paid response,
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
    build_fidelity_repair_prompt,
    build_paraphrase_prompt,
    canonicalize_placeholders,
    json_safe_value,
    protect_tokens,
    result_validation_issues,
    restore_tokens,
)
from watermark_toy import encode_text, score_text


CONFIG_SCHEMA_VERSION = 2
CHECKPOINT_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 2
EXPECTED_METHOD_ID = "paraphrase-verified"
EXPECTED_CALL_GRAPH = ("paraphrase-draft", "fidelity-repair")


class VerifiedExperimentError(Exception):
    """Base class for expected follow-up failures."""


class VerifiedBudgetError(VerifiedExperimentError):
    """Raised before a paid call when the explicit budget is insufficient."""


class VerifiedCheckpointError(VerifiedExperimentError):
    """Raised when a checkpoint cannot be resumed safely."""


class VerifiedResponseContractError(VerifiedExperimentError):
    """Raised after preserving a paid response that violates the frozen route."""


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
    def complete(self, prompt: str, *, model: str) -> ChatCompletion: ...


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
    seed: int
    timeout_seconds: float
    prompt_price_usd_per_million: Decimal
    completion_price_usd_per_million: Decimal
    prompt_token_overhead_reserve: int
    bootstrap_replicates: int
    bootstrap_seed: int
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
    always_run_repair = transform.get("alwaysRunRepair")
    if (
        method_id != EXPECTED_METHOD_ID
        or transform.get("method") != EXPECTED_METHOD_ID
        or call_graph != EXPECTED_CALL_GRAPH
        or transform.get("repairPolicy") != "always_once"
        or always_run_repair is not True
        or transform.get("repairGrounding") != ["masked_source", "draft"]
    ):
        raise ValueError("verified paraphrase must freeze the exact two-pass call graph")

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
        raise ValueError("verified paraphrase must deny data collection and require ZDR")
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
    seed = _nonnegative_int(provider.get("seed"), "provider.seed")
    timeout_seconds = _positive_number(
        provider.get("timeoutSeconds"), "provider.timeoutSeconds"
    )
    prices = _mapping(
        provider.get("pricingUsdPerMillionTokens"),
        "provider.pricingUsdPerMillionTokens",
    )
    prompt_price = _decimal(prices.get("prompt"), "provider prompt price")
    completion_price = _decimal(
        prices.get("completion"), "provider completion price"
    )
    max_prices = _mapping(
        provider.get("maxPriceUsdPerMillionTokens"),
        "provider.maxPriceUsdPerMillionTokens",
    )
    if (
        _decimal(max_prices.get("prompt"), "provider maximum prompt price")
        != prompt_price
        or _decimal(
            max_prices.get("completion"), "provider maximum completion price"
        )
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
    if _positive_int(endpoint.get("maxCompletionTokens"), "endpoint maximum") < max_tokens:
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
    if analysis.get("primaryScoringUnit") != "pooled_corpus":
        raise ValueError("analysis scoring unit must be pooled_corpus")
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

    decision = _mapping(raw.get("decisionPolicy"), "decisionPolicy")
    success_target = decision.get("successTarget")
    retuning = decision.get("retuningAfterResults")
    if success_target is not None or retuning is not False:
        raise ValueError("v2 cannot freeze a success target or retune after results")
    if decision.get("publishAllOutputs") is not True:
        raise ValueError("v2 must publish every output")

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
        seed=seed,
        timeout_seconds=timeout_seconds,
        prompt_price_usd_per_million=prompt_price,
        completion_price_usd_per_million=completion_price,
        prompt_token_overhead_reserve=overhead,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        separate_final_audit=separate_final_audit,
        final_audit_model=final_audit_model,
        success_target=success_target,
        retuning_after_results=retuning,
    )


def expected_verified_call_ids(
    config: VerifiedParaphraseConfig,
) -> tuple[str, ...]:
    """Return the exact 40-call alternating matrix."""
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
    draft_completion_estimate = config.max_tokens * 4
    for document_id in EXPECTED_DOCUMENT_IDS:
        protected = protect_tokens(marked[document_id])
        draft_prompt = build_paraphrase_prompt(protected.masked)
        repair_prompt = build_fidelity_repair_prompt(
            protected.masked,
            "x" * draft_completion_estimate,
        )
        prompt_estimate += len(draft_prompt.encode("utf-8"))
        prompt_estimate += len(repair_prompt.encode("utf-8"))
    call_count = config.document_count * len(config.call_graph)
    prompt_estimate += call_count * config.prompt_token_overhead_reserve
    completion_estimate = call_count * config.max_tokens
    return {
        "callCount": call_count,
        "callsByStage": {
            stage: config.document_count for stage in sorted(config.call_graph)
        },
        "configSha256": config.sha256,
        "documentCount": config.document_count,
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
    """Run or resume the frozen two-pass matrix and return its raw artifact."""
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
        draft_prompt = build_paraphrase_prompt(protected.masked)
        draft, draft_record = manager.complete(
            call_id=f"{document.document_id}:{config.method_id}:paraphrase-draft",
            document_id=document.document_id,
            stage="paraphrase-draft",
            input_text=protected.masked,
            prompt=draft_prompt,
        )
        if draft.finish_reason != "stop":
            issues.append(
                {
                    "code": "finish_reason_contract",
                    "message": f"draft finish reason was {draft.finish_reason!r}",
                    "stage": "paraphrase-draft",
                }
            )

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
                "calls": [draft_record, repair_record],
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
                    "rawFinalMaskedText": repair.content,
                    "restorationMode": restoration_mode,
                    "status": "validation_failure" if issues else "accepted",
                },
            }
        )

    method = {
        "aggregate": _aggregate_method(rows, scores),
        "documents": rows,
        "method": config.method_id,
        "methodId": config.method_id,
        "pivot": None,
    }
    artifact = {
        "baseExperimentConfigSha256": config.base_config_sha256,
        "baseExperimentResultSha256": config.base_result_sha256,
        "configSha256": config.sha256,
        "documentCount": config.document_count,
        "endpointSnapshotSha256": config.endpoint_snapshot_sha256,
        "experimentVersion": config.experiment_version,
        "finalAudit": {
            "model": config.final_audit_model,
            "required": True,
            "separateFromRepair": config.separate_final_audit,
            "status": "pending_frozen_transform_output",
        },
        "methodology": config.methodology,
        "methods": [method],
        "schemaVersion": ARTIFACT_SCHEMA_VERSION,
        "sources": list(config.sources),
        "usage": {
            **aggregate_call_usage(manager.calls),
            "providerCostBudgetCredits": _decimal_text(budget),
        },
        "verifiedAt": config.verified_at,
    }
    canonical_json_bytes(artifact)
    return artifact


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
                raise VerifiedCheckpointError("in-flight call follows a complete matrix")
            if tombstone.get("callId") != expected_call_ids[len(self.calls)]:
                raise VerifiedCheckpointError("in-flight call is not the next matrix call")

    def complete(
        self,
        *,
        call_id: str,
        document_id: str,
        stage: str,
        input_text: str,
        prompt: str,
    ) -> tuple[ChatCompletion, dict[str, object]]:
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
            )
            completion = _completion_from_record(record)
            self._validate_response(completion, prompt)
            return completion, record
        if expected_index != len(self.calls):
            raise VerifiedCheckpointError("attempted to skip an uncheckpointed call")
        if self.max_new_calls is not None and self.new_calls >= self.max_new_calls:
            raise VerifiedCallLimitReached(
                completed_calls=len(self.calls), new_calls=self.new_calls
            )

        spent = _checkpoint_cost(self.calls)
        reserve = _routing_cost(
            len(prompt.encode("utf-8")) + self.config.prompt_token_overhead_reserve,
            self.config.max_tokens,
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
            completion = self.client.complete(prompt, model=self.config.model)
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
                "prompt": prompt,
                "providerError": str(error),
                "rawResponse": raw,
                "recordStatus": "provider_response_invalid",
                "request": {"model": self.config.model},
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
            "prompt": prompt,
            "recordStatus": "accepted_response",
            "request": {"model": self.config.model},
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
        self.calls.append(record)
        self.state["inFlightCall"] = None
        self._save()
        self.new_calls += 1
        self._validate_response(completion, prompt)
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
        prompt: str,
    ) -> None:
        expected = {
            "callId": call_id,
            "documentId": document_id,
            "inputText": input_text,
            "methodId": self.config.method_id,
            "prompt": prompt,
            "stage": stage,
        }
        for field, value in expected.items():
            if record.get(field) != value:
                raise VerifiedCheckpointError(f"saved request mismatch: {field}")
        request = _mapping(record.get("request"), "saved request")
        if request.get("model") != self.config.model:
            raise VerifiedCheckpointError("saved request model mismatch")

    def _validate_response(self, completion: ChatCompletion, prompt: str) -> None:
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
            raise VerifiedResponseContractError("OpenRouter routing metadata is required")
        if metadata.get("strategy") != "direct" or metadata.get("attempt") != 1:
            raise VerifiedResponseContractError("OpenRouter must use one direct attempt")
        if metadata.get("pipeline") not in (None, []):
            raise VerifiedResponseContractError("OpenRouter routing pipeline must be empty")
        endpoints = _mapping(metadata.get("endpoints"), "router endpoints")
        available = endpoints.get("available")
        if not isinstance(available, list) or any(
            not isinstance(item, Mapping) for item in available
        ):
            raise VerifiedResponseContractError("router endpoint metadata is invalid")
        selected = [item for item in available if item.get("selected") is True]
        if len(selected) != 1:
            raise VerifiedResponseContractError("router must select exactly one endpoint")
        provider = _first_present(selected[0], "provider", "provider_name", "providerName")
        model = _first_present(selected[0], "model", "model_id", "modelId")
        if provider != completion.provider:
            raise VerifiedResponseContractError("router selected provider mismatch")
        if model not in self.config.expected_response_models:
            raise VerifiedResponseContractError("router selected model mismatch")
        attempts = metadata.get("attempts")
        if attempts is not None:
            if not isinstance(attempts, list) or len(attempts) != 1:
                raise VerifiedResponseContractError("router attempts must contain one item")
            attempt = _mapping(attempts[0], "router attempt")
            if _first_present(
                attempt, "provider", "provider_name", "providerName"
            ) != completion.provider:
                raise VerifiedResponseContractError("router attempt provider mismatch")
            if _first_present(attempt, "model", "model_id", "modelId") not in (
                self.config.expected_response_models
            ):
                raise VerifiedResponseContractError("router attempt model mismatch")
            if attempt.get("status") != 200 or isinstance(attempt.get("status"), bool):
                raise VerifiedResponseContractError("router attempt status must be 200")
        if completion.usage.prompt_tokens > (
            len(prompt.encode("utf-8")) + self.config.prompt_token_overhead_reserve
        ):
            raise VerifiedResponseContractError("prompt usage exceeds frozen reserve")
        if completion.usage.completion_tokens > self.config.max_tokens:
            raise VerifiedResponseContractError("completion usage exceeds maxTokens")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.path, canonical_json_bytes(self.state))


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
    prompt: str,
    model: str,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "callId": call_id,
                "documentId": document_id,
                "inputSha256": hashlib.sha256(input_text.encode()).hexdigest(),
                "model": model,
                "promptSha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "stage": stage,
            }
        )
    ).hexdigest()


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
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
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
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
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
