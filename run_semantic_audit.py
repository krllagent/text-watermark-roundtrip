"""Run the frozen blinded semantic audit for the transformation artifact."""

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
from typing import Mapping, Protocol, Sequence

from corpus_contract import canonical_json_bytes
from text_contract import find_protected_spans
from unmark import ChatCompletion, OpenRouterClient


CONFIG_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 1
JUDGE_FIELDS = (
    "lostClaims",
    "addedClaims",
    "changedClaims",
    "lostOrChangedExamples",
    "stanceDrift",
    "certaintyDrift",
    "caveatDrift",
    "paragraphRoleOrOrderDrift",
)
VOICE_LEVELS = ("none", "minor", "material")
DEFAULT_METHOD_IDS = ("synonyms", "roundtrip-de", "roundtrip-zh", "paraphrase")


class AuditError(Exception):
    """Base class for semantic-audit failures."""


class AuditBudgetError(AuditError):
    """Raised before a call when the explicit audit budget is insufficient."""


class AuditCheckpointError(AuditError):
    """Raised when a checkpoint cannot be resumed safely."""


class AuditResponseContractError(AuditError):
    """Raised after preserving a response that violates the audit contract."""


class AuditBatchLimitReached(AuditError):
    """Intentional pause after a requested number of new audit batches."""

    def __init__(self, *, completed_batches: int, new_batches: int) -> None:
        super().__init__(
            f"paused after {new_batches} new batch(es); "
            f"{completed_batches} total batch(es) checkpointed"
        )
        self.completed_batches = completed_batches
        self.new_batches = new_batches


class AuditCompletionClient(Protocol):
    def complete(self, prompt: str, *, model: str) -> ChatCompletion: ...


@dataclass(frozen=True)
class AuditConfig:
    path: Path
    root: Path
    raw: dict[str, object]
    sha256: str
    verified_at: str
    methodology: str
    sources: tuple[dict[str, str], ...]
    source_path: Path
    source_sha256: str
    plan_path: Path
    plan_sha256: str
    endpoint_snapshot_path: Path
    endpoint_snapshot_sha256: str
    batch_size: int
    structured_pair_count: int
    method_ids: tuple[str, ...]
    pair_order_seed: str
    model: str
    provider_order: tuple[str, ...]
    expected_response_models: tuple[str, ...]
    expected_response_providers: tuple[str, ...]
    max_tokens: int
    reasoning_effort: str
    prompt_price_usd_per_million: Decimal
    completion_price_usd_per_million: Decimal
    prompt_token_overhead_reserve: int
    seed: int
    timeout_seconds: float


@dataclass(frozen=True)
class AuditPair:
    pair_id: str
    method_id: str
    document_id: str
    source_text: str
    candidate_text: str

    def blinded_dict(self) -> dict[str, str]:
        return {
            "candidateText": self.candidate_text,
            "pairId": self.pair_id,
            "sourceText": self.source_text,
        }


def load_audit_config(path: str | Path) -> AuditConfig:
    config_path = Path(path).resolve()
    root = Path(__file__).resolve().parent
    raw_bytes = config_path.read_bytes()
    raw = _json_object(raw_bytes, "semantic audit config")
    _validate_evidence(raw, "semantic audit config")
    if raw.get("schemaVersion") != CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported semantic audit config schemaVersion")

    source = _mapping(raw.get("sourceArtifact"), "sourceArtifact")
    source_path = _safe_path(root, source.get("path"), "sourceArtifact.path")
    source_sha256 = _sha256_text(source.get("sha256"), "sourceArtifact.sha256")
    _require_sha(source_path, source_sha256, "source artifact")

    audit = _mapping(raw.get("audit"), "audit")
    batch_size = _positive_int(audit.get("batchSize"), "audit.batchSize")
    structured_pair_count = _positive_int(
        audit.get("structuredPairCount"), "audit.structuredPairCount"
    )
    method_ids = _string_tuple(
        audit.get("methodIds", list(DEFAULT_METHOD_IDS)),
        "audit.methodIds",
    )
    if len(method_ids) != len(set(method_ids)):
        raise ValueError("audit.methodIds must not contain duplicates")
    if structured_pair_count != 20 * len(method_ids):
        raise ValueError(
            "audit.structuredPairCount must equal 20 documents per method"
        )
    if structured_pair_count % batch_size:
        raise ValueError("audit batch size must divide the structured pair count")
    pair_order_seed = _text(audit.get("pairOrderSeed"), "audit.pairOrderSeed")
    plan_path = _safe_path(
        root, audit.get("closeReadingPlanPath"), "audit.closeReadingPlanPath"
    )
    plan_sha256 = _sha256_text(
        audit.get("closeReadingPlanSha256"), "audit.closeReadingPlanSha256"
    )
    _require_sha(plan_path, plan_sha256, "semantic audit plan")

    billing = _mapping(raw.get("billing"), "billing")
    if billing.get("creditBaseCurrency") != "USD":
        raise ValueError("audit billing currency must be USD")
    if billing.get("creditUsdBaseUnit") != "1":
        raise ValueError("audit billing credit base unit must be one USD")
    overhead = _positive_int(
        billing.get("promptTokenOverheadReserve"),
        "billing.promptTokenOverheadReserve",
    )

    judge = _mapping(raw.get("judge"), "judge")
    model = _text(judge.get("model"), "judge.model")
    provider_order = _string_tuple(judge.get("providerOrder"), "judge.providerOrder")
    expected_models = _string_tuple(
        judge.get("expectedResponseModels"), "judge.expectedResponseModels"
    )
    expected_providers = _string_tuple(
        judge.get("expectedResponseProviders"), "judge.expectedResponseProviders"
    )
    if judge.get("allowFallbacks") is not False:
        raise ValueError("audit judge fallbacks must be disabled")
    if judge.get("requireParameters") is not True:
        raise ValueError("audit judge must require parameters")
    if judge.get("dataCollection") != "deny" or judge.get("zdr") is not True:
        raise ValueError("audit judge must deny data collection and require ZDR")
    if judge.get("reasoningEffort") != "low" or judge.get("temperature") is not None:
        raise ValueError("audit judge reasoning and temperature are frozen")
    max_tokens = _positive_int(judge.get("maxTokens"), "judge.maxTokens")
    seed = _nonnegative_int(judge.get("seed"), "judge.seed")
    timeout_seconds = _positive_number(
        judge.get("timeoutSeconds"), "judge.timeoutSeconds"
    )
    prices = _mapping(
        judge.get("maxPriceUsdPerMillionTokens"),
        "judge.maxPriceUsdPerMillionTokens",
    )
    prompt_price = _decimal(prices.get("prompt"), "judge prompt price")
    completion_price = _decimal(prices.get("completion"), "judge completion price")

    endpoint_path = _safe_path(
        root,
        judge.get("endpointSnapshotPath"),
        "judge.endpointSnapshotPath",
    )
    endpoint_sha256 = _sha256_text(
        judge.get("endpointSnapshotSha256"),
        "judge.endpointSnapshotSha256",
    )
    _require_sha(endpoint_path, endpoint_sha256, "judge endpoint snapshot")
    endpoint_snapshot = _json_object(endpoint_path.read_bytes(), "judge endpoint snapshot")
    _validate_evidence(endpoint_snapshot, "judge endpoint snapshot")
    endpoint = _mapping(endpoint_snapshot.get("endpoint"), "judge endpoint")
    if endpoint_snapshot.get("requestedModelId") != model:
        raise ValueError("judge endpoint model binding mismatch")
    if endpoint.get("tag") not in provider_order:
        raise ValueError("judge endpoint provider tag mismatch")
    if endpoint.get("providerName") not in expected_providers:
        raise ValueError("judge endpoint provider name mismatch")
    endpoint_prices = _mapping(
        endpoint.get("pricingUsdPerToken"), "judge endpoint pricing"
    )
    if _decimal(endpoint_prices.get("prompt"), "endpoint prompt price") * 1_000_000 != prompt_price:
        raise ValueError("judge endpoint prompt price mismatch")
    if _decimal(endpoint_prices.get("completion"), "endpoint completion price") * 1_000_000 != completion_price:
        raise ValueError("judge endpoint completion price mismatch")

    return AuditConfig(
        path=config_path,
        root=root,
        raw=raw,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        verified_at=_text(raw.get("verifiedAt"), "verifiedAt"),
        methodology=_text(raw.get("methodology"), "methodology"),
        sources=tuple(_sources(raw.get("sources"))),
        source_path=source_path,
        source_sha256=source_sha256,
        plan_path=plan_path,
        plan_sha256=plan_sha256,
        endpoint_snapshot_path=endpoint_path,
        endpoint_snapshot_sha256=endpoint_sha256,
        batch_size=batch_size,
        structured_pair_count=structured_pair_count,
        method_ids=method_ids,
        pair_order_seed=pair_order_seed,
        model=model,
        provider_order=provider_order,
        expected_response_models=expected_models,
        expected_response_providers=expected_providers,
        max_tokens=max_tokens,
        reasoning_effort="low",
        prompt_price_usd_per_million=prompt_price,
        completion_price_usd_per_million=completion_price,
        prompt_token_overhead_reserve=overhead,
        seed=seed,
        timeout_seconds=timeout_seconds,
    )


def load_audit_source(config: AuditConfig) -> dict[str, object]:
    _require_sha(config.source_path, config.source_sha256, "source artifact")
    source = _json_object(config.source_path.read_bytes(), "source artifact")
    methods = source.get("methods")
    if not isinstance(methods, list):
        raise ValueError("source artifact methods must be a list")
    return source


def build_blinded_pairs(
    config: AuditConfig, source: Mapping[str, object]
) -> tuple[AuditPair, ...]:
    methods = source.get("methods")
    if not isinstance(methods, list):
        raise ValueError("source artifact methods must be a list")
    by_method: dict[str, Mapping[str, object]] = {}
    for raw_method in methods:
        method = _mapping(raw_method, "source method")
        method_id = _text(method.get("methodId"), "source methodId")
        if method_id in by_method:
            raise ValueError("source artifact contains a duplicate methodId")
        by_method[method_id] = method
    if any(method_id not in by_method for method_id in config.method_ids):
        raise ValueError("source artifact is missing an audit method")

    pairs: list[AuditPair] = []
    for method_id in config.method_ids:
        documents = by_method[method_id].get("documents")
        if not isinstance(documents, list) or len(documents) != 20:
            raise ValueError(f"{method_id} must contain exactly 20 documents")
        seen_documents: set[str] = set()
        for raw_document in documents:
            document = _mapping(raw_document, "source document")
            document_id = _text(document.get("documentId"), "documentId")
            if document_id in seen_documents:
                raise ValueError("duplicate document ID within audit method")
            seen_documents.add(document_id)
            identity = f"{config.pair_order_seed}\0identity\0{method_id}\0{document_id}"
            pair_id = "pair-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
            pairs.append(
                AuditPair(
                    pair_id=pair_id,
                    method_id=method_id,
                    document_id=document_id,
                    source_text=_text(document.get("markedInputText"), "markedInputText"),
                    candidate_text=_text(document.get("outputText"), "outputText"),
                )
            )
    if len(pairs) != config.structured_pair_count:
        raise ValueError("audit pair count differs from the frozen config")
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError("opaque audit pair IDs collided")
    return tuple(
        sorted(
            pairs,
            key=lambda pair: hashlib.sha256(
                f"{config.pair_order_seed}\0order\0{pair.pair_id}".encode()
            ).digest(),
        )
    )


def build_audit_batches(
    pairs: Sequence[AuditPair], batch_size: int
) -> tuple[tuple[AuditPair, ...], ...]:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    return tuple(
        tuple(pairs[index : index + batch_size])
        for index in range(0, len(pairs), batch_size)
    )


def build_audit_prompt(batch: Sequence[AuditPair]) -> str:
    if not batch:
        raise ValueError("audit batch must not be empty")
    payload = [pair.blinded_dict() for pair in batch]
    fields = ", ".join(JUDGE_FIELDS)
    return (
        "You are a blinded fidelity reviewer. Compare each candidate with its source. "
        "Do not guess how the candidate was produced. Judge meaning and voice, not style "
        "preference. Return one JSON object and no markdown. The object must contain only "
        "a reviews array with one object per pair. Each review must contain pairId; boolean "
        f"fields {fields}; voiceDrift set to none, minor, or material; and evidenceNotes as "
        "a list of short, concrete observations. Set a boolean true only when the source "
        "and candidate provide direct evidence. A wording change alone is not a claim "
        "change. Treat numbers, named entities, examples, qualifications, negation, scope, "
        "and causal direction carefully. Keep the input pair IDs exactly and return every "
        "pair once.\n\nPAIRS_JSON\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def audit_response_format(batch_size: int) -> dict[str, object]:
    properties: dict[str, object] = {
        "pairId": {"type": "string"},
        "voiceDrift": {"enum": list(VOICE_LEVELS), "type": "string"},
        "evidenceNotes": {
            "items": {"maxLength": 500, "minLength": 1, "type": "string"},
            "maxItems": 8,
            "type": "array",
        },
    }
    properties.update({field: {"type": "boolean"} for field in JUDGE_FIELDS})
    return {
        "json_schema": {
            "name": "semantic_audit_batch",
            "schema": {
                "additionalProperties": False,
                "properties": {
                    "reviews": {
                        "items": {
                            "additionalProperties": False,
                            "properties": properties,
                            "required": [
                                "pairId",
                                *JUDGE_FIELDS,
                                "voiceDrift",
                                "evidenceNotes",
                            ],
                            "type": "object",
                        },
                        "maxItems": batch_size,
                        "minItems": batch_size,
                        "type": "array",
                    }
                },
                "required": ["reviews"],
                "type": "object",
            },
            "strict": True,
        },
        "type": "json_schema",
    }


def parse_audit_response(
    content: str, expected_pair_ids: Sequence[str]
) -> list[dict[str, object]]:
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as error:
        raise AuditResponseContractError("audit response must be valid JSON") from error
    if not isinstance(raw, dict) or set(raw) != {"reviews"}:
        raise AuditResponseContractError("audit response must contain only reviews")
    reviews = raw.get("reviews")
    if not isinstance(reviews, list):
        raise AuditResponseContractError("audit reviews must be a list")
    expected = tuple(expected_pair_ids)
    parsed: dict[str, dict[str, object]] = {}
    exact_fields = {
        "pairId",
        *JUDGE_FIELDS,
        "voiceDrift",
        "evidenceNotes",
    }
    for raw_review in reviews:
        if not isinstance(raw_review, dict) or set(raw_review) != exact_fields:
            raise AuditResponseContractError("audit review fields do not match the contract")
        pair_id = raw_review.get("pairId")
        if not isinstance(pair_id, str) or pair_id not in expected or pair_id in parsed:
            raise AuditResponseContractError("audit response pair IDs do not match")
        for field in JUDGE_FIELDS:
            if not isinstance(raw_review.get(field), bool):
                raise AuditResponseContractError(f"audit {field} must be boolean")
        voice = raw_review.get("voiceDrift")
        if voice not in VOICE_LEVELS:
            raise AuditResponseContractError("audit voiceDrift is invalid")
        notes = raw_review.get("evidenceNotes")
        if not isinstance(notes, list) or any(
            not isinstance(note, str) or not note.strip() or len(note) > 500
            for note in notes
        ):
            raise AuditResponseContractError("audit evidenceNotes are invalid")
        parsed[pair_id] = dict(raw_review)
    if set(parsed) != set(expected):
        raise AuditResponseContractError("audit response pair IDs do not match")
    return [parsed[pair_id] for pair_id in expected]


def protected_token_failure(source: str, candidate: str) -> dict[str, object]:
    expected = [source[span.start : span.end] for span in find_protected_spans(source)]
    observed = [
        candidate[span.start : span.end] for span in find_protected_spans(candidate)
    ]
    matched_expected = _ordered_lcs_expected_indices(expected, observed)
    missing = [
        token for index, token in enumerate(expected) if index not in matched_expected
    ]
    return {
        "expectedCount": len(expected),
        "failed": bool(missing),
        "missing": missing,
        "observedCount": len(observed),
    }


def _ordered_lcs_expected_indices(
    expected: Sequence[str],
    observed: Sequence[str],
) -> frozenset[int]:
    """Return expected-token indices in a stable longest ordered match."""
    rows = len(expected) + 1
    columns = len(observed) + 1
    lengths = [[0] * columns for _ in range(rows)]
    for left in range(len(expected) - 1, -1, -1):
        for right in range(len(observed) - 1, -1, -1):
            if expected[left] == observed[right]:
                lengths[left][right] = 1 + lengths[left + 1][right + 1]
            else:
                lengths[left][right] = max(
                    lengths[left + 1][right],
                    lengths[left][right + 1],
                )

    matched: set[int] = set()
    left = 0
    right = 0
    while left < len(expected) and right < len(observed):
        if expected[left] == observed[right]:
            matched.add(left)
            left += 1
            right += 1
        elif lengths[left + 1][right] >= lengths[left][right + 1]:
            left += 1
        else:
            right += 1
    return frozenset(matched)


def run_audit(
    config: AuditConfig,
    source: Mapping[str, object],
    *,
    client: AuditCompletionClient,
    max_provider_cost_credits: Decimal,
    checkpoint_path: str | Path,
    max_new_batches: int | None = None,
    confirm_not_charged_batch_id: str | None = None,
) -> dict[str, object]:
    budget = _required_budget(max_provider_cost_credits)
    if max_new_batches is not None and (
        not isinstance(max_new_batches, int)
        or isinstance(max_new_batches, bool)
        or max_new_batches < 0
    ):
        raise ValueError("max_new_batches must be a nonnegative integer or null")
    pairs = build_blinded_pairs(config, source)
    batches = build_audit_batches(pairs, config.batch_size)
    expected_batch_ids = tuple(
        _batch_id(index, batch) for index, batch in enumerate(batches, start=1)
    )
    path = Path(checkpoint_path)
    state = _load_or_create_checkpoint(
        path,
        config=config,
        expected_batch_ids=expected_batch_ids,
    )
    calls = state.get("calls")
    if not isinstance(calls, list):
        raise AuditCheckpointError("audit checkpoint calls must be a list")
    call_ids = tuple(
        call.get("batchId") if isinstance(call, Mapping) else None for call in calls
    )
    if call_ids != expected_batch_ids[: len(call_ids)]:
        raise AuditCheckpointError("audit checkpoint is not an exact batch prefix")
    in_flight = state.get("inFlightCall")
    if in_flight is None:
        if confirm_not_charged_batch_id is not None:
            raise AuditCheckpointError("no in-flight audit batch exists to resolve")
    else:
        tombstone = _mapping(in_flight, "inFlightCall")
        if tombstone.get("batchId") != confirm_not_charged_batch_id:
            raise AuditCheckpointError(
                "audit in-flight charge is unknown; confirm its exact batch ID only "
                "after checking provider usage"
            )
        if len(calls) >= len(expected_batch_ids):
            raise AuditCheckpointError("in-flight audit batch follows a complete checkpoint")
        if tombstone.get("batchId") != expected_batch_ids[len(calls)]:
            raise AuditCheckpointError("in-flight audit batch is not the next batch")

    new_batches = 0
    reviews_by_pair: dict[str, dict[str, object]] = {}
    for index, batch in enumerate(batches):
        batch_id = expected_batch_ids[index]
        pair_ids = tuple(pair.pair_id for pair in batch)
        prompt = build_audit_prompt(batch)
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        if index < len(calls):
            record = _mapping(calls[index], "audit checkpoint call")
            _validate_saved_request(record, batch_id, pair_ids, prompt_sha256)
            batch_reviews = _validate_audit_record(
                record,
                config=config,
                prompt=prompt,
                expected_pair_ids=pair_ids,
            )
        else:
            if max_new_batches is not None and new_batches >= max_new_batches:
                raise AuditBatchLimitReached(
                    completed_batches=len(calls), new_batches=new_batches
                )
            spent = _checkpoint_cost(calls)
            reserve = _call_reserve(config, prompt)
            if spent >= budget or budget - spent < reserve:
                raise AuditBudgetError(
                    "remaining audit budget is below the conservative next-batch reserve: "
                    f"need {reserve}, have {budget - spent}"
                )
            request_sha256 = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "batchId": batch_id,
                        "model": config.model,
                        "pairIds": list(pair_ids),
                        "promptSha256": prompt_sha256,
                    }
                )
            ).hexdigest()
            current_tombstone = state.get("inFlightCall")
            if current_tombstone is not None:
                tombstone = _mapping(current_tombstone, "inFlightCall")
                if tombstone.get("requestSha256") != request_sha256:
                    raise AuditCheckpointError("audit in-flight request hash mismatch")
            state["inFlightCall"] = {
                "batchId": batch_id,
                "conservativeCostReserveCredits": _decimal_text(reserve),
                "dispatchResolution": (
                    "confirmed_not_charged_redispatch"
                    if current_tombstone is not None
                    else "new_dispatch"
                ),
                "requestSha256": request_sha256,
                "startedAtUnixMs": int(time.time() * 1000),
            }
            _save_checkpoint(path, state)
            started = time.perf_counter()
            completion = client.complete(prompt, model=config.model)
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            record = {
                "batchId": batch_id,
                "conservativeCostReserveCredits": _decimal_text(reserve),
                "latencyMs": latency_ms,
                "pairIds": list(pair_ids),
                "promptSha256": prompt_sha256,
                "recordStatus": "accepted_response",
                "request": {"model": config.model},
                "response": {
                    "content": completion.content,
                    "finishReason": completion.finish_reason,
                    "id": completion.response_id,
                    "model": completion.model,
                    "openrouterMetadata": completion.openrouter_metadata,
                    "provider": completion.provider,
                    "systemFingerprint": completion.system_fingerprint,
                    "usage": completion.usage.to_dict(),
                },
            }
            calls.append(record)
            state["inFlightCall"] = None
            _save_checkpoint(path, state)
            new_batches += 1
            batch_reviews = _validate_audit_record(
                record,
                config=config,
                prompt=prompt,
                expected_pair_ids=pair_ids,
            )
        for review in batch_reviews:
            pair_id = _text(review.get("pairId"), "audit review pairId")
            reviews_by_pair[pair_id] = review

    pair_by_id = {pair.pair_id: pair for pair in pairs}
    reviews: list[dict[str, object]] = []
    for pair in pairs:
        judge_review = reviews_by_pair[pair.pair_id]
        token_check = protected_token_failure(pair.source_text, pair.candidate_text)
        semantic_failure = any(bool(judge_review[field]) for field in JUDGE_FIELDS)
        reviews.append(
            {
                **judge_review,
                "fidelityFailure": semantic_failure or bool(token_check["failed"]),
                "protectedTokenEvidence": token_check,
                "protectedTokenFailure": bool(token_check["failed"]),
                "semanticFidelityFailure": semantic_failure,
            }
        )

    mapping = [
        {
            "documentId": pair.document_id,
            "methodId": pair.method_id,
            "pairId": pair.pair_id,
        }
        for pair in pairs
    ]
    aggregates: list[dict[str, object]] = []
    for method_id in config.method_ids:
        method_pair_ids = {
            pair.pair_id for pair in pair_by_id.values() if pair.method_id == method_id
        }
        method_reviews = [review for review in reviews if review["pairId"] in method_pair_ids]
        aggregates.append(
            {
                "fidelityFailureCount": sum(
                    review["fidelityFailure"] is True for review in method_reviews
                ),
                "methodId": method_id,
                "protectedTokenFailureCount": sum(
                    review["protectedTokenFailure"] is True for review in method_reviews
                ),
                "semanticFidelityFailureCount": sum(
                    review["semanticFidelityFailure"] is True for review in method_reviews
                ),
                "voiceDriftCounts": {
                    level: sum(review["voiceDrift"] == level for review in method_reviews)
                    for level in VOICE_LEVELS
                },
            }
        )

    artifact = {
        "aggregates": aggregates,
        "auditConfigSha256": config.sha256,
        "endpointSnapshotSha256": config.endpoint_snapshot_sha256,
        "judge": {
            "expectedProviders": list(config.expected_response_providers),
            "model": config.model,
            "providerOrder": list(config.provider_order),
        },
        "methodology": config.methodology,
        "opaqueMapping": mapping,
        "planSha256": config.plan_sha256,
        "reviews": reviews,
        "schemaVersion": ARTIFACT_SCHEMA_VERSION,
        "sourceArtifactSha256": config.source_sha256,
        "sources": list(config.sources),
        "usage": _aggregate_usage(calls, budget),
        "verifiedAt": config.verified_at,
    }
    canonical_json_bytes(artifact)
    return artifact


def build_audit_dry_run(config: AuditConfig, source: Mapping[str, object]) -> dict[str, object]:
    pairs = build_blinded_pairs(config, source)
    batches = build_audit_batches(pairs, config.batch_size)
    reserves = [_call_reserve(config, build_audit_prompt(batch)) for batch in batches]
    return {
        "batchCount": len(batches),
        "batchSize": config.batch_size,
        "configSha256": config.sha256,
        "judgeModel": config.model,
        "maxConservativeBatchReserveCredits": _decimal_text(max(reserves)),
        "pairCount": len(pairs),
        "sourceArtifactSha256": config.source_sha256,
        "sumConservativeBatchReservesCredits": _decimal_text(sum(reserves)),
    }


def _validate_audit_record(
    record: Mapping[str, object],
    *,
    config: AuditConfig,
    prompt: str,
    expected_pair_ids: Sequence[str],
) -> list[dict[str, object]]:
    response = _mapping(record.get("response"), "audit checkpoint response")
    if response.get("finishReason") != "stop":
        raise AuditResponseContractError("audit finish reason must be stop")
    model = response.get("model")
    provider = response.get("provider")
    if model not in config.expected_response_models:
        raise AuditResponseContractError("unexpected audit response model")
    if provider not in config.expected_response_providers:
        raise AuditResponseContractError("unexpected audit response provider")
    metadata = _mapping(
        response.get("openrouterMetadata"), "audit OpenRouter metadata"
    )
    if metadata.get("strategy") != "direct":
        raise AuditResponseContractError("audit routing strategy must be direct")
    if metadata.get("attempt") != 1 or isinstance(metadata.get("attempt"), bool):
        raise AuditResponseContractError("audit routing attempt must be one")
    if metadata.get("pipeline") not in (None, []):
        raise AuditResponseContractError("audit routing pipeline must be empty")
    endpoints = _mapping(metadata.get("endpoints"), "audit routing endpoints")
    available = endpoints.get("available")
    if not isinstance(available, list) or any(
        not isinstance(item, Mapping) for item in available
    ):
        raise AuditResponseContractError("audit routing endpoints are invalid")
    selected = [item for item in available if item.get("selected") is True]
    if len(selected) != 1:
        raise AuditResponseContractError("audit must select exactly one endpoint")
    selected_provider = _first_present(selected[0], "provider", "provider_name", "providerName")
    selected_model = _first_present(selected[0], "model", "model_id", "modelId")
    if selected_provider != provider:
        raise AuditResponseContractError("audit selected provider mismatch")
    if selected_model not in config.expected_response_models:
        raise AuditResponseContractError("audit selected model mismatch")
    attempts = metadata.get("attempts")
    if attempts is not None:
        if not isinstance(attempts, list) or len(attempts) != 1:
            raise AuditResponseContractError("audit attempts must contain one item")
        attempt = _mapping(attempts[0], "audit attempt")
        if _first_present(attempt, "provider", "provider_name", "providerName") != provider:
            raise AuditResponseContractError("audit attempt provider mismatch")
        if _first_present(attempt, "model", "model_id", "modelId") not in config.expected_response_models:
            raise AuditResponseContractError("audit attempt model mismatch")
        if attempt.get("status") != 200 or isinstance(attempt.get("status"), bool):
            raise AuditResponseContractError("audit attempt status must be 200")
    usage = _mapping(response.get("usage"), "audit usage")
    prompt_tokens = _nonnegative_int(usage.get("promptTokens"), "promptTokens")
    completion_tokens = _nonnegative_int(
        usage.get("completionTokens"), "completionTokens"
    )
    total_tokens = _nonnegative_int(usage.get("totalTokens"), "totalTokens")
    if total_tokens != prompt_tokens + completion_tokens:
        raise AuditResponseContractError("audit token totals are inconsistent")
    if prompt_tokens > len(prompt.encode()) + config.prompt_token_overhead_reserve:
        raise AuditResponseContractError("audit prompt usage exceeds its reserve")
    if completion_tokens > config.max_tokens:
        raise AuditResponseContractError("audit completion usage exceeds maxTokens")
    cost = _decimal(usage.get("providerCostCredits"), "audit provider cost")
    reserve = _decimal(
        record.get("conservativeCostReserveCredits"), "audit reserve"
    )
    if cost > reserve:
        raise AuditResponseContractError("audit provider cost exceeds its reserve")
    content = response.get("content")
    if not isinstance(content, str):
        raise AuditResponseContractError("audit response content must be text")
    return parse_audit_response(content, expected_pair_ids)


def _load_or_create_checkpoint(
    path: Path,
    *,
    config: AuditConfig,
    expected_batch_ids: Sequence[str],
) -> dict[str, object]:
    if path.exists():
        state = _json_object(path.read_bytes(), "audit checkpoint")
        if state.get("schemaVersion") != CHECKPOINT_SCHEMA_VERSION:
            raise AuditCheckpointError("unsupported audit checkpoint schemaVersion")
        bindings = {
            "auditConfigSha256": config.sha256,
            "endpointSnapshotSha256": config.endpoint_snapshot_sha256,
            "expectedBatchIds": list(expected_batch_ids),
            "planSha256": config.plan_sha256,
            "sourceArtifactSha256": config.source_sha256,
        }
        for key, expected in bindings.items():
            if state.get(key) != expected:
                raise AuditCheckpointError(f"audit checkpoint {key} mismatch")
        return state
    state: dict[str, object] = {
        "auditConfigSha256": config.sha256,
        "calls": [],
        "endpointSnapshotSha256": config.endpoint_snapshot_sha256,
        "expectedBatchIds": list(expected_batch_ids),
        "inFlightCall": None,
        "planSha256": config.plan_sha256,
        "schemaVersion": CHECKPOINT_SCHEMA_VERSION,
        "sourceArtifactSha256": config.source_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(path, state)
    return state


def _validate_saved_request(
    record: Mapping[str, object],
    batch_id: str,
    pair_ids: Sequence[str],
    prompt_sha256: str,
) -> None:
    if record.get("batchId") != batch_id:
        raise AuditCheckpointError("saved audit batch ID mismatch")
    if record.get("pairIds") != list(pair_ids):
        raise AuditCheckpointError("saved audit pair IDs mismatch")
    if record.get("promptSha256") != prompt_sha256:
        raise AuditCheckpointError("saved audit prompt hash mismatch")


def _batch_id(index: int, batch: Sequence[AuditPair]) -> str:
    digest = hashlib.sha256("\0".join(pair.pair_id for pair in batch).encode()).hexdigest()
    return f"batch-{index:02d}-{digest[:12]}"


def _call_reserve(config: AuditConfig, prompt: str) -> Decimal:
    prompt_units = len(prompt.encode()) + config.prompt_token_overhead_reserve
    return (
        Decimal(prompt_units) * config.prompt_price_usd_per_million
        + Decimal(config.max_tokens) * config.completion_price_usd_per_million
    ) / Decimal(1_000_000)


def _checkpoint_cost(calls: Sequence[object]) -> Decimal:
    total = Decimal(0)
    for raw_call in calls:
        call = _mapping(raw_call, "audit call")
        response = _mapping(call.get("response"), "audit response")
        usage = _mapping(response.get("usage"), "audit usage")
        total += _decimal(usage.get("providerCostCredits"), "audit provider cost")
    return total


def _aggregate_usage(calls: Sequence[object], budget: Decimal) -> dict[str, object]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    latency_ms = 0.0
    for raw_call in calls:
        call = _mapping(raw_call, "audit call")
        usage = _mapping(
            _mapping(call.get("response"), "audit response").get("usage"),
            "audit usage",
        )
        prompt_tokens += _nonnegative_int(usage.get("promptTokens"), "promptTokens")
        completion_tokens += _nonnegative_int(
            usage.get("completionTokens"), "completionTokens"
        )
        total_tokens += _nonnegative_int(usage.get("totalTokens"), "totalTokens")
        latency = call.get("latencyMs")
        if not isinstance(latency, (int, float)) or isinstance(latency, bool):
            raise AuditCheckpointError("audit latency must be numeric")
        latency_ms += float(latency)
    return {
        "callCount": len(calls),
        "completionTokens": completion_tokens,
        "latencyMs": round(latency_ms, 3),
        "promptTokens": prompt_tokens,
        "providerCostBudgetCredits": _decimal_text(budget),
        "providerCostCredits": _decimal_text(_checkpoint_cost(calls)),
        "totalTokens": total_tokens,
    }


def _save_checkpoint(path: Path, state: Mapping[str, object]) -> None:
    _atomic_write(path, canonical_json_bytes(state))


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
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


def _required_budget(value: Decimal) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
        raise AuditBudgetError("live audit requires an explicit positive budget")
    return value


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


def _validate_evidence(value: Mapping[str, object], label: str) -> None:
    _text(value.get("verifiedAt"), f"{label}.verifiedAt")
    _text(value.get("methodology"), f"{label}.methodology")
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


def _sha256_text(value: object, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


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


def _build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument(
        "--config",
        default=str(root / "fixtures" / "semantic-audit-config-v1.json"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(root / "results" / "semantic-audit-checkpoint-v1.json"),
    )
    parser.add_argument(
        "--output",
        default=str(root / "results" / "semantic-audit-v1.json"),
    )
    parser.add_argument("--max-provider-cost-credits")
    parser.add_argument("--max-new-batches", type=int)
    parser.add_argument("--confirm-not-charged-batch-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_audit_config(args.config)
    source = load_audit_source(config)
    if args.dry_run:
        if (
            args.max_provider_cost_credits is not None
            or args.max_new_batches is not None
            or args.confirm_not_charged_batch_id is not None
        ):
            raise SystemExit("budget and checkpoint controls are live-only")
        print(canonical_json_bytes(build_audit_dry_run(config, source)).decode(), end="")
        return 0
    if args.max_provider_cost_credits is None:
        raise SystemExit("--live requires --max-provider-cost-credits")
    budget = _decimal(args.max_provider_cost_credits, "audit provider budget")
    client = OpenRouterClient.from_env(
        timeout=config.timeout_seconds,
        provider_order=config.provider_order,
        allow_fallbacks=False,
        require_parameters=True,
        reasoning_effort=config.reasoning_effort,
        temperature=None,
        max_tokens=config.max_tokens,
        seed=config.seed,
        max_prompt_price=float(config.prompt_price_usd_per_million),
        max_completion_price=float(config.completion_price_usd_per_million),
        response_format=audit_response_format(config.batch_size),
    )
    try:
        artifact = run_audit(
            config,
            source,
            client=client,
            max_provider_cost_credits=budget,
            checkpoint_path=args.checkpoint,
            max_new_batches=args.max_new_batches,
            confirm_not_charged_batch_id=args.confirm_not_charged_batch_id,
        )
    except AuditBatchLimitReached as pause:
        print(
            json.dumps(
                {
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "completedBatches": pause.completed_batches,
                    "newBatches": pause.new_batches,
                    "status": "paused_at_batch_limit",
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
                "callCount": artifact["usage"]["callCount"],  # type: ignore[index]
                "output": str(output.resolve()),
                "status": "complete",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AuditBatchLimitReached",
    "AuditBudgetError",
    "AuditCheckpointError",
    "AuditResponseContractError",
    "build_audit_batches",
    "build_audit_dry_run",
    "build_audit_prompt",
    "build_blinded_pairs",
    "audit_response_format",
    "load_audit_config",
    "load_audit_source",
    "parse_audit_response",
    "protected_token_failure",
    "run_audit",
]
