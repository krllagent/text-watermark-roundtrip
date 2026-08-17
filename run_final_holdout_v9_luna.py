"""Run the frozen twenty-document v9 confirmation set with Luna exactly once.

Dry-run, aggregation, packet creation, and review finalization are local-only.
Live mode reuses the already frozen Luna Azure EU request and route contract.
Any request left in flight is a permanent no-redispatch tombstone.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
import time
from typing import Any, Callable

from corpus_contract import canonical_json_bytes
from run_experiment import fidelity_metrics
from run_model_canary_luna import (
    CACHE_READ_PRICE as LUNA_CACHE_READ_PRICE,
    CACHE_WRITE_PRICE as LUNA_CACHE_WRITE_PRICE,
    COMPLETION_PRICE as LUNA_COMPLETION_PRICE,
    CanaryError,
    CaptureTransport,
    EXPECTED_MODELS as LUNA_EXPECTED_MODELS,
    MODEL as LUNA_MODEL,
    PROMPT_PRICE as LUNA_PROMPT_PRICE,
    PROVIDER as LUNA_PROVIDER,
    PROVIDER_TAG as LUNA_PROVIDER_TAG,
    REVIEW_CRITERIA,
    atomic_write,
    build_client as build_luna_client,
    conservative_reserve as luna_conservative_reserve,
    decimal,
    expected_payload as luna_expected_payload,
    fetch_catalog as fetch_luna_catalog,
    object_sha256,
    record_sha256,
    require_mapping,
    validate_completion as validate_luna_completion,
    validate_route_record as validate_luna_route_record,
)
from unmark import (
    ChatCompletion,
    ProviderHTTPError,
    ProviderResponseError,
    build_v4_draft_request,
    canonicalize_placeholders,
    json_safe_value,
    protect_tokens,
    restore_tokens,
    result_validation_issues,
)
from watermark_toy import Document, load_lexicon, score_corpus, score_text


ROOT = Path(__file__).resolve().parent
SCRIPT_VERSION = "final-holdout-v9-luna-v1"
DOCUMENT_IDS = tuple(f"holdout-{index:02d}" for index in range(1, 21))
CONTROLS_PATH = ROOT / "results" / "final-holdout-controls-v9.json"
CONTROLS_SHA256 = "3d37c2c0400888258192e4106e2b349cf06a558bb7e4072e489708c0aaa571c2"
PLAN_PATH = ROOT / "fixtures" / "final-holdout-plan-v9.json"
PLAN_SHA256 = "6b451bcc89a025f8cd4995c47e1d9ecce09e75a5656857f2425cbec20f7c7e70"
KEY_PATH = ROOT / "fixtures" / "final-holdout-key-v9.json"
KEY_SHA256 = "bb7a9fc7f3385249be542bdd7205cf805280fcaf117b1c46a4e4facd107fef66"
MANIFEST_PATH = (
    ROOT
    / "corpus"
    / "holdout-v6"
    / "reviewed-encoder-v9"
    / "marked-1000"
    / "manifest-v9.json"
)
MANIFEST_SHA256 = "83e17ea83f9a9358db764714a714aceeaf3c3b534afa418cf5e3eb60dfd67eff"
PHASE_A_PLAN_PATH = ROOT / "fixtures" / "final-holdout-plan-v8.json"
PHASE_A_PLAN_SHA256 = "ef79dd19985ce8daab9bd14413b4bfb0085a85f8b94a3155720b9e7982c40e4a"
UNMARK_SHA256 = "c474b913d714be820b86f3855c60825e1c11496e331c5fef80b9d64aa67e4665"
DEFAULT_CHECKPOINT = ROOT / "results" / "final-holdout-v9-luna-checkpoint-v1.json"
DEFAULT_AGGREGATE = ROOT / "results" / "final-holdout-v9-luna-automated-v1.json"
DEFAULT_PACKET = ROOT / "results" / "final-holdout-v9-luna-blind-v1.json"
DEFAULT_FINAL = ROOT / "results" / "final-holdout-v9-luna-final-v1.json"
CALL_ID_PREFIX = "luna-final"


@dataclass(frozen=True)
class CandidateContract:
    """Small adapter isolating the selected canary's provider contract."""

    model: str
    route: str
    source_path: Path
    source_sha256: str
    build_client: Callable[[CaptureTransport], Any]
    conservative_reserve: Callable[[Any], Decimal]
    expected_payload: Callable[[Any], dict[str, object]]
    fetch_catalog: Callable[[], dict[str, object]]
    validate_completion: Callable[[ChatCompletion, Any], None]
    validate_route_record: Callable[[object], None]
    expected_models: tuple[str, ...]
    provider: str
    expected_cost: Callable[[Mapping[str, object]], Decimal]


def luna_usage_cost(usage: Mapping[str, object]) -> Decimal:
    """Recompute a stored usage record's provider cost from frozen prices."""
    details = usage.get("promptTokenDetails")
    detail_map = details if isinstance(details, Mapping) else {}
    prompt_tokens = int(usage.get("promptTokens", 0))
    cached = int(detail_map.get("cachedTokens", 0))
    cache_write = int(detail_map.get("cacheWriteTokens", 0))
    completion_tokens = int(usage.get("completionTokens", 0))
    uncached = prompt_tokens - cached - cache_write
    return (
        Decimal(uncached) * LUNA_PROMPT_PRICE
        + Decimal(cached) * LUNA_CACHE_READ_PRICE
        + Decimal(cache_write) * LUNA_CACHE_WRITE_PRICE
        + Decimal(completion_tokens) * LUNA_COMPLETION_PRICE
    ) / Decimal(1_000_000)


CANDIDATE = CandidateContract(
    model=LUNA_MODEL,
    route=LUNA_PROVIDER_TAG,
    source_path=ROOT / "run_model_canary_luna.py",
    source_sha256=hashlib.sha256(
        (ROOT / "run_model_canary_luna.py").read_bytes()
    ).hexdigest(),
    build_client=build_luna_client,
    conservative_reserve=luna_conservative_reserve,
    expected_payload=luna_expected_payload,
    fetch_catalog=fetch_luna_catalog,
    validate_completion=validate_luna_completion,
    validate_route_record=validate_luna_route_record,
    expected_models=LUNA_EXPECTED_MODELS,
    provider=LUNA_PROVIDER,
    expected_cost=luna_usage_cost,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanaryError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise CanaryError(f"{label} must be an object")
    return value


def validate_bound_file(binding: object, label: str) -> dict[str, object]:
    row = dict(require_mapping(binding, label))
    path = row.get("path")
    expected = row.get("sha256")
    if (
        not isinstance(path, str)
        or not isinstance(expected, str)
        or sha256_file(ROOT / path) != expected
    ):
        raise CanaryError(f"bound artifact changed: {label}")
    return row


def load_protocol() -> dict[str, object]:
    """Load and recompute every v9 input without contacting a provider."""
    fixed_hashes = {
        CONTROLS_PATH: CONTROLS_SHA256,
        PLAN_PATH: PLAN_SHA256,
        KEY_PATH: KEY_SHA256,
        MANIFEST_PATH: MANIFEST_SHA256,
        PHASE_A_PLAN_PATH: PHASE_A_PLAN_SHA256,
        CANDIDATE.source_path: CANDIDATE.source_sha256,
        ROOT / "unmark.py": UNMARK_SHA256,
    }
    for path, expected in fixed_hashes.items():
        if sha256_file(path) != expected:
            raise CanaryError(f"frozen input changed: {path.name}")

    controls = load_json(CONTROLS_PATH, "v9 controls")
    bindings = require_mapping(controls.get("artifactBindings"), "artifactBindings")
    plan_binding = validate_bound_file(bindings.get("plan"), "plan")
    key_binding = validate_bound_file(bindings.get("keyArtifact"), "keyArtifact")
    manifest_binding = validate_bound_file(
        bindings.get("markedManifest"), "markedManifest"
    )
    if (
        plan_binding.get("sha256") != PLAN_SHA256
        or key_binding.get("sha256") != KEY_SHA256
        or manifest_binding.get("sha256") != MANIFEST_SHA256
    ):
        raise CanaryError("v9 top-level bindings changed")

    plan = load_json(PLAN_PATH, "v9 plan")
    phase_a = require_mapping(plan.get("phaseA"), "phaseA")
    phase_a_plan_binding = validate_bound_file(phase_a.get("plan"), "phaseA.plan")
    if phase_a_plan_binding.get("sha256") != PHASE_A_PLAN_SHA256:
        raise CanaryError("phase-A plan binding changed")
    phase_a_plan = load_json(PHASE_A_PLAN_PATH, "phase-A plan")
    gate = dict(
        require_mapping(phase_a_plan.get("finalConfirmationGate"), "final gate")
    )
    expected_gate = {
        "callCount": 20,
        "fallbackAllowed": False,
        "maximumPipelineFailures": 0,
        "maximumPlaceholderFailures": 0,
        "minimumEachDocumentWordDistance": 0.15,
        "minimumOutputActivePositions": 20,
        "onePassPerDocument": True,
        "outputDetectorStatus": "not_detected",
        "retryAllowed": False,
    }
    if any(gate.get(key) != value for key, value in expected_gate.items()):
        raise CanaryError("frozen final gate changed")

    detector_binding = validate_bound_file(
        phase_a_plan.get("detectorImplementation"), "detector implementation"
    )
    if detector_binding.get("allowlistAware") is not False:
        raise CanaryError("final detector must be allowlist-unaware")
    lexicon_binding = validate_bound_file(phase_a_plan.get("lexicon"), "lexicon")
    lexicon = load_lexicon(ROOT / str(lexicon_binding["path"]))

    key_artifact = load_json(KEY_PATH, "v9 key")
    key_hex = key_artifact.get("keyHex")
    try:
        key = bytes.fromhex(key_hex) if isinstance(key_hex, str) else b""
    except ValueError as error:
        raise CanaryError("v9 key is invalid") from error
    if len(key) != 32:
        raise CanaryError("v9 key is invalid")
    prepaid = require_mapping(plan.get("prepaidControls"), "prepaidControls")
    density_bps = int(prepaid.get("densityBps", 0))
    context_width = int(prepaid.get("contextWidth", 0))
    min_active = int(prepaid.get("minActivePositions", 0))

    manifest = load_json(MANIFEST_PATH, "v9 marked manifest")
    rows = manifest.get("documents")
    if not isinstance(rows, list) or len(rows) != len(DOCUMENT_IDS):
        raise CanaryError("v9 manifest must contain twenty documents")
    sources: dict[str, str] = {}
    source_sha256s: dict[str, str] = {}
    for expected_id, value in zip(DOCUMENT_IDS, rows, strict=True):
        row = require_mapping(value, f"manifest document {expected_id}")
        if row.get("documentId") != expected_id or not isinstance(row.get("path"), str):
            raise CanaryError("v9 manifest document order changed")
        path = ROOT / str(row["path"])
        expected_sha256 = row.get("sha256")
        if not isinstance(expected_sha256, str) or sha256_file(path) != expected_sha256:
            raise CanaryError(f"v9 marked input changed: {expected_id}")
        text = path.read_text(encoding="utf-8")
        sources[expected_id] = text
        source_sha256s[expected_id] = expected_sha256

    source_score = score_corpus(
        tuple(
            Document(document_id, sources[document_id]) for document_id in DOCUMENT_IDS
        ),
        key=key,
        density_bps=density_bps,
        lexicon=lexicon,
        context_width=context_width,
        min_active_positions=min_active,
    ).to_dict(include_documents=False)
    if (
        source_score.get("status") != "detected"
        or source_score.get("activePositions") != 42
        or source_score.get("hits") != 38
    ):
        raise CanaryError("v9 marked baseline is no longer 38/42 detected")
    prepaid_gate = require_mapping(controls.get("prepaidGate"), "prepaidGate")
    frozen_marked = require_mapping(prepaid_gate.get("marked"), "prepaidGate.marked")
    if prepaid_gate.get("status") != "passed" or any(
        frozen_marked.get(field) != source_score.get(field)
        for field in ("activePositions", "hits", "status")
    ):
        raise CanaryError("recomputed source score differs from frozen controls")

    return {
        "bindings": {
            "candidateContractSha256": CANDIDATE.source_sha256,
            "controlsSha256": CONTROLS_SHA256,
            "keySha256": KEY_SHA256,
            "manifestSha256": MANIFEST_SHA256,
            "phaseAPlanSha256": PHASE_A_PLAN_SHA256,
            "planSha256": PLAN_SHA256,
        },
        "contextWidth": context_width,
        "densityBps": density_bps,
        "finalGate": gate,
        "key": key,
        "lexicon": lexicon,
        "minActivePositions": min_active,
        "sourceScore": source_score,
        "sourceSha256s": source_sha256s,
        "sources": sources,
    }


def request_for(source: str):
    return build_v4_draft_request(protect_tokens(source).masked)


def request_hashes(protocol: Mapping[str, object]) -> dict[str, str]:
    sources = require_mapping(protocol.get("sources"), "protocol sources")
    return {
        document_id: object_sha256(
            CANDIDATE.expected_payload(request_for(str(sources[document_id])))
        )
        for document_id in DOCUMENT_IDS
    }


def initial_checkpoint(protocol: Mapping[str, object]) -> dict[str, object]:
    return {
        "blindMapping": None,
        "calls": [],
        "documentIds": list(DOCUMENT_IDS),
        "importedContractSha256s": {
            "selectedCanaryRunner": CANDIDATE.source_sha256,
            "unmark": UNMARK_SHA256,
        },
        "inFlight": None,
        "model": CANDIDATE.model,
        "protocolBindings": protocol["bindings"],
        "requestSha256s": request_hashes(protocol),
        "route": CANDIDATE.route,
        "runnerSha256": sha256_file(Path(__file__)),
        "schemaVersion": 1,
        "scriptVersion": SCRIPT_VERSION,
        "sourceScore": protocol["sourceScore"],
        "sourceSha256s": protocol["sourceSha256s"],
        "terminalFailure": None,
    }


def validated_blind_mapping(state: Mapping[str, object]) -> dict[str, str]:
    rows = state.get("blindMapping")
    if not isinstance(rows, list) or len(rows) != len(DOCUMENT_IDS):
        raise CanaryError("blind mapping is invalid")
    if not all(isinstance(row, dict) for row in rows):
        raise CanaryError("blind mapping rows must be objects")
    document_ids = [row.get("documentId") for row in rows]
    pair_ids = [row.get("pairId") for row in rows]
    if (
        set(document_ids) != set(DOCUMENT_IDS)
        or len(set(document_ids)) != len(DOCUMENT_IDS)
        or any(not isinstance(pair_id, str) for pair_id in pair_ids)
        or len(set(pair_ids)) != len(DOCUMENT_IDS)
    ):
        raise CanaryError("blind mapping must cover twenty unique pairs")
    return {str(row["documentId"]): str(row["pairId"]) for row in rows}


def validate_stored_completion(completion: Mapping[str, object]) -> None:
    """Re-validate a stored response and its cost without trusting the file."""
    if completion.get("finishReason") != "stop":
        raise CanaryError("stored completion did not finish with stop")
    if (
        completion.get("model") not in CANDIDATE.expected_models
        or completion.get("provider") != CANDIDATE.provider
    ):
        raise CanaryError("stored completion model or provider changed")
    metadata = require_mapping(
        completion.get("openrouterMetadata"), "stored routing metadata"
    )
    if (
        metadata.get("strategy") != "direct"
        or metadata.get("attempt") != 1
        or metadata.get("requested") != CANDIDATE.model
        or metadata.get("is_byok") is not False
        or metadata.get("pipeline") not in (None, [])
    ):
        raise CanaryError("stored routing metadata violates the frozen route")
    usage = require_mapping(completion.get("usage"), "stored usage")
    stored_cost = decimal(usage.get("providerCostCredits"), "stored provider cost")
    if stored_cost != CANDIDATE.expected_cost(usage):
        raise CanaryError("stored provider cost does not match the frozen prices")


def validate_checkpoint(
    state: Mapping[str, object], protocol: Mapping[str, object]
) -> None:
    expected = initial_checkpoint(protocol)
    for field in (
        "documentIds",
        "importedContractSha256s",
        "model",
        "protocolBindings",
        "requestSha256s",
        "route",
        "runnerSha256",
        "schemaVersion",
        "scriptVersion",
        "sourceScore",
        "sourceSha256s",
    ):
        if state.get(field) != expected[field]:
            raise CanaryError(f"checkpoint binding changed: {field}")
    calls = state.get("calls")
    if not isinstance(calls, list) or len(calls) > len(DOCUMENT_IDS):
        raise CanaryError("checkpoint calls are invalid")
    sources = require_mapping(protocol.get("sources"), "protocol sources")
    hashes = require_mapping(expected.get("requestSha256s"), "request hashes")
    for index, value in enumerate(calls):
        row = require_mapping(value, "completed call")
        document_id = DOCUMENT_IDS[index]
        source = str(sources[document_id])
        payload = CANDIDATE.expected_payload(request_for(source))
        if (
            row.get("callId") != f"{CALL_ID_PREFIX}:{document_id}"
            or row.get("documentId") != document_id
            or row.get("sourceText") != source
            or row.get("requestBody") != payload
            or row.get("requestSha256") != hashes[document_id]
        ):
            raise CanaryError("completed call binding changed")
        wire = require_mapping(row.get("wireRequest"), "wire request")
        headers = require_mapping(wire.get("headers"), "wire headers")
        if wire.get("body") != payload or any(
            key.lower() == "authorization" for key in headers
        ):
            raise CanaryError("wire request evidence changed")
        CANDIDATE.validate_route_record(row.get("routePreflight"))
        completion = require_mapping(row.get("completion"), "completion")
        analysis = require_mapping(row.get("analysis"), "analysis")
        validate_stored_completion(completion)
        recomputed = analyze_output(
            protocol, document_id, source, str(completion.get("content", ""))
        )
        if json_safe_value(recomputed) != json_safe_value(analysis):
            raise CanaryError("stored analysis does not recompute from the response")
        if row.get("recordSha256") != record_sha256(row):
            raise CanaryError("completed call digest changed")
    if state.get("blindMapping") is not None:
        validated_blind_mapping(state)
    if state.get("inFlight") is not None or state.get("terminalFailure") is not None:
        raise CanaryError("checkpoint contains a permanent no-redispatch tombstone")


def load_or_create_checkpoint(
    path: Path, protocol: Mapping[str, object]
) -> dict[str, object]:
    if not path.exists():
        state = initial_checkpoint(protocol)
        atomic_write(path, state)
        return state
    state = load_json(path, "final holdout checkpoint")
    validate_checkpoint(state, protocol)
    return state


def detector_score(
    protocol: Mapping[str, object], documents: Sequence[Document], *, details: bool
) -> dict[str, object]:
    return score_corpus(
        tuple(documents),
        key=protocol["key"],
        density_bps=int(protocol["densityBps"]),
        lexicon=protocol["lexicon"],
        context_width=int(protocol["contextWidth"]),
        min_active_positions=int(protocol["minActivePositions"]),
    ).to_dict(include_documents=details)


def analyze_output(
    protocol: Mapping[str, object], document_id: str, source: str, content: str
) -> dict[str, object]:
    protected = protect_tokens(source)
    normalized = content
    issues: list[dict[str, str]] = []
    restored: str | None = None
    try:
        normalized = canonicalize_placeholders(content, protected.tokens)
        issues.extend(result_validation_issues(protected.masked, normalized, None))
        restored = restore_tokens(normalized, protected.tokens)
    except (
        Exception
    ) as error:  # The observed provider output remains a failed document.
        issues.append(
            {
                "code": "placeholder_contract",
                "message": f"{type(error).__name__}: {error}",
            }
        )
        issues.extend(result_validation_issues(protected.masked, content, None))
    evaluated = restored if restored is not None else content
    fidelity = fidelity_metrics(source, evaluated)
    protected_metrics = require_mapping(
        fidelity.get("protectedTokens"), "protected metrics"
    )
    if protected_metrics.get("exactlyRestored") is not True:
        issues.append(
            {"code": "protected_values", "message": "protected values changed"}
        )
    score = score_text(
        evaluated,
        key=protocol["key"],
        density_bps=int(protocol["densityBps"]),
        lexicon=protocol["lexicon"],
        document_id=document_id,
        context_width=int(protocol["contextWidth"]),
        min_active_positions=int(protocol["minActivePositions"]),
    ).to_dict()
    return {
        "detector": score,
        "evaluatedOutputText": evaluated,
        "fidelity": fidelity,
        "maskedOutputText": normalized,
        "pipelineIssues": issues,
        "restoredOutputText": restored,
    }


def run_live(path: Path, budget: Decimal, max_new_calls: int) -> dict[str, object]:
    if budget <= 0 or max_new_calls <= 0:
        raise CanaryError("budget and max-new-calls must be positive")
    protocol = load_protocol()
    state = load_or_create_checkpoint(path, protocol)
    calls = state["calls"]
    assert isinstance(calls, list)
    if len(calls) == len(DOCUMENT_IDS):
        raise CanaryError("all final holdout calls are already complete")
    selected = DOCUMENT_IDS[len(calls) : len(calls) + max_new_calls]
    sources = require_mapping(protocol.get("sources"), "protocol sources")
    requests = [
        (document_id, request_for(str(sources[document_id])))
        for document_id in selected
    ]
    required = sum(
        (CANDIDATE.conservative_reserve(request) for _, request in requests),
        Decimal(0),
    )
    if required > budget:
        raise CanaryError(f"budget {budget} is below conservative reserve {required}")
    route = CANDIDATE.fetch_catalog()
    spent = Decimal(0)
    hashes = require_mapping(state.get("requestSha256s"), "request hashes")
    for document_id, request in requests:
        source = str(sources[document_id])
        call_id = f"{CALL_ID_PREFIX}:{document_id}"
        payload = CANDIDATE.expected_payload(request)
        payload_sha256 = object_sha256(payload)
        if payload_sha256 != hashes[document_id]:
            raise CanaryError("request changed before dispatch")
        reserve = CANDIDATE.conservative_reserve(request)
        transport = CaptureTransport()
        client = CANDIDATE.build_client(transport)
        state["inFlight"] = {
            "callId": call_id,
            "conservativeReserveCredits": format(reserve, "f"),
            "requestBody": payload,
            "requestSha256": payload_sha256,
            "routePreflight": route,
            "startedAtUnixMs": int(time.time() * 1000),
        }
        atomic_write(path, state)
        started = time.monotonic()
        completion: ChatCompletion | None = None
        try:
            completion = client.complete(request, model=CANDIDATE.model)
            in_flight = require_mapping(state.get("inFlight"), "in-flight call")
            state["inFlight"] = {
                **dict(in_flight),
                "receivedAtUnixMs": int(time.time() * 1000),
                "receivedResponse": completion.to_dict(),
                "wireRequest": transport.last_request,
            }
            atomic_write(path, state)
            if (
                transport.last_request is None
                or transport.last_request.get("body") != payload
            ):
                raise CanaryError("wire body differs from the frozen request")
            CANDIDATE.validate_completion(completion, request)
            analysis = analyze_output(protocol, document_id, source, completion.content)
        except Exception as error:
            failure: dict[str, object] = {
                "callId": call_id,
                "error": f"{type(error).__name__}: {error}",
                "failedAtUnixMs": int(time.time() * 1000),
                "wireRequest": transport.last_request,
            }
            if completion is not None:
                failure["receivedResponse"] = completion.to_dict()
            if isinstance(error, ProviderHTTPError):
                failure["httpEvidence"] = error.to_dict()
            if isinstance(error, ProviderResponseError):
                raw = json_safe_value(error.raw_response)
                failure["rawResponse"] = raw if isinstance(raw, dict) else None
            state["terminalFailure"] = failure
            atomic_write(path, state)
            raise CanaryError(
                "call failed after dispatch; it is preserved and will never be retried"
            ) from error
        record: dict[str, object] = {
            "analysis": analysis,
            "callId": call_id,
            "completion": completion.to_dict(),
            "conservativeReserveCredits": format(reserve, "f"),
            "documentId": document_id,
            "latencyMs": round((time.monotonic() - started) * 1000, 3),
            "requestBody": payload,
            "requestSha256": payload_sha256,
            "routePreflight": route,
            "sourceText": source,
            "wireRequest": transport.last_request,
        }
        record["recordSha256"] = record_sha256(record)
        calls.append(record)
        state["inFlight"] = None
        spent += completion.usage.cost
        atomic_write(path, state)
    if spent > budget:
        raise CanaryError("actual provider cost exceeded this invocation budget")
    return {
        "checkpoint": str(path),
        "completedCalls": len(calls),
        "newCalls": len(requests),
        "providerCostCreditsThisInvocation": format(spent, "f"),
        "status": "complete" if len(calls) == len(DOCUMENT_IDS) else "paused",
    }


def automated_aggregate(path: Path) -> dict[str, object]:
    protocol = load_protocol()
    state = load_or_create_checkpoint(path, protocol)
    calls = state.get("calls")
    if not isinstance(calls, list) or len(calls) != len(DOCUMENT_IDS):
        raise CanaryError("automated aggregate requires twenty completed calls")
    outputs: list[Document] = []
    documents: list[dict[str, object]] = []
    pipeline_pass = True
    distance_pass = True
    length_pass = True
    paragraph_pass = True
    protected_pass = True
    provider_cost = Decimal(0)
    for value in calls:
        call = require_mapping(value, "completed call")
        analysis = require_mapping(call.get("analysis"), "analysis")
        fidelity = require_mapping(analysis.get("fidelity"), "fidelity")
        word = require_mapping(fidelity.get("wordLevenshtein"), "word distance")
        length = require_mapping(fidelity.get("length"), "length metrics")
        paragraphs = require_mapping(fidelity.get("paragraphs"), "paragraph metrics")
        protected = require_mapping(
            fidelity.get("protectedTokens"), "protected metrics"
        )
        distance = float(word.get("normalizedDistance", -1))
        length_ratio = float(length.get("outputPerInput", -1))
        paragraph_ratio = float(paragraphs.get("outputPerInput", -1))
        issues = analysis.get("pipelineIssues")
        output = analysis.get("evaluatedOutputText")
        document_id = call.get("documentId")
        if not isinstance(issues, list) or not isinstance(output, str):
            raise CanaryError("completed analysis is invalid")
        if not isinstance(document_id, str):
            raise CanaryError("completed document ID is invalid")
        pipeline_pass = pipeline_pass and not issues
        distance_pass = distance_pass and distance >= 0.15
        length_pass = length_pass and 0.6 <= length_ratio <= 1.6
        paragraph_pass = paragraph_pass and 0.7 <= paragraph_ratio <= 1.3
        protected_pass = protected_pass and protected.get("exactlyRestored") is True
        outputs.append(Document(document_id, output))
        completion = require_mapping(call.get("completion"), "completion")
        usage = require_mapping(completion.get("usage"), "completion usage")
        provider_cost += decimal(usage.get("providerCostCredits"), "provider cost")
        documents.append(
            {
                "documentId": document_id,
                "lengthRatio": length_ratio,
                "paragraphRatio": paragraph_ratio,
                "pipelineIssues": issues,
                "protectedValuesExactlyRestored": protected.get("exactlyRestored"),
                "wordDistance": distance,
            }
        )
    pooled = detector_score(protocol, outputs, details=True)
    detector_pass = (
        pooled.get("status") == "not_detected"
        and int(pooled.get("activePositions", 0)) >= 20
    )
    checks = {
        "allDocumentsWordDistanceAtLeast15Percent": distance_pass,
        "allLengthRatiosWithin0_6To1_6": length_pass,
        "allParagraphRatiosWithin0_7To1_3": paragraph_pass,
        "allProtectedValuesExactlyRestored": protected_pass,
        "noPipelineIssues": pipeline_pass,
        "outputDetectorNotDetectedWithAtLeast20Active": detector_pass,
        "twentyExactOnePassCalls": len(calls) == 20,
    }
    return {
        "automatedGate": {"checks": checks, "passed": all(checks.values())},
        "documents": documents,
        "model": CANDIDATE.model,
        "pooledOutputDetector": pooled,
        "protocolBindings": protocol["bindings"],
        "providerCostCredits": format(provider_cost, "f"),
        "route": CANDIDATE.route,
        "schemaVersion": 1,
        "sourceScore": protocol["sourceScore"],
        "verifiedAt": "2026-08-17",
    }


def write_aggregate(checkpoint: Path, output: Path) -> dict[str, object]:
    aggregate = automated_aggregate(checkpoint)
    atomic_write(output, aggregate)
    return aggregate


def build_blind_packet(checkpoint: Path, output: Path) -> dict[str, object]:
    protocol = load_protocol()
    state = load_or_create_checkpoint(checkpoint, protocol)
    calls = state.get("calls")
    if not isinstance(calls, list) or len(calls) != len(DOCUMENT_IDS):
        raise CanaryError("blind packet requires twenty completed calls")
    if state.get("blindMapping") is None:
        rows = [
            {"documentId": document_id, "pairId": f"pair-{secrets.token_hex(8)}"}
            for document_id in DOCUMENT_IDS
        ]
        secrets.SystemRandom().shuffle(rows)
        state["blindMapping"] = rows
        atomic_write(checkpoint, state)
    pair_by_document = validated_blind_mapping(state)
    call_by_document = {
        str(require_mapping(call, "call")["documentId"]): require_mapping(call, "call")
        for call in calls
    }
    pairs: list[dict[str, object]] = []
    for row in state["blindMapping"]:
        mapping = require_mapping(row, "blind mapping row")
        document_id = str(mapping["documentId"])
        analysis = require_mapping(
            call_by_document[document_id].get("analysis"), "analysis"
        )
        pairs.append(
            {
                "candidateText": analysis.get("evaluatedOutputText"),
                "pairId": pair_by_document[document_id],
                "sourceText": require_mapping(protocol["sources"], "sources")[
                    document_id
                ],
            }
        )
    packet: dict[str, object] = {
        "criteria": list(REVIEW_CRITERIA),
        "instructions": (
            "Read each pair line by line. Ignore stylistic differences. Mark pass only "
            "when claims, numbers, entities, causality, negation, scope, certainty, "
            "examples, caveats, paragraph roles, and exact strings are preserved. "
            "Every non-pass finding needs exact source and candidate substrings."
        ),
        "pairs": pairs,
        "requiredOutput": {
            "findingFields": [
                "criterion",
                "sourceQuote",
                "candidateQuote",
                "explanation",
            ],
            "reviewFields": ["pairId", "verdict", "findings"],
            "verdicts": ["pass", "minor", "major"],
        },
        "schemaVersion": 1,
    }
    packet["packetSha256"] = object_sha256(packet)
    atomic_write(output, packet)
    return packet


def recompute_analysis(path: Path, note: str) -> dict[str, object]:
    """Rescore stored responses after a scoring defect, without new requests.

    Every paid response is one-pass and can never be re-sent, so a defect in
    local scoring has to be repaired by rescoring what is already stored. This
    refuses to touch anything except the analysis: the request, the wire bytes,
    the response, and the route must all still validate first, and the reason
    and the resulting change are recorded in the checkpoint itself.
    """
    if not note.strip():
        raise CanaryError("recomputation requires a written reason")
    protocol = load_protocol()
    state = load_json(path, "final holdout checkpoint")
    calls = state.get("calls")
    if not isinstance(calls, list) or not calls:
        raise CanaryError("recomputation requires at least one completed call")
    sources = require_mapping(protocol.get("sources"), "protocol sources")
    hashes = require_mapping(state.get("requestSha256s"), "request hashes")
    changes: list[dict[str, object]] = []
    for index, value in enumerate(calls):
        row = require_mapping(value, "completed call")
        document_id = DOCUMENT_IDS[index]
        source = str(sources[document_id])
        payload = CANDIDATE.expected_payload(request_for(source))
        wire = require_mapping(row.get("wireRequest"), "wire request")
        if (
            row.get("documentId") != document_id
            or row.get("sourceText") != source
            or row.get("requestBody") != payload
            or row.get("requestSha256") != hashes[document_id]
            or wire.get("body") != payload
        ):
            raise CanaryError("stored request evidence changed; refusing to rescore")
        CANDIDATE.validate_route_record(row.get("routePreflight"))
        completion = require_mapping(row.get("completion"), "completion")
        validate_stored_completion(completion)
        previous = require_mapping(row.get("analysis"), "analysis")
        updated = analyze_output(
            protocol, document_id, source, str(completion.get("content", ""))
        )
        if json_safe_value(updated) == json_safe_value(previous):
            continue
        row["analysis"] = updated
        row["recordSha256"] = record_sha256(row)
        changes.append(
            {
                "documentId": document_id,
                "newIssues": updated.get("pipelineIssues"),
                "previousIssues": previous.get("pipelineIssues"),
            }
        )
    history = state.get("analysisRecomputations")
    entries = list(history) if isinstance(history, list) else []
    previous_runner = state.get("runnerSha256")
    current_runner = sha256_file(Path(__file__))
    state["runnerSha256"] = current_runner
    entries.append(
        {
            "changedDocuments": changes,
            "previousRunnerSha256": previous_runner,
            "reason": note,
            "runnerSha256": current_runner,
        }
    )
    state["analysisRecomputations"] = entries
    # Validate the candidate state before it reaches disk: a rescoring that
    # cannot validate must leave the stored checkpoint untouched.
    validate_checkpoint(state, protocol)
    atomic_write(path, state)
    return {"changed": len(changes), "checkpoint": str(path)}


def require_committed(path: Path, label: str) -> None:
    """Refuse to proceed unless the file is committed exactly as it is on disk.

    The frozen protocol commits the blind packet before anyone sees a review, so
    a later edit cannot quietly reshape what the reviewer was asked to judge.
    """
    try:
        blob = subprocess.run(
            ["git", "rev-parse", f"HEAD:{path.relative_to(ROOT).as_posix()}"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, ValueError) as error:
        raise CanaryError(f"{label} must be committed before review") from error
    current = subprocess.run(
        ["git", "hash-object", str(path)],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    if blob != current:
        raise CanaryError(f"{label} differs from its committed version")


def finalize_review(
    checkpoint: Path,
    packet_path: Path,
    review_path: Path,
    output: Path,
    *,
    enforce_commit: bool = True,
) -> dict[str, object]:
    protocol = load_protocol()
    load_or_create_checkpoint(checkpoint, protocol)
    if enforce_commit:
        require_committed(packet_path, "blind packet")
        require_committed(review_path, "blind review")
    packet = load_json(packet_path, "blind packet")
    review = load_json(review_path, "blind review")
    packet_sha256 = packet.get("packetSha256")
    unsigned_packet = {
        key: value for key, value in packet.items() if key != "packetSha256"
    }
    if (
        not isinstance(packet_sha256, str)
        or object_sha256(unsigned_packet) != packet_sha256
        or review.get("packetSha256") != packet_sha256
    ):
        raise CanaryError("review is bound to a different blind packet")
    expected_packet_path = output.parent / f".{output.name}.expected-packet"
    try:
        expected_packet = build_blind_packet(checkpoint, expected_packet_path)
    finally:
        expected_packet_path.unlink(missing_ok=True)
    if packet != expected_packet:
        raise CanaryError("blind packet differs from the completed calls")
    pairs = packet.get("pairs")
    reviews = review.get("reviews")
    if not isinstance(pairs, list) or not isinstance(reviews, list):
        raise CanaryError("packet or review rows are missing")
    pair_ids = [require_mapping(pair, "pair").get("pairId") for pair in pairs]
    if len(reviews) != 20 or len(set(pair_ids)) != 20:
        raise CanaryError("review must cover twenty unique pairs")
    review_rows = [require_mapping(row, "review row") for row in reviews]
    review_ids = [row.get("pairId") for row in review_rows]
    if len(set(review_ids)) != 20 or set(review_ids) != set(pair_ids):
        raise CanaryError("review must cover each blind pair exactly once")
    pair_by_id = {
        str(require_mapping(pair, "pair")["pairId"]): require_mapping(pair, "pair")
        for pair in pairs
    }
    pass_count = 0
    minor_documents = 0
    minor_findings = 0
    major_findings = 0
    for row in review_rows:
        pair_id = str(row["pairId"])
        verdict = row.get("verdict")
        findings = row.get("findings")
        if verdict not in {"pass", "minor", "major"} or not isinstance(findings, list):
            raise CanaryError("review verdict or findings are invalid")
        if verdict == "pass":
            if findings:
                raise CanaryError("pass review cannot contain findings")
            pass_count += 1
            continue
        if not findings:
            raise CanaryError("non-pass review requires a finding")
        pair = pair_by_id[pair_id]
        for finding_value in findings:
            finding = require_mapping(finding_value, "review finding")
            if set(finding) != {
                "candidateQuote",
                "criterion",
                "explanation",
                "sourceQuote",
            }:
                raise CanaryError("review finding fields changed")
            criterion = finding.get("criterion")
            source_quote = finding.get("sourceQuote")
            candidate_quote = finding.get("candidateQuote")
            explanation = finding.get("explanation")
            if (
                criterion not in REVIEW_CRITERIA
                or not isinstance(source_quote, str)
                or not source_quote
                or source_quote not in str(pair["sourceText"])
                or not isinstance(candidate_quote, str)
                or not candidate_quote
                or candidate_quote not in str(pair["candidateText"])
                or not isinstance(explanation, str)
                or not explanation.strip()
            ):
                raise CanaryError("review finding is not evidence-backed")
        if verdict == "minor":
            minor_documents += 1
            minor_findings += len(findings)
        else:
            major_findings += len(findings)
    review_pass = (
        pass_count >= 19
        and minor_documents <= 1
        and minor_findings <= 1
        and major_findings == 0
    )
    automated = automated_aggregate(checkpoint)
    passed = bool(automated["automatedGate"]["passed"]) and review_pass
    final = {
        "automated": automated,
        "commitGateEnforced": enforce_commit,
        "finalConfirmationPassed": passed,
        "manualReview": review,
        "manualReviewGate": {
            "majorFindings": major_findings,
            "minorDocuments": minor_documents,
            "minorFindings": minor_findings,
            "passDocuments": pass_count,
            "passed": review_pass,
        },
        "packetSha256": packet_sha256,
        "schemaVersion": 1,
        "verifiedAt": "2026-08-17",
    }
    atomic_write(output, final)
    return final


def dry_run() -> dict[str, object]:
    protocol = load_protocol()
    sources = require_mapping(protocol.get("sources"), "protocol sources")
    requests = [request_for(str(sources[document_id])) for document_id in DOCUMENT_IDS]
    return {
        "calls": 20,
        "documentIds": list(DOCUMENT_IDS),
        "maximumConservativeCostCredits": format(
            sum(
                (CANDIDATE.conservative_reserve(request) for request in requests),
                Decimal(0),
            ),
            "f",
        ),
        "model": CANDIDATE.model,
        "payloadSha256s": request_hashes(protocol),
        "protocolBindings": protocol["bindings"],
        "route": CANDIDATE.route,
        "runnerSha256": sha256_file(Path(__file__)),
        "sourceScore": protocol["sourceScore"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--live", action="store_true")
    modes.add_argument("--aggregate", action="store_true")
    modes.add_argument("--blind-packet", action="store_true")
    modes.add_argument("--finalize-review", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--aggregate-output", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--budget")
    parser.add_argument("--max-new-calls", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(canonical_json_bytes(dry_run()).decode("utf-8"), end="")
        return 0
    if args.aggregate:
        result = write_aggregate(args.checkpoint, args.aggregate_output)
        print(json.dumps(result["automatedGate"], sort_keys=True))
        return 0
    if args.blind_packet:
        packet = build_blind_packet(args.checkpoint, args.packet)
        print(
            json.dumps({"packet": str(args.packet), "sha256": packet["packetSha256"]})
        )
        return 0
    if args.finalize_review:
        if args.review is None:
            raise SystemExit("--finalize-review requires --review")
        result = finalize_review(args.checkpoint, args.packet, args.review, args.output)
        print(json.dumps({"passed": result["finalConfirmationPassed"]}))
        return 0
    if args.budget is None or args.max_new_calls is None:
        raise SystemExit("--live requires --budget and --max-new-calls")
    print(
        json.dumps(
            run_live(
                args.checkpoint,
                decimal(args.budget, "budget"),
                args.max_new_calls,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
