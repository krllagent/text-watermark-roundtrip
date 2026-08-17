"""Run the cost-first six-document Luna development gate.

The default mode is local-only. Live mode sends each frozen document at most
once, checkpoints before and after every request, and leaves any uncertain
request as a terminal tombstone. The blind packet contains no model or price
labels. Terra is considered only if the separately reviewed Luna phase fails.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile
import time
from typing import Any
from urllib.request import Request, urlopen

from corpus_contract import canonical_json_bytes
from run_experiment import (
    fidelity_metrics,
    load_experiment_config,
    load_reviewed_corpus,
)
from run_verified_paraphrase import load_verified_paraphrase_config
from unmark import (
    ChatCompletion,
    OpenRouterClient,
    ProviderHTTPError,
    ProviderResponseError,
    StageRequest,
    _urlopen_transport,
    build_v4_draft_request,
    canonicalize_placeholders,
    json_safe_value,
    protect_tokens,
    request_messages,
    request_utf8_size,
    restore_tokens,
    result_validation_issues,
)
from watermark_toy import Document, score_corpus, score_text


ROOT = Path(__file__).resolve().parent
SCRIPT_VERSION = "model-canary-luna-v1"
MODEL = "openai/gpt-5.6-luna"
EXPECTED_MODELS = (MODEL, f"{MODEL}-20260709")
PROVIDER = "Azure"
PROVIDER_TAG = "azure/eu"
DOCUMENT_IDS = ("doc-11", "doc-12", "doc-15", "doc-20", "doc-03", "doc-19")
CATALOG_URL = "https://openrouter.ai/api/v1/endpoints/zdr"
PROMPT_PRICE = Decimal("0.22")
COMPLETION_PRICE = Decimal("1.32")
CACHE_READ_PRICE = Decimal("0.022")
CACHE_WRITE_PRICE = Decimal("0.275")
MAX_COMPLETION_TOKENS = 4096
PROMPT_TOKEN_RESERVE = 2048
SOURCE_ACTIVE = 33
SOURCE_HITS = 33
V9_COMMIT = "e7a8a8e2cabfa8887bcaad0ef5eeb0fa723ebe50"
V9_RESULT = ROOT / "results" / "final-holdout-controls-v9.json"
V9_RESULT_SHA256 = "3d37c2c0400888258192e4106e2b349cf06a558bb7e4072e489708c0aaa571c2"
V9_PLAN_SHA256 = "6b451bcc89a025f8cd4995c47e1d9ecce09e75a5656857f2425cbec20f7c7e70"
V9_MANIFEST_SHA256 = "83e17ea83f9a9358db764714a714aceeaf3c3b534afa418cf5e3eb60dfd67eff"
V4_CONFIG = ROOT / "fixtures" / "verified-paraphrase-config-v4.json"
V4_RESULT = ROOT / "results" / "verified-paraphrase-raw-v4.json"
V4_CONFIG_SHA256 = "f1575a361e47ec8c4275da62113a845998a5e9257d5053afb01203417aa6f32c"
V4_RESULT_SHA256 = "68d488111b069ff8aff46ccc171dca6eb732b23453d5f488c506d7c1c1e9aac4"
SOURCE_SHA256S = {
    "doc-11": "cd664122178c90cfd6f35e0df18e0cdd7c376692d46aacda34405af1af39f142",
    "doc-12": "282e89cf9f7634881ac8008eba2be45f7e243c5a1e5d30c5260461befcef2cb0",
    "doc-15": "4e157e4ee8ad1dff356306b244d33ff5a87ecd94480deb226fec54282522f283",
    "doc-20": "694c238f1c9d9aad4560eb7467706fe9050aad6b9cab1206b57f005381bc50c0",
    "doc-03": "a9ab9578a9dd8aa7a858c565711e67fa69f6ca9680dbc4225928f53faa9f2b00",
    "doc-19": "f550a09791911dfe5ab3ac8bfd955796c9fd15d4ec40e61508779ffd36b2e388",
}
PAYLOAD_SHA256S = {
    "doc-11": "ff353bbfce038207948585d271dfc65aef7652ca98368219da704b62bfdb2b2c",
    "doc-12": "df9d25c5a37fb317b7a506dd07c0092f282c131a410cc86931b57fc753f099e1",
    "doc-15": "a69ad199586b98286494c3f9a4ae24b6a75c8e5f78e14884c1a812ea8591c75c",
    "doc-20": "2b7859cf9b61ae0dae512f87ad36fbce2c30d5e6de0e175f91488878502bde5c",
    "doc-03": "c6eb4cb2fcb49dcaaeb12abdeac49aa1fd7fa01e8effb5046cdf29dbedf8bd46",
    "doc-19": "1d2954d176ebcca4c141e29f1aefb13bb68dc2c6e92d597c169048e6dd147c7d",
}
DEFAULT_CHECKPOINT = ROOT / "results" / "model-canary-luna-checkpoint-v1.json"
DEFAULT_PACKET = ROOT / "results" / "model-canary-luna-blind-v1.json"
DEFAULT_FINAL = ROOT / "results" / "model-canary-luna-final-v1.json"
REVIEW_CRITERIA = (
    "lost_claim",
    "added_claim",
    "changed_claim",
    "number_or_entity",
    "causality",
    "negation",
    "scope_or_certainty",
    "example_or_caveat",
    "paragraph_role_or_order",
    "exact_string",
)


class CanaryError(Exception):
    """Expected refusal before or after a provider call."""


class CaptureTransport:
    """Capture the exact outgoing wire body without retaining Authorization."""

    def __init__(self, delegate: Any = _urlopen_transport) -> None:
        self.delegate = delegate
        self.last_request: dict[str, object] | None = None

    def __call__(
        self,
        endpoint: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> Mapping[str, Any]:
        decoded = json.loads(body.decode("utf-8"))
        self.last_request = {
            "body": decoded,
            "endpoint": endpoint,
            "headers": {
                key: value
                for key, value in headers.items()
                if key.lower() != "authorization"
            },
            "timeoutSeconds": timeout,
        }
        return self.delegate(endpoint, dict(headers), body, timeout)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, value: Mapping[str, object]) -> None:
    content = canonical_json_bytes(dict(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanaryError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise CanaryError(f"{label} must be an object")
    return value


def load_sources() -> dict[str, str]:
    if sha256_file(V4_CONFIG) != V4_CONFIG_SHA256:
        raise CanaryError("v4 config hash changed")
    if sha256_file(V4_RESULT) != V4_RESULT_SHA256:
        raise CanaryError("v4 result hash changed")
    raw = load_json(V4_RESULT, "v4 result")
    methods = raw.get("methods")
    if not isinstance(methods, list) or len(methods) != 1:
        raise CanaryError("v4 result must contain one method")
    method = methods[0]
    if not isinstance(method, dict) or not isinstance(method.get("documents"), list):
        raise CanaryError("v4 documents are missing")
    rows = {
        row.get("documentId"): row
        for row in method["documents"]
        if isinstance(row, dict)
    }
    output: dict[str, str] = {}
    for document_id in DOCUMENT_IDS:
        row = rows.get(document_id)
        text = None if row is None else row.get("markedInputText")
        if not isinstance(text, str) or not text.strip():
            raise CanaryError(f"missing frozen source: {document_id}")
        output[document_id] = text
    actual_hashes = {
        key: hashlib.sha256(output[key].encode("utf-8")).hexdigest()
        for key in DOCUMENT_IDS
    }
    if actual_hashes != SOURCE_SHA256S:
        raise CanaryError("frozen source hashes changed")
    return output


def load_detector() -> tuple[object, object]:
    verified = load_verified_paraphrase_config(V4_CONFIG, root=ROOT)
    base = load_experiment_config(verified.base_config_path, root=ROOT)
    corpus = load_reviewed_corpus(base)
    return base, corpus


def score_sources(sources: Mapping[str, str]) -> dict[str, object]:
    base, corpus = load_detector()
    score = score_corpus(
        tuple(Document(document_id=key, text=sources[key]) for key in DOCUMENT_IDS),
        key=getattr(base, "key"),
        density_bps=getattr(base, "density_bps"),
        lexicon=getattr(corpus, "lexicon"),
        context_width=getattr(base, "context_width"),
        min_active_positions=getattr(base, "min_active_positions"),
    ).to_dict(include_documents=False)
    if (
        score.get("status") != "detected"
        or score.get("activePositions") != SOURCE_ACTIVE
        or score.get("hits") != SOURCE_HITS
    ):
        raise CanaryError("six-document source detector gate changed")
    return score


def score_outputs(outputs: Mapping[str, str]) -> dict[str, object]:
    if set(outputs) != set(DOCUMENT_IDS):
        raise CanaryError("pooled output detector requires all six documents")
    base, corpus = load_detector()
    return score_corpus(
        tuple(Document(document_id=key, text=outputs[key]) for key in DOCUMENT_IDS),
        key=getattr(base, "key"),
        density_bps=getattr(base, "density_bps"),
        lexicon=getattr(corpus, "lexicon"),
        context_width=getattr(base, "context_width"),
        min_active_positions=getattr(base, "min_active_positions"),
    ).to_dict(include_documents=True)


def validate_v9() -> dict[str, object]:
    if sha256_file(V9_RESULT) != V9_RESULT_SHA256:
        raise CanaryError("v9 result hash changed")
    raw = load_json(V9_RESULT, "v9 result")
    bindings = require_mapping(raw.get("artifactBindings"), "artifactBindings")
    for label, binding_value in bindings.items():
        binding = require_mapping(binding_value, f"artifactBindings.{label}")
        relative_path = binding.get("path")
        declared_sha256 = binding.get("sha256")
        if (
            not isinstance(relative_path, str)
            or not isinstance(declared_sha256, str)
            or sha256_file(ROOT / relative_path) != declared_sha256
        ):
            raise CanaryError(f"v9 artifact binding changed: {label}")
    if require_mapping(bindings.get("plan"), "plan").get("sha256") != V9_PLAN_SHA256:
        raise CanaryError("v9 plan binding changed")
    if (
        require_mapping(bindings.get("markedManifest"), "markedManifest").get("sha256")
        != V9_MANIFEST_SHA256
    ):
        raise CanaryError("v9 marked-manifest binding changed")
    marked = require_mapping(
        require_mapping(raw.get("prepaidGate"), "prepaidGate").get("marked"),
        "prepaidGate.marked",
    )
    if (
        marked.get("status") != "detected"
        or marked.get("activePositions") != 42
        or marked.get("hits") != 38
    ):
        raise CanaryError("v9 marked baseline changed")
    return {
        "commit": V9_COMMIT,
        "markedActivePositions": 42,
        "markedHits": 38,
        "markedStatus": "detected",
        "resultSha256": V9_RESULT_SHA256,
    }


def fetch_catalog() -> dict[str, object]:
    request = Request(
        CATALOG_URL,
        headers={"Accept": "application/json", "User-Agent": f"{SCRIPT_VERSION}/1"},
        method="GET",
    )
    with urlopen(request, timeout=30) as response:
        content = response.read(8 * 1024 * 1024 + 1)
    if len(content) > 8 * 1024 * 1024:
        raise CanaryError("ZDR catalog is oversized")
    raw = json.loads(content.decode("utf-8"))
    if not isinstance(raw, dict):
        raise CanaryError("ZDR catalog must be an object")
    rows = raw.get("data")
    if not isinstance(rows, list):
        raise CanaryError("ZDR catalog data is missing")
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("model_id") == MODEL
        and row.get("tag") == PROVIDER_TAG
    ]
    if len(matches) != 1:
        raise CanaryError("exact Luna azure/eu ZDR endpoint is unavailable")
    row = matches[0]
    uptime = row.get("uptime_last_5m")
    if row.get("status") != 0 or uptime is None or float(uptime) != 100.0:
        raise CanaryError("Luna azure/eu ZDR endpoint is unhealthy")
    if (
        row.get("provider_name") != PROVIDER
        or row.get("name") != "Azure | openai/gpt-5.6-luna-20260709"
        or row.get("supports_implicit_caching") is not False
    ):
        raise CanaryError("Luna azure/eu endpoint identity changed")
    supported = row.get("supported_parameters")
    if not isinstance(supported, list) or not {
        "max_completion_tokens",
        "reasoning",
        "reasoning_effort",
        "seed",
    }.issubset(set(supported)):
        raise CanaryError("Luna azure/eu endpoint lacks a required parameter")
    if "temperature" in supported:
        raise CanaryError("Luna azure/eu unexpectedly accepts temperature")
    pricing = require_mapping(row.get("pricing"), "catalog pricing")
    expected = {
        "prompt": PROMPT_PRICE,
        "completion": COMPLETION_PRICE,
        "input_cache_read": CACHE_READ_PRICE,
        "input_cache_write": CACHE_WRITE_PRICE,
    }
    for field, per_million in expected.items():
        if decimal(pricing.get(field), field) * Decimal(1_000_000) != per_million:
            raise CanaryError(f"Luna azure/eu {field} price changed")
    return {
        "catalogUrl": CATALOG_URL,
        "endpointName": row["name"],
        "provider": PROVIDER,
        "status": 0,
        "tag": PROVIDER_TAG,
        "uptimeLast5m": uptime,
    }


def expected_payload(request: StageRequest) -> dict[str, object]:
    return {
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "messages": list(request_messages(request)),
        "model": MODEL,
        "provider": {
            "allow_fallbacks": False,
            "data_collection": "deny",
            "max_price": {
                "completion": float(COMPLETION_PRICE),
                "prompt": float(PROMPT_PRICE),
            },
            "order": [PROVIDER_TAG],
            "require_parameters": True,
            "zdr": True,
        },
        "reasoning": {"effort": "medium"},
        "seed": 20260817,
        "stream": False,
    }


def object_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def record_sha256(record: Mapping[str, object]) -> str:
    unsigned = {key: value for key, value in record.items() if key != "recordSha256"}
    return object_sha256(unsigned)


def validate_route_record(value: object) -> None:
    route = require_mapping(value, "route preflight")
    expected = {
        "catalogUrl": CATALOG_URL,
        "endpointName": "Azure | openai/gpt-5.6-luna-20260709",
        "provider": PROVIDER,
        "status": 0,
        "tag": PROVIDER_TAG,
        "uptimeLast5m": 100,
    }
    if dict(route) != expected:
        raise CanaryError("completed call route preflight changed")


def build_client(transport: CaptureTransport) -> OpenRouterClient:
    return OpenRouterClient.from_env(
        transport=transport,
        timeout=180,
        provider_order=(PROVIDER_TAG,),
        allow_fallbacks=False,
        require_parameters=True,
        reasoning_effort="medium",
        temperature=None,
        max_tokens=MAX_COMPLETION_TOKENS,
        token_cap_field="max_completion_tokens",
        seed=20260817,
        max_prompt_price=float(PROMPT_PRICE),
        max_completion_price=float(COMPLETION_PRICE),
    )


def conservative_reserve(request: StageRequest) -> Decimal:
    prompt_tokens = request_utf8_size(request) + PROMPT_TOKEN_RESERVE
    return (
        Decimal(prompt_tokens) * PROMPT_PRICE
        + Decimal(MAX_COMPLETION_TOKENS) * COMPLETION_PRICE
    ) / Decimal(1_000_000)


def expected_cost(completion: ChatCompletion) -> Decimal:
    usage = completion.usage
    uncached = (
        usage.prompt_tokens - usage.cached_prompt_tokens - usage.cache_write_tokens
    )
    return (
        Decimal(uncached) * PROMPT_PRICE
        + Decimal(usage.cached_prompt_tokens) * CACHE_READ_PRICE
        + Decimal(usage.cache_write_tokens) * CACHE_WRITE_PRICE
        + Decimal(usage.completion_tokens) * COMPLETION_PRICE
    ) / Decimal(1_000_000)


def validate_completion(completion: ChatCompletion, request: StageRequest) -> None:
    if completion.finish_reason != "stop":
        raise CanaryError("completion did not finish with stop")
    if completion.model not in EXPECTED_MODELS or completion.provider != PROVIDER:
        raise CanaryError("provider returned an unexpected model or provider")
    metadata = completion.openrouter_metadata
    if not isinstance(metadata, Mapping):
        raise CanaryError("OpenRouter routing metadata is required")
    if (
        metadata.get("strategy") != "direct"
        or metadata.get("attempt") != 1
        or metadata.get("requested") != MODEL
        or metadata.get("is_byok") is not False
        or metadata.get("pipeline") not in (None, [])
    ):
        raise CanaryError("OpenRouter routing metadata violates the frozen route")
    endpoints = require_mapping(metadata.get("endpoints"), "metadata endpoints")
    available = endpoints.get("available")
    if not isinstance(available, list):
        raise CanaryError("OpenRouter endpoint metadata is missing")
    selected = [
        row for row in available if isinstance(row, dict) and row.get("selected")
    ]
    if len(selected) != 1:
        raise CanaryError("OpenRouter did not select exactly one endpoint")
    selected_provider = first_present(
        selected[0], "provider", "provider_name", "providerName"
    )
    selected_model = first_present(selected[0], "model", "model_id", "modelId")
    if selected_provider != PROVIDER or selected_model not in EXPECTED_MODELS:
        raise CanaryError("OpenRouter selected endpoint metadata changed")
    attempts = metadata.get("attempts")
    if attempts is not None:
        if not isinstance(attempts, list) or len(attempts) != 1:
            raise CanaryError("OpenRouter used more than one attempt")
        attempt = require_mapping(attempts[0], "metadata attempt")
        if attempt.get("status") != 200:
            raise CanaryError("OpenRouter attempt did not return 200")
    if (
        completion.usage.prompt_tokens
        > request_utf8_size(request) + PROMPT_TOKEN_RESERVE
    ):
        raise CanaryError("prompt token usage exceeds the frozen reserve")
    if completion.usage.completion_tokens > MAX_COMPLETION_TOKENS:
        raise CanaryError("completion token usage exceeds the frozen cap")
    if completion.usage.cost != expected_cost(completion):
        raise CanaryError("provider cost differs from the pinned endpoint prices")


def initial_checkpoint(
    sources: Mapping[str, str],
    source_score: Mapping[str, object],
    v9: Mapping[str, object],
) -> dict[str, object]:
    return {
        "blindMapping": None,
        "calls": [],
        "documentIds": list(DOCUMENT_IDS),
        "inFlight": None,
        "model": MODEL,
        "route": PROVIDER_TAG,
        "runnerSha256": sha256_file(Path(__file__)),
        "schemaVersion": 1,
        "scriptVersion": SCRIPT_VERSION,
        "sourceScore": dict(source_score),
        "sourceSha256s": {
            key: hashlib.sha256(sources[key].encode("utf-8")).hexdigest()
            for key in DOCUMENT_IDS
        },
        "terminalFailure": None,
        "unmarkClientSha256": sha256_file(ROOT / "unmark.py"),
        "v9Binding": dict(v9),
    }


def validate_checkpoint(
    state: Mapping[str, object],
    sources: Mapping[str, str],
    source_score: Mapping[str, object],
    v9: Mapping[str, object],
) -> None:
    expected = initial_checkpoint(sources, source_score, v9)
    for field in (
        "documentIds",
        "model",
        "route",
        "runnerSha256",
        "schemaVersion",
        "scriptVersion",
        "sourceScore",
        "sourceSha256s",
        "unmarkClientSha256",
        "v9Binding",
    ):
        if state.get(field) != expected[field]:
            raise CanaryError(f"checkpoint binding changed: {field}")
    calls = state.get("calls")
    if not isinstance(calls, list):
        raise CanaryError("checkpoint calls must be a list")
    if len(calls) > len(DOCUMENT_IDS):
        raise CanaryError("checkpoint has too many completed calls")
    for index, row_value in enumerate(calls):
        if not isinstance(row_value, dict):
            raise CanaryError("checkpoint completed call must be an object")
        document_id = DOCUMENT_IDS[index]
        call_id = f"luna:{document_id}"
        source = sources[document_id]
        request = build_v4_draft_request(protect_tokens(source).masked)
        payload = expected_payload(request)
        payload_sha256 = object_sha256(payload)
        if payload_sha256 != PAYLOAD_SHA256S[document_id]:
            raise CanaryError("frozen request payload changed")
        if (
            row_value.get("callId") != call_id
            or row_value.get("documentId") != document_id
            or row_value.get("sourceText") != source
            or row_value.get("requestBody") != payload
            or row_value.get("requestSha256") != payload_sha256
        ):
            raise CanaryError("checkpoint completed call binding changed")
        wire = require_mapping(row_value.get("wireRequest"), "wire request")
        headers = require_mapping(wire.get("headers"), "wire headers")
        if wire.get("body") != payload or any(
            key.lower() == "authorization" for key in headers
        ):
            raise CanaryError("checkpoint wire evidence changed")
        validate_route_record(row_value.get("routePreflight"))
        require_mapping(row_value.get("completion"), "completion")
        require_mapping(row_value.get("analysis"), "analysis")
        digest = row_value.get("recordSha256")
        if not isinstance(digest, str) or digest != record_sha256(row_value):
            raise CanaryError("checkpoint completed call digest changed")
    if state.get("inFlight") is not None or state.get("terminalFailure") is not None:
        raise CanaryError("checkpoint contains a terminal no-redispatch tombstone")


def load_or_create_checkpoint(
    path: Path,
    sources: Mapping[str, str],
    source_score: Mapping[str, object],
    v9: Mapping[str, object],
) -> dict[str, object]:
    if not path.exists():
        state = initial_checkpoint(sources, source_score, v9)
        atomic_write(path, state)
        return state
    state = load_json(path, "checkpoint")
    validate_checkpoint(state, sources, source_score, v9)
    return state


def validated_blind_mapping(state: Mapping[str, object]) -> dict[str, str]:
    mapping = state.get("blindMapping")
    if not isinstance(mapping, list) or len(mapping) != len(DOCUMENT_IDS):
        raise CanaryError("blind mapping is invalid")
    if not all(isinstance(row, dict) for row in mapping):
        raise CanaryError("blind mapping rows must be objects")
    document_ids = [row.get("documentId") for row in mapping]
    pair_ids = [row.get("pairId") for row in mapping]
    if (
        document_ids != list(DOCUMENT_IDS)
        or any(not isinstance(pair_id, str) for pair_id in pair_ids)
        or len(set(pair_ids)) != len(DOCUMENT_IDS)
    ):
        raise CanaryError("blind mapping is not a unique frozen document mapping")
    return dict(zip(document_ids, pair_ids, strict=True))


def analyze_output(
    document_id: str, source: str, completion: ChatCompletion
) -> dict[str, object]:
    protected = protect_tokens(source)
    normalized = canonicalize_placeholders(completion.content, protected.tokens)
    issues = list(result_validation_issues(protected.masked, normalized, None))
    restored = restore_tokens(normalized, protected.tokens)
    fidelity = fidelity_metrics(source, restored)
    base, corpus = load_detector()
    detector = score_text(
        restored,
        key=getattr(base, "key"),
        density_bps=getattr(base, "density_bps"),
        lexicon=getattr(corpus, "lexicon"),
        document_id=document_id,
        context_width=getattr(base, "context_width"),
        min_active_positions=getattr(base, "min_active_positions"),
    ).to_dict()
    protected_ok = (
        require_mapping(fidelity.get("protectedTokens"), "protected metrics").get(
            "exactlyRestored"
        )
        is True
    )
    if not protected_ok:
        issues.append(
            {"code": "protected_values", "message": "protected values changed"}
        )
    return {
        "detector": detector,
        "fidelity": fidelity,
        "maskedOutputText": normalized,
        "pipelineIssues": issues,
        "restoredOutputText": restored,
    }


def run_live(path: Path, budget: Decimal, max_new_calls: int) -> dict[str, object]:
    if budget <= 0 or max_new_calls <= 0:
        raise CanaryError("budget and max-new-calls must be positive")
    sources = load_sources()
    source_score = score_sources(sources)
    v9 = validate_v9()
    route = fetch_catalog()
    state = load_or_create_checkpoint(path, sources, source_score, v9)
    calls = state["calls"]
    assert isinstance(calls, list)
    if len(calls) >= len(DOCUMENT_IDS):
        raise CanaryError("all Luna calls are already complete")
    selected = DOCUMENT_IDS[len(calls) : len(calls) + max_new_calls]
    requests: list[tuple[str, str, StageRequest, Decimal]] = []
    required = Decimal(0)
    for document_id in selected:
        source = sources[document_id]
        request = build_v4_draft_request(protect_tokens(source).masked)
        reserve = conservative_reserve(request)
        requests.append((document_id, source, request, reserve))
        required += reserve
    if required > budget:
        raise CanaryError(f"budget {budget} is below conservative reserve {required}")
    spent = Decimal(0)
    for document_id, source, request, reserve in requests:
        call_id = f"luna:{document_id}"
        payload = expected_payload(request)
        payload_sha256 = object_sha256(payload)
        if payload_sha256 != PAYLOAD_SHA256S[document_id]:
            raise CanaryError("frozen request payload changed before dispatch")
        transport = CaptureTransport()
        client = build_client(transport)
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
            completion = client.complete(request, model=MODEL)
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
                raise CanaryError(
                    "actual OpenRouter wire body differs from frozen payload"
                )
            validate_completion(completion, request)
            analysis = analyze_output(document_id, source, completion)
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
                "provider call failed after dispatch; checkpoint is terminal and the call will not be retried"
            ) from error
        record = {
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


def build_blind_packet(checkpoint_path: Path, output_path: Path) -> dict[str, object]:
    sources = load_sources()
    source_score = score_sources(sources)
    v9 = validate_v9()
    state = load_or_create_checkpoint(checkpoint_path, sources, source_score, v9)
    calls = state.get("calls")
    if not isinstance(calls, list) or len(calls) != len(DOCUMENT_IDS):
        raise CanaryError("blind packet requires all six accepted Luna calls")
    mapping = state.get("blindMapping")
    if mapping is None:
        mapping = [
            {"documentId": document_id, "pairId": f"pair-{secrets.token_hex(8)}"}
            for document_id in DOCUMENT_IDS
        ]
        state["blindMapping"] = mapping
        atomic_write(checkpoint_path, state)
    pair_by_document = validated_blind_mapping(state)
    pairs: list[dict[str, object]] = []
    for call in calls:
        if not isinstance(call, dict):
            raise CanaryError("checkpoint call is invalid")
        document_id = call.get("documentId")
        analysis = require_mapping(call.get("analysis"), "call analysis")
        pair_id = pair_by_document.get(document_id)
        if not isinstance(pair_id, str) or not isinstance(document_id, str):
            raise CanaryError("blind mapping is incomplete")
        pairs.append(
            {
                "candidateText": analysis.get("restoredOutputText"),
                "pairId": pair_id,
                "sourceText": sources[document_id],
            }
        )
    packet: dict[str, object] = {
        "criteria": list(REVIEW_CRITERIA),
        "instructions": (
            "Read every pair line by line. Mark pass only when every claim, number, "
            "entity, causal direction, negation, scope, certainty, example, caveat, "
            "paragraph role, and exact string is preserved. Ignore style differences."
        ),
        "pairs": pairs,
        "requiredOutput": {
            "reviewFields": ["pairId", "verdict", "findings"],
            "verdicts": ["pass", "minor", "major"],
        },
        "schemaVersion": 1,
    }
    packet["packetSha256"] = hashlib.sha256(canonical_json_bytes(packet)).hexdigest()
    atomic_write(output_path, packet)
    return packet


def finalize_review(
    checkpoint_path: Path,
    packet_path: Path,
    review_path: Path,
    output_path: Path,
) -> dict[str, object]:
    sources = load_sources()
    state = load_or_create_checkpoint(
        checkpoint_path,
        sources,
        score_sources(sources),
        validate_v9(),
    )
    packet = load_json(packet_path, "blind packet")
    review = load_json(review_path, "manual review")
    declared_packet_sha256 = packet.get("packetSha256")
    unsigned_packet = {
        key: value for key, value in packet.items() if key != "packetSha256"
    }
    if (
        not isinstance(declared_packet_sha256, str)
        or declared_packet_sha256 != object_sha256(unsigned_packet)
        or review.get("packetSha256") != declared_packet_sha256
    ):
        raise CanaryError("manual review is bound to a different packet")
    pairs = packet.get("pairs")
    reviews = review.get("reviews")
    if not isinstance(pairs, list) or not isinstance(reviews, list):
        raise CanaryError("packet or review rows are missing")
    if len(pairs) != 6 or len(reviews) != 6:
        raise CanaryError("manual review must contain exactly six unique pairs")
    if not all(isinstance(row, dict) for row in pairs + reviews):
        raise CanaryError("packet and review rows must be objects")
    pair_by_document = validated_blind_mapping(state)
    calls = state.get("calls")
    if not isinstance(calls, list) or len(calls) != len(DOCUMENT_IDS):
        raise CanaryError("manual review requires all six completed calls")
    expected_pairs: list[dict[str, object]] = []
    for call in calls:
        if not isinstance(call, dict):
            raise CanaryError("completed call must be an object")
        document_id = call.get("documentId")
        analysis = require_mapping(call.get("analysis"), "call analysis")
        if not isinstance(document_id, str):
            raise CanaryError("completed call document is invalid")
        expected_pairs.append(
            {
                "candidateText": analysis.get("restoredOutputText"),
                "pairId": pair_by_document[document_id],
                "sourceText": sources[document_id],
            }
        )
    if pairs != expected_pairs:
        raise CanaryError("blind packet no longer matches the completed calls")
    pair_ids = [row.get("pairId") for row in pairs]
    review_ids = [row.get("pairId") for row in reviews]
    if (
        any(not isinstance(pair_id, str) for pair_id in pair_ids + review_ids)
        or len(set(pair_ids)) != 6
        or len(set(review_ids)) != 6
        or set(review_ids) != set(pair_ids)
    ):
        raise CanaryError("manual review must cover every blind pair exactly once")
    review_by_id = {row["pairId"]: row for row in reviews}
    manual_pass = True
    for pair_id in pair_ids:
        row = review_by_id[pair_id]
        verdict = row.get("verdict")
        findings = row.get("findings")
        if verdict not in {"pass", "minor", "major"} or not isinstance(findings, list):
            raise CanaryError("manual review row is invalid")
        if (verdict == "pass") != (len(findings) == 0):
            raise CanaryError("pass must have no findings and findings must not pass")
        manual_pass = manual_pass and verdict == "pass"
    local_pass = True
    distances: list[float] = []
    outputs: dict[str, str] = {}
    for call in calls:
        assert isinstance(call, dict)
        analysis = require_mapping(call.get("analysis"), "call analysis")
        fidelity = require_mapping(analysis.get("fidelity"), "fidelity")
        word = require_mapping(fidelity.get("wordLevenshtein"), "wordLevenshtein")
        distance = float(word.get("normalizedDistance", 0))
        distances.append(distance)
        local_pass = (
            local_pass and not analysis.get("pipelineIssues") and distance >= 0.15
        )
        document_id = call.get("documentId")
        output_text = analysis.get("restoredOutputText")
        if not isinstance(document_id, str) or not isinstance(output_text, str):
            raise CanaryError("completed output text is missing")
        outputs[document_id] = output_text
    pooled_detector = score_outputs(outputs)
    detector_pass = (
        pooled_detector.get("status") == "not_detected"
        and int(pooled_detector.get("activePositions", 0)) >= 20
    )
    passed = manual_pass and local_pass and detector_pass
    final = {
        "checkpoint": state,
        "manualReview": review,
        "packetSha256": packet.get("packetSha256"),
        "selection": {
            "detectorGatePassed": detector_pass,
            "lunaPassed": passed,
            "meanWordDistance": sum(distances) / len(distances),
            "minimumWordDistance": min(distances),
            "nextStep": "final_holdout" if passed else "run_terra_development_gate",
            "pooledOutputDetector": pooled_detector,
            "selectedModel": MODEL if passed else None,
            "terraCalls": 0,
        },
        "verifiedAt": "2026-08-17",
    }
    atomic_write(output_path, final)
    return final


def dry_run() -> dict[str, object]:
    sources = load_sources()
    requests = {
        document_id: build_v4_draft_request(protect_tokens(sources[document_id]).masked)
        for document_id in DOCUMENT_IDS
    }
    payload_hashes = {
        key: object_sha256(expected_payload(value)) for key, value in requests.items()
    }
    if payload_hashes != PAYLOAD_SHA256S:
        raise CanaryError("frozen request payload hashes changed")
    return {
        "calls": 6,
        "documentIds": list(DOCUMENT_IDS),
        "maximumConservativeCostCredits": format(
            sum(
                (conservative_reserve(request) for request in requests.values()),
                Decimal(0),
            ),
            "f",
        ),
        "model": MODEL,
        "payloadSha256s": payload_hashes,
        "route": PROVIDER_TAG,
        "runnerSha256": sha256_file(Path(__file__)),
        "sourceScore": score_sources(sources),
        "unmarkClientSha256": sha256_file(ROOT / "unmark.py"),
        "v9Binding": validate_v9(),
    }


def require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CanaryError(f"{label} must be an object")
    return value


def decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool):
        raise CanaryError(f"{label} must be a decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise CanaryError(f"{label} must be a decimal") from error
    if not parsed.is_finite() or parsed < 0:
        raise CanaryError(f"{label} must be a nonnegative decimal")
    return parsed


def first_present(value: Mapping[str, Any], *names: str) -> object:
    for name in names:
        if name in value:
            return value[name]
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--live", action="store_true")
    modes.add_argument("--blind-packet", action="store_true")
    modes.add_argument("--finalize-review", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
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
    if args.blind_packet:
        packet = build_blind_packet(args.checkpoint, args.packet)
        print(
            json.dumps(
                {"packet": str(args.packet), "packetSha256": packet["packetSha256"]}
            )
        )
        return 0
    if args.finalize_review:
        if args.review is None:
            raise SystemExit("--finalize-review requires --review")
        final = finalize_review(
            args.checkpoint,
            args.packet,
            args.review,
            args.output,
        )
        print(json.dumps(final["selection"], sort_keys=True))
        return 0
    if args.budget is None or args.max_new_calls is None:
        raise SystemExit("--live requires --budget and --max-new-calls")
    result = run_live(
        args.checkpoint,
        decimal(args.budget, "budget"),
        args.max_new_calls,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
