"""Run the frozen CPU-only Stage-1 watermark transformation experiment.

Dry-run reads only frozen local evidence. Live mode uses one pinned OpenRouter
endpoint and durably records an in-flight request before dispatch. It never
redispatches unless the operator explicitly confirms that the prior attempt was
not charged.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time
from typing import Callable, Mapping, Protocol, Sequence

from corpus_contract import canonical_json_bytes, validate_context_reviews
from text_contract import WORD_RE, analyze_text, find_protected_spans
from unmark import (
    ChatCompletion,
    CompletionUsage,
    OpenRouterClient,
    PlaceholderError,
    ProtectedToken,
    ProviderError,
    ProviderResponseError,
    ValidationError,
    build_backward_prompt,
    build_forward_prompt,
    build_paraphrase_prompt,
    build_synonym_prompt,
    json_safe_value,
    protect_tokens,
    result_validation_issues,
    restore_tokens,
    validate_intermediate,
    validate_placeholders,
)
from watermark_toy import (
    DETECTION_ALPHA,
    SCHEME_VERSION,
    Document,
    EncodeResult,
    ScoreResult,
    SynonymLexicon,
    binomial_tail_probability,
    compare_active_fingerprints,
    encode_text,
    load_lexicon,
    run_wrong_key_controls,
    score_text,
)


CONFIG_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 2
PIVOT_REENCODE_MULTIPLIER = 8
PROMPT_BILLING_OVERHEAD_TOKENS = 2_048
EXPECTED_DOCUMENT_IDS = tuple(f"doc-{index:02d}" for index in range(1, 21))
EXPECTED_METHODS = (
    ("none", "none", None),
    ("synonyms", "synonyms", None),
    ("roundtrip-de", "roundtrip", "de"),
    ("roundtrip-zh", "roundtrip", "zh"),
    ("paraphrase", "paraphrase", None),
)
EXPECTED_PROVIDER_ORDER = ("deepinfra/bf16",)
EXPECTED_RESPONSE_PROVIDERS = ("DeepInfra",)
EXPECTED_MODEL = "qwen/qwen3.5-9b"
EXPECTED_ENDPOINT_MODEL = "qwen/qwen3.5-9b-20260310"
EXPECTED_RESPONSE_MODELS = (EXPECTED_MODEL, EXPECTED_ENDPOINT_MODEL)
EXPECTED_PROMPT_PRICE_USD = Decimal("0.10")
EXPECTED_COMPLETION_PRICE_USD = Decimal("0.15")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ExperimentError(Exception):
    """Base class for expected experiment-runner failures."""


class BudgetError(ExperimentError):
    """Raised before a paid call when the explicit budget is insufficient."""


class CheckpointError(ExperimentError):
    """Raised when a checkpoint cannot be safely resumed."""


class ControlGateError(ExperimentError):
    """Raised before any paid call when deterministic controls are not fresh."""


class ResponseContractError(ExperimentError):
    """Raised after preserving a paid response that violates the frozen route."""


class CallLimitReached(ExperimentError):
    """Intentional pause after the requested number of new matrix calls."""

    def __init__(self, *, completed_calls: int, new_calls: int) -> None:
        super().__init__(
            f"paused after {new_calls} new call(s); {completed_calls} total call(s) checkpointed"
        )
        self.completed_calls = completed_calls
        self.new_calls = new_calls


class CompletionClient(Protocol):
    def complete(self, prompt: str, *, model: str) -> ChatCompletion: ...


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    method: str
    pivot: str | None

    @property
    def calls_per_document(self) -> int:
        if self.method == "none":
            return 0
        return 2 if self.method == "roundtrip" else 1


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    root: Path
    raw: dict[str, object]
    sha256: str
    experiment_version: str
    verified_at: str
    methodology: str
    sources: tuple[dict[str, str], ...]
    manifest_path: Path
    manifest_expected_sha256: str
    inventory_path: Path
    inventory_expected_sha256: str
    reviews_directory: Path
    review_bindings: tuple[tuple[Path, str], ...]
    lexicon_path: Path
    lexicon_expected_sha256: str
    endpoint_snapshot_path: Path
    endpoint_snapshot_sha256: str
    semantic_audit_plan_path: Path
    semantic_audit_plan_sha256: str
    key: bytes
    density_bps: int
    context_width: int
    min_active_positions: int
    wrong_key_count: int
    wrong_key_seed: bytes
    methods: tuple[MethodSpec, ...]
    model_forward: str
    model_backward: str
    provider_order: tuple[str, ...]
    expected_response_models: tuple[str, ...]
    expected_response_providers: tuple[str, ...]
    max_tokens: int
    prompt_price_usd_per_million: Decimal
    completion_price_usd_per_million: Decimal
    prompt_token_overhead_reserve: int
    seed: int
    timeout_seconds: float
    bootstrap_replicates: int
    bootstrap_seed: int


@dataclass(frozen=True)
class CorpusItem:
    document_id: str
    path: str
    genre: str
    title: str
    text: str
    sha256: str
    word_count: int
    eligible_positions: int
    protected_span_count: int


@dataclass(frozen=True)
class ReviewedCorpus:
    config: ExperimentConfig
    documents: tuple[CorpusItem, ...]
    manifest: dict[str, object]
    manifest_sha256: str
    inventory_sha256: str
    review_sha256s: tuple[str, ...]
    review_approval: dict[str, object]
    lexicon: SynonymLexicon
    lexicon_file_sha256: str


@dataclass(frozen=True)
class Stage1TransformationOutcome:
    text: str
    raw_final_masked_text: str
    calls: tuple[dict[str, object], ...]
    issues: tuple[dict[str, str], ...]
    restoration_mode: str

    @property
    def failed(self) -> bool:
        return bool(self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "issues": [dict(issue) for issue in self.issues],
            "rawFinalMaskedText": self.raw_final_masked_text,
            "restorationMode": self.restoration_mode,
            "status": "validation_failure" if self.failed else "accepted",
        }


def load_experiment_config(
    path: str | Path,
    *,
    root: str | Path | None = None,
) -> ExperimentConfig:
    """Load and strictly validate the frozen experiment and endpoint contract."""
    config_path = Path(path).resolve()
    root_path = (
        Path(root).resolve() if root is not None else Path(__file__).resolve().parent
    )
    raw_bytes = config_path.read_bytes()
    raw = _load_json_object(raw_bytes, "experiment config")
    if raw.get("schemaVersion") != CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported experiment config schemaVersion")
    _validate_evidence(raw, "experiment config")
    experiment_version = _nonempty_string(raw.get("experimentVersion"), "experimentVersion")

    corpus = _mapping(raw.get("corpus"), "corpus")
    if corpus.get("documentCount") != 20:
        raise ValueError("corpus.documentCount must be exactly 20")
    manifest_path = _safe_relative_path(root_path, corpus.get("manifestPath"), "manifestPath")
    manifest_expected_sha256 = _sha256_string(
        corpus.get("manifestSha256"),
        "corpus.manifestSha256",
    )
    _require_file_sha256(
        manifest_path,
        manifest_expected_sha256,
        "corpus manifest",
    )
    inventory_path = _safe_relative_path(root_path, corpus.get("inventoryPath"), "inventoryPath")
    inventory_expected_sha256 = _sha256_string(
        corpus.get("inventorySha256"),
        "corpus.inventorySha256",
    )
    _require_file_sha256(
        inventory_path,
        inventory_expected_sha256,
        "context inventory",
    )
    reviews_directory = _safe_relative_path(
        root_path,
        corpus.get("reviewsDirectory"),
        "reviewsDirectory",
    )
    raw_reviews = corpus.get("reviews")
    if not isinstance(raw_reviews, list) or not raw_reviews:
        raise ValueError("corpus.reviews must be a nonempty ordered list")
    review_bindings: list[tuple[Path, str]] = []
    seen_review_paths: set[Path] = set()
    for index, raw_review in enumerate(raw_reviews):
        review = _mapping(raw_review, f"corpus.reviews[{index}]")
        review_path = _safe_relative_path(
            root_path,
            review.get("path"),
            f"corpus.reviews[{index}].path",
        )
        if review_path.parent != reviews_directory:
            raise ValueError("every frozen review must be directly under reviewsDirectory")
        if review_path in seen_review_paths:
            raise ValueError("corpus.reviews paths must be unique")
        review_sha256 = _sha256_string(
            review.get("sha256"),
            f"corpus.reviews[{index}].sha256",
        )
        _require_file_sha256(review_path, review_sha256, f"context review {index}")
        review_bindings.append((review_path, review_sha256))
        seen_review_paths.add(review_path)

    lexicon_binding = _mapping(raw.get("lexicon"), "lexicon")
    lexicon_path = _safe_relative_path(
        root_path,
        lexicon_binding.get("path"),
        "lexicon.path",
    )
    lexicon_expected_sha256 = _sha256_string(
        lexicon_binding.get("sha256"),
        "lexicon.sha256",
    )
    _require_file_sha256(lexicon_path, lexicon_expected_sha256, "synonym lexicon")

    marker = _mapping(raw.get("marker"), "marker")
    densities = marker.get("densitiesBps")
    if densities != [500, 1000, 2000]:
        raise ValueError("marker.densitiesBps must be [500, 1000, 2000]")
    density_bps = _positive_int(marker.get("mainDensityBps"), "marker.mainDensityBps")
    if density_bps != 1000 or density_bps not in densities:
        raise ValueError("marker.mainDensityBps must be the frozen 1000 condition")
    context_width = _positive_int(marker.get("contextWidth"), "marker.contextWidth")
    min_active_positions = _positive_int(
        marker.get("minActivePositions"), "marker.minActivePositions"
    )
    key_hex = _nonempty_string(marker.get("keyHex"), "marker.keyHex")
    try:
        key = bytes.fromhex(key_hex)
    except ValueError as error:
        raise ValueError("marker.keyHex must be hexadecimal") from error
    if len(key) < 16 or key.hex() != key_hex.lower():
        raise ValueError("marker.keyHex must canonically encode at least 16 bytes")
    wrong_key_count = _positive_int(marker.get("wrongKeyCount"), "marker.wrongKeyCount")
    wrong_key_seed_hex = _nonempty_string(
        marker.get("wrongKeySeedHex"),
        "marker.wrongKeySeedHex",
    )
    try:
        wrong_key_seed = bytes.fromhex(wrong_key_seed_hex)
    except ValueError as error:
        raise ValueError("marker.wrongKeySeedHex must be hexadecimal") from error
    if len(wrong_key_seed) < 16 or wrong_key_seed.hex() != wrong_key_seed_hex.lower():
        raise ValueError(
            "marker.wrongKeySeedHex must canonically encode at least 16 bytes"
        )

    billing = _mapping(raw.get("billing"), "billing")
    _require_equal(billing.get("creditBaseCurrency"), "USD", "billing.creditBaseCurrency")
    if _decimal(billing.get("creditUsdBaseUnit"), "billing.creditUsdBaseUnit") != 1:
        raise ValueError("billing.creditUsdBaseUnit must be 1")
    _require_equal(
        billing.get("inferencePricingMarkupPercent"),
        0,
        "billing.inferencePricingMarkupPercent",
    )
    _require_equal(billing.get("purchaseFeeExcluded"), True, "billing.purchaseFeeExcluded")
    overhead_reserve = _positive_int(
        billing.get("promptTokenOverheadReserve"),
        "billing.promptTokenOverheadReserve",
    )
    if overhead_reserve != PROMPT_BILLING_OVERHEAD_TOKENS:
        raise ValueError("billing.promptTokenOverheadReserve must be 2048")

    transforms = _mapping(raw.get("transforms"), "transforms")
    _require_equal(transforms.get("allowFallbacks"), False, "transforms.allowFallbacks")
    _require_equal(transforms.get("dataCollection"), "deny", "transforms.dataCollection")
    _require_equal(transforms.get("requireParameters"), True, "transforms.requireParameters")
    _require_equal(transforms.get("reasoningEffort"), "none", "transforms.reasoningEffort")
    _require_equal(transforms.get("temperature"), 0, "transforms.temperature")
    _require_equal(transforms.get("zdr"), True, "transforms.zdr")
    provider_order = _string_tuple(transforms.get("providerOrder"), "providerOrder")
    if provider_order != EXPECTED_PROVIDER_ORDER:
        raise ValueError("transforms.providerOrder must be exactly ['deepinfra/bf16']")
    expected_models = _string_tuple(
        transforms.get("expectedResponseModels"), "expectedResponseModels"
    )
    expected_providers = _string_tuple(
        transforms.get("expectedResponseProviders"), "expectedResponseProviders"
    )
    if expected_models != EXPECTED_RESPONSE_MODELS:
        raise ValueError(
            "expectedResponseModels must freeze the catalog and endpoint-name IDs"
        )
    if expected_providers != EXPECTED_RESPONSE_PROVIDERS:
        raise ValueError("expectedResponseProviders must be exactly ['DeepInfra']")
    model_forward = _nonempty_string(transforms.get("modelForward"), "modelForward")
    model_backward = _nonempty_string(transforms.get("modelBackward"), "modelBackward")
    if model_forward != EXPECTED_MODEL or model_backward != EXPECTED_MODEL:
        raise ValueError("both frozen request models must be qwen/qwen3.5-9b")
    max_tokens = _positive_int(transforms.get("maxTokens"), "transforms.maxTokens")
    seed = _nonnegative_int(transforms.get("seed"), "transforms.seed")
    timeout = transforms.get("timeoutSeconds")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("transforms.timeoutSeconds must be positive")
    max_prices = _mapping(
        transforms.get("maxPriceUsdPerMillionTokens"),
        "maxPriceUsdPerMillionTokens",
    )
    pricing = _mapping(
        transforms.get("pricingUsdPerMillionTokens"),
        "pricingUsdPerMillionTokens",
    )
    prompt_price = _decimal(max_prices.get("prompt"), "max prompt price")
    completion_price = _decimal(max_prices.get("completion"), "max completion price")
    if (
        prompt_price != EXPECTED_PROMPT_PRICE_USD
        or completion_price != EXPECTED_COMPLETION_PRICE_USD
    ):
        raise ValueError(
            "max USD prices must be 0.10 prompt and 0.15 completion per million"
        )
    if _decimal(pricing.get("prompt"), "prompt pricing") != prompt_price:
        raise ValueError("prompt pricing does not match its max price")
    if _decimal(pricing.get("completion"), "completion pricing") != completion_price:
        raise ValueError("completion pricing does not match its max price")

    methods = _parse_methods(transforms.get("methods"))
    endpoint_snapshot_path = _safe_relative_path(
        root_path,
        transforms.get("endpointSnapshotPath"),
        "endpointSnapshotPath",
    )
    endpoint_expected_sha256 = _sha256_string(
        transforms.get("endpointSnapshotSha256"),
        "transforms.endpointSnapshotSha256",
    )
    endpoint_bytes = endpoint_snapshot_path.read_bytes()
    endpoint_actual_sha256 = hashlib.sha256(endpoint_bytes).hexdigest()
    if endpoint_actual_sha256 != endpoint_expected_sha256:
        raise ValueError("endpoint snapshot file hash differs from frozen config binding")
    endpoint = _load_json_object(endpoint_bytes, "endpoint snapshot")
    _validate_endpoint_snapshot(
        endpoint,
        max_tokens=max_tokens,
        prompt_price=prompt_price,
        completion_price=completion_price,
    )

    analysis = _mapping(raw.get("analysis"), "analysis")
    _require_equal(
        analysis.get("primaryScoringUnit"), "pooled_corpus", "primaryScoringUnit"
    )
    _require_equal(
        analysis.get("qualityMetric"),
        "normalized_word_levenshtein",
        "qualityMetric",
    )
    _require_equal(
        analysis.get("resamplingUnit"), "document_id", "resamplingUnit"
    )
    bootstrap_replicates = _positive_int(
        analysis.get("bootstrapReplicates"), "analysis.bootstrapReplicates"
    )
    bootstrap_seed = _nonnegative_int(
        analysis.get("bootstrapSeed"), "analysis.bootstrapSeed"
    )
    semantic_audit_plan_path = _safe_relative_path(
        root_path,
        analysis.get("semanticAuditPlanPath"),
        "analysis.semanticAuditPlanPath",
    )
    semantic_audit_plan_sha256 = _sha256_string(
        analysis.get("semanticAuditPlanSha256"),
        "analysis.semanticAuditPlanSha256",
    )
    _require_file_sha256(
        semantic_audit_plan_path,
        semantic_audit_plan_sha256,
        "semantic audit plan",
    )
    semantic_audit_plan = _load_json_object(
        semantic_audit_plan_path.read_bytes(),
        "semantic audit plan",
    )
    _validate_semantic_audit_plan(semantic_audit_plan)

    return ExperimentConfig(
        path=config_path,
        root=root_path,
        raw=raw,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        experiment_version=experiment_version,
        verified_at=str(raw["verifiedAt"]),
        methodology=str(raw["methodology"]),
        sources=tuple(dict(source) for source in raw["sources"]),  # type: ignore[arg-type]
        manifest_path=manifest_path,
        manifest_expected_sha256=manifest_expected_sha256,
        inventory_path=inventory_path,
        inventory_expected_sha256=inventory_expected_sha256,
        reviews_directory=reviews_directory,
        review_bindings=tuple(review_bindings),
        lexicon_path=lexicon_path,
        lexicon_expected_sha256=lexicon_expected_sha256,
        endpoint_snapshot_path=endpoint_snapshot_path,
        endpoint_snapshot_sha256=endpoint_expected_sha256,
        semantic_audit_plan_path=semantic_audit_plan_path,
        semantic_audit_plan_sha256=semantic_audit_plan_sha256,
        key=key,
        density_bps=density_bps,
        context_width=context_width,
        min_active_positions=min_active_positions,
        wrong_key_count=wrong_key_count,
        wrong_key_seed=wrong_key_seed,
        methods=methods,
        model_forward=model_forward,
        model_backward=model_backward,
        provider_order=provider_order,
        expected_response_models=expected_models,
        expected_response_providers=expected_providers,
        max_tokens=max_tokens,
        prompt_price_usd_per_million=prompt_price,
        completion_price_usd_per_million=completion_price,
        prompt_token_overhead_reserve=overhead_reserve,
        seed=seed,
        timeout_seconds=float(timeout),
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )


def load_reviewed_corpus(config: ExperimentConfig) -> ReviewedCorpus:
    """Load exact manifest bytes and require complete independent reviews."""
    lexicon_bytes = config.lexicon_path.read_bytes()
    lexicon_file_sha256 = hashlib.sha256(lexicon_bytes).hexdigest()
    if lexicon_file_sha256 != config.lexicon_expected_sha256:
        raise ValueError("lexicon file hash differs from frozen config binding")
    lexicon = load_lexicon(config.lexicon_path)
    manifest_bytes = config.manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != config.manifest_expected_sha256:
        raise ValueError("manifest file hash differs from frozen config binding")
    manifest = _load_json_object(manifest_bytes, "corpus manifest")
    if manifest.get("schemaVersion") != 1:
        raise ValueError("unsupported corpus manifest schemaVersion")
    _validate_evidence(manifest, "corpus manifest")
    if manifest.get("documentCount") != 20:
        raise ValueError("corpus manifest must contain exactly 20 documents")
    if manifest.get("lexiconSha256") != lexicon.sha256:
        raise ValueError("corpus manifest lexiconSha256 mismatch")
    raw_documents = manifest.get("documents")
    if not isinstance(raw_documents, list) or len(raw_documents) != 20:
        raise ValueError("corpus manifest documents must contain exactly 20 entries")

    documents: list[CorpusItem] = []
    for expected_id, raw_document in zip(EXPECTED_DOCUMENT_IDS, raw_documents, strict=True):
        document = _mapping(raw_document, "manifest document")
        if document.get("documentId") != expected_id:
            raise ValueError("manifest document IDs must be ordered doc-01 through doc-20")
        relative_path = _nonempty_string(document.get("path"), "manifest document path")
        source_path = _safe_relative_path(config.root, relative_path, "manifest document path")
        if not relative_path.startswith("corpus/original/"):
            raise ValueError("manifest documents must live under corpus/original")
        raw_bytes = source_path.read_bytes()
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if document.get("sha256") != sha256:
            raise ValueError(f"document hash mismatch: {expected_id}")
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"corpus document is not UTF-8: {expected_id}") from error
        analysis = analyze_text(text)
        word_count = len(WORD_RE.findall(text))
        eligible = sum(
            1
            for token in analysis.context_tokens
            if not token.protected
            and token.text is not None
            and (token.text.islower() or token.text.isupper() or token.text.istitle())
            and token.normalized in lexicon.token_to_pair
        )
        expected_counts = {
            "eligiblePositions": eligible,
            "protectedSpanCount": len(analysis.protected_spans),
            "wordCount": word_count,
        }
        for field, expected in expected_counts.items():
            if document.get(field) != expected:
                raise ValueError(f"manifest {field} mismatch: {expected_id}")
        documents.append(
            CorpusItem(
                document_id=expected_id,
                path=relative_path,
                genre=_nonempty_string(document.get("genre"), "manifest genre"),
                title=_nonempty_string(document.get("title"), "manifest title"),
                text=text,
                sha256=sha256,
                word_count=word_count,
                eligible_positions=eligible,
                protected_span_count=len(analysis.protected_spans),
            )
        )
    if manifest.get("wordCount") != sum(item.word_count for item in documents):
        raise ValueError("manifest pooled wordCount mismatch")
    if manifest.get("eligiblePositions") != sum(
        item.eligible_positions for item in documents
    ):
        raise ValueError("manifest pooled eligiblePositions mismatch")

    inventory_bytes = config.inventory_path.read_bytes()
    inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
    if inventory_sha256 != config.inventory_expected_sha256:
        raise ValueError("inventory file hash differs from frozen config binding")
    inventory = _load_json_object(inventory_bytes, "context inventory")
    inventory_documents = inventory.get("documents")
    if not isinstance(inventory_documents, list):
        raise ValueError("context inventory requires documents")
    inventory_ids = tuple(
        item.get("documentId") if isinstance(item, dict) else None
        for item in inventory_documents
    )
    if inventory_ids != EXPECTED_DOCUMENT_IDS:
        raise ValueError("context inventory IDs must match the exact manifest order")

    reviews: list[dict[str, object]] = []
    review_hashes: list[str] = []
    for review_path, expected_sha256 in config.review_bindings:
        review_bytes = review_path.read_bytes()
        review_sha256 = hashlib.sha256(review_bytes).hexdigest()
        if review_sha256 != expected_sha256:
            raise ValueError(
                f"review file hash differs from frozen config binding: {review_path.name}"
            )
        reviews.append(_load_json_object(review_bytes, f"review {review_path.name}"))
        review_hashes.append(review_sha256)
    approval = validate_context_reviews(inventory=inventory, reviews=reviews)
    if approval.get("approvedDocumentCount") != 20:
        raise ValueError("review approval must cover exactly 20 documents")

    return ReviewedCorpus(
        config=config,
        documents=tuple(documents),
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        inventory_sha256=inventory_sha256,
        review_sha256s=tuple(review_hashes),
        review_approval=approval,
        lexicon=lexicon,
        lexicon_file_sha256=lexicon_file_sha256,
    )


def build_dry_run(
    config: ExperimentConfig,
    corpus: ReviewedCorpus,
) -> dict[str, object]:
    """Return exact call counts and a conservative, non-binding token estimate."""
    if corpus.config.sha256 != config.sha256:
        raise ValueError("corpus was loaded under a different experiment config")
    _verify_current_bindings(config, corpus)
    calls_by_method = {
        method.method_id: len(corpus.documents) * method.calls_per_document
        for method in config.methods
    }
    call_count = sum(calls_by_method.values())
    prompt_estimate = 0
    maximum_variant_delta = max(
        abs(len(pair.variants[0]) - len(pair.variants[1]))
        for pair in corpus.lexicon.pairs
    )
    for document in corpus.documents:
        protected = protect_tokens(document.text)
        expansion = document.eligible_positions * maximum_variant_delta
        prompt_estimate += (
            len(build_synonym_prompt(protected.masked).encode("utf-8")) + expansion
        )
        prompt_estimate += (
            len(build_paraphrase_prompt(protected.masked).encode("utf-8")) + expansion
        )
        for pivot in ("de", "zh"):
            prompt_estimate += (
                len(build_forward_prompt(protected.masked, pivot).encode("utf-8"))
                + expansion
            )
            backward_wrapper = len(build_backward_prompt("x", pivot).encode("utf-8")) - 1
            prompt_estimate += (
                backward_wrapper
                + config.max_tokens * PIVOT_REENCODE_MULTIPLIER
            )
    prompt_estimate += call_count * config.prompt_token_overhead_reserve
    completion_at_configured_maximum = call_count * config.max_tokens
    total_estimate = prompt_estimate + completion_at_configured_maximum
    cost_estimate = _routing_cost_usd(
        prompt_estimate,
        completion_at_configured_maximum,
        config=config,
    )
    return {
        "callCount": call_count,
        "callsByMethod": dict(sorted(calls_by_method.items())),
        "configSha256": config.sha256,
        "documentCount": len(corpus.documents),
        "experimentVersion": config.experiment_version,
        "manifestSha256": corpus.manifest_sha256,
        "methodology": (
            "No environment variable, watermark encoding, or network client is used. "
            "This is a conservative planning estimate, not a mathematical upper bound. "
            "Known prompts use UTF-8 bytes as a token estimate. Unknown backward prompts "
            f"use the frozen heuristic of {PIVOT_REENCODE_MULTIPLIER} prompt tokens per "
            "allowed forward completion token. Every call adds "
            f"{config.prompt_token_overhead_reserve} prompt-overhead tokens and assumes "
            "the full max_tokens completion. Live mode independently reserves against the "
            "actual next prompt bytes before dispatch. The frozen billing premise defines "
            "one provider credit base unit as one USD."
        ),
        "routingCostPlanningEstimateUsd": _decimal_text(cost_estimate),
        "schemaVersion": ARTIFACT_SCHEMA_VERSION,
        "semanticAuditPlanSha256": config.semantic_audit_plan_sha256,
        "tokenEstimate": {
            "completionTokensAtConfiguredMaximum": completion_at_configured_maximum,
            "pivotReencodeMultiplierHeuristic": PIVOT_REENCODE_MULTIPLIER,
            "promptTokensPlanningEstimate": prompt_estimate,
            "totalTokensPlanningEstimate": total_estimate,
        },
        "verifiedAt": config.verified_at,
    }


def verify_prepaid_controls(
    config: ExperimentConfig,
    corpus: ReviewedCorpus,
) -> dict[str, object]:
    """Rebuild controls in memory and require byte-exact published outputs."""
    from run_corpus_controls import (
        build_corpus_controls,
        check_control_outputs,
        control_spec_from_config,
    )

    _verify_current_bindings(config, corpus)
    spec = control_spec_from_config(config)
    outputs = build_corpus_controls(config, corpus, spec=spec)
    acceptance = _mapping(outputs.artifact.get("acceptance"), "control acceptance")
    if acceptance.get("passed") is not True:
        raise ControlGateError(
            "deterministic control acceptance failed before paid calls"
        )
    freshness = check_control_outputs(config.root, outputs)
    if freshness.get("passed") is not True:
        raise ControlGateError(
            "deterministic control outputs are not byte-exact on disk"
        )
    files = _mapping(freshness.get("files"), "control freshness files")
    return {
        "acceptancePassed": True,
        "artifactSha256": hashlib.sha256(
            canonical_json_bytes(outputs.artifact)
        ).hexdigest(),
        "checkedFileCount": len(files),
        "outputsMatch": True,
    }


def run_live(
    config: ExperimentConfig,
    corpus: ReviewedCorpus,
    *,
    client: CompletionClient,
    max_provider_cost_credits: Decimal,
    checkpoint_path: str | Path,
    max_new_calls: int | None = None,
    confirm_not_charged_call_id: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Execute or resume the matrix with explicit not-charged redispatch only."""
    if corpus.config.sha256 != config.sha256:
        raise ValueError("corpus was loaded under a different experiment config")
    budget = _required_budget(max_provider_cost_credits)
    build_dry_run(config, corpus)
    if max_new_calls is not None and (
        not isinstance(max_new_calls, int)
        or isinstance(max_new_calls, bool)
        or max_new_calls < 0
    ):
        raise ValueError("max_new_calls must be a nonnegative integer or null")
    prepaid_control_gate = verify_prepaid_controls(config, corpus)

    expected_call_ids = _expected_call_ids(config, corpus)
    manager = _CheckpointManager(
        config=config,
        corpus=corpus,
        client=client,
        checkpoint_path=Path(checkpoint_path),
        expected_call_ids=expected_call_ids,
        budget=budget,
        max_new_calls=max_new_calls,
        confirm_not_charged_call_id=confirm_not_charged_call_id,
        clock=clock,
    )

    marked: dict[str, tuple[EncodeResult, ScoreResult]] = {}
    for document in corpus.documents:
        encoded = encode_text(
            document.text,
            key=config.key,
            document_id=document.document_id,
            density_bps=config.density_bps,
            lexicon=corpus.lexicon,
            context_width=config.context_width,
        )
        baseline_score = score_text(
            encoded.text,
            key=config.key,
            document_id=document.document_id,
            density_bps=config.density_bps,
            lexicon=corpus.lexicon,
            context_width=config.context_width,
            min_active_positions=config.min_active_positions,
        )
        marked[document.document_id] = (encoded, baseline_score)

    rows_by_method: dict[str, list[dict[str, object]]] = {
        method.method_id: [] for method in config.methods
    }
    scores_by_method: dict[str, list[ScoreResult]] = {
        method.method_id: [] for method in config.methods
    }
    for document in corpus.documents:
        encoded, baseline_score = marked[document.document_id]
        for method in config.methods:
            outcome = _run_transformation(
                document=document,
                marked_text=encoded.text,
                method=method,
                config=config,
                manager=manager,
            )
            output = outcome.text
            output_score = (
                baseline_score
                if method.method == "none"
                else score_text(
                    output,
                    key=config.key,
                    document_id=document.document_id,
                    density_bps=config.density_bps,
                    lexicon=corpus.lexicon,
                    context_width=config.context_width,
                    min_active_positions=config.min_active_positions,
                )
            )
            fingerprints = compare_active_fingerprints(
                baseline_score,
                output_score,
            ).to_dict()
            fidelity = fidelity_metrics(encoded.text, output)
            fidelity["failure"] = outcome.failed
            fidelity["failureReasons"] = [
                issue["code"] for issue in outcome.issues
            ]
            row = {
                "calls": list(outcome.calls),
                "detector": output_score.to_dict(),
                "documentId": document.document_id,
                "fidelity": fidelity,
                "fingerprints": fingerprints,
                "genre": document.genre,
                "markedInputText": encoded.text,
                "originalText": document.text,
                "outputText": output,
                "sourceSha256": document.sha256,
                "transformationOutcome": outcome.to_dict(),
            }
            rows_by_method[method.method_id].append(row)
            scores_by_method[method.method_id].append(output_score)

    method_artifacts: list[dict[str, object]] = []
    for method in config.methods:
        rows = rows_by_method[method.method_id]
        wrong_keys = run_wrong_key_controls(
            tuple(
                Document(
                    document_id=_nonempty_string(row.get("documentId"), "documentId"),
                    text=_string(row.get("outputText"), "outputText"),
                )
                for row in rows
            ),
            density_bps=config.density_bps,
            lexicon=corpus.lexicon,
            count=config.wrong_key_count,
            seed=config.wrong_key_seed,
            context_width=config.context_width,
            min_active_positions=config.min_active_positions,
        )
        method_artifacts.append(
            {
                "aggregate": _aggregate_method(
                    rows,
                    scores_by_method[method.method_id],
                ),
                "documents": rows,
                "method": method.method,
                "methodId": method.method_id,
                "pivot": method.pivot,
                "wrongKeyControls": _public_wrong_key_controls(wrong_keys),
            }
        )

    reference_rows = rows_by_method["none"]
    comparisons: list[dict[str, object]] = []
    for method in config.methods[1:]:
        comparison = paired_bootstrap(
            rows_by_method[method.method_id],
            reference_rows,
            replicates=config.bootstrap_replicates,
            seed=config.bootstrap_seed,
        )
        comparisons.append(
            {
                "methodId": method.method_id,
                "referenceMethodId": "none",
                **comparison,
            }
        )

    baseline_scores = [marked[item.document_id][1] for item in corpus.documents]
    artifact = {
        "baseline": {
            "detector": aggregate_precomputed_scores(baseline_scores),
            "documents": [
                {
                    "detector": score.to_dict(),
                    "documentId": item.document_id,
                    "marking": _marking_dict(marked[item.document_id][0]),
                    "markedText": marked[item.document_id][0].text,
                    "originalText": item.text,
                }
                for item, score in zip(corpus.documents, baseline_scores, strict=True)
            ],
        },
        "bootstrapSeed": config.bootstrap_seed,
        "configSha256": config.sha256,
        "densityBps": config.density_bps,
        "documentCount": len(corpus.documents),
        "endpointSnapshotSha256": config.endpoint_snapshot_sha256,
        "experimentVersion": config.experiment_version,
        "keySha256": hashlib.sha256(config.key).hexdigest(),
        "lexiconFileSha256": corpus.lexicon_file_sha256,
        "lexiconSha256": corpus.lexicon.sha256,
        "manifestSha256": corpus.manifest_sha256,
        "methodology": config.methodology,
        "methods": method_artifacts,
        "pairedBootstrap": {
            "comparisons": comparisons,
            "documentIds": list(EXPECTED_DOCUMENT_IDS),
            "replicates": config.bootstrap_replicates,
            "sampleSize": 20,
            "seed": config.bootstrap_seed,
        },
        "prepaidControlGate": prepaid_control_gate,
        "reviewApproval": corpus.review_approval,
        "reviewSha256s": list(corpus.review_sha256s),
        "schemaVersion": ARTIFACT_SCHEMA_VERSION,
        "schemeVersion": SCHEME_VERSION,
        "semanticAuditPlanSha256": config.semantic_audit_plan_sha256,
        "sources": list(config.sources),
        "usage": {
            **aggregate_call_usage(manager.calls),
            "providerCostBudgetCredits": _decimal_text(budget),
        },
        "verifiedAt": config.verified_at,
    }
    canonical_json_bytes(artifact)
    return artifact


def word_levenshtein_metrics(original: str, output: str) -> dict[str, object]:
    """Return an exact word-level distance plus its normalized display value."""
    left = [match.group(0).lower() for match in WORD_RE.finditer(original)]
    right = [match.group(0).lower() for match in WORD_RE.finditer(output)]
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_word in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_word in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_word != right_word),
                )
            )
        previous = current
    distance = previous[-1]
    denominator = max(len(left), len(right))
    return {
        "distance": distance,
        "normalizationDenominator": denominator,
        "normalizedDistance": distance / denominator if denominator else 0.0,
        "originalWordCount": len([*WORD_RE.finditer(original)]),
        "outputWordCount": len([*WORD_RE.finditer(output)]),
    }


def protected_restoration_metrics(original: str, output: str) -> dict[str, object]:
    expected = tuple(original[span.start : span.end] for span in find_protected_spans(original))
    observed = tuple(output[span.start : span.end] for span in find_protected_spans(output))
    return {
        "exactlyRestored": observed == expected,
        "expectedCount": len(expected),
        "observedCount": len(observed),
        "orderedExpectedSha256": hashlib.sha256(
            canonical_json_bytes(list(expected))
        ).hexdigest(),
        "orderedObservedSha256": hashlib.sha256(
            canonical_json_bytes(list(observed))
        ).hexdigest(),
    }


def fidelity_metrics(original: str, output: str) -> dict[str, object]:
    original_paragraphs = _paragraph_count(original)
    output_paragraphs = _paragraph_count(output)
    original_length = len(original)
    output_length = len(output)
    return {
        "length": {
            "absoluteDriftCharacters": abs(output_length - original_length),
            "driftCharacters": output_length - original_length,
            "inputCharacters": original_length,
            "outputCharacters": output_length,
            "outputPerInput": output_length / original_length if original_length else None,
        },
        "paragraphs": {
            "drift": output_paragraphs - original_paragraphs,
            "inputCount": original_paragraphs,
            "outputCount": output_paragraphs,
            "outputPerInput": (
                output_paragraphs / original_paragraphs if original_paragraphs else None
            ),
        },
        "protectedTokens": protected_restoration_metrics(original, output),
        "wordLevenshtein": word_levenshtein_metrics(original, output),
    }


def aggregate_precomputed_scores(scores: Sequence[ScoreResult]) -> dict[str, object]:
    """Pool exact per-document counts without calling the detector again."""
    if len(scores) != 20:
        raise ValueError("pooled detector aggregate requires exactly 20 precomputed scores")
    ids = tuple(score.document_id for score in scores)
    if ids != EXPECTED_DOCUMENT_IDS:
        raise ValueError("precomputed score IDs must be ordered doc-01 through doc-20")
    first = scores[0]
    for score in scores[1:]:
        for field in (
            "density_bps",
            "key_sha256",
            "lexicon_sha256",
            "context_width",
            "min_active_positions",
        ):
            if getattr(score, field) != getattr(first, field):
                raise ValueError(f"precomputed detector metadata mismatch: {field}")
    active = sum(score.active_positions for score in scores)
    hits = sum(score.hits for score in scores)
    insufficient_document_count = sum(
        score.status == "insufficient_evidence" for score in scores
    )
    if active < first.min_active_positions:
        p_value: Fraction | None = None
        z_score: float | None = None
        status = "insufficient_evidence"
    else:
        p_value = binomial_tail_probability(hits, active)
        z_score = (2 * hits - active) / math.sqrt(active)
        status = "detected" if p_value <= DETECTION_ALPHA else "not_detected"
    return {
        "activePerAllWords": _ratio(active, sum(score.all_word_count for score in scores)),
        "activePerEligible": _ratio(
            active,
            sum(score.eligible_positions for score in scores),
        ),
        "activePositions": active,
        "aggregation": "sum_precomputed_document_counts",
        "allWordCount": sum(score.all_word_count for score in scores),
        "contextWidth": first.context_width,
        "densityBps": first.density_bps,
        "documentCount": len(scores),
        "documents": [score.to_dict() for score in scores],
        "eligiblePositions": sum(score.eligible_positions for score in scores),
        "hitRate": _ratio(hits, active),
        "hits": hits,
        "insufficientDocumentCount": insufficient_document_count,
        "insufficientDocumentRate": insufficient_document_count / len(scores),
        "keySha256": first.key_sha256,
        "lexiconSha256": first.lexicon_sha256,
        "minActivePositions": first.min_active_positions,
        "pValue": float(p_value) if p_value is not None else None,
        "pValueExact": (
            {"denominator": p_value.denominator, "numerator": p_value.numerator}
            if p_value is not None
            else None
        ),
        "scorableWordCount": sum(score.scorable_word_count for score in scores),
        "scoringUnit": "pooled_corpus",
        "status": status,
        "zScore": z_score,
    }


def paired_bootstrap(
    method_rows: Sequence[Mapping[str, object]],
    reference_rows: Sequence[Mapping[str, object]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Bootstrap paired document deltas with samples of exactly 20 IDs."""
    if len(method_rows) != 20 or len(reference_rows) != 20:
        raise ValueError("paired bootstrap requires exactly 20 rows per method")
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    method_by_id = _unique_rows(method_rows)
    reference_by_id = _unique_rows(reference_rows)
    if set(method_by_id) != set(reference_by_id):
        raise ValueError("paired bootstrap document IDs do not match")
    ordered_ids = tuple(str(row["documentId"]) for row in reference_rows)
    method = [method_by_id[document_id] for document_id in ordered_ids]
    reference = [reference_by_id[document_id] for document_id in ordered_ids]
    metric_names = (
        "activeSurvivalRateDelta",
        "meanNormalizedWordDistanceDelta",
        "pooledHitRateDelta",
    )
    observed = _paired_metric_delta(method, reference, range(20))
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    rng = random.Random(seed)
    for _ in range(replicates):
        indices = [rng.randrange(20) for _ in range(20)]
        sample = _paired_metric_delta(method, reference, indices)
        for name, value in sample.items():
            if value is not None:
                values[name].append(value)
    metrics: dict[str, object] = {}
    for name in metric_names:
        valid = sorted(values[name])
        metrics[name] = {
            "lower95": _percentile(valid, 0.025),
            "observed": observed[name],
            "upper95": _percentile(valid, 0.975),
            "validReplicates": len(valid),
        }
    return {
        "metrics": metrics,
        "replicates": replicates,
        "sampleSize": 20,
        "seed": seed,
    }


def aggregate_call_usage(calls: Sequence[Mapping[str, object]]) -> dict[str, object]:
    cost = Decimal(0)
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    latency = 0.0
    for call in calls:
        response = _mapping(call.get("response"), "checkpoint response")
        usage = _mapping(response.get("usage"), "checkpoint usage")
        prompt_tokens += _nonnegative_int(usage.get("promptTokens"), "promptTokens")
        completion_tokens += _nonnegative_int(
            usage.get("completionTokens"), "completionTokens"
        )
        total_tokens += _nonnegative_int(usage.get("totalTokens"), "totalTokens")
        cost += _decimal(usage.get("providerCostCredits"), "provider cost")
        latency_value = call.get("latencyMs")
        if not isinstance(latency_value, (int, float)) or isinstance(latency_value, bool):
            raise ValueError("call latencyMs must be numeric")
        latency += float(latency_value)
    return {
        "callCount": len(calls),
        "completionTokens": completion_tokens,
        "latencyMs": round(latency, 3),
        "promptTokens": prompt_tokens,
        "providerCostCredits": _decimal_text(cost),
        "totalTokens": total_tokens,
    }


class _CheckpointManager:
    def __init__(
        self,
        *,
        config: ExperimentConfig,
        corpus: ReviewedCorpus,
        client: CompletionClient,
        checkpoint_path: Path,
        expected_call_ids: tuple[str, ...],
        budget: Decimal,
        max_new_calls: int | None,
        confirm_not_charged_call_id: str | None,
        clock: Callable[[], float],
    ) -> None:
        self.config = config
        self.corpus = corpus
        self.client = client
        self.path = checkpoint_path
        self.expected_call_ids = expected_call_ids
        self.budget = budget
        self.max_new_calls = max_new_calls
        self.confirm_not_charged_call_id = confirm_not_charged_call_id
        self.clock = clock
        self.new_calls = 0
        self.state = self._load_or_create()
        raw_calls = self.state.get("calls")
        if not isinstance(raw_calls, list):
            raise CheckpointError("checkpoint calls must be a list")
        self.calls: list[dict[str, object]] = raw_calls
        ids = tuple(call.get("callId") for call in self.calls if isinstance(call, dict))
        if len(ids) != len(self.calls) or ids != expected_call_ids[: len(ids)]:
            raise CheckpointError("checkpoint calls are not an exact ordered matrix prefix")
        in_flight = self.state.get("inFlightCall")
        if in_flight is None:
            if confirm_not_charged_call_id is not None:
                raise CheckpointError("no in-flight call exists to resolve as not charged")
        else:
            tombstone = _mapping(in_flight, "inFlightCall")
            call_id = tombstone.get("callId")
            if call_id != confirm_not_charged_call_id:
                raise CheckpointError(
                    "in-flight call charge is unknown and blocks resume; pass its exact "
                    "callId only with --confirm-not-charged-call-id after checking provider "
                    "Activity and confirming that it was not charged"
                )
            if len(self.calls) >= len(expected_call_ids):
                raise CheckpointError(
                    "in-flight call cannot follow a complete matrix checkpoint"
                )
            if call_id != expected_call_ids[len(self.calls)]:
                raise CheckpointError("in-flight call is not the next matrix call")

    def complete(
        self,
        *,
        call_id: str,
        document_id: str,
        method_id: str,
        stage: str,
        input_text: str,
        prompt: str,
        requested_model: str,
    ) -> tuple[ChatCompletion, dict[str, object]]:
        try:
            expected_index = self.expected_call_ids.index(call_id)
        except ValueError as error:
            raise CheckpointError(f"unknown matrix call ID: {call_id}") from error
        if expected_index < len(self.calls):
            record = self.calls[expected_index]
            self._validate_request(
                record,
                call_id=call_id,
                document_id=document_id,
                method_id=method_id,
                stage=stage,
                input_text=input_text,
                prompt=prompt,
                requested_model=requested_model,
            )
            if record.get("recordStatus") == "provider_response_invalid":
                raise ResponseContractError(
                    "provider response contract failure was preserved; call will not be "
                    "reissued"
                )
            completion = _completion_from_record(record)
            self._validate_response(record, completion, prompt)
            return completion, record
        if expected_index != len(self.calls):
            raise CheckpointError("attempted to skip an uncheckpointed matrix call")
        request_sha256 = _request_sha256(
            call_id=call_id,
            document_id=document_id,
            method_id=method_id,
            stage=stage,
            input_text=input_text,
            prompt=prompt,
            requested_model=requested_model,
        )
        in_flight = self.state.get("inFlightCall")
        if in_flight is not None:
            tombstone = _mapping(in_flight, "inFlightCall")
            if tombstone.get("callId") != call_id:
                raise CheckpointError("in-flight call ID differs from the next request")
            if tombstone.get("requestSha256") != request_sha256:
                raise CheckpointError("in-flight call request hash mismatch")
        if self.max_new_calls is not None and self.new_calls >= self.max_new_calls:
            raise CallLimitReached(
                completed_calls=len(self.calls),
                new_calls=self.new_calls,
            )
        if _checkpoint_cost(self.calls) >= self.budget:
            raise BudgetError(
                "checkpointed provider cost has reached the explicit credits budget"
            )
        reserve = _routing_cost_usd(
            len(prompt.encode("utf-8")) + self.config.prompt_token_overhead_reserve,
            self.config.max_tokens,
            config=self.config,
        )
        remaining_budget = self.budget - _checkpoint_cost(self.calls)
        if remaining_budget < reserve:
            raise BudgetError(
                "remaining provider-cost-credit budget is below the conservative "
                f"next-call reserve: need {reserve}, have {remaining_budget}"
            )

        self.state["inFlightCall"] = {
            "callId": call_id,
            "conservativeCostReserveCredits": _decimal_text(reserve),
            "dispatchResolution": (
                "confirmed_not_charged_redispatch"
                if in_flight is not None
                else "new_dispatch"
            ),
            "requestSha256": request_sha256,
            "startedAtUnixMs": int(time.time() * 1000),
        }
        self._save()
        started = self.clock()
        try:
            completion = self.client.complete(prompt, model=requested_model)
        except ProviderResponseError as error:
            elapsed_ms = round((self.clock() - started) * 1000, 3)
            raw_response = json_safe_value(error.raw_response)
            if not isinstance(raw_response, dict):
                raise AssertionError("provider raw response must normalize to an object")
            reported_cost = _raw_response_cost(raw_response)
            accounted_cost = reserve if reported_cost is None else reported_cost
            record = {
                "callId": call_id,
                "chargeAccounting": {
                    "providerCostCredits": _decimal_text(accounted_cost),
                    "status": (
                        "conservative_reserve_unknown"
                        if reported_cost is None
                        else "provider_reported"
                    ),
                },
                "conservativeCostReserveCredits": _decimal_text(reserve),
                "documentId": document_id,
                "inputText": input_text,
                "latencyMs": elapsed_ms,
                "methodId": method_id,
                "outputText": _raw_response_output(raw_response),
                "prompt": prompt,
                "providerError": str(error),
                "rawResponse": raw_response,
                "recordStatus": "provider_response_invalid",
                "request": {"model": requested_model},
                "stage": stage,
            }
            self.calls.append(record)
            self.state["inFlightCall"] = None
            self._save()
            self.new_calls += 1
            raise ResponseContractError(
                "provider response violated the contract and was preserved; call will not "
                "be reissued"
            ) from error
        except (ProviderError, TimeoutError):
            raise
        elapsed_ms = round((self.clock() - started) * 1000, 3)
        record = {
            "callId": call_id,
            "conservativeCostReserveCredits": _decimal_text(reserve),
            "documentId": document_id,
            "inputText": input_text,
            "latencyMs": elapsed_ms,
            "methodId": method_id,
            "outputText": completion.content,
            "prompt": prompt,
            "recordStatus": "accepted_response",
            "request": {"model": requested_model},
            "response": {
                "finishReason": completion.finish_reason,
                "id": completion.response_id,
                "model": completion.model,
                "openrouterMetadata": (
                    None
                    if completion.openrouter_metadata is None
                    else json_safe_value(completion.openrouter_metadata)
                ),
                "provider": completion.provider,
                "systemFingerprint": completion.system_fingerprint,
                "usage": completion.usage.to_dict(),
            },
            "routingCostEstimateUsd": _decimal_text(
                _routing_cost_usd(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                    config=self.config,
                )
            ),
            "stage": stage,
        }
        self.calls.append(record)
        self.state["inFlightCall"] = None
        self._save()
        self.new_calls += 1
        self._validate_response(record, completion, prompt)
        if _checkpoint_cost(self.calls) > self.budget:
            raise BudgetError("checkpointed provider cost exceeded the explicit budget")
        return completion, record

    def _load_or_create(self) -> dict[str, object]:
        key_sha256 = hashlib.sha256(self.config.key).hexdigest()
        expected = {
            "configSha256": self.config.sha256,
            "endpointSnapshotSha256": self.config.endpoint_snapshot_sha256,
            "inventorySha256": self.corpus.inventory_sha256,
            "keySha256": key_sha256,
            "lexiconFileSha256": self.corpus.lexicon_file_sha256,
            "manifestSha256": self.corpus.manifest_sha256,
            "reviewSha256s": list(self.corpus.review_sha256s),
            "schemaVersion": CHECKPOINT_SCHEMA_VERSION,
            "semanticAuditPlanSha256": self.config.semantic_audit_plan_sha256,
        }
        if not self.path.exists():
            return {**expected, "inFlightCall": None, "calls": []}
        state = _load_json_object(self.path.read_bytes(), "checkpoint")
        labels = {
            "configSha256": "config",
            "endpointSnapshotSha256": "endpoint snapshot",
            "inventorySha256": "inventory",
            "keySha256": "watermark key",
            "lexiconFileSha256": "lexicon file",
            "manifestSha256": "manifest",
            "reviewSha256s": "reviews",
            "schemaVersion": "schema",
            "semanticAuditPlanSha256": "semantic audit plan",
        }
        for field, value in expected.items():
            if state.get(field) != value:
                raise CheckpointError(f"checkpoint {labels[field]} binding mismatch")
        return state

    def _validate_request(self, record: Mapping[str, object], **expected: object) -> None:
        request = _mapping(record.get("request"), "checkpoint request")
        checks = {
            "callId": expected["call_id"],
            "documentId": expected["document_id"],
            "inputText": expected["input_text"],
            "methodId": expected["method_id"],
            "prompt": expected["prompt"],
            "stage": expected["stage"],
        }
        for field, value in checks.items():
            if record.get(field) != value:
                raise CheckpointError(f"checkpoint request mismatch: {field}")
        if request.get("model") != expected["requested_model"]:
            raise CheckpointError("checkpoint request mismatch: model")

    def _validate_response(
        self,
        record: Mapping[str, object],
        completion: ChatCompletion,
        prompt: str,
    ) -> None:
        if completion.model not in self.config.expected_response_models:
            raise ResponseContractError(
                f"unexpected response model: {completion.model}"
            )
        if completion.provider not in self.config.expected_response_providers:
            raise ResponseContractError(
                f"unexpected selected provider: {completion.provider}"
            )
        metadata = completion.openrouter_metadata
        if not isinstance(metadata, Mapping):
            raise ResponseContractError("OpenRouter routing metadata is required")
        if metadata.get("strategy") != "direct":
            raise ResponseContractError("OpenRouter routing strategy must be direct")
        attempt_number = metadata.get("attempt")
        if (
            not isinstance(attempt_number, int)
            or isinstance(attempt_number, bool)
            or attempt_number != 1
        ):
            raise ResponseContractError("OpenRouter routing attempt must be 1")
        if metadata.get("pipeline") not in (None, []):
            raise ResponseContractError("OpenRouter routing pipeline must be empty or absent")
        endpoints = _mapping(metadata.get("endpoints"), "router metadata endpoints")
        available = endpoints.get("available")
        if not isinstance(available, list) or any(
            not isinstance(item, Mapping) for item in available
        ):
            raise ResponseContractError(
                "router metadata endpoints.available must be a list of objects"
            )
        selected_endpoints = [item for item in available if item.get("selected") is True]
        if len(selected_endpoints) != 1:
            raise ResponseContractError(
                "router metadata must contain exactly one selected endpoint"
            )
        selected_endpoint = selected_endpoints[0]
        selected_provider = _first_present(
            selected_endpoint,
            "provider",
            "provider_name",
            "providerName",
        )
        selected_model = _first_present(
            selected_endpoint,
            "model",
            "model_id",
            "modelId",
        )
        if selected_provider != completion.provider:
            raise ResponseContractError("router selected provider metadata mismatch")
        if selected_model not in self.config.expected_response_models:
            raise ResponseContractError("router selected model metadata mismatch")
        top_level_selected = metadata.get(
            "selected_provider", metadata.get("selectedProvider")
        )
        if top_level_selected is not None and top_level_selected != completion.provider:
            raise ResponseContractError("router top-level selected provider mismatch")
        attempts = metadata.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != 1:
            raise ResponseContractError("router attempts must contain exactly one item")
        if not isinstance(attempts[0], Mapping):
            raise ResponseContractError("router attempt must be an object")
        attempt = attempts[0]
        attempt_provider = _first_present(
            attempt,
            "provider",
            "provider_name",
            "providerName",
        )
        attempt_model = _first_present(attempt, "model", "model_id", "modelId")
        if attempt_provider != selected_provider:
            raise ResponseContractError("router attempt provider metadata mismatch")
        if attempt_model != selected_model:
            raise ResponseContractError("router attempt model metadata mismatch")
        if attempt.get("status") != 200 or isinstance(attempt.get("status"), bool):
            raise ResponseContractError("router attempt status must be HTTP 200")
        usage = completion.usage
        if usage.total_tokens != usage.prompt_tokens + usage.completion_tokens:
            raise ResponseContractError("response token totals are inconsistent")
        if usage.prompt_tokens > (
            len(prompt.encode("utf-8")) + self.config.prompt_token_overhead_reserve
        ):
            raise ResponseContractError(
                "prompt usage exceeds the UTF-8 plus billing-overhead token ceiling"
            )
        if usage.completion_tokens > self.config.max_tokens:
            raise ResponseContractError("completion usage exceeds frozen maxTokens")
        reserve = _decimal(
            record.get("conservativeCostReserveCredits"),
            "conservative cost reserve",
        )
        if usage.cost > reserve:
            raise ResponseContractError(
                "reported provider cost exceeds the checkpointed next-call reserve"
            )
        response = _mapping(record.get("response"), "checkpoint response")
        if response.get("id") != completion.response_id:
            raise CheckpointError("checkpoint response ID mismatch")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.path, canonical_json_bytes(self.state))


def _run_transformation(
    *,
    document: CorpusItem,
    marked_text: str,
    method: MethodSpec,
    config: ExperimentConfig,
    manager: _CheckpointManager,
) -> Stage1TransformationOutcome:
    if method.method == "none":
        return Stage1TransformationOutcome(
            text=marked_text,
            raw_final_masked_text=marked_text,
            calls=(),
            issues=(),
            restoration_mode="not_applicable",
        )
    protected = protect_tokens(marked_text)
    calls: list[dict[str, object]] = []
    issues: list[dict[str, str]] = []
    if method.method in {"synonyms", "paraphrase"}:
        stage = method.method
        prompt = (
            build_synonym_prompt(protected.masked)
            if method.method == "synonyms"
            else build_paraphrase_prompt(protected.masked)
        )
        completion, record = manager.complete(
            call_id=_call_id(document.document_id, method.method_id, stage),
            document_id=document.document_id,
            method_id=method.method_id,
            stage=stage,
            input_text=protected.masked,
            prompt=prompt,
            requested_model=config.model_forward,
        )
        calls.append(record)
        _append_finish_reason_issue(stage, completion, issues)
        final_masked = completion.content
    else:
        assert method.pivot is not None
        forward_stage = f"forward-{method.pivot}"
        forward_prompt = build_forward_prompt(protected.masked, method.pivot)
        forward, forward_record = manager.complete(
            call_id=_call_id(document.document_id, method.method_id, forward_stage),
            document_id=document.document_id,
            method_id=method.method_id,
            stage=forward_stage,
            input_text=protected.masked,
            prompt=forward_prompt,
            requested_model=config.model_forward,
        )
        calls.append(forward_record)
        _append_finish_reason_issue(forward_stage, forward, issues)
        try:
            validate_intermediate(forward.content, method.pivot, protected.tokens)
        except (PlaceholderError, ValidationError) as error:
            issues.append(_validation_issue(f"intermediate-{method.pivot}", error))
        backward_stage = f"backward-{method.pivot}"
        backward_prompt = build_backward_prompt(forward.content, method.pivot)
        backward, backward_record = manager.complete(
            call_id=_call_id(document.document_id, method.method_id, backward_stage),
            document_id=document.document_id,
            method_id=method.method_id,
            stage=backward_stage,
            input_text=forward.content,
            prompt=backward_prompt,
            requested_model=config.model_backward,
        )
        calls.append(backward_record)
        _append_finish_reason_issue(backward_stage, backward, issues)
        final_masked = backward.content
    placeholders_valid = True
    try:
        validate_placeholders(final_masked, protected.tokens)
    except PlaceholderError as error:
        placeholders_valid = False
        issues.append(_validation_issue("final", error))
    for issue in result_validation_issues(protected.masked, final_masked, method.pivot):
        issues.append({"stage": "final", **issue})
    if placeholders_valid:
        output = restore_tokens(final_masked, protected.tokens)
        restoration_mode = "exact"
    else:
        output = _best_effort_restore(final_masked, protected.tokens)
        restoration_mode = "best_effort"
    if not protected_restoration_metrics(marked_text, output)["exactlyRestored"]:
        issues.append(
            {
                "code": "protected_restoration_mismatch",
                "message": "protected tokens were not restored byte-exactly",
                "stage": "final",
            }
        )
    return Stage1TransformationOutcome(
        text=output,
        raw_final_masked_text=final_masked,
        calls=tuple(calls),
        issues=tuple(issues),
        restoration_mode=restoration_mode,
    )


def _best_effort_restore(
    text: str,
    tokens: Sequence[ProtectedToken],
) -> str:
    restored = text
    for token in tokens:
        restored = restored.replace(token.placeholder, token.original)
    return restored


def _validation_issue(stage: str, error: Exception) -> dict[str, str]:
    message = str(error)
    if isinstance(error, PlaceholderError):
        code = "placeholder_contract"
    elif "byte-identical" in message:
        code = "unchanged_output"
    elif "length" in message:
        code = "length_contract"
    elif "paragraph" in message:
        code = "paragraph_contract"
    elif "pivot" in message or "German" in message or "Chinese" in message:
        code = "pivot_language_contract"
    else:
        code = "validation_contract"
    return {"code": code, "message": message, "stage": stage}


def _append_finish_reason_issue(
    stage: str,
    completion: ChatCompletion,
    issues: list[dict[str, str]],
) -> None:
    if completion.finish_reason != "stop":
        issues.append(
            {
                "code": "finish_reason_contract",
                "message": (
                    "provider returned a paid partial completion with finish reason "
                    f"{completion.finish_reason!r}"
                ),
                "stage": stage,
            }
        )


def _aggregate_method(
    rows: Sequence[Mapping[str, object]],
    scores: Sequence[ScoreResult],
) -> dict[str, object]:
    if len(rows) != 20:
        raise ValueError("method aggregate requires exactly 20 document rows")
    calls = [
        call
        for row in rows
        for call in _list_of_mappings(row.get("calls"), "document calls")
    ]
    word_distances = [
        float(
            _mapping(
                _mapping(row["fidelity"], "fidelity").get("wordLevenshtein"),
                "wordLevenshtein",
            )["normalizedDistance"]
        )
        for row in rows
    ]
    length_ratios = [
        float(
            _mapping(
                _mapping(row["fidelity"], "fidelity").get("length"), "length"
            )["outputPerInput"]
        )
        for row in rows
    ]
    paragraph_ratios = [
        float(
            _mapping(
                _mapping(row["fidelity"], "fidelity").get("paragraphs"),
                "paragraphs",
            )["outputPerInput"]
        )
        for row in rows
    ]
    fingerprint_rows = [
        _mapping(row.get("fingerprints"), "fingerprints") for row in rows
    ]
    fingerprints = {
        field: sum(int(row[field]) for row in fingerprint_rows)
        for field in (
            "baselineActive",
            "lostActive",
            "newActive",
            "outputActive",
            "survivingActive",
        )
    }
    fingerprints["lostActiveRate"] = _ratio(
        fingerprints["lostActive"], fingerprints["baselineActive"]
    )
    fingerprints["survivingActiveRate"] = _ratio(
        fingerprints["survivingActive"], fingerprints["baselineActive"]
    )
    document_usages = [
        aggregate_call_usage(
            _list_of_mappings(row.get("calls"), "document calls")
        )
        for row in rows
    ]
    total_input_word_count = sum(
        len(WORD_RE.findall(_nonempty_string(row.get("markedInputText"), "markedInputText")))
        for row in rows
    )
    usage = aggregate_call_usage(calls)
    document_costs = [
        _decimal(item["providerCostCredits"], "document provider cost")
        for item in document_usages
    ]
    document_latencies = [float(item["latencyMs"]) for item in document_usages]
    total_cost = _decimal(usage["providerCostCredits"], "method provider cost")
    fidelity_failure_count = sum(
        _mapping(row.get("fidelity"), "fidelity").get("failure") is True
        for row in rows
    )
    usage.update(
        {
            "meanPerDocumentProviderCostCredits": _decimal_text(
                total_cost / len(rows)
            ),
            "medianPerDocumentLatencyMs": round(_median_float(document_latencies), 3),
            "medianPerDocumentProviderCostCredits": _decimal_text(
                _median_decimal(document_costs)
            ),
            "providerCostCreditsPer1000Documents": _decimal_text(
                total_cost * Decimal(1000) / len(rows)
            ),
            "providerCostCreditsPer1000MarkedInputWords": (
                _decimal_text(total_cost * Decimal(1000) / total_input_word_count)
                if total_input_word_count
                else None
            ),
            "totalInputWordCount": total_input_word_count,
        }
    )
    return {
        "detector": aggregate_precomputed_scores(scores),
        "fidelity": {
            "allProtectedTokensExactlyRestored": all(
                bool(
                    _mapping(
                        _mapping(row["fidelity"], "fidelity").get("protectedTokens"),
                        "protectedTokens",
                    )["exactlyRestored"]
                )
                for row in rows
            ),
            "meanLengthRatio": sum(length_ratios) / len(length_ratios),
            "meanNormalizedWordDistance": sum(word_distances) / len(word_distances),
            "meanParagraphRatio": sum(paragraph_ratios) / len(paragraph_ratios),
            "totalInputWordCount": total_input_word_count,
        },
        "fidelityFailureCount": fidelity_failure_count,
        "fidelityFailureRate": fidelity_failure_count / len(rows),
        "fingerprints": fingerprints,
        "usage": usage,
    }


def _public_wrong_key_controls(controls: object) -> dict[str, object]:
    to_dict = getattr(controls, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("wrong-key controls must support to_dict")
    raw = to_dict(include_scores=True)
    public = _remove_key_hashes(raw)
    if not isinstance(public, dict):
        raise TypeError("wrong-key controls must serialize to an object")
    scores = public.get("scores")
    if not isinstance(scores, list):
        raise TypeError("wrong-key controls must include a score distribution")
    for index, score in enumerate(scores):
        item = _mapping(score, "wrong-key score")
        score_with_index = dict(item)
        score_with_index["wrongKeyIndex"] = index
        scores[index] = score_with_index
    return public


def _remove_key_hashes(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _remove_key_hashes(item)
            for key, item in value.items()
            if key != "keySha256"
        }
    if isinstance(value, list):
        return [_remove_key_hashes(item) for item in value]
    return value


def _paired_metric_delta(
    method: Sequence[Mapping[str, object]],
    reference: Sequence[Mapping[str, object]],
    indices: Sequence[int] | range,
) -> dict[str, float | None]:
    chosen_method = [method[index] for index in indices]
    chosen_reference = [reference[index] for index in indices]

    def pooled_hit(rows: Sequence[Mapping[str, object]]) -> float | None:
        detector_rows = [_mapping(row.get("detector"), "detector") for row in rows]
        active = sum(int(row["activePositions"]) for row in detector_rows)
        return _ratio(sum(int(row["hits"]) for row in detector_rows), active)

    def survival(rows: Sequence[Mapping[str, object]]) -> float | None:
        values = [_mapping(row.get("fingerprints"), "fingerprints") for row in rows]
        baseline = sum(int(row["baselineActive"]) for row in values)
        return _ratio(sum(int(row["survivingActive"]) for row in values), baseline)

    def word_mean(rows: Sequence[Mapping[str, object]]) -> float:
        values = [
            float(
                _mapping(
                    _mapping(row.get("fidelity"), "fidelity").get("wordLevenshtein"),
                    "wordLevenshtein",
                )["normalizedDistance"]
            )
            for row in rows
        ]
        return sum(values) / len(values)

    method_hit = pooled_hit(chosen_method)
    reference_hit = pooled_hit(chosen_reference)
    method_survival = survival(chosen_method)
    reference_survival = survival(chosen_reference)
    return {
        "activeSurvivalRateDelta": (
            method_survival - reference_survival
            if method_survival is not None and reference_survival is not None
            else None
        ),
        "meanNormalizedWordDistanceDelta": (
            word_mean(chosen_method) - word_mean(chosen_reference)
        ),
        "pooledHitRateDelta": (
            method_hit - reference_hit
            if method_hit is not None and reference_hit is not None
            else None
        ),
    }


def _unique_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    output: dict[str, Mapping[str, object]] = {}
    for row in rows:
        document_id = row.get("documentId")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("bootstrap rows require documentId")
        if document_id in output:
            raise ValueError("paired bootstrap document IDs must be unique")
        output[document_id] = row
    return output


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    index = int(round(probability * (len(values) - 1)))
    return values[max(0, min(index, len(values) - 1))]


def _median_float(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _median_decimal(values: Sequence[Decimal]) -> Decimal:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _expected_call_ids(
    config: ExperimentConfig,
    corpus: ReviewedCorpus,
) -> tuple[str, ...]:
    calls: list[str] = []
    for document in corpus.documents:
        for method in config.methods:
            if method.method in {"synonyms", "paraphrase"}:
                calls.append(
                    _call_id(document.document_id, method.method_id, method.method)
                )
            elif method.method == "roundtrip":
                assert method.pivot is not None
                calls.append(
                    _call_id(
                        document.document_id,
                        method.method_id,
                        f"forward-{method.pivot}",
                    )
                )
                calls.append(
                    _call_id(
                        document.document_id,
                        method.method_id,
                        f"backward-{method.pivot}",
                    )
                )
    return tuple(calls)


def _call_id(document_id: str, method_id: str, stage: str) -> str:
    return f"{document_id}:{method_id}:{stage}"


def _request_sha256(
    *,
    call_id: str,
    document_id: str,
    method_id: str,
    stage: str,
    input_text: str,
    prompt: str,
    requested_model: str,
) -> str:
    value = {
        "callId": call_id,
        "documentId": document_id,
        "inputSha256": hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
        "methodId": method_id,
        "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "requestedModel": requested_model,
        "stage": stage,
    }
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _completion_from_record(record: Mapping[str, object]) -> ChatCompletion:
    response = _mapping(record.get("response"), "checkpoint response")
    usage = _mapping(response.get("usage"), "checkpoint usage")
    metadata = response.get("openrouterMetadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise CheckpointError("checkpoint OpenRouter metadata must be an object or null")
    fingerprint = response.get("systemFingerprint")
    if fingerprint is not None and not isinstance(fingerprint, str):
        raise CheckpointError("checkpoint system fingerprint is invalid")
    return ChatCompletion(
        content=_string(record.get("outputText"), "checkpoint outputText"),
        finish_reason=_nonempty_string(response.get("finishReason"), "finishReason"),
        model=_nonempty_string(response.get("model"), "response model"),
        openrouter_metadata=metadata,
        provider=_nonempty_string(response.get("provider"), "response provider"),
        response_id=_nonempty_string(response.get("id"), "response id"),
        system_fingerprint=fingerprint,
        usage=CompletionUsage(
            prompt_tokens=_nonnegative_int(usage.get("promptTokens"), "promptTokens"),
            completion_tokens=_nonnegative_int(
                usage.get("completionTokens"), "completionTokens"
            ),
            total_tokens=_nonnegative_int(usage.get("totalTokens"), "totalTokens"),
            cost=_decimal(usage.get("providerCostCredits"), "provider cost"),
        ),
    )


def _checkpoint_cost(calls: Sequence[Mapping[str, object]]) -> Decimal:
    total = Decimal(0)
    for call in calls:
        if call.get("recordStatus") == "provider_response_invalid":
            accounting = _mapping(
                call.get("chargeAccounting"),
                "invalid response charge accounting",
            )
            total += _decimal(
                accounting.get("providerCostCredits"),
                "invalid response accounted cost",
            )
            continue
        response = _mapping(call.get("response"), "checkpoint response")
        usage = _mapping(response.get("usage"), "checkpoint usage")
        total += _decimal(usage.get("providerCostCredits"), "provider cost")
    return total


def _raw_response_cost(raw_response: Mapping[str, object]) -> Decimal | None:
    usage = raw_response.get("usage")
    if not isinstance(usage, Mapping) or usage.get("cost") is None:
        return None
    try:
        return _decimal(usage.get("cost"), "raw provider response cost")
    except ValueError:
        return None


def _raw_response_output(raw_response: Mapping[str, object]) -> str | None:
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, Mapping):
        return None
    message = first.get("message")
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def _marking_dict(result: EncodeResult) -> dict[str, object]:
    output = result.to_dict()
    output.pop("markedText", None)
    return output


def _routing_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    config: ExperimentConfig,
) -> Decimal:
    million = Decimal(1_000_000)
    return (
        Decimal(prompt_tokens) * config.prompt_price_usd_per_million
        + Decimal(completion_tokens) * config.completion_price_usd_per_million
    ) / million


def _parse_methods(value: object) -> tuple[MethodSpec, ...]:
    if not isinstance(value, list) or len(value) != len(EXPECTED_METHODS):
        raise ValueError("transforms.methods must contain the exact five-method matrix")
    parsed: list[MethodSpec] = []
    for raw, (expected_id, expected_method, expected_pivot) in zip(
        value,
        EXPECTED_METHODS,
        strict=True,
    ):
        method = _mapping(raw, "transform method")
        if method.get("id") != expected_id or method.get("pivot") != expected_pivot:
            raise ValueError("transforms.methods order or pivot differs from frozen matrix")
        declared = method.get("method", expected_id)
        if declared != expected_method:
            raise ValueError(f"transform method mismatch for {expected_id}")
        parsed.append(MethodSpec(expected_id, expected_method, expected_pivot))
    return tuple(parsed)


def _validate_endpoint_snapshot(
    snapshot: Mapping[str, object],
    *,
    max_tokens: int,
    prompt_price: Decimal,
    completion_price: Decimal,
) -> None:
    if snapshot.get("schemaVersion") != 1:
        raise ValueError("unsupported endpoint snapshot schemaVersion")
    _validate_evidence(snapshot, "endpoint snapshot")
    if snapshot.get("requestedModelId") != EXPECTED_MODEL:
        raise ValueError("endpoint snapshot requested model mismatch")
    if snapshot.get("catalogModelId") != EXPECTED_MODEL:
        raise ValueError("endpoint snapshot catalog model mismatch")
    endpoint = _mapping(snapshot.get("endpoint"), "endpoint snapshot endpoint")
    expected = {
        "providerName": "DeepInfra",
        "quantization": "bf16",
        "tag": "deepinfra/bf16",
    }
    for field, value in expected.items():
        if endpoint.get(field) != value:
            raise ValueError(f"endpoint snapshot {field} mismatch")
    maximum = _positive_int(endpoint.get("maxCompletionTokens"), "maxCompletionTokens")
    if max_tokens > maximum:
        raise ValueError("frozen maxTokens exceeds endpoint maximum")
    parameters = endpoint.get("supportedParameters")
    if not isinstance(parameters, list) or "seed" not in parameters:
        raise ValueError("endpoint snapshot must support seed")
    per_token = _mapping(endpoint.get("pricingUsdPerToken"), "endpoint pricing")
    if _decimal(per_token.get("prompt"), "endpoint prompt price") * Decimal(
        1_000_000
    ) != prompt_price:
        raise ValueError("endpoint prompt price mismatch")
    if _decimal(per_token.get("completion"), "endpoint completion price") * Decimal(
        1_000_000
    ) != completion_price:
        raise ValueError("endpoint completion price mismatch")


def _validate_semantic_audit_plan(plan: Mapping[str, object]) -> None:
    if plan.get("schemaVersion") != 1:
        raise ValueError("unsupported semantic audit plan schemaVersion")
    _validate_evidence(plan, "semantic audit plan")
    _nonempty_string(plan.get("auditVersion"), "semantic audit plan auditVersion")
    scope = _mapping(plan.get("scope"), "semantic audit plan scope")
    _require_equal(
        scope.get("documentCountPerMethod"),
        20,
        "semantic audit plan documentCountPerMethod",
    )
    _require_equal(
        scope.get("structuredPairCount"),
        80,
        "semantic audit plan structuredPairCount",
    )
    expected_methods = [
        "synonyms",
        "roundtrip-de",
        "roundtrip-zh",
        "paraphrase",
    ]
    if scope.get("methods") != expected_methods:
        raise ValueError("semantic audit plan methods differ from the frozen matrix")
    close_reading = _mapping(
        plan.get("closeReadingSample"),
        "semantic audit plan closeReadingSample",
    )
    _require_equal(
        close_reading.get("documentsPerMethod"),
        3,
        "semantic audit plan close-reading sample size",
    )


def _validate_evidence(value: Mapping[str, object], label: str) -> None:
    for field in ("verifiedAt", "methodology"):
        _nonempty_string(value.get(field), f"{label}.{field}")
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"{label}.sources must be a nonempty list")
    for source in sources:
        item = _mapping(source, f"{label} source")
        _nonempty_string(item.get("title"), "source title")
        url = _nonempty_string(item.get("url"), "source URL")
        if not url.startswith(("https://", "http://")):
            raise ValueError("source URL must be HTTP(S)")


def _load_json_object(raw_bytes: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sha256_string(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_file_sha256(path: Path, expected: str, label: str) -> None:
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"cannot read frozen {label}: {path}") from error
    if actual != expected:
        raise ValueError(f"{label} file hash differs from frozen config binding")


def _verify_current_bindings(
    config: ExperimentConfig,
    corpus: ReviewedCorpus,
) -> None:
    """Close the load/run gap by rechecking every pre-registered file."""
    _require_file_sha256(config.path, config.sha256, "experiment config")
    _require_file_sha256(
        config.manifest_path,
        config.manifest_expected_sha256,
        "corpus manifest",
    )
    _require_file_sha256(
        config.inventory_path,
        config.inventory_expected_sha256,
        "context inventory",
    )
    _require_file_sha256(
        config.lexicon_path,
        config.lexicon_expected_sha256,
        "synonym lexicon",
    )
    _require_file_sha256(
        config.endpoint_snapshot_path,
        config.endpoint_snapshot_sha256,
        "endpoint snapshot",
    )
    _require_file_sha256(
        config.semantic_audit_plan_path,
        config.semantic_audit_plan_sha256,
        "semantic audit plan",
    )
    for index, (review_path, expected_sha256) in enumerate(config.review_bindings):
        _require_file_sha256(review_path, expected_sha256, f"context review {index}")
    if corpus.manifest_sha256 != config.manifest_expected_sha256:
        raise ValueError("loaded corpus manifest binding mismatch")
    if corpus.inventory_sha256 != config.inventory_expected_sha256:
        raise ValueError("loaded corpus inventory binding mismatch")
    if corpus.lexicon_file_sha256 != config.lexicon_expected_sha256:
        raise ValueError("loaded corpus lexicon binding mismatch")
    if corpus.review_sha256s != tuple(
        expected for _, expected in config.review_bindings
    ):
        raise ValueError("loaded corpus review bindings mismatch")


def _safe_relative_path(root: Path, value: object, label: str) -> Path:
    relative = _nonempty_string(value, label)
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{label} must be a safe relative path")
    resolved = (root / relative_path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} escapes repository root")
    return resolved


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _first_present(value: Mapping[str, object], *fields: str) -> object:
    for field in fields:
        if field in value:
            return value[field]
    return None


def _list_of_mappings(value: object, label: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{label} must be a list of objects")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"transforms.{label} must be a list of strings")
    return tuple(value)


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
    if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
        raise BudgetError("live mode requires an explicit positive provider-cost budget")
    return value


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise ValueError(f"{label} must be {expected!r}")


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _paragraph_count(text: str) -> int:
    return len(re.split(r"\n\s*\n", text.strip())) if text.strip() else 0


def _atomic_write(path: Path, content: bytes) -> None:
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
        default=str(root / "fixtures" / "experiment-config-v1.json"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(root / "results" / "experiment-checkpoint-v1.json"),
    )
    parser.add_argument(
        "--output",
        default=str(root / "results" / "experiment-raw-v1.json"),
    )
    parser.add_argument("--max-provider-cost-credits")
    parser.add_argument("--max-new-calls", type=int)
    parser.add_argument("--confirm-not-charged-call-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    corpus = load_reviewed_corpus(config)
    if args.dry_run:
        if (
            args.max_provider_cost_credits is not None
            or args.max_new_calls is not None
            or args.confirm_not_charged_call_id is not None
        ):
            raise SystemExit(
                "budget, max-new-calls, and not-charged resolution are live-only options"
            )
        print(canonical_json_bytes(build_dry_run(config, corpus)).decode("utf-8"), end="")
        return 0
    if args.max_provider_cost_credits is None:
        raise SystemExit("--live requires --max-provider-cost-credits")
    budget = _decimal(args.max_provider_cost_credits, "provider budget")
    client = OpenRouterClient.from_env(
        timeout=config.timeout_seconds,
        provider_order=config.provider_order,
        allow_fallbacks=False,
        require_parameters=True,
        temperature=0,
        max_tokens=config.max_tokens,
        seed=config.seed,
        max_prompt_price=float(config.prompt_price_usd_per_million),
        max_completion_price=float(config.completion_price_usd_per_million),
    )
    try:
        artifact = run_live(
            config,
            corpus,
            client=client,
            max_provider_cost_credits=budget,
            checkpoint_path=args.checkpoint,
            max_new_calls=args.max_new_calls,
            confirm_not_charged_call_id=args.confirm_not_charged_call_id,
        )
    except CallLimitReached as pause:
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
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(artifact)
    _atomic_write(output_path, content)
    print(
        json.dumps(
            {
                "artifactSha256": hashlib.sha256(content).hexdigest(),
                "callCount": artifact["usage"]["callCount"],  # type: ignore[index]
                "output": str(output_path.resolve()),
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
    "ARTIFACT_SCHEMA_VERSION",
    "BudgetError",
    "CallLimitReached",
    "CheckpointError",
    "ControlGateError",
    "ExperimentConfig",
    "MethodSpec",
    "ResponseContractError",
    "ReviewedCorpus",
    "aggregate_call_usage",
    "aggregate_precomputed_scores",
    "build_dry_run",
    "fidelity_metrics",
    "load_experiment_config",
    "load_reviewed_corpus",
    "main",
    "paired_bootstrap",
    "protected_restoration_metrics",
    "run_live",
    "verify_prepaid_controls",
    "word_levenshtein_metrics",
]
