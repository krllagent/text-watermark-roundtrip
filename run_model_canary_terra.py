"""Run the Terra fallback on the frozen six-document development canary.

This is a thin candidate configuration over the audited Luna execution engine.
It preserves the same inputs, prompt, one-call policy, checkpoint contract, and
blind review. The six-document detector result is descriptive; only the later
twenty-document holdout can support a removal claim.
"""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import tempfile

import run_model_canary_luna as engine


MODEL = "openai/gpt-5.6-terra"
EXPECTED_MODELS = (MODEL, f"{MODEL}-20260709")
ENDPOINT_NAME = "Azure | openai/gpt-5.6-terra-20260709"
PROMPT_PRICE = Decimal("2.20")
COMPLETION_PRICE = Decimal("13.20")
CACHE_READ_PRICE = Decimal("0.22")
CACHE_WRITE_PRICE = Decimal("2.75")
PAYLOAD_SHA256S = {
    "doc-11": "cbf6e6ea1d855f07273b3ed1a47207fd1ad7a210d49fed5114d04b658c41c83e",
    "doc-12": "2b42a9060e442ad37341b38b6d44f7bc1f7b8fe247d13dc120e252f3e1bf27f0",
    "doc-15": "a9cbf9bfd3721551305951b431ad502fe91ed8846b1e328b65dc795cea27bd05",
    "doc-20": "fccd9fd9410cbc5cf2ba0b0bc206042b46fbf60bed4efed7fbaaa6cef1832a97",
    "doc-03": "2de67d7b32b91b25376780fd0201d087765d8c47dfe44f6404d4aed5012a68d7",
    "doc-19": "d08aa19a0c652f99501916bf49fdecd572b1beb0141c642df37ba9a5225fa791",
}
ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "results" / "model-canary-terra-checkpoint-v1.json"
DEFAULT_PACKET = ROOT / "results" / "model-canary-terra-blind-v1.json"
DEFAULT_FINAL = ROOT / "results" / "model-canary-terra-final-v1.json"
LUNA_FINAL = ROOT / "results" / "model-canary-luna-final-v1.json"
LUNA_FINAL_SHA256 = "dd551d96e9fc7a0caff550bc732e76e21ba7a25b1151bf93b3bde711bc537056"
ORIGINAL_FINALIZE_REVIEW = engine.finalize_review


def configure_engine() -> None:
    luna_binding = validate_luna_rejection()
    engine.__file__ = __file__
    engine.SCRIPT_VERSION = "model-canary-terra-v1"
    engine.CANDIDATE_LABEL = "terra"
    engine.MODEL = MODEL
    engine.EXPECTED_MODELS = EXPECTED_MODELS
    engine.PROMPT_PRICE = PROMPT_PRICE
    engine.COMPLETION_PRICE = COMPLETION_PRICE
    engine.CACHE_READ_PRICE = CACHE_READ_PRICE
    engine.CACHE_WRITE_PRICE = CACHE_WRITE_PRICE
    engine.PAYLOAD_SHA256S = PAYLOAD_SHA256S
    engine.PREREQUISITE_BINDINGS = luna_binding
    engine.DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT
    engine.DEFAULT_PACKET = DEFAULT_PACKET
    engine.DEFAULT_FINAL = DEFAULT_FINAL
    engine.fetch_catalog = fetch_catalog
    engine.validate_route_record = validate_route_record
    engine.finalize_review = finalize_review


def validate_luna_rejection() -> dict[str, object]:
    if engine.sha256_file(LUNA_FINAL) != LUNA_FINAL_SHA256:
        raise engine.CanaryError("Luna rejection artifact hash changed")
    result = engine.load_json(LUNA_FINAL, "Luna rejection artifact")
    selection = engine.require_mapping(result.get("selection"), "Luna selection")
    if (
        selection.get("lunaPassed") is not False
        or selection.get("nextStep") != "run_terra_development_gate"
        or selection.get("selectedModel") is not None
    ):
        raise engine.CanaryError("Luna rejection no longer authorizes Terra")
    return {
        "lunaFinalPath": "results/model-canary-luna-final-v1.json",
        "lunaFinalSha256": LUNA_FINAL_SHA256,
        "lunaPassed": False,
        "nextStep": "run_terra_development_gate",
    }


def fetch_catalog() -> dict[str, object]:
    request = engine.Request(
        engine.CATALOG_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": "model-canary-terra-v1/1",
        },
        method="GET",
    )
    with engine.urlopen(request, timeout=30) as response:
        content = response.read(8 * 1024 * 1024 + 1)
    if len(content) > 8 * 1024 * 1024:
        raise engine.CanaryError("ZDR catalog is oversized")
    raw = json.loads(content.decode("utf-8"))
    rows = raw.get("data") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        raise engine.CanaryError("ZDR catalog data is missing")
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("model_id") == MODEL
        and row.get("tag") == engine.PROVIDER_TAG
    ]
    if len(matches) != 1:
        raise engine.CanaryError("exact Terra azure/eu ZDR endpoint is unavailable")
    row = matches[0]
    uptime = row.get("uptime_last_5m")
    if row.get("status") != 0 or (uptime is not None and float(uptime) != 100.0):
        raise engine.CanaryError("Terra azure/eu ZDR endpoint is unhealthy")
    if (
        row.get("provider_name") != engine.PROVIDER
        or row.get("name") != ENDPOINT_NAME
        or row.get("supports_implicit_caching") is not False
    ):
        raise engine.CanaryError("Terra azure/eu endpoint identity changed")
    supported = row.get("supported_parameters")
    required = {
        "max_completion_tokens",
        "reasoning",
        "reasoning_effort",
        "seed",
    }
    if not isinstance(supported, list) or not required.issubset(set(supported)):
        raise engine.CanaryError("Terra azure/eu endpoint lacks a required parameter")
    if "temperature" in supported:
        raise engine.CanaryError("Terra azure/eu unexpectedly accepts temperature")
    pricing = engine.require_mapping(row.get("pricing"), "catalog pricing")
    expected = {
        "prompt": PROMPT_PRICE,
        "completion": COMPLETION_PRICE,
        "input_cache_read": CACHE_READ_PRICE,
        "input_cache_write": CACHE_WRITE_PRICE,
    }
    for field, per_million in expected.items():
        if (
            engine.decimal(pricing.get(field), field) * Decimal(1_000_000)
            != per_million
        ):
            raise engine.CanaryError(f"Terra azure/eu {field} price changed")
    return {
        "catalogUrl": engine.CATALOG_URL,
        "endpointName": ENDPOINT_NAME,
        "provider": engine.PROVIDER,
        "status": 0,
        "tag": engine.PROVIDER_TAG,
        "uptimeLast5m": uptime,
    }


def validate_route_record(value: object) -> None:
    route = engine.require_mapping(value, "route preflight")
    uptime = route.get("uptimeLast5m")
    expected = {
        "catalogUrl": engine.CATALOG_URL,
        "endpointName": ENDPOINT_NAME,
        "provider": engine.PROVIDER,
        "status": 0,
        "tag": engine.PROVIDER_TAG,
    }
    stable = {key: value for key, value in route.items() if key != "uptimeLast5m"}
    if stable != expected or uptime not in (None, 100, 100.0):
        raise engine.CanaryError("completed Terra call route preflight changed")


def finalize_review(
    checkpoint_path: Path,
    packet_path: Path,
    review_path: Path,
    output_path: Path,
) -> dict[str, object]:
    validation_path = Path(tempfile.gettempdir()) / "terra-strict-validation.json"
    validated = ORIGINAL_FINALIZE_REVIEW(
        checkpoint_path,
        packet_path,
        review_path,
        validation_path,
    )
    checkpoint = engine.require_mapping(validated.get("checkpoint"), "checkpoint")
    calls = checkpoint.get("calls")
    review = engine.require_mapping(validated.get("manualReview"), "manual review")
    reviews = review.get("reviews")
    if not isinstance(calls, list) or not isinstance(reviews, list):
        raise engine.CanaryError("validated Terra review is incomplete")
    manual_pass = all(
        isinstance(row, dict) and row.get("verdict") == "pass" for row in reviews
    )
    distances: list[float] = []
    local_pass = True
    for call_value in calls:
        call = engine.require_mapping(call_value, "call")
        analysis = engine.require_mapping(call.get("analysis"), "analysis")
        fidelity = engine.require_mapping(analysis.get("fidelity"), "fidelity")
        word = engine.require_mapping(fidelity.get("wordLevenshtein"), "word distance")
        distance = float(word.get("normalizedDistance", 0))
        distances.append(distance)
        local_pass = (
            local_pass and not analysis.get("pipelineIssues") and distance >= 0.15
        )
    passed = manual_pass and local_pass
    descriptive_detector = engine.require_mapping(
        engine.require_mapping(validated.get("selection"), "selection").get(
            "pooledOutputDetector"
        ),
        "pooled output detector",
    )
    provider_cost = Decimal(0)
    for call_value in calls:
        call = engine.require_mapping(call_value, "call")
        completion = engine.require_mapping(call.get("completion"), "completion")
        usage = engine.require_mapping(completion.get("usage"), "usage")
        provider_cost += engine.decimal(
            usage.get("providerCostCredits"), "provider cost"
        )
    validated["selection"] = {
        "candidateOutputDetectorRole": "descriptive_only",
        "candidateOutputRemovalClaimAllowed": False,
        "lunaCalls": 6,
        "manualFidelityPassed": manual_pass,
        "meanWordDistance": sum(distances) / len(distances),
        "minimumWordDistance": min(distances),
        "nextStep": "final_holdout" if passed else "stop_without_demo",
        "pipelineAndDistancePassed": local_pass,
        "pooledOutputDetector": dict(descriptive_detector),
        "providerCostCredits": format(provider_cost, "f"),
        "selectedModel": MODEL if passed else None,
        "terraCalls": 6,
        "terraPassed": passed,
    }
    engine.atomic_write(output_path, validated)
    return validated


def main(argv: list[str] | None = None) -> int:
    configure_engine()
    return engine.main(argv)


configure_engine()


if __name__ == "__main__":
    raise SystemExit(main())
