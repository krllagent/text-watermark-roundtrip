"""Run the frozen twenty-document v9 confirmation set with the v2 method once.

The development canary chain selected no model: Luna, plain Terra, and the
masked-anchor v1 method each failed the six-document gate. The visible-anchor
v2 method fixed v1's negation garble and left one blind-review minor finding.
The twenty-document v9 corpus is still untouched, so it is the only unbiased
measurement of that method. This runner reuses the audited holdout protocol
and binds it to the Terra provider contract and the v2 transformation.

Configuration is process-scoped: nothing is patched at import time, and the
resulting engine contract is verified against the frozen constants before any
request is built.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path

from corpus_contract import canonical_json_bytes
import run_final_holdout_v9_luna as base
import run_model_canary_luna as engine
import run_model_canary_terra as terra
import run_model_canary_terra_locked_v2 as locked_v2
from run_model_canary_terra_locked_v2 import (
    anchor_alignment_issues,
    build_visible_locked_request,
    protect_visible_anchors,
    strip_markers,
)
import unmark
from watermark_toy import score_text


ROOT = Path(__file__).resolve().parent
SCRIPT_VERSION = "final-holdout-v9-terra-locked-v2-v1"
DEFAULT_CHECKPOINT = ROOT / "results" / "final-holdout-v9-locked-v2-checkpoint-v1.json"
DEFAULT_AGGREGATE = ROOT / "results" / "final-holdout-v9-locked-v2-automated-v1.json"
DEFAULT_PACKET = ROOT / "results" / "final-holdout-v9-locked-v2-blind-v1.json"
DEFAULT_FINAL = ROOT / "results" / "final-holdout-v9-locked-v2-final-v1.json"
CANARY_FINAL = ROOT / "results" / "model-canary-terra-locked-v2-final-v1.json"
CANARY_FINAL_SHA256 = "ab10b3cbc1d891c1b1b41b7618a282f87b227e9eff4a9b68ee098d1cf29cfd32"


def terra_usage_cost(usage: Mapping[str, object]) -> Decimal:
    details = usage.get("promptTokenDetails")
    detail_map = details if isinstance(details, Mapping) else {}
    prompt_tokens = int(usage.get("promptTokens", 0))
    cached = int(detail_map.get("cachedTokens", 0))
    cache_write = int(detail_map.get("cacheWriteTokens", 0))
    completion_tokens = int(usage.get("completionTokens", 0))
    uncached = prompt_tokens - cached - cache_write
    return (
        Decimal(uncached) * terra.PROMPT_PRICE
        + Decimal(cached) * terra.CACHE_READ_PRICE
        + Decimal(cache_write) * terra.CACHE_WRITE_PRICE
        + Decimal(completion_tokens) * terra.COMPLETION_PRICE
    ) / Decimal(1_000_000)


def request_for(source: str):
    return build_visible_locked_request(protect_visible_anchors(source).masked)


def restored_anchor_issues(protected, content: str) -> list[dict[str, str]]:
    """Compare anchors per sentence with placeholders restored on both sides.

    Masking hides sentence terminators that live inside a protected string, for
    example a quoted sentence, so a masked source can hold fewer sentence pieces
    than the document it came from. Comparing a masked source against a raw
    response therefore reports a misalignment when the model punctuated
    correctly. Restoring placeholders on both sides compares like with like. If
    the response violates the placeholder contract, restoring is impossible and
    the marker-level comparison stands; the placeholder failure is recorded
    separately by the caller.
    """
    anchored_source = unmark.restore_tokens(protected.masked, protected.tokens)
    try:
        anchored_output = unmark.restore_tokens(
            unmark.canonicalize_placeholders(content, protected.tokens),
            protected.tokens,
        )
    except Exception:
        return anchor_alignment_issues(protected.masked, content)
    return anchor_alignment_issues(anchored_source, anchored_output)


def analyze_output(
    protocol: Mapping[str, object], document_id: str, source: str, content: str
) -> dict[str, object]:
    """Score one response with the v2 anchor contract and the v9 holdout key."""
    protected = protect_visible_anchors(source)
    issues: list[dict[str, str]] = list(restored_anchor_issues(protected, content))
    stripped_masked = strip_markers(protected.masked)
    normalized = strip_markers(content)
    restored: str | None = None
    try:
        normalized = unmark.canonicalize_placeholders(normalized, protected.tokens)
        issues.extend(
            unmark.result_validation_issues(stripped_masked, normalized, None)
        )
        restored = unmark.restore_tokens(normalized, protected.tokens)
    except Exception as error:  # The paid response stays a failed document.
        issues.append(
            {
                "code": "placeholder_contract",
                "message": f"{type(error).__name__}: {error}",
            }
        )
        issues.extend(
            unmark.result_validation_issues(
                stripped_masked, strip_markers(content), None
            )
        )
    evaluated = restored if restored is not None else strip_markers(content)
    fidelity = base.fidelity_metrics(source, evaluated)
    protected_metrics = base.require_mapping(
        fidelity.get("protectedTokens"), "protected metrics"
    )
    if protected_metrics.get("exactlyRestored") is not True:
        issues.append(
            {"code": "protected_values", "message": "protected values changed"}
        )
    source_sentences = locked_v2.split_sentences(source)
    output_sentences = locked_v2.split_sentences(evaluated)
    if len(source_sentences) != len(output_sentences):
        issues.append(
            {
                "code": "sentence_alignment",
                "message": (
                    f"expected {len(source_sentences)} sentences and observed "
                    f"{len(output_sentences)}"
                ),
            }
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
        "sourceSentenceCount": len(source_sentences),
        "visibleAnchorCount": len(locked_v2._MARK_RE.findall(protected.masked)),
    }


def validate_canary_rejection() -> dict[str, object]:
    """Bind this run to the exact artifact that selected the v2 method."""
    if base.sha256_file(CANARY_FINAL) != CANARY_FINAL_SHA256:
        raise base.CanaryError("v2 canary artifact hash changed")
    result = base.load_json(CANARY_FINAL, "v2 canary artifact")
    selection = base.require_mapping(result.get("selection"), "v2 canary selection")
    if selection.get("advancedMethod") != "visible_anchor_sentence_aligned_v2":
        raise base.CanaryError("v2 canary artifact does not describe this method")
    return {
        "canaryFinalPath": "results/model-canary-terra-locked-v2-final-v1.json",
        "canaryFinalSha256": CANARY_FINAL_SHA256,
        "canaryManualFidelityPassed": selection.get("manualFidelityPassed"),
        "canaryPipelineAndDistancePassed": selection.get("pipelineAndDistancePassed"),
        "method": "visible_anchor_sentence_aligned_v2",
    }


def verify_engine_contract() -> None:
    """Fail closed if configuration left the engine on another candidate."""
    if engine.MODEL != terra.MODEL or engine.EXPECTED_MODELS != terra.EXPECTED_MODELS:
        raise base.CanaryError("engine model is not the Terra candidate")
    if (
        engine.PROMPT_PRICE != terra.PROMPT_PRICE
        or engine.COMPLETION_PRICE != terra.COMPLETION_PRICE
        or engine.CACHE_READ_PRICE != terra.CACHE_READ_PRICE
        or engine.CACHE_WRITE_PRICE != terra.CACHE_WRITE_PRICE
    ):
        raise base.CanaryError("engine prices are not the Terra prices")
    if (
        unmark.request_messages is not locked_v2.locked_request_messages
        or engine.request_messages is not locked_v2.locked_request_messages
    ):
        raise base.CanaryError("engine is not using the v2 locked request builder")
    probe = request_for("A short probe sentence must not change.")
    payload = engine.expected_payload(probe)
    if payload.get("model") != terra.MODEL:
        raise base.CanaryError("frozen payload model differs from the contract model")
    messages = payload.get("messages")
    if (
        not isinstance(messages, list)
        or not messages
        or messages[0].get("content") != locked_v2.LOCKED_SYSTEM_INSTRUCTION
    ):
        raise base.CanaryError("frozen payload does not carry the v2 instruction")


def configure() -> None:
    """Bind the shared holdout protocol to Terra and the v2 transformation."""
    locked_v2.configure_engine()
    verify_engine_contract()
    prerequisite = validate_canary_rejection()
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base.CALL_ID_PREFIX = "locked-v2-final"
    base.DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT
    base.DEFAULT_AGGREGATE = DEFAULT_AGGREGATE
    base.DEFAULT_PACKET = DEFAULT_PACKET
    base.DEFAULT_FINAL = DEFAULT_FINAL
    base.PREREQUISITE_BINDINGS = prerequisite
    base.request_for = request_for
    base.analyze_output = analyze_output
    base.CANDIDATE = replace(
        base.CANDIDATE,
        model=terra.MODEL,
        source_path=ROOT / "run_model_canary_terra_locked_v2.py",
        source_sha256=base.sha256_file(ROOT / "run_model_canary_terra_locked_v2.py"),
        fetch_catalog=terra.fetch_catalog,
        validate_route_record=terra.validate_route_record,
        expected_models=terra.EXPECTED_MODELS,
        expected_cost=terra_usage_cost,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--live", action="store_true")
    modes.add_argument("--aggregate", action="store_true")
    modes.add_argument("--blind-packet", action="store_true")
    modes.add_argument("--finalize-review", action="store_true")
    modes.add_argument("--recompute-analysis", action="store_true")
    parser.add_argument("--note")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--aggregate-output", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--budget")
    parser.add_argument("--max-new-calls", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure()
    args = build_parser().parse_args(argv)
    if args.dry_run:
        result = dict(base.dry_run())
        result["method"] = "visible_anchor_sentence_aligned_v2"
        result["runnerSha256"] = base.sha256_file(Path(__file__))
        print(canonical_json_bytes(result).decode("utf-8"), end="")
        return 0
    if args.recompute_analysis:
        if not args.note:
            raise SystemExit("--recompute-analysis requires --note")
        print(
            json.dumps(
                base.recompute_analysis(args.checkpoint, args.note), sort_keys=True
            )
        )
        return 0
    if args.aggregate:
        result = base.write_aggregate(args.checkpoint, args.aggregate_output)
        print(json.dumps(result["automatedGate"], sort_keys=True))
        return 0
    if args.blind_packet:
        packet = base.build_blind_packet(args.checkpoint, args.packet)
        print(
            json.dumps({"packet": str(args.packet), "sha256": packet["packetSha256"]})
        )
        return 0
    if args.finalize_review:
        if args.review is None:
            raise SystemExit("--finalize-review requires --review")
        result = base.finalize_review(
            args.checkpoint, args.packet, args.review, args.output
        )
        print(json.dumps({"passed": result["finalConfirmationPassed"]}))
        return 0
    if args.budget is None or args.max_new_calls is None:
        raise SystemExit("--live requires --budget and --max-new-calls")
    print(
        json.dumps(
            base.run_live(
                args.checkpoint,
                base.decimal(args.budget, "budget"),
                args.max_new_calls,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
