"""Build and aggregate the 10-document curated percentage evaluation.

This module owns the frozen design validation, direct DIPPER input, blinded
five-candidate panel batches, percentage formulas, and final table assembly.
Paid transport is implemented separately so all prompts and controls can be
frozen and tested before the first call.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from decimal import Decimal
import math
from pathlib import Path
import random
import statistics
from typing import Mapping, Sequence

import dipper_smoke


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "curated-percent-eval-v1.json"
DEFAULT_DIPPER_INPUT = ROOT / "results" / "curated-dipper-inputs-v1.json"
DEFAULT_METHODS = ROOT / "results" / "curated-methods-v1.json"
DEFAULT_DIPPER = ROOT / "results" / "curated-dipper-v1.json"
DEFAULT_PAIRS = ROOT / "results" / "curated-percent-pairs-v1.json"
DEFAULT_PANEL_INPUT = ROOT / "results" / "curated-panel-input-v1.json"
DEFAULT_FINAL = ROOT / "results" / "curated-percent-table-v1.json"
PANEL_SEED = 20260821
ALLOWED_PERCENTAGES = tuple(range(0, 101, 10))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_design(config_path: Path = DEFAULT_CONFIG):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    corpus_path = ROOT / config["corpus"]["path"]
    calibration_path = ROOT / config["calibration"]["path"]
    claims_path = ROOT / config["claims"]["path"]
    for path, expected in (
        (corpus_path, config["corpus"]["sha256"]),
        (calibration_path, config["calibration"]["sha256"]),
        (claims_path, config["claims"]["sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"frozen input hash mismatch: {path}")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    document_ids = list(config["documentIds"])
    if [document["documentId"] for document in corpus["documents"]] != document_ids:
        raise ValueError("corpus document order differs from frozen design")
    expected_claim_ids = [f"c{index:02d}" for index in range(1, 11)]
    if sorted(claims["documents"]) != sorted(document_ids):
        raise ValueError("claims document set differs from frozen design")
    for document_id in document_ids:
        rows = claims["documents"][document_id]
        if [row["id"] for row in rows] != expected_claim_ids:
            raise ValueError(f"claim IDs are not c01..c10 for {document_id}")
    if config["percentageContract"]["allowedJudgePercentages"] != list(
        ALLOWED_PERCENTAGES
    ):
        raise ValueError("percentage grid must use ten-point increments")
    if len({judge["vendor"] for judge in config["judges"]}) != 4:
        raise ValueError("panel must contain four different vendors")
    threshold = float(config["calibration"]["singleDetectionThreshold"])
    if threshold != float(calibration["singleThreshold"]["threshold"]):
        raise ValueError("frozen threshold differs from calibration artifact")
    budget_sum = sum(float(value) for key, value in config["budgetsUsd"].items() if key != "totalAdditional")
    if budget_sum > float(config["budgetsUsd"]["totalAdditional"]) + 1e-12:
        raise ValueError("component budgets exceed total additional budget")
    return config, corpus, calibration, claims


def watermark_removal_percent(
    *, source_mean: float, candidate_mean: float, clean_mean: float
) -> float:
    denominator = source_mean - clean_mean
    if denominator <= 0:
        raise ValueError("marked source score must exceed paired clean score")
    value = 100.0 * (source_mean - candidate_mean) / denominator
    return max(0.0, min(100.0, value))


def pipeline_failure_watermark_removal_percent() -> float:
    """Frozen denominator policy for methods that return no usable final text."""
    return 0.0


def lower_median_percent(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("percentage aggregation requires at least one value")
    if any(value not in ALLOWED_PERCENTAGES for value in values):
        raise ValueError("judge percentage is outside the 10-percent grid")
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def validate_panel_candidate(
    candidate: Mapping[str, object],
    *,
    expected_candidate_id: str,
    expected_claim_ids: Sequence[str],
) -> dict[str, object]:
    if candidate.get("candidateId") != expected_candidate_id:
        raise ValueError("panel candidate ID mismatch")
    claims = candidate.get("claims")
    if not isinstance(claims, list) or [row.get("id") for row in claims] != list(
        expected_claim_ids
    ):
        raise ValueError("panel claim IDs or order mismatch")
    normalized_claims = []
    for row in claims:
        status = row.get("status")
        if status not in ("preserved", "changed", "missing"):
            raise ValueError("invalid panel claim status")
        normalized_claims.append({"id": row["id"], "status": status})
    result = {
        "candidateId": expected_candidate_id,
        "claims": normalized_claims,
    }
    for key in ("readabilityPercent", "usabilityPercent"):
        value = candidate.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value not in ALLOWED_PERCENTAGES:
            raise ValueError(f"{key} must lie on the 10-percent grid")
        result[key] = value
    errors = candidate.get("materialErrors")
    if not isinstance(errors, list) or len(errors) > 5 or not all(
        isinstance(value, str) and 0 < len(value) <= 240 for value in errors
    ):
        raise ValueError("materialErrors must be an array of at most five short strings")
    result["materialErrors"] = errors
    return result


def aggregate_pair_panel(
    verdicts: Sequence[Mapping[str, object]], *, claim_ids: Sequence[str]
) -> dict[str, object]:
    if len(verdicts) != 4:
        raise ValueError("a complete panel requires four verdicts")
    claim_results = []
    for claim_id in claim_ids:
        statuses = []
        for verdict in verdicts:
            by_id = {row["id"]: row["status"] for row in verdict["claims"]}
            statuses.append(by_id[claim_id])
        preserved = statuses.count("preserved")
        claim_results.append(
            {
                "id": claim_id,
                "majorityPreserved": preserved >= 3,
                "preservedVotes": preserved,
                "statuses": statuses,
            }
        )
    return {
        "claimPreservationPercent": 10
        * sum(row["majorityPreserved"] for row in claim_results),
        "claimResults": claim_results,
        "panelComplete": True,
        "readabilityPercent": lower_median_percent(
            [int(verdict["readabilityPercent"]) for verdict in verdicts]
        ),
        "usabilityPercent": lower_median_percent(
            [int(verdict["usabilityPercent"]) for verdict in verdicts]
        ),
    }


def build_direct_dipper_input(
    *, config: Mapping[str, object], corpus: Mapping[str, object]
) -> dict[str, object]:
    by_id = {document["documentId"]: document for document in corpus["documents"]}
    cases = []
    for document_id in config["documentIds"]:
        marked = by_id[document_id]["marked"]
        text = str(marked["text"])
        prompt = str(marked["prompt"])
        cases.append(
            {
                "caseId": f"{document_id}::marked-source",
                "documentId": document_id,
                "inputKind": "marked-source",
                "inputText": text,
                "inputTextSha256": sha256_text(text),
                "methodBeforeDipper": None,
                "preDipperMeanG": float(marked["meanG"]),
                "prefix": prompt,
                "prefixSha256": sha256_text(prompt),
            }
        )
    now = utc_now()
    return {
        "attack": dipper_smoke.attack_contract(),
        "cases": cases,
        "createdAt": now,
        "inputs": {
            "corpusSha256": config["corpus"]["sha256"],
            "documentIds": list(config["documentIds"]),
        },
        "methodology": (
            "Apply the pinned published DIPPER-11B baseline directly to each of the ten "
            "curated marked sources. No prior transformation is used."
        ),
        "schemaVersion": 1,
        "sources": dipper_smoke.evidence_sources(),
        "verifiedAt": now,
    }


def build_pairs(
    *,
    config: Mapping[str, object],
    corpus: Mapping[str, object],
    methods: Mapping[str, object],
    dipper: Mapping[str, object],
) -> dict[str, object]:
    by_id = {document["documentId"]: document for document in corpus["documents"]}
    rows_by_key = {
        (row["documentId"], row["method"]): row for row in methods["documents"]
    }
    dipper_by_id = {
        row["documentId"]: row
        for row in dipper["documents"]
        if row["inputKind"] == "marked-source"
    }
    pairs = []
    for document_id in config["documentIds"]:
        source = by_id[document_id]["marked"]
        clean = by_id[document_id]["unmarked"]
        for method in config["methods"]:
            if method == "dipper":
                row = dipper_by_id.get(document_id)
                if row is None:
                    raise ValueError(f"missing DIPPER row for {document_id}")
                candidate_text = str(row["dipperOutputText"])
                detector = row["afterDetector"]
                word_distance = float(row["wordDistanceFromMarkedSource"])
            else:
                row = rows_by_key.get((document_id, method))
                if row is None:
                    raise ValueError(f"missing method row {document_id}::{method}")
                if row.get("outcome") == "completed":
                    candidate_text = str(row["evaluatedOutputText"])
                    detector = row["transformedDetector"]
                    word_distance = float(row["wordDistance"])
                else:
                    # A rejected pivot produced no usable final English text. Keep the
                    # case in all denominators, expose an empty candidate to the blinded
                    # quality panel, and assign zero product-level signal removal rather
                    # than pretending that failure is a successful scrub.
                    candidate_text = ""
                    detector = {"meanG": None, "pipelineOutcome": row.get("outcome")}
                    word_distance = 1.0
            candidate_mean = detector.get("meanG")
            removal = (
                pipeline_failure_watermark_removal_percent()
                if candidate_mean is None
                else watermark_removal_percent(
                    source_mean=float(source["meanG"]),
                    candidate_mean=float(candidate_mean),
                    clean_mean=float(clean["meanG"]),
                )
            )
            pairs.append(
                {
                    "candidateDetector": detector,
                    "candidateText": candidate_text,
                    "candidateTextSha256": sha256_text(candidate_text),
                    "cleanMeanG": clean["meanG"],
                    "detectedAtThreshold": (
                        None
                        if candidate_mean is None
                        else float(candidate_mean)
                        >= float(config["calibration"]["singleDetectionThreshold"])
                    ),
                    "documentId": document_id,
                    "method": method,
                    "pipelineCompleted": candidate_mean is not None,
                    "pairId": f"{document_id}::{method}",
                    "sourceMeanG": source["meanG"],
                    "sourceText": source["text"],
                    "sourceTextSha256": source["textSha256"],
                    "watermarkRemovalPercent": removal,
                    "wordDistancePercent": 100.0 * word_distance,
                }
            )
    if len(pairs) != 50 or len({pair["pairId"] for pair in pairs}) != 50:
        raise ValueError("percentage evaluation requires exactly fifty unique pairs")
    now = utc_now()
    return {
        "createdAt": now,
        "inputs": {
            "corpusSha256": config["corpus"]["sha256"],
            "dipperSha256": None,
            "methodsSha256": None,
        },
        "methodology": (
            "Join ten curated marked sources to four frozen OpenRouter transformations "
            "and direct DIPPER. Calculate watermark removal as the clipped fraction of "
            "the paired source-to-clean signal removed."
        ),
        "pairs": pairs,
        "schemaVersion": 1,
        "sources": config["sources"],
        "verifiedAt": now,
    }


def build_panel_input(
    *,
    config: Mapping[str, object],
    claims: Mapping[str, object],
    pairs: Mapping[str, object],
    seed: int = PANEL_SEED,
) -> dict[str, object]:
    by_document: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for pair in pairs["pairs"]:
        by_document[str(pair["documentId"])].append(pair)
    batches = []
    blind_map = {}
    for index, document_id in enumerate(config["documentIds"]):
        candidates = list(by_document[document_id])
        random.Random(seed + index).shuffle(candidates)
        source_hashes = {pair["sourceTextSha256"] for pair in candidates}
        if len(candidates) != 5 or len(source_hashes) != 1:
            raise ValueError(f"panel batch is incomplete for {document_id}")
        blind_candidates = []
        for candidate_index, pair in enumerate(candidates, start=1):
            blind_id = f"candidate-{candidate_index:02d}"
            blind_candidates.append(
                {
                    "candidateId": blind_id,
                    "text": pair["candidateText"],
                    "textSha256": pair["candidateTextSha256"],
                }
            )
            blind_map[f"{document_id}::{blind_id}"] = pair["pairId"]
        batches.append(
            {
                "batchId": document_id,
                "candidates": blind_candidates,
                "claims": claims["documents"][document_id],
                "documentId": document_id,
                "sourceText": candidates[0]["sourceText"],
                "sourceTextSha256": candidates[0]["sourceTextSha256"],
            }
        )
    now = utc_now()
    return {
        "batches": batches,
        "blindMap": blind_map,
        "createdAt": now,
        "methodology": (
            "Blind method identity and deterministically shuffle five candidates per "
            "source. Supply exactly ten frozen claims and require 10-point percentage "
            "scores from each judge."
        ),
        "schemaVersion": 1,
        "seed": seed,
        "sources": config["sources"],
        "verifiedAt": now,
    }


def split_panel_input_one_candidate(
    panel_input: Mapping[str, object], *, label: str
) -> dict[str, object]:
    batches = []
    blind_map = {}
    for batch in panel_input["batches"]:
        for candidate in batch["candidates"]:
            batch_id = f"{batch['batchId']}-{label}-{candidate['candidateId']}"
            batches.append(
                {
                    "batchId": batch_id,
                    "candidates": [candidate],
                    "claims": batch["claims"],
                    "documentId": batch["documentId"],
                    "sourceText": batch["sourceText"],
                    "sourceTextSha256": batch["sourceTextSha256"],
                }
            )
            original_key = f"{batch['batchId']}::{candidate['candidateId']}"
            blind_map[f"{batch_id}::{candidate['candidateId']}"] = panel_input[
                "blindMap"
            ][original_key]
    now = utc_now()
    return {
        "batches": batches,
        "blindMap": blind_map,
        "createdAt": now,
        "derivedFromSha256": None,
        "methodology": (
            "Split each blinded five-candidate source batch into one-candidate batches "
            "without changing candidate IDs, text, source, claims, or score contract."
        ),
        "schemaVersion": 1,
        "sources": panel_input["sources"],
        "verifiedAt": now,
    }


def combine_panel_artifacts(
    *,
    batched_input: Mapping[str, object],
    split_input: Mapping[str, object],
    batched_output: Mapping[str, object],
    split_output: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    blind_map = {**batched_input["blindMap"], **split_input["blindMap"]}
    calls = {**batched_output["calls"], **split_output["calls"]}
    if len(calls) != len(batched_output["calls"]) + len(split_output["calls"]):
        raise ValueError("panel call keys overlap while combining")
    now = utc_now()
    combined_input = {
        "batches": list(batched_input["batches"]) + list(split_input["batches"]),
        "blindMap": blind_map,
        "createdAt": now,
        "methodology": "Union of frozen batched three-vendor input and one-candidate xAI input.",
        "schemaVersion": 1,
        "sources": batched_input["sources"],
        "verifiedAt": now,
    }
    combined_output = {
        "calls": calls,
        "createdAt": now,
        "methodology": (
            "Combine three blinded five-candidate judges with one blinded single-"
            "candidate xAI judge. Every final pair has exactly four vendor-independent "
            "verdicts under the same claims and ten-percent contract."
        ),
        "schemaVersion": 1,
        "sources": batched_output["sources"],
        "status": "complete",
        "totalCostUsd": split_output["totalCostUsd"],
        "verifiedAt": now,
    }
    return combined_input, combined_output


def merge_single_candidate_panels(
    *,
    panel_input: Mapping[str, object],
    outputs: Sequence[Mapping[str, object]],
    judge_models: Sequence[str],
) -> dict[str, object]:
    """Union several one-candidate panel checkpoints over the same frozen input.

    Every checkpoint must have been produced against ``panel_input`` (same
    ``panelInputSha256``), call keys must not overlap, and after the union every
    batch must carry exactly one valid verdict from each configured judge.
    """
    input_sha = sha256_text(
        json.dumps(panel_input, ensure_ascii=False, sort_keys=True)
    )
    calls: dict[str, object] = {}
    total = Decimal(0)
    evidence = []
    for output in outputs:
        overlap = set(calls) & set(output["calls"])
        if overlap:
            raise ValueError(f"panel call keys overlap while merging: {sorted(overlap)[:3]}")
        calls.update(output["calls"])
        total += Decimal(str(output["totalCostUsd"]))
        evidence.append(
            {
                "panelInputSha256": output.get("panelInputSha256"),
                "selectedModels": output.get("selectedModels"),
                "status": output.get("status"),
                "totalCostUsd": output.get("totalCostUsd"),
            }
        )
    expected_models = set(judge_models)
    for batch in panel_input["batches"]:
        if len(batch["candidates"]) != 1:
            raise ValueError(f"batch {batch['batchId']} is not single-candidate")
        seen = set()
        for model in expected_models:
            call = calls.get(f"{batch['batchId']}::{model}")
            if not call or "candidates" not in call:
                raise ValueError(f"missing valid verdict {batch['batchId']}::{model}")
            seen.add(model)
        if seen != expected_models:
            raise ValueError(f"incomplete judge set for {batch['batchId']}")
    now = utc_now()
    return {
        "calls": calls,
        "createdAt": now,
        "inputDigestSha256": input_sha,
        "mergedFrom": evidence,
        "methodology": (
            "Union of one-candidate panel checkpoints over one frozen single-candidate "
            "input. Every judge saw exactly one source and one candidate per prompt, so "
            "no verdict could be contaminated by a neighbouring candidate. Every final "
            "pair has exactly four vendor-independent verdicts under the same claims "
            "and ten-percent contract."
        ),
        "schemaVersion": 1,
        "sources": panel_input["sources"],
        "status": "complete",
        "totalCostUsd": format(total, "f"),
        "verifiedAt": now,
    }


def summarize_final_pairs(
    pairs: Sequence[Mapping[str, object]], methods: Sequence[str]
) -> dict[str, object]:
    metrics = (
        "watermarkRemovalPercent",
        "pspPercent",
        "claimPreservationPercent",
        "readabilityPercent",
        "usabilityPercent",
        "finalPercent",
    )
    output = {}
    for method in methods:
        rows = [row for row in pairs if row["method"] == method]
        if len(rows) != 10:
            raise ValueError(f"final summary requires ten rows for {method}")
        output[method] = {
            key: statistics.mean(float(row[key]) for row in rows) for key in metrics
        }
        output[method].update(
            {
                "detectorRemovedCount": sum(
                    row.get("detectedAtThreshold") is False for row in rows
                ),
                "documentCount": len(rows),
                "pipelineCompletedCount": sum(
                    row.get("pipelineCompleted") is True for row in rows
                ),
            }
        )
    return output


def finalize(
    *,
    config: Mapping[str, object],
    pairs_artifact: Mapping[str, object],
    psp_artifact: Mapping[str, object],
    panel_input: Mapping[str, object],
    panel_output: Mapping[str, object],
) -> dict[str, object]:
    psp_by_pair = {row["pairId"]: row for row in psp_artifact["rows"]}
    verdicts: dict[str, list[dict[str, object]]] = defaultdict(list)
    for call in panel_output["calls"].values():
        if "terminalError" in call:
            raise ValueError("full panel contains a terminal provider error")
        batch_id = str(call["batchId"])
        for candidate in call["candidates"]:
            pair_id = panel_input["blindMap"][
                f"{batch_id}::{candidate['candidateId']}"
            ]
            verdicts[pair_id].append(
                {
                    **candidate,
                    "judge": call["judge"],
                }
            )
    claim_ids = [f"c{index:02d}" for index in range(1, 11)]
    output_pairs = []
    for pair in pairs_artifact["pairs"]:
        pair_id = pair["pairId"]
        if pair_id not in psp_by_pair:
            raise ValueError(f"missing P-SP row for {pair_id}")
        panel = aggregate_pair_panel(verdicts[pair_id], claim_ids=claim_ids)
        row = {
            **pair,
            "claimPreservationPercent": panel["claimPreservationPercent"],
            "panel": panel,
            "psp": psp_by_pair[pair_id]["psp"],
            "pspPercent": psp_by_pair[pair_id]["pspPercent"],
            "readabilityPercent": panel["readabilityPercent"],
            "usabilityPercent": panel["usabilityPercent"],
        }
        row["finalPercent"] = min(
            float(row["watermarkRemovalPercent"]),
            float(row["claimPreservationPercent"]),
            float(row["readabilityPercent"]),
            float(row["usabilityPercent"]),
        )
        output_pairs.append(row)
    summary = summarize_final_pairs(output_pairs, config["methods"])
    now = utc_now()
    return {
        "createdAt": now,
        "methodology": (
            "Combine paired normalized watermark-signal removal, official P-SP, a "
            "majority vote over ten frozen claims, and conservative lower-median "
            "readability/usability scores restricted to ten-point increments. The "
            "per-document final percentage is the minimum of the four required axes."
        ),
        "pairs": output_pairs,
        "percentageContract": config["percentageContract"],
        "schemaVersion": 1,
        "sources": config["sources"],
        "summary": summary,
        "verifiedAt": now,
    }


def _prepare_dipper(args: argparse.Namespace) -> int:
    config, corpus, _, _ = load_design(args.config)
    artifact = build_direct_dipper_input(config=config, corpus=corpus)
    write_json_atomic(args.output, artifact)
    return 0


def _build_pairs_command(args: argparse.Namespace) -> int:
    config, corpus, _, claims = load_design(args.config)
    methods = json.loads(args.methods.read_text(encoding="utf-8"))
    dipper = json.loads(args.dipper.read_text(encoding="utf-8"))
    artifact = build_pairs(config=config, corpus=corpus, methods=methods, dipper=dipper)
    artifact["inputs"]["methodsSha256"] = sha256_file(args.methods)
    artifact["inputs"]["dipperSha256"] = sha256_file(args.dipper)
    write_json_atomic(args.output, artifact)
    panel = build_panel_input(config=config, claims=claims, pairs=artifact)
    write_json_atomic(args.panel_output, panel)
    return 0


def _finalize_command(args: argparse.Namespace) -> int:
    config, _, _, _ = load_design(args.config)
    inputs = {
        "pairs": json.loads(args.pairs.read_text(encoding="utf-8")),
        "psp": json.loads(args.psp.read_text(encoding="utf-8")),
        "panelInput": json.loads(args.panel_input.read_text(encoding="utf-8")),
        "panelOutput": json.loads(args.panel_output.read_text(encoding="utf-8")),
    }
    artifact = finalize(
        config=config,
        pairs_artifact=inputs["pairs"],
        psp_artifact=inputs["psp"],
        panel_input=inputs["panelInput"],
        panel_output=inputs["panelOutput"],
    )
    artifact["inputs"] = {
        "pairsSha256": sha256_file(args.pairs),
        "panelInputSha256": sha256_file(args.panel_input),
        "panelOutputSha256": sha256_file(args.panel_output),
        "pspSha256": sha256_file(args.psp),
    }
    methods = json.loads(args.methods.read_text(encoding="utf-8"))
    lifecycle = json.loads(args.dipper_lifecycle.read_text(encoding="utf-8"))
    transformation_cost = float(methods["budget"]["spentUsd"])
    dipper_cost = float(lifecycle["estimatedCostUpperBoundUsd"])
    panel_cost = float(inputs["panelOutput"]["totalCostUsd"])
    artifact["cost"] = {
        "dipperGpuUpperBoundUsd": dipper_cost,
        "hardAdditionalCeilingUsd": float(config["budgetsUsd"]["totalAdditional"]),
        "panelConservativeUpperBoundUsd": panel_cost,
        "pspUsd": 0.0,
        "totalUpperBoundUsd": transformation_cost + dipper_cost + panel_cost,
        "transformationsUsd": transformation_cost,
    }
    artifact["inputs"]["dipperLifecycleSha256"] = sha256_file(
        args.dipper_lifecycle
    )
    artifact["inputs"]["methodsSha256"] = sha256_file(args.methods)
    write_json_atomic(args.output, artifact)
    return 0


def _split_panel_command(args: argparse.Namespace) -> int:
    source = json.loads(args.input.read_text(encoding="utf-8"))
    artifact = split_panel_input_one_candidate(source, label=args.label)
    artifact["derivedFromSha256"] = sha256_file(args.input)
    write_json_atomic(args.output, artifact)
    return 0


def _merge_single_panel_command(args: argparse.Namespace) -> int:
    config, _, _, _ = load_design(args.config)
    panel_input = json.loads(args.input.read_text(encoding="utf-8"))
    outputs = [json.loads(path.read_text(encoding="utf-8")) for path in args.outputs]
    merged = merge_single_candidate_panels(
        panel_input=panel_input,
        outputs=outputs,
        judge_models=[judge["model"] for judge in config["judges"]],
    )
    merged["mergedFromFiles"] = [
        {"path": str(path), "sha256": sha256_file(path)} for path in args.outputs
    ]
    merged["panelInputSha256"] = sha256_file(args.input)
    write_json_atomic(args.output_panel, merged)
    return 0


def _combine_panel_command(args: argparse.Namespace) -> int:
    values = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (
            args.batched_input,
            args.split_input,
            args.batched_output,
            args.split_output,
        )
    ]
    combined_input, combined_output = combine_panel_artifacts(
        batched_input=values[0],
        split_input=values[1],
        batched_output=values[2],
        split_output=values[3],
    )
    combined_output["inputs"] = {
        "batchedOutputSha256": sha256_file(args.batched_output),
        "splitOutputSha256": sha256_file(args.split_output),
    }
    write_json_atomic(args.output_input, combined_input)
    write_json_atomic(args.output_panel, combined_output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-dipper")
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare.add_argument("--output", type=Path, default=DEFAULT_DIPPER_INPUT)
    prepare.set_defaults(handler=_prepare_dipper)
    pairs = commands.add_parser("build-pairs")
    pairs.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pairs.add_argument("--methods", type=Path, default=DEFAULT_METHODS)
    pairs.add_argument("--dipper", type=Path, default=DEFAULT_DIPPER)
    pairs.add_argument("--output", type=Path, default=DEFAULT_PAIRS)
    pairs.add_argument("--panel-output", type=Path, default=DEFAULT_PANEL_INPUT)
    pairs.set_defaults(handler=_build_pairs_command)
    finish = commands.add_parser("finalize")
    finish.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    finish.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    finish.add_argument("--psp", type=Path, required=True)
    finish.add_argument("--panel-input", type=Path, default=DEFAULT_PANEL_INPUT)
    finish.add_argument("--panel-output", type=Path, required=True)
    finish.add_argument("--output", type=Path, default=DEFAULT_FINAL)
    finish.add_argument("--methods", type=Path, default=DEFAULT_METHODS)
    finish.add_argument(
        "--dipper-lifecycle",
        type=Path,
        default=ROOT / "results" / "curated-dipper-runpod-v2-lifecycle-v3.json",
    )
    finish.set_defaults(handler=_finalize_command)
    split = commands.add_parser("split-panel-input")
    split.add_argument("--input", type=Path, default=DEFAULT_PANEL_INPUT)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--label", default="single")
    split.set_defaults(handler=_split_panel_command)
    merge = commands.add_parser("merge-single-panel")
    merge.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    merge.add_argument("--input", type=Path, required=True)
    merge.add_argument("--outputs", type=Path, nargs="+", required=True)
    merge.add_argument("--output-panel", type=Path, required=True)
    merge.set_defaults(handler=_merge_single_panel_command)

    combine = commands.add_parser("combine-panel")
    combine.add_argument("--batched-input", type=Path, required=True)
    combine.add_argument("--split-input", type=Path, required=True)
    combine.add_argument("--batched-output", type=Path, required=True)
    combine.add_argument("--split-output", type=Path, required=True)
    combine.add_argument("--output-input", type=Path, required=True)
    combine.add_argument("--output-panel", type=Path, required=True)
    combine.set_defaults(handler=_combine_panel_command)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
