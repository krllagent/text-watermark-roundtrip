"""Freeze prospective-key controls from the already committed phase-A review."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

from corpus_contract import canonical_json_bytes
from freeze_final_holdout_v8 import (
    _atomic_write,
    _encode_documents,
    _fraction,
    _json_object,
    _load_bound_json,
    _load_source_documents,
    validate_review_bound_allowlist,
)
from watermark_toy import Document, load_lexicon, run_wrong_key_controls, score_corpus


SCHEMA_VERSION = 1
PLAN_PATH = "fixtures/final-holdout-plan-v9.json"
KEY_PATH = "fixtures/final-holdout-key-v9.json"
MARKED_DIRECTORY = "corpus/holdout-v6/reviewed-encoder-v9/marked-1000"
MARKED_MANIFEST_PATH = f"{MARKED_DIRECTORY}/manifest-v9.json"
ARTIFACT_PATH = "results/final-holdout-controls-v9.json"
ARTIFACT_VERSION = "final-holdout-controls-v9-prospective-key"


@dataclass(frozen=True)
class FinalHoldoutV9Package:
    plan: dict[str, object]
    key_artifact: dict[str, object]
    artifact: dict[str, object]
    files: dict[str, bytes]


def build_final_holdout_v9_package(
    plan_path: str | Path,
    *,
    root: str | Path | None = None,
) -> FinalHoldoutV9Package:
    root_path = (
        Path(root).resolve() if root is not None else Path(__file__).resolve().parent
    )
    selected_plan_path = Path(plan_path).resolve()
    plan_bytes = selected_plan_path.read_bytes()
    plan = _json_object(plan_bytes, "v9 final holdout plan")
    _validate_evidence(plan, "v9 final holdout plan")
    if plan.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unsupported v9 plan schemaVersion")
    if (
        plan.get("experimentVersion")
        != "text-watermark-final-holdout-v9-prospective-key"
    ):
        raise ValueError("unexpected v9 experimentVersion")

    phase_a = plan["phaseA"]
    if phase_a.get("commit") != "559a1d2749a1f0468f12bf3a51d81eca644164fe":
        raise ValueError("phase-A commit binding changed")
    phase_a_plan = _load_bound_json(root_path, phase_a["plan"], "phase-A plan")
    inventory = _load_bound_json(root_path, phase_a["inventory"], "phase-A inventory")
    review = _load_bound_json(root_path, phase_a["review"], "phase-A review")
    allowlist = _load_bound_json(root_path, phase_a["allowlist"], "phase-A allowlist")
    developmental = _load_bound_json(
        root_path,
        phase_a["developmentalControls"],
        "developmental predecessor",
    )
    if (
        phase_a["developmentalControls"].get("classification")
        != "developmental_not_confirmatory"
    ):
        raise ValueError("old-key controls must be classified as developmental")
    if developmental.get("prepaidGate", {}).get("status") != "passed":
        raise ValueError("phase-A developmental controls changed")
    validate_review_bound_allowlist(phase_a_plan, inventory, review, allowlist)

    key_artifact = _load_bound_json(root_path, plan["keyArtifact"], "v9 key artifact")
    _validate_key_artifact(plan, key_artifact)
    key = bytes.fromhex(key_artifact["keyHex"])

    source_plan = _load_bound_json(
        root_path, phase_a_plan["sourceCorpus"]["plan"], "source corpus plan"
    )
    source_manifest = _load_bound_json(
        root_path,
        phase_a_plan["sourceCorpus"]["manifest"],
        "source corpus manifest",
    )
    lexicon_path = root_path / phase_a_plan["lexicon"]["path"]
    if (
        hashlib.sha256(lexicon_path.read_bytes()).hexdigest()
        != phase_a_plan["lexicon"]["sha256"]
    ):
        raise ValueError("phase-A lexicon SHA mismatch")
    lexicon = load_lexicon(lexicon_path)
    detector = phase_a_plan["detectorImplementation"]
    detector_path = root_path / detector["path"]
    if hashlib.sha256(detector_path.read_bytes()).hexdigest() != detector["sha256"]:
        raise ValueError("phase-A detector implementation SHA mismatch")
    if detector.get("allowlistAware") is not False:
        raise ValueError("full detector must remain allowlist-unaware")
    documents = _load_source_documents(root_path, source_plan, source_manifest)

    controls = plan["prepaidControls"]
    density_bps = int(controls["densityBps"])
    context_width = int(controls["contextWidth"])
    min_active = int(controls["minActivePositions"])
    marked, encoding = _encode_documents(
        documents,
        inventory,
        allowlist,
        key=key,
        density_bps=density_bps,
        context_width=context_width,
        lexicon=lexicon,
    )
    original_documents = tuple(
        Document(item["documentId"], item["text"]) for item in documents
    )
    marked_documents = tuple(
        Document(item["documentId"], item["text"]) for item in marked
    )
    unmarked_score = score_corpus(
        original_documents,
        key=key,
        density_bps=density_bps,
        lexicon=lexicon,
        context_width=context_width,
        min_active_positions=min_active,
    )
    marked_score = score_corpus(
        marked_documents,
        key=key,
        density_bps=density_bps,
        lexicon=lexicon,
        context_width=context_width,
        min_active_positions=min_active,
    )
    wrong_keys = run_wrong_key_controls(
        marked_documents,
        density_bps=density_bps,
        lexicon=lexicon,
        count=int(controls["wrongKeyCount"]),
        seed=bytes.fromhex(controls["wrongKeySeedHex"]),
        context_width=context_width,
        min_active_positions=min_active,
    )

    evidence = _evidence(plan)
    marked_manifest = {
        **evidence,
        "artifactVersion": "final-holdout-marked-inputs-v9-prospective-key",
        "allowlistSha256": phase_a["allowlist"]["sha256"],
        "densityBps": density_bps,
        "documentCount": len(marked),
        "documents": [
            {
                "documentId": item["documentId"],
                "path": f"{MARKED_DIRECTORY}/{item['documentId']}.md",
                "sha256": hashlib.sha256(item["text"].encode("utf-8")).hexdigest(),
                "sourceSha256": item["sourceSha256"],
            }
            for item in marked
        ],
        "keyArtifactSha256": plan["keyArtifact"]["sha256"],
        "phaseACommit": phase_a["commit"],
        "reviewSha256": phase_a["review"]["sha256"],
    }
    marked_manifest_bytes = canonical_json_bytes(marked_manifest)
    wrong_key_rows = [
        {
            "activePositions": score.active_positions,
            "hits": score.hits,
            "index": index,
            "pValueExact": _fraction(score.p_value),
            "status": score.status,
            "zScore": score.z_score,
        }
        for index, score in enumerate(wrong_keys.scores)
    ]
    expected = plan["expectedBaselineAfterFirstDraw"]
    checks = {
        "approvedActiveFavored": encoding["approvedActiveFavored"]
        == expected["approvedActiveFavored"],
        "firstDrawBound": key_artifact["draw"]["drawCount"] == 1
        and key_artifact["draw"]["redrawAllowed"] is False,
        "markedActiveAtLeast20": marked_score.active_positions >= min_active,
        "markedDetected": marked_score.status == "detected",
        "markedMatchesFirstDraw": _score_matches(marked_score, expected["marked"]),
        "noRejectedPhysicalChanges": encoding["rejectedPhysicalChanges"] == 0,
        "reviewAndMaskPredateKey": key_artifact["phaseA"]["commit"]
        == phase_a["commit"],
        "unmarkedMatchesFirstDraw": _score_matches(
            unmarked_score, expected["unmarked"]
        ),
        "unmarkedNotDetected": unmarked_score.status == "not_detected",
        "wrongKeyDetectionRateAtMost2_5Percent": wrong_keys.sufficient_count > 0
        and wrong_keys.detected_count * 40 <= wrong_keys.sufficient_count,
        "wrongKeysAllSufficient": wrong_keys.insufficient_count == 0,
    }
    artifact = {
        **evidence,
        "artifactVersion": ARTIFACT_VERSION,
        "artifactBindings": {
            "keyArtifact": plan["keyArtifact"],
            "markedManifest": {
                "path": MARKED_MANIFEST_PATH,
                "sha256": hashlib.sha256(marked_manifest_bytes).hexdigest(),
            },
            "phaseAAllowlist": phase_a["allowlist"],
            "phaseAInventory": phase_a["inventory"],
            "phaseAReview": phase_a["review"],
            "plan": {
                "path": PLAN_PATH,
                "sha256": hashlib.sha256(plan_bytes).hexdigest(),
            },
        },
        "developmentalPredecessor": {
            **phase_a["developmentalControls"],
            "phaseACommit": phase_a["commit"],
        },
        "encoding": encoding,
        "prepaidGate": {
            "checks": checks,
            "marked": marked_score.to_dict(include_documents=True),
            "status": "passed" if all(checks.values()) else "failed",
            "unmarked": unmarked_score.to_dict(include_documents=True),
        },
        "prospectiveKeyProtocol": {
            "drawCount": 1,
            "keyGeneratedAfterPhaseACommit": phase_a["commit"],
            "outcomeKnownAtDraw": False,
            "redrawAllowed": False,
        },
        "providerCalls": 0,
        "providerExecution": plan["providerExecution"],
        "wrongKeyControlsOnMarked": {
            **wrong_keys.to_dict(include_scores=False),
            "scores": wrong_key_rows,
            "scoresSha256": hashlib.sha256(
                canonical_json_bytes(wrong_key_rows)
            ).hexdigest(),
        },
    }
    files = {
        PLAN_PATH: plan_bytes,
        KEY_PATH: canonical_json_bytes(key_artifact),
        MARKED_MANIFEST_PATH: marked_manifest_bytes,
        ARTIFACT_PATH: canonical_json_bytes(artifact),
    }
    for item in marked:
        files[f"{MARKED_DIRECTORY}/{item['documentId']}.md"] = item["text"].encode(
            "utf-8"
        )
    return FinalHoldoutV9Package(plan, key_artifact, artifact, files)


def _validate_key_artifact(plan, key_artifact):
    _validate_evidence(key_artifact, "v9 key artifact")
    if key_artifact.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unsupported v9 key schemaVersion")
    if key_artifact.get("keyVersion") != "final-holdout-public-key-v9":
        raise ValueError("unexpected v9 keyVersion")
    phase_a = plan["phaseA"]
    key_phase = key_artifact.get("phaseA")
    if not isinstance(key_phase, dict) or key_phase.get("commit") != phase_a["commit"]:
        raise ValueError("v9 key does not bind phase A")
    for field in ("inventory", "review", "allowlist"):
        if key_phase.get(field) != phase_a[field]:
            raise ValueError(f"v9 key phase-A {field} binding mismatch")
    if key_phase.get("densityBps") != plan["prepaidControls"]["densityBps"]:
        raise ValueError("v9 key density binding mismatch")
    draw = key_artifact.get("draw")
    if not isinstance(draw, dict):
        raise ValueError("v9 key requires draw metadata")
    if (
        draw.get("byteCount") != 32
        or draw.get("drawCount") != 1
        or draw.get("redrawAllowed") is not False
        or draw.get("outcomeKnownAtDraw") is not False
    ):
        raise ValueError("v9 key must be one unrepeated 32-byte draw")
    key_hex = key_artifact.get("keyHex")
    if not isinstance(key_hex, str) or len(bytes.fromhex(key_hex)) != 32:
        raise ValueError("v9 keyHex must encode exactly 32 bytes")


def _score_matches(score, expected):
    return (
        score.active_positions == expected["activePositions"]
        and score.hits == expected["hits"]
        and score.status == expected["status"]
    )


def _validate_evidence(value, label):
    for field in ("verifiedAt", "methodology"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ValueError(f"{label} requires {field}")
    if not isinstance(value.get("sources"), list) or not value["sources"]:
        raise ValueError(f"{label} requires sources")


def _evidence(plan):
    return {
        "methodology": plan["methodology"],
        "schemaVersion": SCHEMA_VERSION,
        "sources": plan["sources"],
        "verifiedAt": plan["verifiedAt"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=PLAN_PATH)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    package = build_final_holdout_v9_package(root / args.plan, root=root)
    if args.freeze:
        for relative_path, data in package.files.items():
            _atomic_write(root / relative_path, data)
    else:
        print(canonical_json_bytes(package.artifact).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
