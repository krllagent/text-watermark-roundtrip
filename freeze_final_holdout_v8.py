"""Freeze the reviewed encoder mask and CPU-only final-holdout controls.

The mask limits only which source occurrences the encoder may change.  The
public toy scorer is called unchanged on the full lexicon.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

from corpus_contract import canonical_json_bytes
from text_contract import analyze_text
from watermark_toy import (
    Document,
    inspect_positions,
    load_lexicon,
    run_wrong_key_controls,
    score_corpus,
)


SCHEMA_VERSION = 1
ARTIFACT_VERSION = "final-holdout-controls-v8"
PLAN_PATH = "fixtures/final-holdout-plan-v8.json"
DERIVED_DIRECTORY = "corpus/holdout-v6/reviewed-encoder-v8"
INVENTORY_PATH = f"{DERIVED_DIRECTORY}/context-inventory-v8.json"
REVIEW_PATH = f"{DERIVED_DIRECTORY}/context-review-v8.json"
ALLOWLIST_PATH = f"{DERIVED_DIRECTORY}/encoder-allowlist-v8.json"
MARKED_DIRECTORY = f"{DERIVED_DIRECTORY}/marked-1000"
MARKED_MANIFEST_PATH = f"{MARKED_DIRECTORY}/manifest-v8.json"
ARTIFACT_PATH = "results/final-holdout-controls-v8.json"


# Exhaustive manual review of all 411 source occurrences.  Entries omitted
# from these groups were explicitly approved.  This table was recorded before
# the real watermark key was consulted for the final encoding pass.
_REJECTION_GROUPS: Mapping[str, tuple[str, ...]] = {
    "grammar_or_part_of_speech": (
        "01:01",
        "01:12",
        "01:13",
        "01:15",
        "01:23",
        "01:30",
        "02:07",
        "02:08",
        "02:09",
        "02:10",
        "02:16",
        "03:05",
        "03:29",
        "04:04",
        "04:08",
        "04:15",
        "04:20",
        "05:08",
        "05:15",
        "06:05",
        "06:06",
        "06:07",
        "06:14",
        "07:15",
        "07:17",
        "08:04",
        "08:08",
        "08:14",
        "08:15",
        "08:19",
        "09:03",
        "09:05",
        "10:14",
        "11:17",
        "12:01",
        "12:09",
        "12:12",
        "13:19",
        "14:03",
        "15:05",
        "15:07",
        "15:10",
        "15:13",
        "15:19",
        "16:02",
        "16:07",
        "16:09",
        "16:15",
        "17:00",
        "17:03",
        "17:11",
        "18:03",
        "18:05",
        "18:14",
        "18:16",
        "19:06",
        "19:09",
        "19:10",
        "19:18",
        "19:20",
        "19:21",
        "20:02",
        "20:06",
        "20:12",
        "20:13",
    ),
    "meaning_or_reference": (
        "01:11",
        "01:19",
        "01:21",
        "02:01",
        "02:17",
        "02:19",
        "03:03",
        "03:06",
        "03:15",
        "03:21",
        "03:26",
        "03:30",
        "03:32",
        "03:33",
        "04:09",
        "04:21",
        "04:22",
        "05:05",
        "05:10",
        "05:19",
        "05:20",
        "07:04",
        "07:09",
        "08:05",
        "08:07",
        "08:17",
        "09:07",
        "09:10",
        "10:03",
        "10:10",
        "10:12",
        "11:05",
        "11:10",
        "12:02",
        "12:05",
        "12:06",
        "12:08",
        "13:05",
        "13:10",
        "13:17",
        "14:04",
        "14:13",
        "14:14",
        "15:03",
        "15:14",
        "16:03",
        "16:04",
        "16:12",
        "17:02",
        "17:08",
        "17:12",
        "19:17",
        "20:07",
    ),
    "named_label_integrity": ("18:00",),
    "register_or_collocation": (
        "01:06",
        "06:03",
        "06:12",
        "13:00",
        "14:02",
        "17:16",
        "19:00",
        "20:01",
    ),
    "technical_term_integrity": (
        "02:02",
        "02:03",
        "02:04",
        "03:08",
        "03:13",
        "13:06",
        "15:15",
        "19:11",
        "19:12",
    ),
}

_REASON_TEXT = {
    "grammar_or_part_of_speech": (
        "the case-preserving partner does not fit the token's grammatical role "
        "or leaves the sentence syntactically incomplete"
    ),
    "meaning_or_reference": (
        "the partner changes the intended proposition, causal relation, referent "
        "identity, or operational meaning"
    ),
    "named_label_integrity": (
        "the occurrence continues a named interface label and the partner would "
        "corrupt that label"
    ),
    "register_or_collocation": (
        "the partner is materially unnatural in this exact collocation or shifts "
        "the intended register"
    ),
    "technical_term_integrity": (
        "the partner loses the source's domain-specific technical meaning"
    ),
}


@dataclass(frozen=True)
class FinalHoldoutPackage:
    plan: dict[str, object]
    inventory: dict[str, object]
    review: dict[str, object]
    allowlist: dict[str, object]
    artifact: dict[str, object]
    files: dict[str, bytes]


def build_final_holdout_package(
    plan_path: str | Path,
    *,
    root: str | Path | None = None,
) -> FinalHoldoutPackage:
    root_path = (
        Path(root).resolve() if root is not None else Path(__file__).resolve().parent
    )
    selected_plan_path = Path(plan_path).resolve()
    plan_bytes = selected_plan_path.read_bytes()
    plan = _json_object(plan_bytes, "final holdout plan")
    _validate_plan(root_path, plan)

    source_plan = _load_bound_json(
        root_path, plan["sourceCorpus"]["plan"], "source plan"
    )
    source_manifest = _load_bound_json(
        root_path, plan["sourceCorpus"]["manifest"], "source manifest"
    )
    lexicon = load_lexicon(root_path / plan["lexicon"]["path"])
    marker_config = _load_bound_json(root_path, plan["markerConfig"], "marker config")
    documents = _load_source_documents(root_path, source_plan, source_manifest)

    inventory = _build_inventory(plan, documents, lexicon)
    review = _build_review(plan, inventory)
    allowlist = _build_allowlist(plan, inventory, review)
    validate_review_bound_allowlist(plan, inventory, review, allowlist)

    key = bytes.fromhex(marker_config["marker"]["keyHex"])
    density_bps = int(plan["prepaidControls"]["densityBps"])
    context_width = int(plan["prepaidControls"]["contextWidth"])
    min_active = int(plan["prepaidControls"]["minActivePositions"])
    marked, encoding = _encode_documents(
        documents,
        inventory,
        allowlist,
        key=key,
        density_bps=density_bps,
        context_width=context_width,
        lexicon=lexicon,
    )

    original_score = score_corpus(
        tuple(Document(item["documentId"], item["text"]) for item in documents),
        key=key,
        density_bps=density_bps,
        lexicon=lexicon,
        context_width=context_width,
        min_active_positions=min_active,
    )
    marked_score = score_corpus(
        tuple(Document(item["documentId"], item["text"]) for item in marked),
        key=key,
        density_bps=density_bps,
        lexicon=lexicon,
        context_width=context_width,
        min_active_positions=min_active,
    )
    wrong_keys = run_wrong_key_controls(
        tuple(Document(item["documentId"], item["text"]) for item in marked),
        density_bps=density_bps,
        lexicon=lexicon,
        count=int(plan["prepaidControls"]["wrongKeyCount"]),
        seed=bytes.fromhex(plan["prepaidControls"]["wrongKeySeedHex"]),
        context_width=context_width,
        min_active_positions=min_active,
    )

    evidence = _evidence(plan)
    marked_manifest = {
        **evidence,
        "artifactVersion": "final-holdout-marked-inputs-v8",
        "allowlistSha256": _sha_json(allowlist),
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
        "inventorySha256": _sha_json(inventory),
        "reviewSha256": _sha_json(review),
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
    checks = {
        "all411ContextsReviewed": review["reviewedOccurrences"] == 411,
        "allRejectedContextsSkipped": encoding["rejectedPhysicalChanges"] == 0,
        "approvedActiveFavored": encoding["approvedActiveFavored"] == 18,
        "markedActiveAtLeast20": marked_score.active_positions >= min_active,
        "markedDetected": marked_score.status == "detected",
        "markedExpected25Of34": marked_score.active_positions == 34
        and marked_score.hits == 25,
        "unmarkedNotDetected": original_score.status == "not_detected",
        "wrongKeyDetectionRateAtMost2_5Percent": (
            wrong_keys.sufficient_count > 0
            and wrong_keys.detected_count * 40 <= wrong_keys.sufficient_count
        ),
        "wrongKeysAllSufficient": wrong_keys.insufficient_count == 0,
    }
    artifact = {
        **evidence,
        "artifactVersion": ARTIFACT_VERSION,
        "artifactBindings": {
            "allowlist": {"path": ALLOWLIST_PATH, "sha256": _sha_json(allowlist)},
            "inventory": {"path": INVENTORY_PATH, "sha256": _sha_json(inventory)},
            "markedManifest": {
                "path": MARKED_MANIFEST_PATH,
                "sha256": hashlib.sha256(marked_manifest_bytes).hexdigest(),
            },
            "plan": {
                "path": PLAN_PATH,
                "sha256": hashlib.sha256(plan_bytes).hexdigest(),
            },
            "review": {"path": REVIEW_PATH, "sha256": _sha_json(review)},
        },
        "encoding": encoding,
        "prepaidGate": {
            "checks": checks,
            "marked": marked_score.to_dict(include_documents=True),
            "status": "passed" if all(checks.values()) else "failed",
            "unmarked": original_score.to_dict(include_documents=True),
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

    files: dict[str, bytes] = {
        PLAN_PATH: plan_bytes,
        INVENTORY_PATH: canonical_json_bytes(inventory),
        REVIEW_PATH: canonical_json_bytes(review),
        ALLOWLIST_PATH: canonical_json_bytes(allowlist),
        MARKED_MANIFEST_PATH: marked_manifest_bytes,
        ARTIFACT_PATH: canonical_json_bytes(artifact),
    }
    for item in marked:
        files[f"{MARKED_DIRECTORY}/{item['documentId']}.md"] = item["text"].encode(
            "utf-8"
        )
    return FinalHoldoutPackage(plan, inventory, review, allowlist, artifact, files)


def validate_review_bound_allowlist(
    plan: Mapping[str, object],
    inventory: Mapping[str, object],
    review: Mapping[str, object],
    allowlist: Mapping[str, object],
) -> None:
    if allowlist.get("inventorySha256") != _sha_json(inventory):
        raise ValueError("allowlist inventorySha256 mismatch")
    if allowlist.get("reviewSha256") != _sha_json(review):
        raise ValueError("allowlist reviewSha256 mismatch")
    inventory_docs = _documents_by_id(inventory, "inventory")
    review_docs = _documents_by_id(review, "review")
    allowlist_docs = _documents_by_id(allowlist, "allowlist")
    if set(inventory_docs) != set(review_docs) or set(inventory_docs) != set(
        allowlist_docs
    ):
        raise ValueError("allowlist document set mismatch")
    total_approved = 0
    total_rejected = 0
    for document_id, source in inventory_docs.items():
        decisions = review_docs[document_id].get("decisions")
        if not isinstance(decisions, list) or len(decisions) != len(
            source["occurrences"]
        ):
            raise ValueError(f"review decisions incomplete for {document_id}")
        by_fingerprint = {
            occurrence["occurrenceFingerprint"]: occurrence
            for occurrence in source["occurrences"]
        }
        if len(by_fingerprint) != len(source["occurrences"]):
            raise ValueError(f"duplicate inventory fingerprint for {document_id}")
        approved: list[str] = []
        rejected: list[str] = []
        for occurrence, decision in zip(source["occurrences"], decisions, strict=True):
            fingerprint = occurrence["occurrenceFingerprint"]
            if decision.get("occurrenceFingerprint") != fingerprint:
                raise ValueError(f"review fingerprint mismatch for {document_id}")
            if decision.get("decision") == "approved":
                approved.append(fingerprint)
            elif decision.get("decision") == "rejected":
                rejected.append(fingerprint)
            else:
                raise ValueError(f"invalid review decision for {document_id}")
        mask = allowlist_docs[document_id]
        if mask.get("documentSha256") != source.get("documentSha256"):
            raise ValueError(f"allowlist document SHA mismatch for {document_id}")
        if mask.get("approvedOccurrenceFingerprints") != approved:
            raise ValueError(
                f"allowlist approved fingerprint mismatch for {document_id}"
            )
        if mask.get("rejectedOccurrenceFingerprints") != rejected:
            raise ValueError(
                f"allowlist rejected fingerprint mismatch for {document_id}"
            )
        total_approved += len(approved)
        total_rejected += len(rejected)
    if allowlist.get("approvedOccurrences") != total_approved:
        raise ValueError("allowlist approved occurrence count mismatch")
    if allowlist.get("rejectedOccurrences") != total_rejected:
        raise ValueError("allowlist rejected occurrence count mismatch")
    expected = plan["contextReview"]
    if total_approved != expected["expectedApprovedOccurrences"]:
        raise ValueError("allowlist approved occurrence contract mismatch")
    if total_rejected != expected["expectedRejectedOccurrences"]:
        raise ValueError("allowlist rejected occurrence contract mismatch")


def _build_inventory(plan, documents, lexicon):
    evidence = _evidence(plan)
    inventory_docs = []
    total = 0
    for item in documents:
        occurrences = []
        for token in analyze_text(item["text"]).context_tokens:
            if token.protected or token.text is None or not _supported_case(token.text):
                continue
            pair = lexicon.token_to_pair.get(token.normalized)
            if pair is None:
                continue
            partner = (
                pair.variants[1]
                if pair.variants[0] == token.normalized
                else pair.variants[0]
            )
            partner = _apply_case(partner, token.text)
            paragraph_start = item["text"].rfind("\n\n", 0, token.start) + 2
            paragraph_end = item["text"].find("\n\n", token.end)
            if paragraph_end < 0:
                paragraph_end = len(item["text"])
            binding = {
                "classId": pair.class_id,
                "documentId": item["documentId"],
                "documentSha256": item["sha256"],
                "end": token.end,
                "ordinal": len(occurrences),
                "partner": partner,
                "start": token.start,
                "token": token.text,
            }
            occurrences.append(
                {
                    **binding,
                    "candidateParagraph": (
                        item["text"][paragraph_start : token.start]
                        + partner
                        + item["text"][token.end : paragraph_end]
                    ),
                    "occurrenceFingerprint": _sha_json(binding),
                    "sourceParagraph": item["text"][paragraph_start:paragraph_end],
                }
            )
        total += len(occurrences)
        inventory_docs.append(
            {
                "documentId": item["documentId"],
                "documentSha256": item["sha256"],
                "eligibleOccurrences": len(occurrences),
                "occurrences": occurrences,
            }
        )
    if total != plan["sourceCorpus"]["expectedScorerEligibleOccurrences"]:
        raise ValueError("unexpected scorer-eligible occurrence count")
    return {
        **evidence,
        "artifactVersion": "final-holdout-context-inventory-v8",
        "documents": inventory_docs,
        "eligibleOccurrences": total,
        "lexiconSha256": plan["lexicon"]["sha256"],
        "sourceManifestSha256": plan["sourceCorpus"]["manifest"]["sha256"],
    }


def _build_review(plan, inventory):
    rejection_map = _rejection_map()
    evidence = _evidence(plan)
    documents = []
    approved_total = 0
    rejected_total = 0
    for source in inventory["documents"]:
        decisions = []
        rejected_count = 0
        for occurrence in source["occurrences"]:
            reference = (
                f"{int(source['documentId'][-2:]):02d}:{occurrence['ordinal']:02d}"
            )
            criterion = rejection_map.get(reference)
            decision = {
                "decision": "rejected" if criterion else "approved",
                "occurrenceFingerprint": occurrence["occurrenceFingerprint"],
                "ordinal": occurrence["ordinal"],
            }
            if criterion:
                decision["criterion"] = criterion
                decision["reason"] = (
                    f"Rejected {occurrence['token']!r} -> {occurrence['partner']!r}: "
                    f"{_REASON_TEXT[criterion]}."
                )
                rejected_count += 1
            decisions.append(decision)
        approved_count = len(decisions) - rejected_count
        approved_total += approved_count
        rejected_total += rejected_count
        documents.append(
            {
                "approvedOccurrences": approved_count,
                "decisions": decisions,
                "documentId": source["documentId"],
                "documentSha256": source["documentSha256"],
                "rejectedOccurrences": rejected_count,
                "reviewedOccurrences": len(decisions),
            }
        )
    expected = plan["contextReview"]
    if (
        approved_total != expected["expectedApprovedOccurrences"]
        or rejected_total != expected["expectedRejectedOccurrences"]
    ):
        raise ValueError("manual review table does not match frozen counts")
    return {
        **evidence,
        "approvedOccurrences": approved_total,
        "artifactVersion": "final-holdout-context-review-v8",
        "documents": documents,
        "inventorySha256": _sha_json(inventory),
        "rejectedOccurrences": rejected_total,
        "reviewMethod": plan["contextReview"]["reviewerExposure"],
        "reviewedOccurrences": approved_total + rejected_total,
    }


def _build_allowlist(plan, inventory, review):
    evidence = _evidence(plan)
    review_docs = _documents_by_id(review, "review")
    documents = []
    for source in inventory["documents"]:
        decisions = review_docs[source["documentId"]]["decisions"]
        approved = [
            occurrence["occurrenceFingerprint"]
            for occurrence, decision in zip(
                source["occurrences"], decisions, strict=True
            )
            if decision["decision"] == "approved"
        ]
        rejected = [
            occurrence["occurrenceFingerprint"]
            for occurrence, decision in zip(
                source["occurrences"], decisions, strict=True
            )
            if decision["decision"] == "rejected"
        ]
        documents.append(
            {
                "approvedOccurrenceFingerprints": approved,
                "documentId": source["documentId"],
                "documentSha256": source["documentSha256"],
                "rejectedOccurrenceFingerprints": rejected,
            }
        )
    return {
        **evidence,
        "approvedOccurrences": review["approvedOccurrences"],
        "artifactVersion": "final-holdout-encoder-allowlist-v8",
        "detectorUsesAllowlist": False,
        "documents": documents,
        "inventorySha256": _sha_json(inventory),
        "maskScope": "source_encoder_only",
        "rejectedOccurrences": review["rejectedOccurrences"],
        "reviewSha256": _sha_json(review),
    }


def _encode_documents(
    documents, inventory, allowlist, *, key, density_bps, context_width, lexicon
):
    inventory_docs = _documents_by_id(inventory, "inventory")
    mask_docs = _documents_by_id(allowlist, "allowlist")
    marked = []
    rejected_rows = []
    approved_active = 0
    approved_favored = 0
    rejected_active = 0
    physical_changes = 0
    for item in documents:
        document_id = item["documentId"]
        occurrences = inventory_docs[document_id]["occurrences"]
        approved = set(mask_docs[document_id]["approvedOccurrenceFingerprints"])
        positions = inspect_positions(
            item["text"],
            key=key,
            document_id=document_id,
            density_bps=density_bps,
            lexicon=lexicon,
            context_width=context_width,
        )
        if len(positions) != len(occurrences):
            raise ValueError(f"encoder inventory mismatch for {document_id}")
        chunks = []
        cursor = 0
        changed_here = 0
        for occurrence, position in zip(occurrences, positions, strict=True):
            _validate_position_binding(occurrence, position)
            fingerprint = occurrence["occurrenceFingerprint"]
            is_approved = fingerprint in approved
            if position.active and is_approved:
                approved_active += 1
                approved_favored += 1
                replacement = _apply_case(position.favored_variant, position.token)
                if replacement != position.token:
                    chunks.extend((item["text"][cursor : position.start], replacement))
                    cursor = position.end
                    changed_here += 1
                    physical_changes += 1
            elif position.active:
                rejected_active += 1
            if not is_approved:
                rejected_rows.append(
                    {
                        "active": position.active,
                        "changed": False,
                        "documentId": document_id,
                        "occurrenceFingerprint": fingerprint,
                        "ordinal": occurrence["ordinal"],
                    }
                )
        chunks.append(item["text"][cursor:])
        marked.append(
            {
                "changedPositions": changed_here,
                "documentId": document_id,
                "sourceSha256": item["sha256"],
                "text": "".join(chunks),
            }
        )
    return marked, {
        "approvedActiveFavored": approved_favored,
        "approvedActiveOpportunities": approved_active,
        "physicalChanges": physical_changes,
        "rejectedActiveSkipped": rejected_active,
        "rejectedPhysicalChanges": 0,
        "rejectedPositions": rejected_rows,
    }


def _validate_plan(root, plan):
    if plan.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unsupported final holdout plan schemaVersion")
    _validate_evidence(plan, "final holdout plan")
    if plan.get("experimentVersion") != "text-watermark-final-holdout-v8":
        raise ValueError("unexpected final holdout experimentVersion")
    if (
        plan.get("providerExecution", {}).get("status")
        != "blocked_pending_committed_v7_winner_binding"
    ):
        raise ValueError("provider execution must remain blocked")
    for field in (
        plan["sourceCorpus"]["plan"],
        plan["sourceCorpus"]["manifest"],
        plan["lexicon"],
        plan["markerConfig"],
        plan["detectorImplementation"],
    ):
        path = _safe_path(root, field["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != field["sha256"]:
            raise ValueError(f"bound input SHA mismatch: {field['path']}")


def _load_source_documents(root, source_plan, source_manifest):
    manifest_docs = {item["documentId"]: item for item in source_manifest["documents"]}
    documents = []
    for planned in source_plan["documents"]:
        document_id = planned["documentId"]
        manifest = manifest_docs.get(document_id)
        if manifest is None or manifest["path"] != planned["path"]:
            raise ValueError(f"source manifest mismatch for {document_id}")
        path = _safe_path(root, planned["path"])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
            raise ValueError(f"source document SHA mismatch for {document_id}")
        documents.append(
            {
                "documentId": document_id,
                "path": planned["path"],
                "sha256": manifest["sha256"],
                "text": raw.decode("utf-8"),
            }
        )
    return documents


def _load_bound_json(root, binding, label):
    path = _safe_path(root, binding["path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError(f"{label} SHA mismatch")
    return _json_object(raw, label)


def _safe_path(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path must be nonempty")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("artifact path escapes repository root")
    return path


def _validate_position_binding(occurrence, position):
    for field, value in (
        ("start", position.start),
        ("end", position.end),
        ("token", position.token),
        ("classId", position.class_id),
    ):
        if occurrence[field] != value:
            raise ValueError(f"encoder occurrence binding mismatch: {field}")


def _documents_by_id(value, label):
    documents = value.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"{label} requires documents")
    result = {}
    for document in documents:
        document_id = document.get("documentId")
        if not isinstance(document_id, str) or document_id in result:
            raise ValueError(f"{label} has invalid documentId")
        result[document_id] = document
    return result


def _rejection_map():
    output = {}
    for criterion, references in _REJECTION_GROUPS.items():
        for reference in references:
            if reference in output:
                raise ValueError(f"duplicate rejection reference: {reference}")
            output[reference] = criterion
    if len(output) != 136:
        raise ValueError(f"expected 136 rejection references, got {len(output)}")
    return output


def _evidence(plan):
    return {
        "methodology": plan["methodology"],
        "schemaVersion": SCHEMA_VERSION,
        "sources": plan["sources"],
        "verifiedAt": plan["verifiedAt"],
    }


def _validate_evidence(value, label):
    for field in ("verifiedAt", "methodology"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ValueError(f"{label} requires {field}")
    if not isinstance(value.get("sources"), list) or not value["sources"]:
        raise ValueError(f"{label} requires sources")


def _json_object(raw, label):
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sha_json(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _fraction(value: Fraction | None):
    if value is None:
        return None
    return {"denominator": value.denominator, "numerator": value.numerator}


def _supported_case(value):
    return value.islower() or value.isupper() or value.istitle()


def _apply_case(value, source):
    if source.isupper():
        return value.upper()
    if source.istitle():
        return value.title()
    return value


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(data)
        temporary.flush()
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=PLAN_PATH)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    package = build_final_holdout_package(root / args.plan, root=root)
    if args.freeze:
        for relative_path, data in package.files.items():
            _atomic_write(root / relative_path, data)
    else:
        print(canonical_json_bytes(package.artifact).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
