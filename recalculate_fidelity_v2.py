"""Recalculate v1 fidelity counts after fixing protected-value diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from corpus_contract import canonical_json_bytes
from run_semantic_audit import protected_token_failure
from text_contract import find_protected_spans


EXPECTED_METHODS = ("synonyms", "roundtrip-de", "roundtrip-zh", "paraphrase")
EXPECTED_TRANSFORM_SHA256 = "7cbebefa7e871032e9ac80125cd95d0a9cb2a733c99324f150640ab26744e723"
EXPECTED_AUDIT_SHA256 = "5143583fc30e790b6211432d925e3fd806c08931db87f22bccbaeebab5f42ea6"
_DUPLICATED_TERMINAL_PUNCTUATION = re.compile(r"([,.;:!?])\1")


def build_fidelity_recalculation(
    transform: Mapping[str, object],
    audit: Mapping[str, object],
) -> dict[str, object]:
    mappings = audit.get("opaqueMapping")
    reviews = audit.get("reviews")
    if not isinstance(mappings, list) or not isinstance(reviews, list):
        raise ValueError("audit requires opaqueMapping and reviews")
    review_by_pair = {
        _text(review.get("pairId"), "review pairId"): review
        for review in (_mapping(item, "review") for item in reviews)
    }
    semantic_by_document: dict[tuple[str, str], bool] = {}
    for raw_mapping in mappings:
        mapping = _mapping(raw_mapping, "opaque mapping")
        pair_id = _text(mapping.get("pairId"), "mapping pairId")
        review = _mapping(review_by_pair.get(pair_id), "mapped review")
        semantic_by_document[
            (
                _text(mapping.get("methodId"), "mapping methodId"),
                _text(mapping.get("documentId"), "mapping documentId"),
            )
        ] = review.get("semanticFidelityFailure") is True

    raw_methods = transform.get("methods")
    if not isinstance(raw_methods, list):
        raise ValueError("transform artifact methods must be a list")
    by_method = {
        _text(method.get("methodId"), "methodId"): method
        for method in (_mapping(item, "method") for item in raw_methods)
    }
    results: list[dict[str, object]] = []
    for method_id in EXPECTED_METHODS:
        method = _mapping(by_method.get(method_id), f"method {method_id}")
        documents = method.get("documents")
        if not isinstance(documents, list) or len(documents) != 20:
            raise ValueError(f"{method_id} must contain 20 documents")
        rows: list[dict[str, object]] = []
        for raw_document in documents:
            document = _mapping(raw_document, "method document")
            document_id = _text(document.get("documentId"), "documentId")
            source = _text(document.get("markedInputText"), "markedInputText")
            candidate = _text(document.get("outputText"), "outputText")
            exact_check = protected_token_failure(source, candidate)
            protected_values = [
                source[span.start : span.end] for span in find_protected_spans(source)
            ]
            duplicated_values = [
                value
                for value in protected_values
                if any(
                    value + punctuation * 2 in candidate
                    for punctuation in ",.;:!?"
                )
            ]
            duplicate_surface = bool(
                duplicated_values
                and _DUPLICATED_TERMINAL_PUNCTUATION.search(candidate)
            )
            pipeline_defect = bool(exact_check["failed"]) or duplicate_surface
            semantic_failure = semantic_by_document[(method_id, document_id)]
            rows.append(
                {
                    "documentId": document_id,
                    "semanticFailure": semantic_failure,
                    "pipelineDefect": pipeline_defect,
                    "fidelityFailure": semantic_failure or pipeline_defect,
                    "exactProtectedValueMissing": bool(exact_check["failed"]),
                    "duplicatedTerminalPunctuation": duplicate_surface,
                }
            )
        results.append(
            {
                "methodId": method_id,
                "semanticFailureCount": sum(row["semanticFailure"] for row in rows),
                "pipelineDefectCount": sum(row["pipelineDefect"] for row in rows),
                "fidelityFailureCount": sum(row["fidelityFailure"] for row in rows),
                "documents": rows,
            }
        )

    return {
        "schemaVersion": 1,
        "verifiedAt": "2026-08-17",
        "methodology": (
            "Keep the frozen Gemini semantic judgments from semantic-audit-v1. "
            "Recompute exact protected-value loss with a longest ordered match so one "
            "missing value cannot cascade into later false misses. Separately count the "
            "duplicated punctuation created when the old money placeholder swallowed a "
            "source comma. A document fails overall when either the semantic judgment or "
            "the corrected pipeline check fails."
        ),
        "sources": [
            {
                "title": "Experiment repository",
                "url": "https://github.com/krllagent/text-watermark-roundtrip",
            }
        ],
        "sourceBindings": {
            "transformSha256": EXPECTED_TRANSFORM_SHA256,
            "semanticAuditSha256": EXPECTED_AUDIT_SHA256,
        },
        "methods": results,
    }


def load_bound_sources(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    transform_path = root / "results" / "experiment-raw-v1.json"
    audit_path = root / "results" / "semantic-audit-v1.json"
    for path, expected in (
        (transform_path, EXPECTED_TRANSFORM_SHA256),
        (audit_path, EXPECTED_AUDIT_SHA256),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"source SHA-256 mismatch: {path.name}")
    return _json_object(transform_path), _json_object(audit_path)


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(root / "results" / "fidelity-recalculation-v2.json"),
    )
    args = parser.parse_args(argv)
    transform, audit = load_bound_sources(root)
    artifact = build_fidelity_recalculation(transform, audit)
    Path(args.output).write_bytes(canonical_json_bytes(artifact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
