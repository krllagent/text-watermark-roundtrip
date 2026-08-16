"""Run the frozen CPU-only detector controls on the reviewed article corpus.

The runner never calls a model provider.  It reads the hash-bound experiment
config and independently reviewed corpus, encodes all three frozen densities,
and emits the 1000-bps marked corpus plus a canonical evidence artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from corpus_contract import canonical_json_bytes, validate_context_reviews
from text_contract import TEXT_CONTRACT_VERSION, TOKENIZER_VERSION
from watermark_toy import (
    SCHEME_VERSION,
    Document,
    EncodeResult,
    encode_text,
    run_wrong_key_controls,
    score_corpus,
)


ARTIFACT_SCHEMA_VERSION = 1
MARKED_MANIFEST_SCHEMA_VERSION = 1
ARTIFACT_VERSION = "corpus-controls-v1"
ARTIFACT_PATH = "results/corpus-controls-v1.json"
MARKED_DIRECTORY = "corpus/marked-1000"
MARKED_MANIFEST_PATH = f"{MARKED_DIRECTORY}/manifest-v1.json"
FROZEN_DENSITIES_BPS = (500, 1_000, 2_000)
FROZEN_MAIN_DENSITY_BPS = 1_000
FROZEN_WRONG_KEY_COUNT = 1_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ControlSpec:
    densities_bps: tuple[int, ...]
    main_density_bps: int
    wrong_key_count: int
    wrong_key_seed: bytes


@dataclass(frozen=True)
class ControlOutputs:
    artifact: dict[str, object]
    files: dict[str, bytes]


def control_spec_from_config(
    config: Any,
    *,
    require_production_wrong_key_count: bool = True,
) -> ControlSpec:
    """Read the marker controls from the already validated experiment config."""
    raw = getattr(config, "raw", None)
    if not isinstance(raw, dict):
        raise ValueError("experiment config raw value must be an object")
    marker = raw.get("marker")
    if not isinstance(marker, dict):
        raise ValueError("experiment config requires marker")

    densities = marker.get("densitiesBps")
    if densities != list(FROZEN_DENSITIES_BPS):
        raise ValueError("marker.densitiesBps must remain [500, 1000, 2000]")
    main_density = marker.get("mainDensityBps")
    if main_density != FROZEN_MAIN_DENSITY_BPS:
        raise ValueError("marker.mainDensityBps must remain 1000")
    if getattr(config, "density_bps", None) != main_density:
        raise ValueError("loaded config density does not match marker.mainDensityBps")

    wrong_key_count = marker.get("wrongKeyCount")
    if (
        not isinstance(wrong_key_count, int)
        or isinstance(wrong_key_count, bool)
        or wrong_key_count <= 0
    ):
        raise ValueError("marker.wrongKeyCount must be a positive integer")
    if (
        require_production_wrong_key_count
        and wrong_key_count != FROZEN_WRONG_KEY_COUNT
    ):
        raise ValueError("marker.wrongKeyCount must be exactly 1000")

    seed_hex = marker.get("wrongKeySeedHex")
    if not isinstance(seed_hex, str) or not seed_hex:
        raise ValueError("marker.wrongKeySeedHex must be nonempty hexadecimal")
    try:
        seed = bytes.fromhex(seed_hex)
    except ValueError as error:
        raise ValueError("marker.wrongKeySeedHex must be hexadecimal") from error
    if len(seed) < 16 or seed.hex() != seed_hex.lower():
        raise ValueError(
            "marker.wrongKeySeedHex must canonically encode at least 16 bytes"
        )

    return ControlSpec(
        densities_bps=FROZEN_DENSITIES_BPS,
        main_density_bps=FROZEN_MAIN_DENSITY_BPS,
        wrong_key_count=wrong_key_count,
        wrong_key_seed=seed,
    )


def load_control_inputs(
    config_path: str | Path,
    *,
    root: str | Path | None = None,
) -> tuple[Any, Any]:
    """Load every frozen input and explicitly re-run context-review validation."""
    # Keep this import local so the pure control builder remains independently
    # testable with tiny in-memory fixtures.
    from run_experiment import load_experiment_config, load_reviewed_corpus

    config = load_experiment_config(config_path, root=root)
    corpus = load_reviewed_corpus(config)

    inventory = _load_bound_json_object(
        config.inventory_path,
        "context inventory",
        config.inventory_expected_sha256,
    )
    review_bindings = getattr(config, "review_bindings", ())
    if not review_bindings:
        raise ValueError("corpus controls require frozen context review files")
    reviews = [
        _load_bound_json_object(
            review_path,
            f"review {review_path.name}",
            expected_sha256,
        )
        for review_path, expected_sha256 in review_bindings
    ]
    approval = validate_context_reviews(inventory=inventory, reviews=reviews)
    if canonical_json_bytes(approval) != canonical_json_bytes(corpus.review_approval):
        raise ValueError("context review approval differs from reviewed corpus loader")
    _validate_reviewed_inputs(config, corpus)
    return config, corpus


def build_corpus_controls(
    config: Any,
    corpus: Any,
    *,
    spec: ControlSpec | None = None,
) -> ControlOutputs:
    """Build deterministic control results and every canonical output byte."""
    _validate_reviewed_inputs(config, corpus)
    selected_spec = spec if spec is not None else control_spec_from_config(config)
    _validate_control_spec(config, selected_spec)

    source_items = tuple(corpus.documents)
    original_documents = tuple(
        Document(document_id=item.document_id, text=item.text) for item in source_items
    )
    _validate_source_items(source_items)
    original_corpus_sha256 = _corpus_hash_from_source_items(source_items)
    key_sha256 = hashlib.sha256(config.key).hexdigest()
    review_approval_sha256 = _sha256_json(corpus.review_approval)

    results: list[dict[str, object]] = []
    acceptance_by_density: list[dict[str, object]] = []
    main_encodings: tuple[EncodeResult, ...] | None = None
    main_marked_documents: tuple[Document, ...] | None = None
    main_marking: dict[str, object] | None = None
    main_marked_corpus_sha256: str | None = None

    for density_bps in selected_spec.densities_bps:
        encodings = tuple(
            encode_text(
                document.text,
                key=config.key,
                document_id=document.document_id,
                density_bps=density_bps,
                lexicon=corpus.lexicon,
                context_width=config.context_width,
            )
            for document in original_documents
        )
        _validate_encoding_bindings(source_items, encodings)
        marked_documents = tuple(
            Document(document_id=encoding.document_id, text=encoding.text)
            for encoding in encodings
        )
        marking = _marking_summary(encodings)
        marked_corpus_sha256 = _corpus_hash_from_documents(marked_documents)

        unmarked_score = score_corpus(
            original_documents,
            key=config.key,
            density_bps=density_bps,
            lexicon=corpus.lexicon,
            context_width=config.context_width,
            min_active_positions=config.min_active_positions,
        )
        marked_score = score_corpus(
            marked_documents,
            key=config.key,
            density_bps=density_bps,
            lexicon=corpus.lexicon,
            context_width=config.context_width,
            min_active_positions=config.min_active_positions,
        )
        wrong_keys = run_wrong_key_controls(
            marked_documents,
            density_bps=density_bps,
            lexicon=corpus.lexicon,
            count=selected_spec.wrong_key_count,
            seed=selected_spec.wrong_key_seed,
            context_width=config.context_width,
            min_active_positions=config.min_active_positions,
        )
        wrong_key_result = _wrong_key_result(wrong_keys)

        checks = {
            "markedActiveMatchesEncoding": (
                marked_score.active_positions == marking["activePositions"]
            ),
            "markedDetected": marked_score.status == "detected",
            "markedHitsEqualActivePositions": (
                marked_score.hits == marked_score.active_positions
            ),
            "unmarkedNotDetected": unmarked_score.status == "not_detected",
            "wrongKeyDetectionRateAtMost2_5Percent": (
                wrong_keys.sufficient_count > 0
                and wrong_keys.detected_count * 40 <= wrong_keys.sufficient_count
            ),
            "wrongKeysAllSufficient": wrong_keys.insufficient_count == 0,
        }
        density_passed = all(checks.values())
        acceptance_by_density.append(
            {
                "checks": checks,
                "densityBps": density_bps,
                "passed": density_passed,
            }
        )
        results.append(
            {
                "checks": checks,
                "densityBps": density_bps,
                "markedCorpusSha256": marked_corpus_sha256,
                "marking": marking,
                "trueKey": {
                    "marked": _public_score(marked_score, include_documents=True),
                    "unmarked": _public_score(unmarked_score, include_documents=True),
                },
                "wrongKeysOnMarked": wrong_key_result,
            }
        )

        if density_bps == selected_spec.main_density_bps:
            main_encodings = encodings
            main_marked_documents = marked_documents
            main_marking = marking
            main_marked_corpus_sha256 = marked_corpus_sha256

    if (
        main_encodings is None
        or main_marked_documents is None
        or main_marking is None
        or main_marked_corpus_sha256 is None
    ):
        raise AssertionError("frozen main density was not generated")

    marked_files = {
        _marked_document_path(document.document_id): document.text.encode("utf-8")
        for document in main_marked_documents
    }
    marked_manifest = _build_marked_manifest(
        config=config,
        corpus=corpus,
        source_items=source_items,
        encodings=main_encodings,
        marked_files=marked_files,
        original_corpus_sha256=original_corpus_sha256,
        marked_corpus_sha256=main_marked_corpus_sha256,
        review_approval_sha256=review_approval_sha256,
        key_sha256=key_sha256,
    )
    marked_manifest_bytes = canonical_json_bytes(marked_manifest)
    marked_manifest_sha256 = hashlib.sha256(marked_manifest_bytes).hexdigest()

    acceptance_passed = all(
        bool(item["passed"]) for item in acceptance_by_density
    )
    artifact: dict[str, object] = {
        "acceptance": {
            "densities": acceptance_by_density,
            "passed": acceptance_passed,
        },
        "artifactVersion": ARTIFACT_VERSION,
        "configSha256": config.sha256,
        "contextInventorySha256": corpus.inventory_sha256,
        "controlParameters": {
            "contextWidth": config.context_width,
            "densitiesBps": list(selected_spec.densities_bps),
            "densityDenominator": "eligible synonym positions",
            "mainDensityBps": selected_spec.main_density_bps,
            "minActivePositions": config.min_active_positions,
            "wrongKeyCount": selected_spec.wrong_key_count,
        },
        "corpusManifestSha256": corpus.manifest_sha256,
        "experimentVersion": config.experiment_version,
        "keySha256": key_sha256,
        "lexiconFileSha256": corpus.lexicon_file_sha256,
        "lexiconSha256": corpus.lexicon.sha256,
        "mainMarkedCorpus": {
            "densityBps": selected_spec.main_density_bps,
            "directory": MARKED_DIRECTORY,
            "manifestPath": MARKED_MANIFEST_PATH,
            "manifestSha256": marked_manifest_sha256,
            "markedCorpusSha256": main_marked_corpus_sha256,
            "marking": {
                key: value for key, value in main_marking.items() if key != "documents"
            },
        },
        "methodology": (
            "CPU-only deterministic controls over the exact hash-bound and independently "
            "reviewed article corpus. Density is measured over eligible synonym positions; "
            "coverage is also reported over all words. Each density scores the unmarked "
            "corpus and its true-key encoding, then evaluates independently derived wrong "
            "keys on that marked corpus. Wrong keys are null controls, not bootstrap "
            "samples. Only the 1000-bps marked texts are retained for transformations."
        ),
        "originalCorpusSha256": original_corpus_sha256,
        "results": results,
        "reviewApproval": dict(corpus.review_approval),
        "reviewApprovalSha256": review_approval_sha256,
        "reviewSha256s": list(corpus.review_sha256s),
        "schemaVersion": ARTIFACT_SCHEMA_VERSION,
        "schemeVersion": SCHEME_VERSION,
        "sources": [dict(source) for source in config.sources],
        "textContractVersion": TEXT_CONTRACT_VERSION,
        "tokenizerVersion": TOKENIZER_VERSION,
        "verifiedAt": config.verified_at,
        "wrongKeySeedSha256": hashlib.sha256(
            selected_spec.wrong_key_seed
        ).hexdigest(),
    }
    artifact_bytes = canonical_json_bytes(artifact)
    files = dict(marked_files)
    files[MARKED_MANIFEST_PATH] = marked_manifest_bytes
    files[ARTIFACT_PATH] = artifact_bytes
    return ControlOutputs(artifact=artifact, files=files)


def check_control_outputs(
    root: str | Path,
    outputs: ControlOutputs,
) -> dict[str, object]:
    """Compare every generated output against disk without writing anything."""
    root_path = Path(root).resolve()
    matches: dict[str, bool] = {}
    for relative_path, expected_bytes in sorted(outputs.files.items()):
        target = _safe_output_path(root_path, relative_path)
        matches[relative_path] = target.exists() and target.read_bytes() == expected_bytes
    expected_marked_paths = {
        relative_path
        for relative_path in outputs.files
        if relative_path.startswith(f"{MARKED_DIRECTORY}/")
    }
    marked_root = _safe_output_path(root_path, MARKED_DIRECTORY)
    actual_marked_paths = (
        {
            path.relative_to(root_path).as_posix()
            for path in marked_root.rglob("*")
            if path.is_file()
        }
        if marked_root.exists()
        else set()
    )
    unexpected_files = sorted(actual_marked_paths - expected_marked_paths)
    return {
        "files": matches,
        "passed": all(matches.values()) and not unexpected_files,
        "unexpectedFiles": unexpected_files,
    }


def write_control_outputs(root: str | Path, outputs: ControlOutputs) -> None:
    """Write all generated files, with the evidence artifact written last."""
    root_path = Path(root).resolve()
    ordered_paths = sorted(
        outputs.files,
        key=lambda relative_path: (relative_path == ARTIFACT_PATH, relative_path),
    )
    for relative_path in ordered_paths:
        target = _safe_output_path(root_path, relative_path)
        expected_bytes = outputs.files[relative_path]
        if target.exists() and target.read_bytes() == expected_bytes:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(expected_bytes)


def _validate_reviewed_inputs(config: Any, corpus: Any) -> None:
    if getattr(getattr(corpus, "config", None), "sha256", None) != getattr(
        config, "sha256", None
    ):
        raise ValueError("reviewed corpus was loaded under a different config")
    documents = tuple(getattr(corpus, "documents", ()))
    if not documents:
        raise ValueError("reviewed corpus must contain documents")

    review_sha256s = tuple(getattr(corpus, "review_sha256s", ()))
    if not review_sha256s:
        raise ValueError("corpus controls require at least one frozen review")
    for review_sha256 in review_sha256s:
        _require_sha256(review_sha256, "review sha256")
    review_bindings = getattr(config, "review_bindings", None)
    if review_bindings is not None:
        expected_review_sha256s = tuple(binding[1] for binding in review_bindings)
        if review_sha256s != expected_review_sha256s:
            raise ValueError("review hashes differ from frozen config bindings")

    approval = getattr(corpus, "review_approval", None)
    if not isinstance(approval, dict):
        raise ValueError("reviewed corpus requires validate_context_reviews approval")
    if approval.get("schemaVersion") != 1:
        raise ValueError("review approval schemaVersion must be 1")
    if approval.get("approvedDocumentCount") != len(documents):
        raise ValueError("review approvedDocumentCount must cover every document")
    if approval.get("inventorySha256") != getattr(corpus, "inventory_sha256", None):
        raise ValueError("review approval inventorySha256 mismatch")
    lexicon = getattr(corpus, "lexicon", None)
    if approval.get("lexiconSha256") != getattr(lexicon, "sha256", None):
        raise ValueError("review approval lexiconSha256 mismatch")
    manifest = getattr(corpus, "manifest", None)
    if not isinstance(manifest, dict):
        raise ValueError("reviewed corpus manifest must be an object")
    if approval.get("corpusVersion") != manifest.get("corpusVersion"):
        raise ValueError("review approval corpusVersion mismatch")

    for value, label in (
        (getattr(config, "sha256", None), "config sha256"),
        (getattr(corpus, "manifest_sha256", None), "manifest sha256"),
        (getattr(corpus, "inventory_sha256", None), "inventory sha256"),
        (getattr(corpus, "lexicon_file_sha256", None), "lexicon file sha256"),
    ):
        _require_sha256(value, label)


def _validate_control_spec(config: Any, spec: ControlSpec) -> None:
    if spec.densities_bps != FROZEN_DENSITIES_BPS:
        raise ValueError("control densities must remain 500, 1000, and 2000 bps")
    if spec.main_density_bps != FROZEN_MAIN_DENSITY_BPS:
        raise ValueError("main control density must remain 1000 bps")
    if getattr(config, "density_bps", None) != spec.main_density_bps:
        raise ValueError("config main density does not match control spec")
    if (
        not isinstance(spec.wrong_key_count, int)
        or isinstance(spec.wrong_key_count, bool)
        or spec.wrong_key_count <= 0
    ):
        raise ValueError("wrong key count must be a positive integer")
    if not isinstance(spec.wrong_key_seed, bytes) or len(spec.wrong_key_seed) < 16:
        raise ValueError("wrong key seed must contain at least 16 bytes")
    configured = control_spec_from_config(
        config,
        require_production_wrong_key_count=False,
    )
    if spec != configured:
        raise ValueError("control spec must exactly match the frozen experiment config")


def _validate_source_items(source_items: Sequence[Any]) -> None:
    document_ids: set[str] = set()
    for item in source_items:
        if item.document_id in document_ids:
            raise ValueError(f"duplicate source document ID: {item.document_id}")
        document_ids.add(item.document_id)
        source_sha256 = hashlib.sha256(item.text.encode("utf-8")).hexdigest()
        if source_sha256 != item.sha256:
            raise ValueError(f"source text hash mismatch: {item.document_id}")


def _validate_encoding_bindings(
    source_items: Sequence[Any],
    encodings: Sequence[EncodeResult],
) -> None:
    for item, encoding in zip(source_items, encodings, strict=True):
        if encoding.document_id != item.document_id:
            raise ValueError("encoding document order differs from source corpus")
        if encoding.eligible_positions != item.eligible_positions:
            raise ValueError(
                f"encoding eligible position count differs from manifest: {item.document_id}"
            )


def _marking_summary(encodings: Sequence[EncodeResult]) -> dict[str, object]:
    documents = [_marking_counts(encoding) for encoding in encodings]
    all_word_count = sum(encoding.all_word_count for encoding in encodings)
    scorable_word_count = sum(encoding.scorable_word_count for encoding in encodings)
    eligible_positions = sum(encoding.eligible_positions for encoding in encodings)
    active_positions = sum(encoding.active_positions for encoding in encodings)
    changed_positions = sum(encoding.changed_positions for encoding in encodings)
    return {
        "activePositions": active_positions,
        "allWordCount": all_word_count,
        "changedPositions": changed_positions,
        "coverage": _coverage(
            active_positions=active_positions,
            changed_positions=changed_positions,
            eligible_positions=eligible_positions,
            all_word_count=all_word_count,
        ),
        "documents": documents,
        "eligiblePositions": eligible_positions,
        "scorableWordCount": scorable_word_count,
    }


def _marking_counts(encoding: EncodeResult) -> dict[str, object]:
    return {
        "activePositions": encoding.active_positions,
        "allWordCount": encoding.all_word_count,
        "changedPositions": encoding.changed_positions,
        "coverage": _coverage(
            active_positions=encoding.active_positions,
            changed_positions=encoding.changed_positions,
            eligible_positions=encoding.eligible_positions,
            all_word_count=encoding.all_word_count,
        ),
        "documentId": encoding.document_id,
        "eligiblePositions": encoding.eligible_positions,
        "scorableWordCount": encoding.scorable_word_count,
    }


def _coverage(
    *,
    active_positions: int,
    changed_positions: int,
    eligible_positions: int,
    all_word_count: int,
) -> dict[str, object]:
    return {
        "activePerAllWords": _exact_ratio(active_positions, all_word_count),
        "activePerEligible": _exact_ratio(active_positions, eligible_positions),
        "changedPerAllWords": _exact_ratio(changed_positions, all_word_count),
        "changedPerEligible": _exact_ratio(changed_positions, eligible_positions),
    }


def _exact_ratio(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "denominator": denominator,
        "numerator": numerator,
        "value": numerator / denominator if denominator else None,
    }


def _wrong_key_result(wrong_keys: Any) -> dict[str, object]:
    result = wrong_keys.to_dict(include_scores=False)
    result["scores"] = []
    scores = result["scores"]
    assert isinstance(scores, list)
    for index, score in enumerate(wrong_keys.scores):
        public = _public_score(score, include_documents=False)
        public["wrongKeyIndex"] = index
        scores.append(public)
    return result


def _public_score(score: Any, *, include_documents: bool) -> dict[str, object]:
    return _remove_key_hashes(score.to_dict(include_documents=include_documents))


def _remove_key_hashes(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _remove_key_hashes(item)
            for key, item in value.items()
            if key != "keySha256"
        }
    if isinstance(value, list):
        return [_remove_key_hashes(item) for item in value]
    return value


def _build_marked_manifest(
    *,
    config: Any,
    corpus: Any,
    source_items: Sequence[Any],
    encodings: Sequence[EncodeResult],
    marked_files: Mapping[str, bytes],
    original_corpus_sha256: str,
    marked_corpus_sha256: str,
    review_approval_sha256: str,
    key_sha256: str,
) -> dict[str, object]:
    source_version = corpus.manifest.get("corpusVersion")
    if not isinstance(source_version, str) or not source_version:
        raise ValueError("source corpus manifest requires corpusVersion")
    documents: list[dict[str, object]] = []
    for item, encoding in zip(source_items, encodings, strict=True):
        marked_path = _marked_document_path(item.document_id)
        marked_bytes = marked_files[marked_path]
        counts = _marking_counts(encoding)
        documents.append(
            {
                **counts,
                "genre": item.genre,
                "markedPath": marked_path,
                "markedSha256": hashlib.sha256(marked_bytes).hexdigest(),
                "sourcePath": item.path,
                "sourceSha256": item.sha256,
                "title": item.title,
            }
        )
    aggregate = _marking_summary(encodings)
    return {
        "activePositions": aggregate["activePositions"],
        "allWordCount": aggregate["allWordCount"],
        "changedPositions": aggregate["changedPositions"],
        "configSha256": config.sha256,
        "contextInventorySha256": corpus.inventory_sha256,
        "contextWidth": config.context_width,
        "coverage": aggregate["coverage"],
        "densityBps": FROZEN_MAIN_DENSITY_BPS,
        "documentCount": len(documents),
        "documents": documents,
        "eligiblePositions": aggregate["eligiblePositions"],
        "keySha256": key_sha256,
        "lexiconFileSha256": corpus.lexicon_file_sha256,
        "lexiconSha256": corpus.lexicon.sha256,
        "markedCorpusSha256": marked_corpus_sha256,
        "markedCorpusVersion": f"{source_version}-marked-1000-v1",
        "methodology": (
            "Deterministic 1000-bps true-key encoding of the exact reviewed source "
            "corpus. Every marked file is bound to its source and marked SHA-256; "
            "density is defined over eligible synonym positions."
        ),
        "minActivePositions": config.min_active_positions,
        "originalCorpusSha256": original_corpus_sha256,
        "reviewApprovalSha256": review_approval_sha256,
        "reviewSha256s": list(corpus.review_sha256s),
        "schemaVersion": MARKED_MANIFEST_SCHEMA_VERSION,
        "schemeVersion": SCHEME_VERSION,
        "scorableWordCount": aggregate["scorableWordCount"],
        "sourceCorpusVersion": source_version,
        "sourceManifestSha256": corpus.manifest_sha256,
        "sources": [dict(source) for source in config.sources],
        "textContractVersion": TEXT_CONTRACT_VERSION,
        "tokenizerVersion": TOKENIZER_VERSION,
        "verifiedAt": config.verified_at,
    }


def _corpus_hash_from_source_items(source_items: Sequence[Any]) -> str:
    return _sha256_json(
        [
            {"documentId": item.document_id, "sha256": item.sha256}
            for item in source_items
        ]
    )


def _corpus_hash_from_documents(documents: Sequence[Document]) -> str:
    return _sha256_json(
        [
            {
                "documentId": document.document_id,
                "sha256": hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
            }
            for document in documents
        ]
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _marked_document_path(document_id: str) -> str:
    return f"{MARKED_DIRECTORY}/{document_id}.md"


def _safe_output_path(root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe output path: {relative_path!r}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"output path escapes repository root: {relative_path!r}")
    return resolved


def _load_bound_json_object(
    path: str | Path,
    label: str,
    expected_sha256: str,
) -> dict[str, object]:
    _require_sha256(expected_sha256, f"{label} expected sha256")
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {label}") from error
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{label} hash differs from frozen config binding")
    return _load_json_bytes(raw, label)


def _load_json_bytes(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be lowercase SHA-256")


def _build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(root), help="repository root")
    parser.add_argument(
        "--config",
        default="fixtures/experiment-config-v1.json",
        help="absolute path or path relative to --root",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in memory and byte-check every canonical output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config, corpus = load_control_inputs(config_path, root=root)
    spec = control_spec_from_config(config)
    outputs = build_corpus_controls(config, corpus, spec=spec)
    freshness = check_control_outputs(root, outputs)
    if not args.check:
        write_control_outputs(root, outputs)
        freshness = check_control_outputs(root, outputs)
    passed = bool(outputs.artifact["acceptance"]["passed"]) and bool(
        freshness["passed"]
    )
    files = freshness["files"]
    assert isinstance(files, dict)
    print(
        json.dumps(
            {
                "acceptancePassed": outputs.artifact["acceptance"]["passed"],
                "artifactMatches": files.get(ARTIFACT_PATH, False),
                "configSha256": outputs.artifact["configSha256"],
                "mismatchedFiles": sorted(
                    path for path, matches in files.items() if not matches
                ),
                "outputsMatch": freshness["passed"],
                "passed": passed,
                "unexpectedFiles": freshness["unexpectedFiles"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
