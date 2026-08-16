"""Validate and freeze the original article corpus without using a watermark key."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

from text_contract import WORD_RE, analyze_text
from watermark_toy import SynonymLexicon


CORPUS_SCHEMA_VERSION = 1
_DOCUMENT_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,63})$")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
_RESERVED_EMAIL_DOMAINS = ("example.com", "example.org", "example.net", "example.invalid")


@dataclass(frozen=True)
class CorpusDocument:
    document_id: str
    path: str
    genre: str
    title: str
    text: str
    sha256: str
    word_count: int
    eligible_positions: int
    protected_span_count: int

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "documentId": self.document_id,
            "eligiblePositions": self.eligible_positions,
            "genre": self.genre,
            "path": self.path,
            "protectedSpanCount": self.protected_span_count,
            "sha256": self.sha256,
            "title": self.title,
            "wordCount": self.word_count,
        }


def load_corpus_plan(path: str | Path) -> dict[str, object]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("corpus plan must be a JSON object")
    if raw.get("schemaVersion") != CORPUS_SCHEMA_VERSION:
        raise ValueError("unsupported corpus plan schemaVersion")
    for field in ("corpusVersion", "verifiedAt", "methodology"):
        if not isinstance(raw.get(field), str) or not raw[field].strip():
            raise ValueError(f"corpus plan requires non-empty {field}")
    _validate_sources(raw.get("sources"))

    contract = raw.get("documentContract")
    if not isinstance(contract, dict):
        raise ValueError("corpus plan requires documentContract")
    for field in ("count", "minWords", "maxWords", "minEligiblePositionsPerDocument"):
        value = contract.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"documentContract.{field} must be a positive integer")
    if contract["minWords"] > contract["maxWords"]:
        raise ValueError("documentContract minWords cannot exceed maxWords")

    documents = raw.get("documents")
    if not isinstance(documents, list) or len(documents) != contract["count"]:
        raise ValueError("corpus plan document count does not match its contract")
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("every planned corpus document must be an object")
        document_id = document.get("documentId")
        relative_path = document.get("path")
        genre = document.get("genre")
        if not isinstance(document_id, str) or not _DOCUMENT_ID_RE.fullmatch(document_id):
            raise ValueError(f"invalid corpus document ID: {document_id!r}")
        if document_id in seen_ids:
            raise ValueError(f"duplicate corpus document ID: {document_id}")
        if not isinstance(relative_path, str) or not relative_path.startswith(
            "corpus/original/"
        ):
            raise ValueError(f"invalid corpus document path: {relative_path!r}")
        path_parts = Path(relative_path).parts
        if Path(relative_path).is_absolute() or ".." in path_parts:
            raise ValueError(f"unsafe corpus document path: {relative_path!r}")
        if relative_path in seen_paths:
            raise ValueError(f"duplicate corpus document path: {relative_path}")
        if not isinstance(genre, str) or not genre.strip():
            raise ValueError(f"corpus document {document_id} requires a genre")
        seen_ids.add(document_id)
        seen_paths.add(relative_path)
    return raw


def inspect_corpus(
    root: str | Path,
    *,
    plan: dict[str, object],
    lexicon: SynonymLexicon,
) -> tuple[CorpusDocument, ...]:
    root_path = Path(root).resolve()
    contract = plan["documentContract"]
    planned_documents = plan["documents"]
    assert isinstance(contract, dict) and isinstance(planned_documents, list)
    inspected: list[CorpusDocument] = []

    for planned in planned_documents:
        assert isinstance(planned, dict)
        relative_path = planned["path"]
        assert isinstance(relative_path, str)
        source_path = (root_path / relative_path).resolve()
        if not source_path.is_relative_to(root_path):
            raise ValueError(f"corpus path escapes repository root: {relative_path}")
        raw_bytes = source_path.read_bytes()
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"corpus document is not UTF-8: {relative_path}") from error
        _validate_document_surface(text, relative_path)
        word_count = sum(1 for _ in WORD_RE.finditer(text))
        if not contract["minWords"] <= word_count <= contract["maxWords"]:
            raise ValueError(
                f"{relative_path} has {word_count} words; expected "
                f"{contract['minWords']}..{contract['maxWords']}"
            )
        analysis = analyze_text(text)
        eligible_positions = sum(
            1
            for token in analysis.context_tokens
            if not token.protected
            and token.text is not None
            and _supported_case(token.text)
            and token.normalized in lexicon.token_to_pair
        )
        if eligible_positions < contract["minEligiblePositionsPerDocument"]:
            raise ValueError(
                f"{relative_path} has {eligible_positions} eligible positions; expected at "
                f"least {contract['minEligiblePositionsPerDocument']}"
            )
        title = text.splitlines()[0][2:].strip()
        inspected.append(
            CorpusDocument(
                document_id=str(planned["documentId"]),
                path=relative_path,
                genre=str(planned["genre"]),
                title=title,
                text=text,
                sha256=hashlib.sha256(raw_bytes).hexdigest(),
                word_count=word_count,
                eligible_positions=eligible_positions,
                protected_span_count=len(analysis.protected_spans),
            )
        )

    hashes = [document.sha256 for document in inspected]
    if len(hashes) != len(set(hashes)):
        raise ValueError("corpus documents must have unique byte hashes")
    return tuple(inspected)


def build_manifest(
    *,
    plan: dict[str, object],
    plan_sha256: str,
    lexicon: SynonymLexicon,
    documents: Iterable[CorpusDocument],
) -> dict[str, object]:
    frozen = tuple(documents)
    return {
        "corpusVersion": plan["corpusVersion"],
        "documentCount": len(frozen),
        "documents": [document.to_manifest_dict() for document in frozen],
        "eligiblePositions": sum(document.eligible_positions for document in frozen),
        "lexiconSha256": lexicon.sha256,
        "methodology": plan["methodology"],
        "planSha256": plan_sha256,
        "schemaVersion": CORPUS_SCHEMA_VERSION,
        "sources": plan["sources"],
        "verifiedAt": plan["verifiedAt"],
        "wordCount": sum(document.word_count for document in frozen),
    }


def build_context_inventory(
    *,
    plan: dict[str, object],
    lexicon: SynonymLexicon,
    documents: Iterable[CorpusDocument],
) -> dict[str, object]:
    inventory_documents: list[dict[str, object]] = []
    for document in documents:
        occurrences: list[dict[str, object]] = []
        for token in analyze_text(document.text).context_tokens:
            if token.protected or token.text is None or not _supported_case(token.text):
                continue
            pair = lexicon.token_to_pair.get(token.normalized)
            if pair is None:
                continue
            snippet_start = max(0, token.start - 55)
            snippet_end = min(len(document.text), token.end + 55)
            occurrences.append(
                {
                    "classId": pair.class_id,
                    "end": token.end,
                    "ordinal": len(occurrences),
                    "snippet": " ".join(
                        document.text[snippet_start:snippet_end].split()
                    ),
                    "start": token.start,
                    "token": token.text,
                }
            )
        inventory_documents.append(
            {
                "documentId": document.document_id,
                "eligibleOccurrences": len(occurrences),
                "occurrences": occurrences,
                "reviewStatus": "pending_manual_context_review",
                "sha256": document.sha256,
            }
        )
    return {
        "corpusVersion": plan["corpusVersion"],
        "documents": inventory_documents,
        "lexiconSha256": lexicon.sha256,
        "methodology": (
            "Key-free inventory of every candidate synonym occurrence. Each document must "
            "receive an independent manual context review before marking or model calls."
        ),
        "schemaVersion": CORPUS_SCHEMA_VERSION,
        "sources": plan["sources"],
        "verifiedAt": plan["verifiedAt"],
    }


def validate_context_reviews(
    *,
    inventory: dict[str, object],
    reviews: Iterable[dict[str, object]],
) -> dict[str, object]:
    """Validate independent reviews against the exact key-free inventory."""
    if inventory.get("schemaVersion") != CORPUS_SCHEMA_VERSION:
        raise ValueError("unsupported context inventory schemaVersion")
    inventory_documents = inventory.get("documents")
    if not isinstance(inventory_documents, list) or not inventory_documents:
        raise ValueError("context inventory requires documents")

    expected: dict[str, dict[str, object]] = {}
    for document in inventory_documents:
        if not isinstance(document, dict):
            raise ValueError("context inventory document must be an object")
        document_id = document.get("documentId")
        if not isinstance(document_id, str) or document_id in expected:
            raise ValueError("context inventory has an invalid or duplicate documentId")
        expected[document_id] = document

    inventory_sha256 = hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()
    reviewed: dict[str, dict[str, object]] = {}
    review_files = tuple(reviews)
    if not review_files:
        raise ValueError("at least one context review is required")

    for review in review_files:
        if not isinstance(review, dict):
            raise ValueError("every context review must be a JSON object")
        if review.get("schemaVersion") != CORPUS_SCHEMA_VERSION:
            raise ValueError("unsupported context review schemaVersion")
        for field in ("reviewVersion", "verifiedAt", "methodology", "reviewer"):
            if not isinstance(review.get(field), str) or not review[field].strip():
                raise ValueError(f"context review requires non-empty {field}")
        _validate_sources(review.get("sources"))
        if review.get("corpusVersion") != inventory.get("corpusVersion"):
            raise ValueError("context review corpusVersion does not match inventory")
        if review.get("lexiconSha256") != inventory.get("lexiconSha256"):
            raise ValueError("context review lexiconSha256 does not match inventory")
        if review.get("inventorySha256") != inventory_sha256:
            raise ValueError("context review inventorySha256 does not match inventory")

        decisions = review.get("documents")
        if not isinstance(decisions, list) or not decisions:
            raise ValueError("context review requires documents")
        for decision in decisions:
            if not isinstance(decision, dict):
                raise ValueError("context review document must be an object")
            document_id = decision.get("documentId")
            if not isinstance(document_id, str) or document_id not in expected:
                raise ValueError(f"context review has unknown documentId: {document_id!r}")
            if document_id in reviewed:
                raise ValueError(f"duplicate context review for document: {document_id}")
            source = expected[document_id]
            if decision.get("documentSha256") != source.get("sha256"):
                raise ValueError(f"context review hash mismatch for document: {document_id}")
            eligible = source.get("eligibleOccurrences")
            if decision.get("eligibleOccurrences") != eligible:
                raise ValueError(
                    f"context review occurrence count mismatch for document: {document_id}"
                )
            reviewed_count = decision.get("reviewedOccurrences")
            if reviewed_count != eligible:
                raise ValueError(
                    f"context review is incomplete for document: {document_id}"
                )
            findings = decision.get("findings")
            if not isinstance(findings, list) or any(
                not isinstance(finding, str) or not finding.strip()
                for finding in findings
            ):
                raise ValueError(f"context review findings are invalid for: {document_id}")
            if decision.get("decision") != "approved" or findings:
                raise ValueError(f"context review did not approve document: {document_id}")
            reviewed[document_id] = decision

    missing = sorted(set(expected) - set(reviewed))
    if missing:
        raise ValueError(f"context review is missing document: {missing[0]}")

    return {
        "approvedDocumentCount": len(reviewed),
        "corpusVersion": inventory["corpusVersion"],
        "inventorySha256": inventory_sha256,
        "lexiconSha256": inventory["lexiconSha256"],
        "reviewVersions": sorted(
            str(review["reviewVersion"]) for review in review_files
        ),
        "schemaVersion": CORPUS_SCHEMA_VERSION,
    }


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _validate_document_surface(text: str, relative_path: str) -> None:
    if text.startswith("\ufeff"):
        raise ValueError(f"{relative_path} starts with a UTF-8 BOM")
    if "\r" in text:
        raise ValueError(f"{relative_path} must use LF line endings")
    if not text.endswith("\n"):
        raise ValueError(f"{relative_path} must end with a newline")
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# ") or not lines[0][2:].strip():
        raise ValueError(f"{relative_path} must start with one Markdown H1")
    if any(line.startswith("# ") for line in lines[1:]):
        raise ValueError(f"{relative_path} contains more than one Markdown H1")
    if "—" in text or "–" in text:
        raise ValueError(f"{relative_path} contains a prohibited dash character")
    if "⟦" in text or "⟧" in text:
        raise ValueError(f"{relative_path} collides with transform placeholders")
    for match in _EMAIL_RE.finditer(text):
        domain = match.group(1).lower()
        if domain not in _RESERVED_EMAIL_DOMAINS and not domain.endswith(".example"):
            raise ValueError(
                f"{relative_path} contains a non-reserved email domain: {domain}"
            )


def _validate_sources(raw_sources: object) -> None:
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("corpus plan requires at least one source")
    for source in raw_sources:
        if not isinstance(source, dict):
            raise ValueError("every corpus source must be an object")
        if not isinstance(source.get("title"), str) or not source["title"].strip():
            raise ValueError("every corpus source requires a title")
        url = source.get("url")
        if not isinstance(url, str) or not url.startswith(("https://", "http://")):
            raise ValueError("every corpus source requires an HTTP URL")


def _supported_case(token: str) -> bool:
    return token.islower() or token.isupper() or token.istitle()


__all__ = [
    "CORPUS_SCHEMA_VERSION",
    "CorpusDocument",
    "build_context_inventory",
    "build_manifest",
    "canonical_json_bytes",
    "inspect_corpus",
    "load_corpus_plan",
    "validate_context_reviews",
]
