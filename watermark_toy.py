"""Transparent CPU-only keyed lexical watermark for EXP-002.

This is a teaching model. It is not a Claude, Gemini, or SynthID detector.
Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import statistics
from typing import Iterable, Sequence

from text_contract import (
    PROTECTED_SENTINEL,
    TEXT_CONTRACT_VERSION,
    TOKENIZER_VERSION,
    TextAnalysis,
    analyze_text,
)


SCHEME_VERSION = "toy-lexical-v1"
DEFAULT_CONTEXT_WIDTH = 4
DEFAULT_MIN_ACTIVE_POSITIONS = 20
DETECTION_ALPHA = Fraction(1, 100)
_LEXICON_WORD_RE = re.compile(r"^[a-z]+$")
_FRAME_MAGIC = b"text-watermark-roundtrip\x00"


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str

    def __post_init__(self) -> None:
        _validate_document_id(self.document_id)
        if not isinstance(self.text, str):
            raise TypeError("document text must be a string")


@dataclass(frozen=True)
class SynonymPair:
    class_id: str
    variants: tuple[str, str]


class SynonymLexicon:
    """Validated two-member lexical classes with a canonical content hash."""

    def __init__(self, pairs: Iterable[SynonymPair]) -> None:
        ordered = tuple(sorted(pairs, key=lambda pair: pair.class_id))
        if not ordered:
            raise ValueError("lexicon must contain at least one pair")

        token_to_pair: dict[str, SynonymPair] = {}
        class_ids: set[str] = set()
        for pair in ordered:
            if pair.class_id in class_ids:
                raise ValueError(f"duplicate class id: {pair.class_id}")
            class_ids.add(pair.class_id)
            if len(pair.variants) != 2 or pair.variants[0] == pair.variants[1]:
                raise ValueError(f"class {pair.class_id} must have two distinct variants")
            for variant in pair.variants:
                if not _LEXICON_WORD_RE.fullmatch(variant):
                    raise ValueError(f"invalid lowercase ASCII variant: {variant!r}")
                if variant in token_to_pair:
                    raise ValueError(f"duplicate lexicon variant: {variant}")
                token_to_pair[variant] = pair

        canonical = json.dumps(
            {
                "pairs": [
                    {"classId": pair.class_id, "variants": list(pair.variants)}
                    for pair in ordered
                ],
                "schemeVersion": SCHEME_VERSION,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.pairs = ordered
        self.token_to_pair = token_to_pair
        self.digest = hashlib.sha256(canonical).digest()
        self.sha256 = self.digest.hex()

    @classmethod
    def from_pairs(cls, pairs: Iterable[Sequence[str]]) -> "SynonymLexicon":
        parsed: list[SynonymPair] = []
        for raw_pair in pairs:
            if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
                raise ValueError("every synonym class must contain exactly two variants")
            first, second = raw_pair
            if not isinstance(first, str) or not isinstance(second, str):
                raise TypeError("synonym variants must be strings")
            parsed.append(
                SynonymPair(
                    class_id=f"{first}|{second}",
                    variants=(first, second),
                )
            )
        return cls(parsed)


@dataclass(frozen=True)
class PositionDecision:
    start: int
    end: int
    token: str
    class_id: str
    context: tuple[str, ...]
    occurrence_rank: int
    active: bool
    favored_index: int
    favored_variant: str
    fingerprint: str


@dataclass(frozen=True)
class EncodeResult:
    document_id: str
    text: str
    density_bps: int
    all_word_count: int
    scorable_word_count: int
    eligible_positions: int
    active_positions: int
    changed_positions: int
    positions: tuple[PositionDecision, ...]
    key_sha256: str
    lexicon_sha256: str
    context_width: int

    def to_dict(self) -> dict[str, object]:
        return {
            "activePerAllWords": _ratio(self.active_positions, self.all_word_count),
            "activePerEligible": _ratio(self.active_positions, self.eligible_positions),
            "activePositions": self.active_positions,
            "allWordCount": self.all_word_count,
            "changedPerAllWords": _ratio(self.changed_positions, self.all_word_count),
            "changedPositions": self.changed_positions,
            "contextWidth": self.context_width,
            "densityBps": self.density_bps,
            "documentId": self.document_id,
            "eligiblePositions": self.eligible_positions,
            "keySha256": self.key_sha256,
            "lexiconSha256": self.lexicon_sha256,
            "markedText": self.text,
            "scorableWordCount": self.scorable_word_count,
            "schemeVersion": SCHEME_VERSION,
            "textContractVersion": TEXT_CONTRACT_VERSION,
            "tokenizerVersion": TOKENIZER_VERSION,
        }


@dataclass(frozen=True)
class ScoreResult:
    document_id: str
    density_bps: int
    all_word_count: int
    scorable_word_count: int
    eligible_positions: int
    active_positions: int
    hits: int
    p_value: Fraction | None
    z_score: float | None
    status: str
    active_fingerprints: tuple[str, ...]
    key_sha256: str
    lexicon_sha256: str
    context_width: int
    min_active_positions: int

    def to_dict(self) -> dict[str, object]:
        return _score_dict(
            document_id=self.document_id,
            density_bps=self.density_bps,
            all_word_count=self.all_word_count,
            scorable_word_count=self.scorable_word_count,
            eligible_positions=self.eligible_positions,
            active_positions=self.active_positions,
            hits=self.hits,
            p_value=self.p_value,
            z_score=self.z_score,
            status=self.status,
            key_sha256=self.key_sha256,
            lexicon_sha256=self.lexicon_sha256,
            context_width=self.context_width,
            min_active_positions=self.min_active_positions,
            scoring_unit="document_diagnostic",
        )


@dataclass(frozen=True)
class CorpusScore:
    density_bps: int
    document_count: int
    all_word_count: int
    scorable_word_count: int
    eligible_positions: int
    active_positions: int
    hits: int
    p_value: Fraction | None
    z_score: float | None
    status: str
    documents: tuple[ScoreResult, ...]
    active_fingerprints: tuple[str, ...]
    document_ids: tuple[str, ...]
    key_sha256: str
    lexicon_sha256: str
    context_width: int
    min_active_positions: int

    def to_dict(self, *, include_documents: bool = True) -> dict[str, object]:
        output = _score_dict(
            document_id=None,
            density_bps=self.density_bps,
            all_word_count=self.all_word_count,
            scorable_word_count=self.scorable_word_count,
            eligible_positions=self.eligible_positions,
            active_positions=self.active_positions,
            hits=self.hits,
            p_value=self.p_value,
            z_score=self.z_score,
            status=self.status,
            key_sha256=self.key_sha256,
            lexicon_sha256=self.lexicon_sha256,
            context_width=self.context_width,
            min_active_positions=self.min_active_positions,
            scoring_unit="pooled_corpus",
        )
        output["documentCount"] = self.document_count
        if include_documents:
            output["documents"] = [document.to_dict() for document in self.documents]
        return output


@dataclass(frozen=True)
class WrongKeyControls:
    count: int
    sufficient_count: int
    insufficient_count: int
    detected_count: int
    detected_rate: float | None
    median_z_score: float | None
    max_z_score: float | None
    scores: tuple[CorpusScore, ...]

    def to_dict(self, *, include_scores: bool = False) -> dict[str, object]:
        output: dict[str, object] = {
            "count": self.count,
            "detectedCount": self.detected_count,
            "detectedRateAmongSufficient": self.detected_rate,
            "insufficientCount": self.insufficient_count,
            "maxZScore": self.max_z_score,
            "medianZScore": self.median_z_score,
            "sufficientCount": self.sufficient_count,
        }
        if include_scores:
            output["scores"] = [score.to_dict(include_documents=False) for score in self.scores]
        return output


@dataclass(frozen=True)
class FingerprintComparison:
    baseline_active: int
    output_active: int
    surviving_active: int
    lost_active: int
    new_active: int

    def to_dict(self) -> dict[str, object]:
        return {
            "baselineActive": self.baseline_active,
            "lostActive": self.lost_active,
            "lostActiveRate": _ratio(self.lost_active, self.baseline_active),
            "newActive": self.new_active,
            "outputActive": self.output_active,
            "survivingActive": self.surviving_active,
        }


def load_lexicon(path: str | Path) -> SynonymLexicon:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("synonym lexicon must be a JSON object")
    if raw.get("schemaVersion") != 1:
        raise ValueError("unsupported synonym lexicon schemaVersion")
    for field in ("lexiconVersion", "verifiedAt", "methodology"):
        if not isinstance(raw.get(field), str) or not raw[field].strip():
            raise ValueError(f"synonym lexicon requires non-empty {field}")
    sources = raw.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("synonym lexicon requires at least one source")
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("every synonym lexicon source must be an object")
        if not isinstance(source.get("title"), str) or not source["title"].strip():
            raise ValueError("every synonym lexicon source requires a title")
        if not isinstance(source.get("url"), str) or not source["url"].startswith(
            ("https://", "http://")
        ):
            raise ValueError("every synonym lexicon source requires an HTTP URL")
    if raw.get("manualContextReviewRequired") is not True:
        raise ValueError("synonym lexicon must require manual context review")
    if not isinstance(raw.get("pairs"), list):
        raise ValueError("synonym lexicon pairs must be a list")
    return SynonymLexicon.from_pairs(raw["pairs"])


def inspect_positions(
    text: str,
    *,
    key: bytes,
    document_id: str,
    density_bps: int,
    lexicon: SynonymLexicon,
    context_width: int = DEFAULT_CONTEXT_WIDTH,
) -> tuple[PositionDecision, ...]:
    """Return every eligible position and its deterministic keyed decision."""
    return _scan(
        text,
        key=key,
        document_id=document_id,
        density_bps=density_bps,
        lexicon=lexicon,
        context_width=context_width,
    )[1]


def encode_text(
    text: str,
    *,
    key: bytes,
    document_id: str,
    density_bps: int,
    lexicon: SynonymLexicon,
    context_width: int = DEFAULT_CONTEXT_WIDTH,
) -> EncodeResult:
    """Choose the favored synonym at every active eligible position."""
    analysis, positions = _scan(
        text,
        key=key,
        document_id=document_id,
        density_bps=density_bps,
        lexicon=lexicon,
        context_width=context_width,
    )
    chunks: list[str] = []
    cursor = 0
    changed_positions = 0
    for position in positions:
        if not position.active:
            continue
        replacement = _apply_case(position.favored_variant, position.token)
        if replacement == position.token:
            continue
        chunks.append(text[cursor : position.start])
        chunks.append(replacement)
        cursor = position.end
        changed_positions += 1
    chunks.append(text[cursor:])

    return EncodeResult(
        document_id=document_id,
        text="".join(chunks),
        density_bps=density_bps,
        all_word_count=analysis.all_word_count,
        scorable_word_count=analysis.scorable_word_count,
        eligible_positions=len(positions),
        active_positions=sum(position.active for position in positions),
        changed_positions=changed_positions,
        positions=positions,
        key_sha256=hashlib.sha256(key).hexdigest(),
        lexicon_sha256=lexicon.sha256,
        context_width=context_width,
    )


def score_text(
    text: str,
    *,
    key: bytes,
    document_id: str,
    density_bps: int,
    lexicon: SynonymLexicon,
    context_width: int = DEFAULT_CONTEXT_WIDTH,
    min_active_positions: int = DEFAULT_MIN_ACTIVE_POSITIONS,
) -> ScoreResult:
    analysis, positions = _scan(
        text,
        key=key,
        document_id=document_id,
        density_bps=density_bps,
        lexicon=lexicon,
        context_width=context_width,
    )
    active = tuple(position for position in positions if position.active)
    hits = sum(position.token.lower() == position.favored_variant for position in active)
    p_value, z_score, status = _decision(hits, len(active), min_active_positions)
    return ScoreResult(
        document_id=document_id,
        density_bps=density_bps,
        all_word_count=analysis.all_word_count,
        scorable_word_count=analysis.scorable_word_count,
        eligible_positions=len(positions),
        active_positions=len(active),
        hits=hits,
        p_value=p_value,
        z_score=z_score,
        status=status,
        active_fingerprints=tuple(position.fingerprint for position in active),
        key_sha256=hashlib.sha256(key).hexdigest(),
        lexicon_sha256=lexicon.sha256,
        context_width=context_width,
        min_active_positions=min_active_positions,
    )


def score_corpus(
    documents: Sequence[Document],
    *,
    key: bytes,
    density_bps: int,
    lexicon: SynonymLexicon,
    context_width: int = DEFAULT_CONTEXT_WIDTH,
    min_active_positions: int = DEFAULT_MIN_ACTIVE_POSITIONS,
) -> CorpusScore:
    if not documents:
        raise ValueError("corpus must contain at least one document")
    document_ids = [document.document_id for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("document IDs must be unique within a corpus")

    scores = tuple(
        score_text(
            document.text,
            key=key,
            document_id=document.document_id,
            density_bps=density_bps,
            lexicon=lexicon,
            context_width=context_width,
            min_active_positions=min_active_positions,
        )
        for document in documents
    )
    active_positions = sum(score.active_positions for score in scores)
    hits = sum(score.hits for score in scores)
    p_value, z_score, status = _decision(hits, active_positions, min_active_positions)
    return CorpusScore(
        density_bps=density_bps,
        document_count=len(scores),
        all_word_count=sum(score.all_word_count for score in scores),
        scorable_word_count=sum(score.scorable_word_count for score in scores),
        eligible_positions=sum(score.eligible_positions for score in scores),
        active_positions=active_positions,
        hits=hits,
        p_value=p_value,
        z_score=z_score,
        status=status,
        documents=scores,
        active_fingerprints=tuple(
            fingerprint
            for score in scores
            for fingerprint in score.active_fingerprints
        ),
        document_ids=tuple(document_ids),
        key_sha256=hashlib.sha256(key).hexdigest(),
        lexicon_sha256=lexicon.sha256,
        context_width=context_width,
        min_active_positions=min_active_positions,
    )


def compare_active_fingerprints(
    baseline: ScoreResult | CorpusScore,
    output: ScoreResult | CorpusScore,
) -> FingerprintComparison:
    """Compare original and output keyed opportunities as multisets.

    New active positions are reported separately. They are not silently
    treated as surviving positions from the marked input.
    """
    if type(baseline) is not type(output):
        raise ValueError("fingerprint comparisons require the same scoring-unit type")
    shared_fields = (
        "density_bps",
        "key_sha256",
        "lexicon_sha256",
        "context_width",
    )
    for field in shared_fields:
        if getattr(baseline, field) != getattr(output, field):
            raise ValueError(f"fingerprint comparison mismatch: {field}")
    if isinstance(baseline, ScoreResult):
        assert isinstance(output, ScoreResult)
        if baseline.document_id != output.document_id:
            raise ValueError(
                "document fingerprint comparisons require the same document ID"
            )
    else:
        assert isinstance(baseline, CorpusScore) and isinstance(output, CorpusScore)
        if baseline.document_ids != output.document_ids:
            raise ValueError(
                "corpus fingerprint comparisons require identical ordered document IDs"
            )

    baseline_counts = Counter(baseline.active_fingerprints)
    output_counts = Counter(output.active_fingerprints)
    surviving = sum((baseline_counts & output_counts).values())
    baseline_active = sum(baseline_counts.values())
    output_active = sum(output_counts.values())
    return FingerprintComparison(
        baseline_active=baseline_active,
        output_active=output_active,
        surviving_active=surviving,
        lost_active=baseline_active - surviving,
        new_active=output_active - surviving,
    )


def binomial_tail_probability(hits: int, trials: int) -> Fraction:
    """Return exact P[Binomial(trials, 0.5) >= hits]."""
    if trials < 0 or hits < 0 or hits > trials:
        raise ValueError("require 0 <= hits <= trials")
    numerator = sum(math.comb(trials, successes) for successes in range(hits, trials + 1))
    return Fraction(numerator, 1 << trials)


def run_wrong_key_controls(
    documents: Sequence[Document],
    *,
    density_bps: int,
    lexicon: SynonymLexicon,
    count: int = 1_000,
    seed: bytes = b"exp002-wrong-key-controls-v1",
    context_width: int = DEFAULT_CONTEXT_WIDTH,
    min_active_positions: int = DEFAULT_MIN_ACTIVE_POSITIONS,
) -> WrongKeyControls:
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")
    _validate_key(seed)

    scores = tuple(
        score_corpus(
            documents,
            key=hmac.new(
                seed,
                b"wrong-key-v1\x00" + index.to_bytes(8, "big"),
                hashlib.sha256,
            ).digest(),
            density_bps=density_bps,
            lexicon=lexicon,
            context_width=context_width,
            min_active_positions=min_active_positions,
        )
        for index in range(count)
    )
    sufficient = tuple(score for score in scores if score.status != "insufficient_evidence")
    detected_count = sum(score.status == "detected" for score in sufficient)
    z_scores = [score.z_score for score in sufficient if score.z_score is not None]
    return WrongKeyControls(
        count=count,
        sufficient_count=len(sufficient),
        insufficient_count=count - len(sufficient),
        detected_count=detected_count,
        detected_rate=detected_count / len(sufficient) if sufficient else None,
        median_z_score=statistics.median(z_scores) if z_scores else None,
        max_z_score=max(z_scores) if z_scores else None,
        scores=scores,
    )


def _scan(
    text: str,
    *,
    key: bytes,
    document_id: str,
    density_bps: int,
    lexicon: SynonymLexicon,
    context_width: int,
) -> tuple[TextAnalysis, tuple[PositionDecision, ...]]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    key = _validate_key(key)
    _validate_document_id(document_id)
    _validate_density_bps(density_bps)
    if not isinstance(context_width, int) or isinstance(context_width, bool):
        raise TypeError("context_width must be an integer")
    if not 1 <= context_width <= 32:
        raise ValueError("context_width must be between 1 and 32")

    analysis = analyze_text(text)
    context: deque[str] = deque(maxlen=context_width)
    occurrence_counts: defaultdict[tuple[str, tuple[str, ...]], int] = defaultdict(int)
    positions: list[PositionDecision] = []
    activation_threshold = density_bps * (1 << 256) // 10_000

    for token in analysis.context_tokens:
        if token.protected:
            context.append(PROTECTED_SENTINEL)
            continue

        assert token.text is not None
        raw = token.text
        normalized = token.normalized
        pair = lexicon.token_to_pair.get(normalized) if _has_supported_case_shape(raw) else None
        if pair is None:
            context.append(normalized)
            continue

        normalized_context = tuple(context)
        occurrence_key = (pair.class_id, normalized_context)
        occurrence_rank = occurrence_counts[occurrence_key]
        occurrence_counts[occurrence_key] += 1
        activation_message = _position_message(
            domain=b"activate-v1",
            lexicon=lexicon,
            document_id=document_id,
            class_id=pair.class_id,
            context=normalized_context,
            occurrence_rank=occurrence_rank,
        )
        favor_message = _position_message(
            domain=b"favor-v1",
            lexicon=lexicon,
            document_id=document_id,
            class_id=pair.class_id,
            context=normalized_context,
            occurrence_rank=occurrence_rank,
        )
        activation_value = int.from_bytes(
            hmac.new(key, activation_message, hashlib.sha256).digest(),
            "big",
        )
        favored_index = hmac.new(key, favor_message, hashlib.sha256).digest()[-1] & 1
        fingerprint = hashlib.sha256(
            _position_message(
                domain=b"fingerprint-v1",
                lexicon=lexicon,
                document_id=document_id,
                class_id=pair.class_id,
                context=normalized_context,
                occurrence_rank=occurrence_rank,
            )
        ).hexdigest()
        positions.append(
            PositionDecision(
                start=token.start,
                end=token.end,
                token=raw,
                class_id=pair.class_id,
                context=normalized_context,
                occurrence_rank=occurrence_rank,
                active=activation_value < activation_threshold,
                favored_index=favored_index,
                favored_variant=pair.variants[favored_index],
                fingerprint=fingerprint,
            )
        )
        context.append(f"<{pair.class_id}>")

    return analysis, tuple(positions)


def _position_message(
    *,
    domain: bytes,
    lexicon: SynonymLexicon,
    document_id: str,
    class_id: str,
    context: tuple[str, ...],
    occurrence_rank: int,
) -> bytes:
    fields = [
        SCHEME_VERSION.encode("ascii"),
        domain,
        lexicon.digest,
        document_id.encode("utf-8"),
        class_id.encode("ascii"),
        occurrence_rank.to_bytes(8, "big"),
        len(context).to_bytes(2, "big"),
        *(atom.encode("utf-8") for atom in context),
    ]
    framed = bytearray(_FRAME_MAGIC)
    for field in fields:
        framed.extend(len(field).to_bytes(4, "big"))
        framed.extend(field)
    return bytes(framed)


def _decision(
    hits: int,
    active_positions: int,
    min_active_positions: int,
) -> tuple[Fraction | None, float | None, str]:
    if not isinstance(min_active_positions, int) or isinstance(min_active_positions, bool):
        raise TypeError("min_active_positions must be an integer")
    if min_active_positions <= 0:
        raise ValueError("min_active_positions must be positive")
    if active_positions < min_active_positions:
        return None, None, "insufficient_evidence"

    p_value = binomial_tail_probability(hits, active_positions)
    z_score = (2 * hits - active_positions) / math.sqrt(active_positions)
    status = "detected" if p_value <= DETECTION_ALPHA else "not_detected"
    return p_value, z_score, status


def _validate_key(key: bytes) -> bytes:
    if not isinstance(key, bytes):
        raise TypeError("key must be bytes")
    if len(key) < 16:
        raise ValueError("key must contain at least 16 bytes")
    return key


def _validate_document_id(document_id: str) -> None:
    if not isinstance(document_id, str):
        raise TypeError("document_id must be a string")
    if not document_id or len(document_id.encode("utf-8")) > 256:
        raise ValueError("document_id must contain 1 to 256 UTF-8 bytes")
    if any(character in document_id for character in ("\x00", "\n", "\r")):
        raise ValueError("document_id cannot contain NUL or line breaks")


def _validate_density_bps(density_bps: int) -> None:
    if not isinstance(density_bps, int) or isinstance(density_bps, bool):
        raise TypeError("density_bps must be an integer")
    if not 1 <= density_bps <= 10_000:
        raise ValueError("density_bps must be between 1 and 10000")


def _has_supported_case_shape(token: str) -> bool:
    return token.islower() or token.isupper() or token.istitle()


def _apply_case(replacement: str, original: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original.istitle():
        return replacement.capitalize()
    return replacement


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _score_dict(
    *,
    document_id: str | None,
    density_bps: int,
    all_word_count: int,
    scorable_word_count: int,
    eligible_positions: int,
    active_positions: int,
    hits: int,
    p_value: Fraction | None,
    z_score: float | None,
    status: str,
    key_sha256: str,
    lexicon_sha256: str,
    context_width: int,
    min_active_positions: int,
    scoring_unit: str,
) -> dict[str, object]:
    output: dict[str, object] = {
        "activePerAllWords": _ratio(active_positions, all_word_count),
        "activePerEligible": _ratio(active_positions, eligible_positions),
        "activePositions": active_positions,
        "allWordCount": all_word_count,
        "contextWidth": context_width,
        "detectionAlphaExact": {
            "denominator": DETECTION_ALPHA.denominator,
            "numerator": DETECTION_ALPHA.numerator,
        },
        "densityBps": density_bps,
        "eligiblePositions": eligible_positions,
        "hitRate": _ratio(hits, active_positions),
        "hits": hits,
        "keySha256": key_sha256,
        "lexiconSha256": lexicon_sha256,
        "minActivePositions": min_active_positions,
        "pValue": float(p_value) if p_value is not None else None,
        "pValueExact": (
            {"denominator": p_value.denominator, "numerator": p_value.numerator}
            if p_value is not None
            else None
        ),
        "schemeVersion": SCHEME_VERSION,
        "scoringUnit": scoring_unit,
        "scorableWordCount": scorable_word_count,
        "status": status,
        "textContractVersion": TEXT_CONTRACT_VERSION,
        "tokenizerVersion": TOKENIZER_VERSION,
        "zScore": z_score,
    }
    if document_id is not None:
        output["documentId"] = document_id
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("encode", "detect"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument(
            "--lexicon",
            default=str(Path(__file__).with_name("fixtures") / "synonym_pairs-v1.json"),
        )
        command_parser.add_argument("--key-hex", required=True)
        command_parser.add_argument("--document-id", required=True)
        command_parser.add_argument("--density-bps", type=int, default=1_000)
        command_parser.add_argument("--input", required=True, help="UTF-8 text file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        lexicon = load_lexicon(args.lexicon)
        key = bytes.fromhex(args.key_hex)
        text = Path(args.input).read_text(encoding="utf-8")
        if args.command == "encode":
            output = encode_text(
                text,
                key=key,
                document_id=args.document_id,
                density_bps=args.density_bps,
                lexicon=lexicon,
            ).to_dict()
        else:
            output = score_text(
                text,
                key=key,
                document_id=args.document_id,
                density_bps=args.density_bps,
                lexicon=lexicon,
            ).to_dict()
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(output, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CorpusScore",
    "DEFAULT_CONTEXT_WIDTH",
    "DEFAULT_MIN_ACTIVE_POSITIONS",
    "DETECTION_ALPHA",
    "Document",
    "EncodeResult",
    "FingerprintComparison",
    "PositionDecision",
    "SCHEME_VERSION",
    "ScoreResult",
    "SynonymLexicon",
    "WrongKeyControls",
    "binomial_tail_probability",
    "compare_active_fingerprints",
    "encode_text",
    "inspect_positions",
    "load_lexicon",
    "run_wrong_key_controls",
    "score_corpus",
    "score_text",
]
