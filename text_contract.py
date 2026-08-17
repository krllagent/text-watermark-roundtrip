"""Shared token and protected-span contract for the toy experiment.

The marker and later text transformations must use this module rather than
inventing separate definitions of protected content.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


WORD_RE = re.compile(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b")
PROTECTED_SENTINEL = "<protected>"
TEXT_CONTRACT_VERSION = "protected-spans-v1"
TOKENIZER_VERSION = "ascii-word-v1"

_PROTECTED_PATTERNS = (
    re.compile(r'"[^"\n]*"'),
    re.compile(r"“[^”\n]*”"),
    re.compile(r"(?<![A-Za-z])'[^'\n]+'(?![A-Za-z])"),
    re.compile(r"‘[^’\n]*’"),
    re.compile(r"`[^`\n]*`"),
    re.compile(r"https?://[^\s<>\"“”`]*[^\s<>\"“”`.,;:!?\)\]\}]", re.IGNORECASE),
    re.compile(r"www\.[^\s<>\"“”`]*[^\s<>\"“”`.,;:!?\)\]\}]", re.IGNORECASE),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"(?<![A-Za-z0-9_])@[A-Za-z0-9_]+"),
    re.compile(r"(?<![A-Za-z0-9_])#[A-Za-z0-9_]+"),
    re.compile(
        r"(?:"
        r"[$£€]\s?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
        r"|"
        r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s?%"
        r")"
    ),
)


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("span must have 0 <= start < end")


@dataclass(frozen=True)
class ContextToken:
    normalized: str
    start: int
    end: int
    text: str | None
    protected: bool


@dataclass(frozen=True)
class TextAnalysis:
    protected_spans: tuple[Span, ...]
    context_tokens: tuple[ContextToken, ...]
    all_word_count: int
    scorable_word_count: int


def find_protected_spans(text: str) -> tuple[Span, ...]:
    """Return sorted, non-overlapping protected spans.

    Overlaps are merged for marker exclusion, so every byte covered by any
    protected class stays outside the lexical signal. The later placeholder
    transform may apply its URL-over-quote naming priority without shrinking
    this exclusion union.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    matches: list[Span] = []
    for pattern in _PROTECTED_PATTERNS:
        matches.extend(Span(match.start(), match.end()) for match in pattern.finditer(text))

    if not matches:
        return ()

    matches.sort(key=lambda span: (span.start, span.end))
    merged: list[Span] = [matches[0]]
    for span in matches[1:]:
        previous = merged[-1]
        if _overlaps(span, previous):
            merged[-1] = Span(previous.start, max(previous.end, span.end))
        else:
            merged.append(span)
    return tuple(merged)


def analyze_text(text: str) -> TextAnalysis:
    """Tokenize immutable input and replace each protected span with a sentinel."""
    protected_spans = find_protected_spans(text)
    context_tokens: list[ContextToken] = []
    cursor = 0

    for span in protected_spans:
        context_tokens.extend(_word_tokens(text, cursor, span.start))
        context_tokens.append(
            ContextToken(
                normalized=PROTECTED_SENTINEL,
                start=span.start,
                end=span.end,
                text=None,
                protected=True,
            )
        )
        cursor = span.end

    context_tokens.extend(_word_tokens(text, cursor, len(text)))
    scorable_word_count = sum(not token.protected for token in context_tokens)
    return TextAnalysis(
        protected_spans=protected_spans,
        context_tokens=tuple(context_tokens),
        all_word_count=sum(1 for _ in WORD_RE.finditer(text)),
        scorable_word_count=scorable_word_count,
    )


def _word_tokens(text: str, start: int, end: int) -> list[ContextToken]:
    tokens: list[ContextToken] = []
    for match in WORD_RE.finditer(text, start, end):
        raw = match.group(0)
        tokens.append(
            ContextToken(
                normalized=raw.lower(),
                start=match.start(),
                end=match.end(),
                text=raw,
                protected=False,
            )
        )
    return tokens


def _overlaps(left: Span, right: Span) -> bool:
    return left.start < right.end and right.start < left.end


__all__ = [
    "ContextToken",
    "PROTECTED_SENTINEL",
    "TEXT_CONTRACT_VERSION",
    "TOKENIZER_VERSION",
    "Span",
    "TextAnalysis",
    "WORD_RE",
    "analyze_text",
    "find_protected_spans",
]
