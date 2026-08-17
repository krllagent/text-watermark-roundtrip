"""Stdlib-only text transformations for the Stage-1 toy experiment.

The transformation API deliberately has no marker key, density, or encoder
lexicon arguments. It can only see the text that it is asked to rewrite.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import math
import os
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from text_contract import Span, find_protected_spans


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3.5-9b"
SUPPORTED_METHODS = frozenset(
    {"none", "synonyms", "roundtrip", "paraphrase", "paraphrase-verified"}
)
SUPPORTED_PIVOTS = frozenset({"de", "zh"})
SUPPORTED_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)

_PLACEHOLDER_RE = re.compile(r"⟦T([1-9][0-9]*)⟧")
_BRACKET_TOKEN_RE = re.compile(r"⟦[^\n⟦⟧]*⟧")
_PLACEHOLDER_VARIANT_RE = re.compile(
    r"(?P<open>⟦|\[)\s*T(?P<number>[1-9][0-9]*)\s*(?P<close>⟧|\])"
)
_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")
_GERMAN_MARKERS = frozenset(
    {
        "aber",
        "als",
        "auf",
        "das",
        "dem",
        "den",
        "der",
        "des",
        "die",
        "eine",
        "einem",
        "einen",
        "einer",
        "eines",
        "für",
        "ich",
        "im",
        "ist",
        "mit",
        "nicht",
        "oder",
        "und",
        "von",
        "wir",
        "zu",
    }
)


class TransformationError(Exception):
    """Base class for safe, expected transformation failures."""


class ConfigurationError(TransformationError):
    """Raised before a request when provider configuration is invalid."""


class PlaceholderError(TransformationError):
    """Raised when protected placeholders are lost, copied, or invented."""


class ValidationError(TransformationError):
    """Raised when a model output violates the frozen text contract."""


class ProviderError(TransformationError):
    """Raised for sanitized provider transport or response failures."""


class ProviderResponseError(ProviderError):
    """A provider response was obtained but violated the response contract."""

    def __init__(self, message: str, *, raw_response: Mapping[str, Any]) -> None:
        super().__init__(message)
        normalized = json_safe_value(raw_response)
        if not isinstance(normalized, dict):
            raise TypeError("raw_response must normalize to an object")
        self.raw_response: dict[str, Any] = normalized


@dataclass(frozen=True)
class ProtectedToken:
    placeholder: str
    original: str
    start: int
    end: int


@dataclass(frozen=True)
class ProtectedText:
    masked: str
    tokens: tuple[ProtectedToken, ...]


@dataclass(frozen=True)
class CompletionUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: Decimal

    def to_dict(self) -> dict[str, object]:
        return {
            "completionTokens": self.completion_tokens,
            "providerCostCredits": str(self.cost),
            "promptTokens": self.prompt_tokens,
            "totalTokens": self.total_tokens,
        }


@dataclass(frozen=True)
class ChatCompletion:
    content: str
    finish_reason: str
    model: str
    openrouter_metadata: Mapping[str, Any] | None
    provider: str
    response_id: str
    system_fingerprint: str | None
    usage: CompletionUsage

    def to_dict(self) -> dict[str, object]:
        return {
            "content": self.content,
            "finishReason": self.finish_reason,
            "id": self.response_id,
            "model": self.model,
            "openrouterMetadata": (
                None if self.openrouter_metadata is None else dict(self.openrouter_metadata)
            ),
            "provider": self.provider,
            "systemFingerprint": self.system_fingerprint,
            "usage": self.usage.to_dict(),
        }


@dataclass(frozen=True)
class TransformCall:
    stage: str
    prompt: str
    completion: ChatCompletion


@dataclass(frozen=True)
class TransformationResult:
    method: str
    pivot: str | None
    text: str
    calls: tuple[TransformCall, ...]


Transport = Callable[[str, dict[str, str], bytes, float], Mapping[str, Any]]


def protect_tokens(text: str) -> ProtectedText:
    """Mask protected unions and existing bracket tokens deterministically."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    spans = list(find_protected_spans(text))
    spans.extend(Span(match.start(), match.end()) for match in _BRACKET_TOKEN_RE.finditer(text))
    spans.extend(
        Span(match.start(), match.end())
        for match in _PLACEHOLDER_VARIANT_RE.finditer(text)
        if _matching_placeholder_brackets(match)
    )
    merged = _merge_spans(spans)
    occupied_numbers = {int(match.group(1)) for match in _PLACEHOLDER_RE.finditer(text)}

    tokens: list[ProtectedToken] = []
    next_number = 1
    for span in merged:
        while next_number in occupied_numbers:
            next_number += 1
        placeholder = f"⟦T{next_number}⟧"
        occupied_numbers.add(next_number)
        tokens.append(
            ProtectedToken(
                placeholder=placeholder,
                original=text[span.start : span.end],
                start=span.start,
                end=span.end,
            )
        )
        next_number += 1

    masked = text
    for token in reversed(tokens):
        masked = masked[: token.start] + token.placeholder + masked[token.end :]
    return ProtectedText(masked=masked, tokens=tuple(tokens))


def canonicalize_placeholders(
    text: str,
    tokens: Sequence[ProtectedToken],
) -> str:
    """Normalize only obvious variants of known placeholders, then validate.

    Models occasionally rewrite ``⟦T1⟧`` as ``[T1]`` or add whitespace inside
    the brackets. Source text with those shapes is masked by :func:`protect_tokens`,
    so an alias that remains in model output must refer to the current protected map.
    Unknown, duplicated, missing, or reordered aliases remain hard failures.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    expected = {token.placeholder for token in tokens}
    replacements: list[tuple[int, int, str]] = []
    for match in _PLACEHOLDER_VARIANT_RE.finditer(text):
        if not _matching_placeholder_brackets(match):
            continue
        canonical = f"⟦T{match.group('number')}⟧"
        raw = match.group(0)
        if raw == canonical:
            continue
        if canonical not in expected:
            raise PlaceholderError(f"unknown placeholder variant: {raw}")
        replacements.append((match.start(), match.end(), canonical))

    normalized = text
    for start, end, canonical in reversed(replacements):
        normalized = normalized[:start] + canonical + normalized[end:]
    validate_placeholders(normalized, tokens)
    return normalized


def validate_placeholders(text: str, tokens: Sequence[ProtectedToken]) -> None:
    """Require each known placeholder exactly once and reject invented ones."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    expected = [token.placeholder for token in tokens]
    if len(expected) != len(set(expected)):
        raise PlaceholderError("placeholder map contains duplicate keys")

    found = [match.group(0) for match in _BRACKET_TOKEN_RE.finditer(text)]
    for placeholder in expected:
        count = found.count(placeholder)
        if count == 0:
            raise PlaceholderError(f"missing placeholder: {placeholder}")
        if count > 1:
            raise PlaceholderError(f"duplicated placeholder: {placeholder}")

    unknown = sorted(set(found) - set(expected))
    if unknown:
        raise PlaceholderError(f"unknown placeholder: {unknown[0]}")
    if found != expected:
        raise PlaceholderError("protected placeholders were reordered")


def _matching_placeholder_brackets(match: re.Match[str]) -> bool:
    return (match.group("open"), match.group("close")) in {
        ("⟦", "⟧"),
        ("[", "]"),
    }


def restore_tokens(text: str, tokens: Sequence[ProtectedToken]) -> str:
    """Restore a validated model output without normalizing original bytes."""
    validate_placeholders(text, tokens)
    restored = text
    for token in tokens:
        restored = restored.replace(token.placeholder, token.original)
    return restored


def build_synonym_prompt(masked: str) -> str:
    _require_nonempty_text(masked, "masked text")
    return _prompt(
        "Rewrite the English text by replacing only a limited number of content words "
        "with natural, context-appropriate synonyms. Keep the edit light.",
        masked,
    )


def build_paraphrase_prompt(masked: str) -> str:
    _require_nonempty_text(masked, "masked text")
    return _prompt(
        "Fully rephrase the English text in natural English. Change the wording and "
        "sentence construction while keeping the complete meaning.",
        masked,
    )


def build_fidelity_repair_prompt(source_masked: str, draft_masked: str) -> str:
    """Build the mandatory source-grounded second pass for verified paraphrase."""
    _require_nonempty_text(source_masked, "authoritative source")
    _require_nonempty_text(draft_masked, "draft to repair")
    instruction = (
        "Repair the draft against the authoritative source. Restore every missing or "
        "changed claim, caveat, example, named entity, number, author stance, degree of "
        "certainty, negation, scope, causal direction, paragraph role, and paragraph "
        "order. Remove anything the source does not support. Keep the final wording and "
        "sentence construction materially different from the source: do not copy source "
        "phrases merely to make the comparison easier. Preserve every placeholder from "
        "the source exactly once and in its relevant position. Return only the repaired "
        "English text."
    )
    return (
        f"{instruction}\n\n"
        "--- BEGIN AUTHORITATIVE SOURCE ---\n"
        f"{source_masked}\n"
        "--- END AUTHORITATIVE SOURCE ---\n\n"
        "--- BEGIN DRAFT TO REPAIR ---\n"
        f"{draft_masked}\n"
        "--- END DRAFT TO REPAIR ---"
    )


def build_forward_prompt(masked: str, pivot: str) -> str:
    _require_nonempty_text(masked, "masked text")
    language = _pivot_language(pivot)
    return _prompt(
        f"Translate the complete English text into {language}. Preserve its full meaning "
        "without shortening, adding, or rearranging it.",
        masked,
    )


def build_backward_prompt(pivot_text: str, pivot: str) -> str:
    _require_nonempty_text(pivot_text, "pivot text")
    language = _pivot_language(pivot)
    return _prompt(
        f"Read the complete {language} text and express all of its meaning anew in natural "
        "English. Do not translate phrase by phrase, and do not mention translation. Keep "
        "approximately the same length.",
        pivot_text,
    )


def validate_intermediate(
    pivot_text: str,
    pivot: str,
    tokens: Sequence[ProtectedToken] = (),
) -> None:
    """Reject empty, wrong-language, or placeholder-corrupt pivot output."""
    _require_nonempty_text(pivot_text, "pivot text", ValidationError)
    _pivot_language(pivot, ValidationError)
    validate_placeholders(pivot_text, tokens)

    without_placeholders = _PLACEHOLDER_RE.sub(" ", pivot_text)
    if pivot == "de" and not _looks_german(without_placeholders):
        raise ValidationError("pivot text does not pass the German language heuristic")
    if pivot == "zh" and not _looks_chinese(without_placeholders):
        raise ValidationError("pivot text does not pass the Chinese language heuristic")


def result_validation_issues(
    original: str,
    result: str,
    pivot: str | None,
) -> tuple[dict[str, str], ...]:
    """Return every observable final-result contract issue in stable order."""
    _require_nonempty_text(original, "original text", ValidationError)
    if not isinstance(result, str):
        raise ValidationError("result text must be a string")
    if pivot is not None:
        _pivot_language(pivot, ValidationError)

    issues: list[dict[str, str]] = []
    if not result.strip():
        issues.append(
            {
                "code": "empty_output",
                "message": "result text must be nonempty",
            }
        )
    length_ratio = len(result) / len(original)
    if not 0.6 <= length_ratio <= 1.6:
        issues.append(
            {
                "code": "length_contract",
                "message": "result length must be between 0.6x and 1.6x the original",
            }
        )

    original_paragraphs = _paragraph_count(original)
    result_paragraphs = _paragraph_count(result)
    paragraph_ratio = result_paragraphs / original_paragraphs
    if not 0.7 <= paragraph_ratio <= 1.3:
        issues.append(
            {
                "code": "paragraph_contract",
                "message": "result paragraph count must stay within 30% of the original",
            }
        )

    if result == original:
        issues.append(
            {
                "code": "unchanged_output",
                "message": "result is byte-identical to the original",
            }
        )

    without_placeholders = _PLACEHOLDER_RE.sub(" ", result)
    if pivot == "de" and _looks_german(without_placeholders):
        issues.append(
            {
                "code": "pivot_language_contract",
                "message": "result still appears to contain German pivot prose",
            }
        )
    if pivot == "zh" and _contains_cjk(without_placeholders):
        issues.append(
            {
                "code": "pivot_language_contract",
                "message": "result still contains Chinese pivot characters",
            }
        )
    return tuple(issues)


def validate_result(original: str, result: str, pivot: str | None) -> None:
    """Validate a still-masked final result before protected-token restore."""
    issues = result_validation_issues(original, result, pivot)
    if issues:
        raise ValidationError(issues[0]["message"])


class OpenRouterClient:
    """Small injectable client with one preselected endpoint and no fallback."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        transport: Transport | None = None,
        timeout: float = 60.0,
        provider_order: Sequence[str] = (),
        allow_fallbacks: bool = False,
        require_parameters: bool = True,
        reasoning_effort: str = "none",
        temperature: float | None = 0.0,
        max_tokens: int = 4_096,
        seed: int | None = None,
        max_prompt_price: float | None = None,
        max_completion_price: float | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ConfigurationError("OPENROUTER_API_KEY must be nonempty")
        if "\r" in api_key or "\n" in api_key:
            raise ConfigurationError("OPENROUTER_API_KEY contains invalid characters")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ConfigurationError("timeout must be a positive number")
        if isinstance(provider_order, (str, bytes)) or not isinstance(
            provider_order, Sequence
        ):
            raise ConfigurationError("provider_order must be a sequence of provider slugs")
        normalized_providers: list[str] = []
        for provider in provider_order:
            if (
                not isinstance(provider, str)
                or not provider.strip()
                or provider != provider.strip()
                or any(character in provider for character in ("\r", "\n"))
            ):
                raise ConfigurationError("provider_order contains an invalid provider slug")
            normalized_providers.append(provider)
        if len(normalized_providers) != len(set(normalized_providers)):
            raise ConfigurationError("provider_order contains duplicate provider slugs")
        if not isinstance(allow_fallbacks, bool):
            raise ConfigurationError("allow_fallbacks must be boolean")
        if not isinstance(require_parameters, bool):
            raise ConfigurationError("require_parameters must be boolean")
        if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
            raise ConfigurationError(
                "reasoning_effort must be one of none, minimal, low, medium, high, "
                "xhigh, or max"
            )
        if temperature is not None and (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
        ):
            raise ConfigurationError("temperature must be numeric or null")
        if temperature is not None and not 0 <= temperature <= 2:
            raise ConfigurationError("temperature must be between 0 and 2")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ConfigurationError("max_tokens must be a positive integer")
        if seed is not None and (
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        ):
            raise ConfigurationError("seed must be a nonnegative integer or null")
        if (max_prompt_price is None) != (max_completion_price is None):
            raise ConfigurationError("both max price fields must be set together")
        for label, price in (
            ("max_prompt_price", max_prompt_price),
            ("max_completion_price", max_completion_price),
        ):
            if price is not None and (
                not isinstance(price, (int, float))
                or isinstance(price, bool)
                or price < 0
            ):
                raise ConfigurationError(f"{label} must be a nonnegative number or null")
        normalized_response_format: dict[str, Any] | None = None
        if response_format is not None:
            if not isinstance(response_format, Mapping) or not response_format:
                raise ConfigurationError("response_format must be a nonempty object or null")
            normalized = json_safe_value(response_format)
            if not isinstance(normalized, dict):
                raise ConfigurationError("response_format must normalize to an object")
            normalized_response_format = normalized

        self._api_key = api_key
        self.endpoint = resolve_chat_completions_url(base_url)
        self._transport = transport or _urlopen_transport
        self.timeout = float(timeout)
        self.provider_order = tuple(normalized_providers)
        self.allow_fallbacks = allow_fallbacks
        self.require_parameters = require_parameters
        self.reasoning_effort = reasoning_effort
        self.temperature = None if temperature is None else float(temperature)
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_prompt_price = (
            None if max_prompt_price is None else float(max_prompt_price)
        )
        self.max_completion_price = (
            None if max_completion_price is None else float(max_completion_price)
        )
        self.response_format = normalized_response_format

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        transport: Transport | None = None,
        timeout: float = 60.0,
        provider_order: Sequence[str] = (),
        allow_fallbacks: bool = False,
        require_parameters: bool = True,
        reasoning_effort: str = "none",
        temperature: float | None = 0.0,
        max_tokens: int = 4_096,
        seed: int | None = None,
        max_prompt_price: float | None = None,
        max_completion_price: float | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> "OpenRouterClient":
        source = os.environ if environ is None else environ
        api_key = source.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ConfigurationError("OPENROUTER_API_KEY is required")
        base_url = source.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)
        return cls(
            api_key,
            base_url=base_url,
            transport=transport,
            timeout=timeout,
            provider_order=provider_order,
            allow_fallbacks=allow_fallbacks,
            require_parameters=require_parameters,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            max_prompt_price=max_prompt_price,
            max_completion_price=max_completion_price,
            response_format=response_format,
        )

    def complete(self, prompt: str, *, model: str) -> ChatCompletion:
        _require_nonempty_text(prompt, "prompt")
        _require_nonempty_text(model, "model")
        payload = {
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "model": model,
            "provider": {
                "allow_fallbacks": self.allow_fallbacks,
                "data_collection": "deny",
                "require_parameters": self.require_parameters,
                "zdr": True,
            },
            "reasoning": {"effort": self.reasoning_effort},
            "stream": False,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.provider_order:
            payload["provider"]["order"] = list(self.provider_order)
        if self.max_prompt_price is not None:
            assert self.max_completion_price is not None
            payload["provider"]["max_price"] = {
                "completion": self.max_completion_price,
                "prompt": self.max_prompt_price,
            }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.response_format is not None:
            payload["response_format"] = dict(self.response_format)
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Metadata": "enabled",
        }
        try:
            raw_response = self._transport(self.endpoint, headers, body, self.timeout)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("OpenRouter request failed") from exc
        try:
            return _parse_completion(raw_response)
        except ProviderResponseError:
            raise
        except ProviderError as exc:
            raise ProviderResponseError(
                str(exc),
                raw_response=raw_response,
            ) from exc


def resolve_chat_completions_url(base_url: str) -> str:
    """Resolve provider-root and OpenAI-compatible `/api/v1` base shapes."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise ConfigurationError("OPENROUTER_BASE_URL must be nonempty")
    parsed = urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError("OPENROUTER_BASE_URL must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ConfigurationError("OPENROUTER_BASE_URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ConfigurationError("OPENROUTER_BASE_URL must not contain query or fragment")

    path = parsed.path.rstrip("/")
    if path.endswith("/api/v1"):
        endpoint_path = f"{path}/chat/completions"
    elif "/api/v1/" in path:
        raise ConfigurationError("OPENROUTER_BASE_URL must be a provider root or end in /api/v1")
    else:
        endpoint_path = f"{path}/api/v1/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, endpoint_path, "", ""))


def transform_text(
    text: str,
    *,
    method: str,
    client: OpenRouterClient | None = None,
    pivot: str | None = None,
    model_forward: str | None = None,
    model_backward: str | None = None,
) -> TransformationResult:
    """Run exactly the call graph belonging to one frozen experiment method."""
    _require_nonempty_text(text, "text")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported method: {method!r}")
    if method == "roundtrip":
        if pivot not in SUPPORTED_PIVOTS:
            raise ValueError("roundtrip requires pivot='de' or pivot='zh'")
    elif pivot is not None:
        raise ValueError("pivot is only valid for the roundtrip method")

    if method == "none":
        return TransformationResult(method=method, pivot=None, text=text, calls=())
    if client is None:
        raise ConfigurationError("a configured OpenRouterClient is required")

    forward_model = (
        model_forward
        if model_forward is not None
        else os.environ.get("UNMARK_MODEL_FORWARD", DEFAULT_MODEL)
    )
    _require_nonempty_text(forward_model, "forward model", ConfigurationError)

    protected = protect_tokens(text)
    calls: list[TransformCall] = []

    if method == "synonyms":
        prompt = build_synonym_prompt(protected.masked)
        completion = client.complete(prompt, model=forward_model)
        _require_stop_completion(completion)
        calls.append(TransformCall(stage="synonyms", prompt=prompt, completion=completion))
        final_masked = completion.content
    elif method == "paraphrase":
        prompt = build_paraphrase_prompt(protected.masked)
        completion = client.complete(prompt, model=forward_model)
        _require_stop_completion(completion)
        calls.append(TransformCall(stage="paraphrase", prompt=prompt, completion=completion))
        final_masked = completion.content
    elif method == "paraphrase-verified":
        draft_prompt = build_paraphrase_prompt(protected.masked)
        draft_completion = client.complete(draft_prompt, model=forward_model)
        _require_stop_completion(draft_completion)
        calls.append(
            TransformCall(
                stage="paraphrase-draft",
                prompt=draft_prompt,
                completion=draft_completion,
            )
        )
        repair_prompt = build_fidelity_repair_prompt(
            protected.masked,
            draft_completion.content,
        )
        repair_completion = client.complete(repair_prompt, model=forward_model)
        _require_stop_completion(repair_completion)
        calls.append(
            TransformCall(
                stage="fidelity-repair",
                prompt=repair_prompt,
                completion=repair_completion,
            )
        )
        final_masked = repair_completion.content
    else:
        assert pivot is not None
        backward_model = (
            model_backward
            if model_backward is not None
            else os.environ.get("UNMARK_MODEL_BACKWARD", DEFAULT_MODEL)
        )
        _require_nonempty_text(backward_model, "backward model", ConfigurationError)
        forward_prompt = build_forward_prompt(protected.masked, pivot)
        forward_completion = client.complete(forward_prompt, model=forward_model)
        _require_stop_completion(forward_completion)
        calls.append(
            TransformCall(
                stage=f"forward-{pivot}",
                prompt=forward_prompt,
                completion=forward_completion,
            )
        )
        forward_masked = canonicalize_placeholders(
            forward_completion.content,
            protected.tokens,
        )
        validate_intermediate(forward_masked, pivot, protected.tokens)

        backward_prompt = build_backward_prompt(forward_masked, pivot)
        backward_completion = client.complete(backward_prompt, model=backward_model)
        _require_stop_completion(backward_completion)
        calls.append(
            TransformCall(
                stage=f"backward-{pivot}",
                prompt=backward_prompt,
                completion=backward_completion,
            )
        )
        final_masked = backward_completion.content

    final_masked = canonicalize_placeholders(final_masked, protected.tokens)
    validate_result(protected.masked, final_masked, pivot)
    restored = restore_tokens(final_masked, protected.tokens)
    return TransformationResult(
        method=method,
        pivot=pivot,
        text=restored,
        calls=tuple(calls),
    )


def _require_stop_completion(completion: ChatCompletion) -> None:
    """Keep the product helper fail-closed while experiments retain paid partials."""
    if completion.finish_reason != "stop":
        raise ProviderError(
            f"OpenRouter response finish reason must be stop, got "
            f"{completion.finish_reason!r}"
        )


def _prompt(instruction: str, text: str) -> str:
    fidelity = (
        "Preserve every claim, caveat, example, author stance, degree of certainty, and "
        "the exact number of paragraphs, paragraph order, and each paragraph's role. Preserve "
        "every placeholder exactly once and in its relevant "
        "position. Do not summarize, omit, add facts, improve the argument, or add a "
        "preface. Return only the transformed text."
    )
    return f"{instruction}\n\n{fidelity}\n\n--- BEGIN TEXT ---\n{text}\n--- END TEXT ---"


def _pivot_language(
    pivot: str,
    error_type: type[Exception] = ValueError,
) -> str:
    if pivot == "de":
        return "German"
    if pivot == "zh":
        return "Simplified Chinese"
    raise error_type("pivot must be 'de' or 'zh'")


def _merge_spans(spans: Sequence[Span]) -> tuple[Span, ...]:
    if not spans:
        return ()
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    merged: list[Span] = [ordered[0]]
    for span in ordered[1:]:
        previous = merged[-1]
        if span.start < previous.end:
            merged[-1] = Span(previous.start, max(previous.end, span.end))
        else:
            merged.append(span)
    return tuple(merged)


def _looks_german(text: str) -> bool:
    words = [word.lower() for word in _WORD_RE.findall(text)]
    if not words:
        return False
    hits = [word for word in words if word in _GERMAN_MARKERS]
    return len(hits) >= 3 and len(set(hits)) >= 2 and len(hits) / len(words) >= 0.08


def _contains_cjk(text: str) -> bool:
    return any("\u3400" <= character <= "\u9fff" for character in text)


def _looks_chinese(text: str) -> bool:
    letters = [character for character in text if character.isalpha()]
    cjk_count = sum("\u3400" <= character <= "\u9fff" for character in letters)
    return cjk_count >= 4 and cjk_count / max(1, len(letters)) >= 0.2


def _paragraph_count(text: str) -> int:
    return len(re.split(r"\n\s*\n", text.strip())) if text.strip() else 0


def _require_nonempty_text(
    value: object,
    label: str,
    error_type: type[Exception] = ValueError,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{label} must be a nonempty string")


def _urlopen_transport(
    url: str,
    headers: dict[str, str],
    body: bytes,
    timeout: float,
) -> Mapping[str, Any]:
    request = Request(url, data=body, headers=headers, method="POST")
    opener = build_opener(_RejectRedirectHandler())
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.geturl() != url:
                raise ProviderError("OpenRouter redirected the configured endpoint")
            raw = response.read(5 * 1024 * 1024 + 1)
    except HTTPError as exc:
        raise ProviderError(f"OpenRouter request failed with HTTP {exc.code}") from exc
    except (URLError, OSError) as exc:
        raise ProviderError("OpenRouter request failed") from exc
    if len(raw) > 5 * 1024 * 1024:
        raise ProviderError("OpenRouter response exceeded 5 MiB")
    try:
        decoded = json.loads(raw.decode("utf-8"), parse_float=Decimal)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderError("OpenRouter returned invalid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ProviderError("OpenRouter returned an invalid response object")
    return decoded


def _parse_completion(raw: Mapping[str, Any]) -> ChatCompletion:
    try:
        response_id = raw["id"]
        model = raw["model"]
        choices = raw["choices"]
        choice = choices[0]
        message = choice["message"]
        content = message["content"]
        finish_reason = choice["finish_reason"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("OpenRouter response is missing completion fields") from exc

    if not isinstance(response_id, str) or not response_id:
        raise ProviderError("OpenRouter response has an invalid id")
    if not isinstance(model, str) or not model:
        raise ProviderError("OpenRouter response has an invalid model")
    if not isinstance(content, str):
        raise ProviderError("OpenRouter response has invalid message content")
    if not isinstance(finish_reason, str) or not finish_reason:
        raise ProviderError("OpenRouter response has an invalid finish_reason")
    provider = raw.get("provider")
    if not isinstance(provider, str) or not provider:
        raise ProviderError("OpenRouter response has an invalid provider")
    system_fingerprint = raw.get("system_fingerprint")
    if system_fingerprint is not None and (
        not isinstance(system_fingerprint, str) or not system_fingerprint
    ):
        raise ProviderError("OpenRouter response has an invalid system_fingerprint")

    raw_usage = raw.get("usage")
    if not isinstance(raw_usage, Mapping):
        raise ProviderError("OpenRouter response has invalid usage")
    prompt_tokens = _required_nonnegative_int(
        raw_usage.get("prompt_tokens"), "prompt_tokens"
    )
    completion_tokens = _required_nonnegative_int(
        raw_usage.get("completion_tokens"), "completion_tokens"
    )
    total_tokens = _required_nonnegative_int(
        raw_usage.get("total_tokens"), "total_tokens"
    )
    if total_tokens != prompt_tokens + completion_tokens:
        raise ProviderError("OpenRouter response has inconsistent usage.total_tokens")
    usage = CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost=_required_nonnegative_decimal(raw_usage.get("cost"), "cost"),
    )
    raw_metadata = raw.get("openrouter_metadata")
    if raw_metadata is not None and not isinstance(raw_metadata, Mapping):
        raise ProviderError("OpenRouter response has invalid openrouter_metadata")
    openrouter_metadata = (
        None if raw_metadata is None else json_safe_value(raw_metadata)
    )
    if openrouter_metadata is not None and not isinstance(openrouter_metadata, dict):
        raise ProviderError("OpenRouter response has invalid openrouter_metadata")
    return ChatCompletion(
        content=content,
        finish_reason=finish_reason,
        model=model,
        openrouter_metadata=openrouter_metadata,
        provider=provider,
        response_id=response_id,
        system_fingerprint=system_fingerprint,
        usage=usage,
    )


def _required_nonnegative_int(value: object, label: str) -> int:
    if value is None:
        raise ProviderError(f"OpenRouter response is missing usage.{label}")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProviderError(f"OpenRouter response has invalid usage.{label}")
    return value


def _required_nonnegative_decimal(value: object, label: str) -> Decimal:
    if value is None:
        raise ProviderError(f"OpenRouter response is missing usage.{label}")
    if isinstance(value, bool):
        raise ProviderError(f"OpenRouter response has invalid usage.{label}")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ProviderError(f"OpenRouter response has invalid usage.{label}") from exc
    if not parsed.is_finite() or parsed < 0:
        raise ProviderError(f"OpenRouter response has invalid usage.{label}")
    return parsed


def json_safe_value(value: Any) -> Any:
    """Recursively normalize provider JSON while preserving decimal text exactly."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ProviderError("OpenRouter response contains a non-finite decimal")
        return format(value, "f")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProviderError("OpenRouter response contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProviderError("OpenRouter response contains a non-string object key")
            normalized[key] = json_safe_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    raise ProviderError("OpenRouter response contains a non-JSON value")


class _RejectRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


__all__ = [
    "ChatCompletion",
    "CompletionUsage",
    "ConfigurationError",
    "DEFAULT_MODEL",
    "DEFAULT_OPENROUTER_BASE_URL",
    "OpenRouterClient",
    "PlaceholderError",
    "ProtectedText",
    "ProtectedToken",
    "ProviderError",
    "ProviderResponseError",
    "SUPPORTED_METHODS",
    "SUPPORTED_PIVOTS",
    "TransformCall",
    "TransformationError",
    "TransformationResult",
    "ValidationError",
    "build_backward_prompt",
    "build_fidelity_repair_prompt",
    "build_forward_prompt",
    "build_paraphrase_prompt",
    "build_synonym_prompt",
    "canonicalize_placeholders",
    "json_safe_value",
    "protect_tokens",
    "resolve_chat_completions_url",
    "result_validation_issues",
    "restore_tokens",
    "transform_text",
    "validate_intermediate",
    "validate_placeholders",
    "validate_result",
]
