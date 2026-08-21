"""Small paid smoke test for four SynthID removal methods.

The runner uses marked documents from the valid GPU corpus, checkpoints every
paid OpenRouter response before moving on, and scores the transformed text with
the exact sampling table stored in that corpus.  It is intentionally sequential
and stops before any call whose conservative cost ceiling would exceed the
explicit run budget.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from run_experiment import fidelity_metrics
from run_model_canary_luna import atomic_write
import unmark


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "results" / "synthid-corpus-gpu-v2.json"
DEFAULT_OUTPUT = ROOT / "results" / "synthid-smoke-v1.json"
DEFAULT_CHECKPOINT = ROOT / "results" / "synthid-smoke-checkpoint-v1.json"
DEFAULT_MODEL = "qwen/qwen3.7-plus"
DEFAULT_DOCUMENT_IDS = ("doc-01", "doc-04")
DEFAULT_BUDGET_USD = Decimal("0.50")
DEFAULT_MAX_TOKENS = 2_500
DEFAULT_THRESHOLD = 0.5067

METHOD_SPECS = (
    ("synonyms", "synonyms", None, ("synonyms",)),
    ("paraphrase", "paraphrase", None, ("paraphrase",)),
    ("roundtrip-de", "roundtrip", "de", ("forward-de", "backward-de")),
    ("roundtrip-zh", "roundtrip", "zh", ("forward-zh", "backward-zh")),
)


class SmokeError(Exception):
    """Base class for fail-closed smoke-run errors."""


class BudgetExceeded(SmokeError):
    """Raised before a paid call whose ceiling would exceed the run budget."""


class UncertainInFlight(SmokeError):
    """Raised when a prior call may have been billed but was not checkpointed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _evidence_sources() -> list[dict[str, str]]:
    return [
        {
            "title": "SynthID-Text: watermarking large language model output",
            "url": "https://doi.org/10.1038/s41586-024-08025-4",
        },
        {
            "title": "Transformers SynthID text reference implementation",
            "url": "https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/generation/logits_process.py",
        },
        {
            "title": "OpenRouter models API",
            "url": "https://openrouter.ai/api/v1/models",
        },
    ]


def new_checkpoint_state(
    *,
    corpus_sha256: str,
    model: str,
    budget_usd: Decimal,
) -> dict[str, object]:
    now = utc_now()
    return {
        "budgetUsd": format(budget_usd, "f"),
        "calls": {},
        "corpusSha256": corpus_sha256,
        "createdAt": now,
        "inFlight": None,
        "methodology": (
            "Sequential paid-call checkpoint for a two-document SynthID smoke test. "
            "Each request is written as in-flight before dispatch and each response, "
            "including provider usage cost, is atomically stored before another call."
        ),
        "model": model,
        "schemaVersion": 1,
        "sources": _evidence_sources(),
        "status": "running",
        "totalCostUsd": "0",
        "transformations": {},
        "verifiedAt": now,
    }


def checkpoint_spend(state: Mapping[str, object]) -> Decimal:
    calls = state.get("calls")
    if not isinstance(calls, Mapping):
        raise SmokeError("checkpoint calls must be an object")
    total = Decimal(0)
    for record in calls.values():
        if not isinstance(record, Mapping):
            raise SmokeError("checkpoint call record must be an object")
        completion = record.get("completion")
        if not isinstance(completion, Mapping):
            raise SmokeError("checkpoint completion must be an object")
        usage = completion.get("usage")
        if not isinstance(usage, Mapping):
            raise SmokeError("checkpoint usage must be an object")
        try:
            total += Decimal(str(usage["providerCostCredits"]))
        except (KeyError, InvalidOperation, ValueError) as error:
            raise SmokeError("checkpoint contains invalid provider cost") from error
    return total


def call_cost_ceiling(
    *,
    request_bytes: int,
    max_tokens: int,
    prompt_usd_per_token: Decimal,
    completion_usd_per_token: Decimal,
) -> Decimal:
    if request_bytes < 0 or max_tokens <= 0:
        raise ValueError("request_bytes must be nonnegative and max_tokens positive")
    # A tokenizer cannot emit more tokens than there are UTF-8 bytes.  Treating every
    # byte as a prompt token is therefore a deliberately conservative upper bound.
    return (
        Decimal(request_bytes) * prompt_usd_per_token
        + Decimal(max_tokens) * completion_usd_per_token
    )


def _completion_from_dict(value: Mapping[str, object]) -> unmark.ChatCompletion:
    usage = value.get("usage")
    if not isinstance(usage, Mapping):
        raise SmokeError("cached completion usage is invalid")
    metadata = value.get("openrouterMetadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise SmokeError("cached OpenRouter metadata is invalid")
    return unmark.ChatCompletion(
        content=str(value["content"]),
        finish_reason=str(value["finishReason"]),
        model=str(value["model"]),
        openrouter_metadata=None if metadata is None else dict(metadata),
        provider=str(value["provider"]),
        response_id=str(value["id"]),
        system_fingerprint=(
            None
            if value.get("systemFingerprint") is None
            else str(value["systemFingerprint"])
        ),
        usage=unmark.CompletionUsage(
            prompt_tokens=int(usage["promptTokens"]),
            completion_tokens=int(usage["completionTokens"]),
            total_tokens=int(usage["totalTokens"]),
            cost=Decimal(str(usage["providerCostCredits"])),
            cached_prompt_tokens=int(
                (usage.get("promptTokenDetails") or {}).get("cachedTokens", 0)
            ),
            cache_write_tokens=int(
                (usage.get("promptTokenDetails") or {}).get("cacheWriteTokens", 0)
            ),
        ),
    )


class CheckpointedClient:
    """A transform_text-compatible client with budget and paid-call durability."""

    def __init__(
        self,
        *,
        delegate: object,
        checkpoint_path: Path,
        state: dict[str, object],
        budget_usd: Decimal,
        prompt_usd_per_token: Decimal,
        completion_usd_per_token: Decimal,
    ) -> None:
        self.delegate = delegate
        self.checkpoint_path = checkpoint_path
        self.state = state
        self.budget_usd = budget_usd
        self.prompt_usd_per_token = prompt_usd_per_token
        self.completion_usd_per_token = completion_usd_per_token
        self.max_tokens = int(getattr(delegate, "max_tokens"))
        self._document_id: str | None = None
        self._method_id: str | None = None
        self._stages: tuple[str, ...] = ()
        self._call_index = 0

    def begin_transform(
        self,
        document_id: str,
        method_id: str,
        stages: Sequence[str],
    ) -> None:
        self._document_id = document_id
        self._method_id = method_id
        self._stages = tuple(stages)
        self._call_index = 0

    def _current_call_key(self) -> tuple[str, str]:
        if self._document_id is None or self._method_id is None:
            raise SmokeError("begin_transform must be called before complete")
        if self._call_index >= len(self._stages):
            raise SmokeError("transform made more calls than its frozen graph")
        stage = self._stages[self._call_index]
        return f"{self._document_id}::{self._method_id}::{stage}", stage

    def complete(
        self,
        request: unmark.RequestInput,
        *,
        model: str,
        max_tokens: int | None = None,
        response_format: Mapping[str, object] | None = None,
    ) -> unmark.ChatCompletion:
        call_key, stage = self._current_call_key()
        effective_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        request_value = {
            "maxTokens": effective_max_tokens,
            "messages": list(unmark.request_messages(request)),
            "model": model,
            "responseFormat": None if response_format is None else dict(response_format),
        }
        request_hash = _sha256_json(request_value)
        request_bytes = len(
            json.dumps(
                request_value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )

        calls = self.state.get("calls")
        if not isinstance(calls, dict):
            raise SmokeError("checkpoint calls must be mutable")
        cached = calls.get(call_key)
        if cached is not None:
            if not isinstance(cached, Mapping):
                raise SmokeError("cached call record is invalid")
            if cached.get("requestSha256") != request_hash:
                raise SmokeError(f"cached request mismatch for {call_key}")
            completion = cached.get("completion")
            if not isinstance(completion, Mapping):
                raise SmokeError("cached completion is invalid")
            self._call_index += 1
            print(
                json.dumps({"event": "cached_call", "key": call_key}),
                flush=True,
            )
            return _completion_from_dict(completion)

        if self.state.get("inFlight") is not None:
            raise UncertainInFlight(
                "checkpoint has an unresolved in-flight call; refusing a possible retry"
            )
        reserve = call_cost_ceiling(
            request_bytes=request_bytes,
            max_tokens=effective_max_tokens,
            prompt_usd_per_token=self.prompt_usd_per_token,
            completion_usd_per_token=self.completion_usd_per_token,
        )
        spent = checkpoint_spend(self.state)
        if spent + reserve > self.budget_usd:
            raise BudgetExceeded(
                f"{call_key} ceiling {reserve} would exceed budget {self.budget_usd} "
                f"after spend {spent}"
            )
        self.state["inFlight"] = {
            "callKey": call_key,
            "requestSha256": request_hash,
            "reserveCostUsd": format(reserve, "f"),
            "startedAt": utc_now(),
        }
        self.state["verifiedAt"] = utc_now()
        atomic_write(self.checkpoint_path, self.state)
        print(
            json.dumps(
                {
                    "event": "dispatch",
                    "key": call_key,
                    "reserveUsd": format(reserve, "f"),
                    "spentUsd": format(spent, "f"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        completion = self.delegate.complete(
            request,
            model=model,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        calls[call_key] = {
            "completedAt": utc_now(),
            "completion": completion.to_dict(),
            "documentId": self._document_id,
            "method": self._method_id,
            "requestBytes": request_bytes,
            "requestSha256": request_hash,
            "reserveCostUsd": format(reserve, "f"),
            "stage": stage,
        }
        self.state["inFlight"] = None
        self.state["totalCostUsd"] = format(checkpoint_spend(self.state), "f")
        self.state["verifiedAt"] = utc_now()
        atomic_write(self.checkpoint_path, self.state)
        self._call_index += 1
        print(
            json.dumps(
                {
                    "actualCostUsd": format(completion.usage.cost, "f"),
                    "event": "completed_call",
                    "key": call_key,
                    "provider": completion.provider,
                    "totalSpentUsd": self.state["totalCostUsd"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return completion


def _models_url(base_url: str) -> str:
    parsed = urlsplit(base_url.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SmokeError("OPENROUTER_BASE_URL must be an absolute HTTP(S) URL")
    path = parsed.path.rstrip("/")
    endpoint_path = f"{path}/models" if path.endswith("/api/v1") else f"{path}/api/v1/models"
    return urlunsplit((parsed.scheme, parsed.netloc, endpoint_path, "", ""))


def fetch_model_pricing(
    *,
    base_url: str,
    api_key: str,
    model: str,
) -> dict[str, object]:
    request = Request(
        _models_url(base_url),
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with urlopen(request, timeout=60) as response:
        payload = json.load(response, parse_float=Decimal)
    records = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(records, list):
        raise SmokeError("OpenRouter models response is invalid")
    selected = next(
        (record for record in records if isinstance(record, Mapping) and record.get("id") == model),
        None,
    )
    if selected is None:
        raise SmokeError(f"model is absent from OpenRouter models API: {model}")
    pricing = selected.get("pricing")
    if not isinstance(pricing, Mapping):
        raise SmokeError("OpenRouter model pricing is absent")
    try:
        prompt = Decimal(str(pricing["prompt"]))
        completion = Decimal(str(pricing["completion"]))
    except (KeyError, InvalidOperation, ValueError) as error:
        raise SmokeError("OpenRouter model pricing is invalid") from error
    if prompt < 0 or completion < 0:
        raise SmokeError("OpenRouter model pricing must be nonnegative")
    return {
        "completionUsdPerToken": format(completion, "f"),
        "contextLength": selected.get("context_length"),
        "model": model,
        "promptUsdPerToken": format(prompt, "f"),
        "verifiedAt": utc_now(),
    }


class ExplicitNonZdrTransport:
    """Limit non-ZDR routing to this non-sensitive smoke experiment only.

    The shared client intentionally defaults to ZDR and is a frozen protocol input.
    Alibaba is the sole endpoint for the selected model and does not satisfy that
    route flag, so this adapter changes exactly that one provider field while keeping
    data_collection=deny and every other request field untouched.
    """

    def __init__(self, delegate=None) -> None:
        self.delegate = delegate or unmark._urlopen_transport

    def __call__(
        self,
        url: str,
        headers: dict[str, str],
        body: bytes,
        timeout: float,
    ) -> Mapping[str, object]:
        payload = json.loads(body.decode("utf-8"))
        provider = payload.get("provider")
        if not isinstance(provider, dict):
            raise SmokeError("OpenRouter request is missing provider controls")
        if provider.get("data_collection") != "deny" or provider.get("zdr") is not True:
            raise SmokeError("unexpected shared-client privacy controls")
        provider["zdr"] = False
        rewritten = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return self.delegate(url, headers, rewritten, timeout)


class SynthIDDetector:
    """Reference detector bound to the corpus's exact serialized table."""

    def __init__(self, corpus: Mapping[str, object]) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, SynthIDTextWatermarkLogitsProcessor
        except ImportError as error:
            raise SmokeError("run this script with the repository .venv-wm") from error

        self.torch = torch
        self.ngram_len = int(corpus["ngramLen"])
        self.tokenizer = AutoTokenizer.from_pretrained(str(corpus["model"]))
        self.processor = SynthIDTextWatermarkLogitsProcessor(
            ngram_len=self.ngram_len,
            keys=corpus["keys"],
            sampling_table_size=len(corpus["samplingTable"]),
            sampling_table_seed=0,
            context_history_size=int(corpus["contextHistorySize"]),
            device=torch.device("cpu"),
        )
        table = torch.tensor(
            corpus["samplingTable"],
            dtype=self.processor.sampling_table.dtype,
            device=torch.device("cpu"),
        )
        if table.shape != self.processor.sampling_table.shape:
            raise SmokeError("serialized SynthID sampling table has the wrong shape")
        self.processor.sampling_table.copy_(table)

    def score(self, text: str, *, marked_source: str) -> dict[str, object]:
        torch = self.torch
        ids = self.tokenizer(
            [text], return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        source_ids = self.tokenizer(
            [marked_source], return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].tolist()
        if ids.shape[1] <= self.ngram_len:
            return {"meanG": None, "scoredTokens": 0, "tokens": int(ids.shape[1])}
        g_values = self.processor.compute_g_values(ids)[0]
        context_mask = self.processor.compute_context_repetition_mask(ids)[0].bool()
        output_ids = ids[0].tolist()
        source_ngrams = {
            tuple(source_ids[index : index + self.ngram_len])
            for index in range(len(source_ids) - self.ngram_len + 1)
        }
        copied = torch.tensor(
            [
                tuple(output_ids[index : index + self.ngram_len]) in source_ngrams
                for index in range(len(output_ids) - self.ngram_len + 1)
            ],
            dtype=torch.bool,
        )
        reused_mask = context_mask & copied
        novel_mask = context_mask & ~copied
        depth = int(g_values.shape[-1])
        valid_positions = int(context_mask.sum().item())
        reused_positions = int(reused_mask.sum().item())
        novel_positions = int(novel_mask.sum().item())
        g_count = valid_positions * depth
        reused_g_count = reused_positions * depth
        novel_g_count = novel_positions * depth
        g_ones = int(g_values[context_mask].sum().item())
        reused_g_ones = int(g_values[reused_mask].sum().item())
        novel_g_ones = int(g_values[novel_mask].sum().item())

        def ratio(numerator: int, denominator: int) -> float | None:
            return numerator / denominator if denominator else None

        return {
            "exactNgramReuseFraction": ratio(reused_positions, valid_positions),
            "gOneCount": g_ones,
            "gValueCount": g_count,
            "meanG": ratio(g_ones, g_count),
            "novelGOneCount": novel_g_ones,
            "novelGValueCount": novel_g_count,
            "novelMeanG": ratio(novel_g_ones, novel_g_count),
            "novelPositions": novel_positions,
            "reusedGOneCount": reused_g_ones,
            "reusedGValueCount": reused_g_count,
            "reusedMeanG": ratio(reused_g_ones, reused_g_count),
            "reusedPositions": reused_positions,
            "scoredTokens": valid_positions,
            "tokens": int(ids.shape[1]),
        }


def _call_summaries(
    state: Mapping[str, object],
    *,
    document_id: str,
    method_id: str,
    stages: Sequence[str],
) -> list[dict[str, object]]:
    calls = state["calls"]
    assert isinstance(calls, Mapping)
    output = []
    for stage in stages:
        key = f"{document_id}::{method_id}::{stage}"
        record = calls.get(key)
        if not isinstance(record, Mapping):
            continue
        completion = record["completion"]
        assert isinstance(completion, Mapping)
        output.append(
            {
                "callKey": key,
                "finishReason": completion.get("finishReason"),
                "id": completion.get("id"),
                "model": completion.get("model"),
                "provider": completion.get("provider"),
                "requestSha256": record.get("requestSha256"),
                "stage": stage,
                "usage": completion.get("usage"),
            }
        )
    return output


def _last_completion_text(
    state: Mapping[str, object],
    *,
    document_id: str,
    method_id: str,
    stages: Sequence[str],
) -> str | None:
    calls = state["calls"]
    assert isinstance(calls, Mapping)
    for stage in reversed(tuple(stages)):
        record = calls.get(f"{document_id}::{method_id}::{stage}")
        if not isinstance(record, Mapping):
            continue
        completion = record.get("completion")
        if isinstance(completion, Mapping) and isinstance(completion.get("content"), str):
            return str(completion["content"])
    return None


def run_transformation(
    *,
    client: CheckpointedClient,
    detector: SynthIDDetector,
    document: Mapping[str, object],
    method_id: str,
    method: str,
    pivot: str | None,
    stages: Sequence[str],
    model: str,
    threshold: float,
) -> dict[str, object]:
    document_id = str(document["documentId"])
    marked = document["marked"]
    assert isinstance(marked, Mapping)
    source = str(marked["text"])
    client.begin_transform(document_id, method_id, stages)
    issues: list[dict[str, str]] = []
    outcome = "completed"
    try:
        result = unmark.transform_text(
            source,
            method=method,
            client=client,
            pivot=pivot,
            model_forward=model,
            model_backward=model,
        )
        evaluated = result.text
    except (unmark.PlaceholderError, unmark.ValidationError) as error:
        raw = _last_completion_text(
            client.state,
            document_id=document_id,
            method_id=method_id,
            stages=stages,
        )
        if raw is None or (method == "roundtrip" and len(_call_summaries(
            client.state,
            document_id=document_id,
            method_id=method_id,
            stages=stages,
        )) < 2):
            return {
                "calls": _call_summaries(
                    client.state,
                    document_id=document_id,
                    method_id=method_id,
                    stages=stages,
                ),
                "detail": f"{type(error).__name__}: {str(error)[:240]}",
                "documentId": document_id,
                "method": method_id,
                "outcome": "pivot_rejected" if method == "roundtrip" else "validation_failure",
            }
        protected = unmark.protect_tokens(source)
        try:
            normalized = unmark.canonicalize_placeholders(raw, protected.tokens)
            issues.extend(unmark.result_validation_issues(protected.masked, normalized, pivot))
            evaluated = unmark.restore_tokens(normalized, protected.tokens)
        except unmark.PlaceholderError as placeholder_error:
            evaluated = raw
            issues.append(
                {
                    "code": "placeholder_contract",
                    "message": str(placeholder_error)[:240],
                }
            )
        issues.append({"code": "strict_validation", "message": str(error)[:240]})
        outcome = "completed_with_validation_issues"

    source_score = detector.score(source, marked_source=source)
    reported_source_mean = float(marked["meanG"])
    source_delta = abs(float(source_score["meanG"]) - reported_source_mean)
    if source_delta > 1e-12:
        raise SmokeError(
            f"detector/table mismatch for {document_id}: delta={source_delta}"
        )
    output_score = detector.score(evaluated, marked_source=source)
    fidelity = fidelity_metrics(source, evaluated)
    word_distance = float(
        fidelity["wordLevenshtein"]["normalizedDistance"]  # type: ignore[index]
    )
    output_mean = output_score["meanG"]
    removed = output_mean is not None and float(output_mean) < threshold
    return {
        "calls": _call_summaries(
            client.state,
            document_id=document_id,
            method_id=method_id,
            stages=stages,
        ),
        "documentId": document_id,
        "evaluatedOutputText": evaluated,
        "fidelity": fidelity,
        "method": method_id,
        "outcome": outcome,
        "pipelineIssues": issues,
        "publishedThreshold": threshold,
        "removedAtPublishedThreshold": removed,
        "scoreDeltaFromMarkedSource": (
            None if output_mean is None else float(output_mean) - float(source_score["meanG"])
        ),
        "sourceDetector": source_score,
        "sourceText": source,
        "transformedDetector": output_score,
        "wordDistance": word_distance,
    }


def summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for method_id, _, _, _ in METHOD_SPECS:
        method_rows = [
            row
            for row in rows
            if row.get("method") == method_id
            and isinstance(row.get("transformedDetector"), Mapping)
        ]
        means = [
            float(row["transformedDetector"]["meanG"])  # type: ignore[index]
            for row in method_rows
            if row["transformedDetector"].get("meanG") is not None  # type: ignore[union-attr]
        ]
        reuse = [
            float(row["transformedDetector"]["exactNgramReuseFraction"])  # type: ignore[index]
            for row in method_rows
            if row["transformedDetector"].get("exactNgramReuseFraction") is not None  # type: ignore[union-attr]
        ]
        distances = [float(row["wordDistance"]) for row in method_rows]
        output[method_id] = {
            "completedDocumentCount": len(method_rows),
            "documentCount": len([row for row in rows if row.get("method") == method_id]),
            "meanExactNgramReuseFraction": statistics.mean(reuse) if reuse else None,
            "meanG": statistics.mean(means) if means else None,
            "meanWordDistance": statistics.mean(distances) if distances else None,
            "removedDocumentCount": sum(
                row.get("removedAtPublishedThreshold") is True for row in method_rows
            ),
        }
    return output


def smoke_methodology(*, document_count: int, threshold: float) -> str:
    return (
        "Apply the repository's frozen synonym, full-paraphrase, English-to-German-"
        "to-English, and English-to-Simplified-Chinese-to-English prompts to "
        f"{document_count} marked documents. Use one current OpenRouter model for "
        "every stage. Score source and output locally with the exact serialized "
        "SynthID sampling table, report exact tokenizer 5-gram reuse, and classify "
        f"removal at the explicitly supplied mean-g threshold {threshold:.12g}. "
        "Document counts are exact sample counts, not a population estimate."
    )


def _load_checkpoint(
    path: Path,
    *,
    corpus_sha256: str,
    model: str,
    budget_usd: Decimal,
) -> dict[str, object]:
    if not path.exists():
        return new_checkpoint_state(
            corpus_sha256=corpus_sha256,
            model=model,
            budget_usd=budget_usd,
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SmokeError("checkpoint must be an object")
    expected = {
        "budgetUsd": format(budget_usd, "f"),
        "corpusSha256": corpus_sha256,
        "model": model,
        "schemaVersion": 1,
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise SmokeError(f"checkpoint {field} does not match this run")
    if value.get("inFlight") is not None:
        raise UncertainInFlight(
            "checkpoint contains an unresolved in-flight call; inspect before retrying"
        )
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--documents", nargs="+", default=list(DEFAULT_DOCUMENT_IDS))
    parser.add_argument("--budget-usd", type=Decimal, default=DEFAULT_BUDGET_USD)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)

    if args.budget_usd <= 0 or args.budget_usd > Decimal("5"):
        raise SmokeError("budget must be positive and no more than $5")
    corpus_path = args.corpus.resolve()
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus_sha256 = sha256_file(corpus_path)
    documents = {
        str(document["documentId"]): document for document in corpus["documents"]
    }
    unknown = [document_id for document_id in args.documents if document_id not in documents]
    if unknown:
        raise SmokeError(f"unknown document IDs: {', '.join(unknown)}")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    base_url = os.environ.get("OPENROUTER_BASE_URL")
    if not api_key or not base_url:
        raise SmokeError("OPENROUTER_API_KEY and OPENROUTER_BASE_URL are required")
    pricing = fetch_model_pricing(
        base_url=base_url,
        api_key=api_key,
        model=args.model,
    )
    prompt_rate = Decimal(str(pricing["promptUsdPerToken"]))
    completion_rate = Decimal(str(pricing["completionUsdPerToken"]))
    state = _load_checkpoint(
        args.checkpoint,
        corpus_sha256=corpus_sha256,
        model=args.model,
        budget_usd=args.budget_usd,
    )
    state["modelPricing"] = pricing
    state["verifiedAt"] = utc_now()
    atomic_write(args.checkpoint, state)

    delegate = unmark.OpenRouterClient.from_env(
        transport=ExplicitNonZdrTransport(),
        timeout=300,
        allow_fallbacks=True,
        require_parameters=False,
        reasoning_effort="none",
        temperature=0.0,
        max_tokens=args.max_tokens,
        max_prompt_price=float(prompt_rate * Decimal(1_000_000)),
        max_completion_price=float(completion_rate * Decimal(1_000_000)),
    )
    client = CheckpointedClient(
        delegate=delegate,
        checkpoint_path=args.checkpoint,
        state=state,
        budget_usd=args.budget_usd,
        prompt_usd_per_token=prompt_rate,
        completion_usd_per_token=completion_rate,
    )
    detector = SynthIDDetector(corpus)
    rows: list[dict[str, object]] = []
    print(
        json.dumps(
            {
                "budgetUsd": format(args.budget_usd, "f"),
                "documents": args.documents,
                "event": "start",
                "maximumCalls": len(args.documents) * 6,
                "model": args.model,
                "pricing": pricing,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for document_id in args.documents:
        for method_id, method, pivot, stages in METHOD_SPECS:
            row = run_transformation(
                client=client,
                detector=detector,
                document=documents[document_id],
                method_id=method_id,
                method=method,
                pivot=pivot,
                stages=stages,
                model=args.model,
                threshold=args.threshold,
            )
            rows.append(row)
            transformations = state.get("transformations")
            assert isinstance(transformations, dict)
            transformations[f"{document_id}::{method_id}"] = row
            state["totalCostUsd"] = format(checkpoint_spend(state), "f")
            state["verifiedAt"] = utc_now()
            atomic_write(args.checkpoint, state)
            transformed = row.get("transformedDetector")
            print(
                json.dumps(
                    {
                        "documentId": document_id,
                        "event": "completed_transform",
                        "meanG": (
                            transformed.get("meanG")
                            if isinstance(transformed, Mapping)
                            else None
                        ),
                        "method": method_id,
                        "outcome": row.get("outcome"),
                        "removed": row.get("removedAtPublishedThreshold"),
                        "reuse": (
                            transformed.get("exactNgramReuseFraction")
                            if isinstance(transformed, Mapping)
                            else None
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = summarize(rows)
    now = utc_now()
    artifact = {
        "budget": {
            "ceilingUsd": format(args.budget_usd, "f"),
            "spentUsd": format(checkpoint_spend(state), "f"),
        },
        "checkpoint": {
            "callCount": len(state["calls"]),
            "callsSha256": _sha256_json(state["calls"]),
            "path": str(args.checkpoint.resolve().relative_to(ROOT)),
        },
        "createdAt": now,
        "documents": rows,
        "inputs": {
            "corpus": {
                "path": str(corpus_path.relative_to(ROOT)),
                "sha256": corpus_sha256,
            },
            "documentIds": list(args.documents),
        },
        "methodology": smoke_methodology(
            document_count=len(args.documents), threshold=args.threshold
        ),
        "model": {
            **pricing,
            "maxCompletionTokensPerCall": args.max_tokens,
            "reasoningEffort": "none",
            "temperature": 0.0,
            "zeroDataRetentionRequired": False,
        },
        "publishedDetectionThreshold": args.threshold,
        "schemaVersion": 1,
        "sources": _evidence_sources(),
        "summary": summary,
        "verifiedAt": now,
        "watermark": {
            "contextHistorySize": corpus["contextHistorySize"],
            "depth": len(corpus["keys"]),
            "model": corpus["model"],
            "ngramLen": corpus["ngramLen"],
            "samplingTableSha256": _sha256_json(corpus["samplingTable"]),
        },
    }
    atomic_write(args.output, artifact)
    state["status"] = "complete"
    state["totalCostUsd"] = format(checkpoint_spend(state), "f")
    state["verifiedAt"] = utc_now()
    state["finalArtifact"] = {
        "path": str(args.output.resolve().relative_to(ROOT)),
        "sha256": sha256_file(args.output),
    }
    atomic_write(args.checkpoint, state)
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(args.output),
                "spentUsd": state["totalCostUsd"],
                "summary": summary,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
