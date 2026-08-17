"""Screen candidate models on the v2 visible-anchor method, six development docs.

This is a screen, not a frozen measurement. It reuses the exact v2 system
instruction, payload shape, masking, and automatic checks that the Terra run
used, so the only thing that changes between rows is the model. Providers are
not pinned to one endpoint per model; the provider that actually served each
call is recorded instead. Only the automatic gate runs here: anchor retention
per sentence, placeholder restoration, sentence alignment, teaching-marker
removal, cost. Blind reading is deliberately not part of a screen.

Requests go out over plain HTTP rather than the frozen canary client so that a
truncated or empty response is recorded as a result, with its token usage and
finish reason, instead of raising. Reasoning models can spend an entire token
budget thinking and return no text; that is a finding, not an error to hide.

The twenty-document holdout is untouched. Screening runs on the six
development documents, which is what a development set is for.
"""

from __future__ import annotations

import argparse
from concurrent import futures
from decimal import Decimal
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from run_experiment import fidelity_metrics
import run_model_canary_luna as engine
import run_model_canary_terra_locked_v2 as locked_v2
import unmark
from watermark_toy import Document, score_corpus, score_text


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "model-screen-v2.json"
MAX_COMPLETION_TOKENS = 32_768
REASONING_EFFORT = "medium"

CANDIDATES = (
    "tencent/hy3",
    "xiaomi/mimo-v2.5",
    "inclusionai/ling-3.0-flash",
    "nvidia/nemotron-3.5-lightning",
    "inception/mercury-2",
    "deepseek/deepseek-v4-flash",
    "meta/muse-spark-1.2",
    "z-ai/glm-5.2",
    "minimax/minimax-m3",
    "qwen/qwen3.7-plus",
)


def base_url() -> str:
    return os.environ.get(
        "OPENROUTER_BASE_URL", unmark.DEFAULT_OPENROUTER_BASE_URL
    ).rstrip("/")


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise unmark.ConfigurationError("OPENROUTER_API_KEY is required")
    return key


def fetch_catalog() -> dict[str, dict[str, object]]:
    request = urllib.request.Request(
        f"{base_url()}/models",
        headers={"Authorization": f"Bearer {api_key()}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        rows = json.loads(response.read())["data"]
    return {row["id"]: row for row in rows}


def build_body(model: str, catalog_row: dict[str, object]) -> dict[str, object]:
    supported = set(catalog_row.get("supported_parameters") or ())
    request = locked_v2.build_visible_locked_request("PLACEHOLDER")
    body: dict[str, object] = {
        "model": model,
        "messages": [],
        "stream": False,
        "provider": {
            "allow_fallbacks": True,
            "data_collection": "deny",
            "require_parameters": False,
        },
    }
    if "max_completion_tokens" in supported:
        body["max_completion_tokens"] = MAX_COMPLETION_TOKENS
    else:
        body["max_tokens"] = MAX_COMPLETION_TOKENS
    if "reasoning" in supported or "reasoning_effort" in supported:
        body["reasoning"] = {"effort": REASONING_EFFORT}
    if "seed" in supported:
        body["seed"] = 20260817
    del request
    return body


def call_model(
    model: str, template: dict[str, object], messages: list[dict[str, str]]
) -> dict[str, object]:
    body = {**template, "messages": messages}
    request = urllib.request.Request(
        f"{base_url()}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    payload = None
    last_error = ""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                payload = json.loads(response.read())
            break
        except urllib.error.HTTPError as error:
            detail = error.read(4096).decode("utf-8", "replace")
            last_error = f"HTTP {error.code}: {detail[:400]}"
            # 429 here is the provider concurrency cap, not a content problem:
            # back off and retry rather than scoring an unsent request.
            if error.code != 429 or attempt == 4:
                return {"transportError": last_error}
            time.sleep(5 * (attempt + 1))
        except Exception as error:  # network-level failure
            return {"transportError": f"{type(error).__name__}: {str(error)[:400]}"}
    if payload is None:
        return {"transportError": last_error or "no response"}
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return {"transportError": "response carried no choice", "raw": payload}
    choice = choices[0]
    message = choice.get("message") or {}
    usage = payload.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "content": message.get("content") or "",
        "reasoningTokens": int(details.get("reasoning_tokens", 0) or 0),
        "finishReason": choice.get("finish_reason"),
        "costCredits": str(usage.get("cost", "0")),
        "promptTokens": int(usage.get("prompt_tokens", 0) or 0),
        "completionTokens": int(usage.get("completion_tokens", 0) or 0),
        "provider": payload.get("provider"),
        "servedModel": payload.get("model"),
    }


def analyze(document_id: str, source: str, content: str) -> dict[str, object]:
    """Run the same automatic checks the Terra run used, on one response."""
    protected = locked_v2.protect_visible_anchors(source)
    anchored_source = unmark.restore_tokens(protected.masked, protected.tokens)
    issues: list[dict[str, str]] = []
    restored: str | None = None
    try:
        normalized_marked = unmark.canonicalize_placeholders(content, protected.tokens)
        anchored_output = unmark.restore_tokens(normalized_marked, protected.tokens)
        issues.extend(
            locked_v2.anchor_alignment_issues(anchored_source, anchored_output)
        )
        stripped = locked_v2.strip_markers(normalized_marked)
        issues.extend(
            unmark.result_validation_issues(
                locked_v2.strip_markers(protected.masked), stripped, None
            )
        )
        restored = unmark.restore_tokens(stripped, protected.tokens)
    except Exception as error:
        issues.append(
            {
                "code": "placeholder_contract",
                "message": f"{type(error).__name__}: {str(error)[:200]}",
            }
        )
    evaluated = restored if restored is not None else locked_v2.strip_markers(content)
    fidelity = fidelity_metrics(source, evaluated)
    protected_metrics = engine.require_mapping(
        fidelity.get("protectedTokens"), "protected metrics"
    )
    if protected_metrics.get("exactlyRestored") is not True:
        issues.append(
            {"code": "protected_values", "message": "protected values changed"}
        )
    source_sentences = len(locked_v2.split_sentences(source))
    output_sentences = len(locked_v2.split_sentences(evaluated))
    if source_sentences != output_sentences:
        issues.append(
            {
                "code": "sentence_alignment",
                "message": (
                    f"expected {source_sentences} sentences and observed "
                    f"{output_sentences}"
                ),
            }
        )
    base, corpus = engine.load_detector()
    detector = score_text(
        evaluated,
        key=getattr(base, "key"),
        density_bps=getattr(base, "density_bps"),
        lexicon=getattr(corpus, "lexicon"),
        document_id=document_id,
        context_width=getattr(base, "context_width"),
        min_active_positions=getattr(base, "min_active_positions"),
    ).to_dict()
    return {
        "anchorCount": len(locked_v2._MARK_RE.findall(protected.masked)),
        "detector": detector,
        "evaluatedOutputText": evaluated,
        "pipelineIssues": issues,
        "wordDistance": float(
            engine.require_mapping(fidelity.get("wordLevenshtein"), "distance").get(
                "normalizedDistance", 0
            )
        ),
    }


def screen_one_document(
    model: str, template: dict[str, object], document_id: str, source: str
) -> dict[str, object]:
    """Run and score a single (model, document) pair. Never raises."""
    request = locked_v2.build_visible_locked_request(
        locked_v2.protect_visible_anchors(source).masked
    )
    messages = [dict(row) for row in locked_v2.locked_request_messages(request)]
    started = time.monotonic()
    response = call_model(model, template, messages)
    latency = round((time.monotonic() - started) * 1000, 1)
    if "transportError" in response:
        return {
            "documentId": document_id,
            "latencyMs": latency,
            "outcome": "transport_error",
            "detail": response["transportError"],
        }
    row: dict[str, object] = {
        "completionTokens": response["completionTokens"],
        "costCredits": response["costCredits"],
        "documentId": document_id,
        "finishReason": response["finishReason"],
        "latencyMs": latency,
        "promptTokens": response["promptTokens"],
        "provider": response.get("provider"),
        "reasoningTokens": response["reasoningTokens"],
        "servedModel": response["servedModel"],
    }
    content = str(response["content"])
    if not content.strip():
        row["outcome"] = "empty_content"
        row["pipelineIssues"] = [
            {
                "code": "empty_content",
                "message": (
                    "model returned no text; "
                    f"{response['reasoningTokens']} reasoning tokens spent"
                ),
            }
        ]
        return row
    try:
        analysis = analyze(document_id, source, content)
    except Exception as error:  # scoring must not lose a paid response
        row["outcome"] = "analysis_error"
        row["detail"] = f"{type(error).__name__}: {str(error)[:300]}"
        return row
    row.update(
        {
            "anchorCount": analysis["anchorCount"],
            "evaluatedOutputText": analysis["evaluatedOutputText"],
            "outcome": "completed",
            "pipelineIssues": analysis["pipelineIssues"],
            "wordDistance": analysis["wordDistance"],
        }
    )
    return row


def summarize_model(
    model: str, documents: list[dict[str, object]]
) -> dict[str, object]:
    order = {value: index for index, value in enumerate(engine.DOCUMENT_IDS)}
    documents = sorted(documents, key=lambda row: order[str(row["documentId"])])
    outputs: list[Document] = []
    providers: set[str] = set()
    spent = Decimal(0)
    prompt_tokens = completion_tokens = reasoning_tokens = 0
    for row in documents:
        spent += Decimal(str(row.get("costCredits", "0")))
        prompt_tokens += int(row.get("promptTokens", 0) or 0)
        completion_tokens += int(row.get("completionTokens", 0) or 0)
        reasoning_tokens += int(row.get("reasoningTokens", 0) or 0)
        if row.get("provider"):
            providers.add(str(row["provider"]))
        if row.get("outcome") == "completed":
            # Keep the text in the artifact: a later blind reading must judge
            # exactly the output that produced these automatic scores.
            outputs.append(
                Document(str(row["documentId"]), str(row["evaluatedOutputText"]))
            )
    pooled = None
    if len(outputs) == len(engine.DOCUMENT_IDS):
        base, corpus = engine.load_detector()
        pooled = score_corpus(
            tuple(outputs),
            key=getattr(base, "key"),
            density_bps=getattr(base, "density_bps"),
            lexicon=getattr(corpus, "lexicon"),
            context_width=getattr(base, "context_width"),
            min_active_positions=getattr(base, "min_active_positions"),
        ).to_dict(include_documents=False)
    produced = [row for row in documents if row.get("outcome") == "completed"]
    clean = [row for row in produced if not row["pipelineIssues"]]
    distances = [float(row["wordDistance"]) for row in produced]
    return {
        "cleanDocuments": len(clean),
        "completionTokens": completion_tokens,
        "documents": documents,
        "meanWordDistance": (sum(distances) / len(distances)) if distances else None,
        "model": model,
        "pooledDetector": pooled,
        "producedDocuments": len(produced),
        "promptTokens": prompt_tokens,
        "providers": sorted(providers),
        "reasoningTokens": reasoning_tokens,
        "totalCostCredits": format(spent, "f"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(CANDIDATES))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    catalog = fetch_catalog()
    sources = engine.load_sources()
    templates: dict[str, dict[str, object]] = {}
    jobs: list[tuple[str, str]] = []
    for model in args.models:
        catalog_row = catalog.get(model)
        if catalog_row is None:
            print(
                json.dumps({"model": model, "skipped": "absent from OpenRouter"}),
                flush=True,
            )
            continue
        templates[model] = build_body(model, catalog_row)
        jobs.extend((model, document_id) for document_id in engine.DOCUMENT_IDS)

    # Every (model, document) pair is independent: no shared state, no ordering
    # requirement, and one model's failure must not hold up another's row.
    collected: dict[str, list[dict[str, object]]] = {model: [] for model in templates}
    done = 0
    print(
        json.dumps({"event": "start", "calls": len(jobs), "workers": args.workers}),
        flush=True,
    )
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(
                screen_one_document,
                model,
                templates[model],
                document_id,
                sources[document_id],
            ): (model, document_id)
            for model, document_id in jobs
        }
        for future in futures.as_completed(pending):
            model, document_id = pending[future]
            row = future.result()
            collected[model].append(row)
            done += 1
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "done": f"{done}/{len(jobs)}",
                        "model": model,
                        "document": document_id,
                        "outcome": row.get("outcome"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    rows = []
    for model in templates:
        row = summarize_model(model, collected[model])
        rows.append(row)
        pooled = row["pooledDetector"] or {}
        print(
            json.dumps(
                {
                    "result": model,
                    "clean": f"{row['cleanDocuments']}/{row['producedDocuments']}",
                    "cost": row["totalCostCredits"],
                    "detector": pooled.get("status"),
                    "reasoningTokens": row["reasoningTokens"],
                    "wordDistance": row["meanWordDistance"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    result = {
        "candidates": rows,
        "developmentDocuments": list(engine.DOCUMENT_IDS),
        "maxCompletionTokens": MAX_COMPLETION_TOKENS,
        "method": "visible_anchor_sentence_aligned_v2",
        "note": (
            "Screen only: providers are not pinned and no blind reading was "
            "performed. The twenty-document holdout was not used."
        ),
        "reasoningEffort": REASONING_EFFORT,
        "schemaVersion": 1,
        "systemInstructionSha256": engine.object_sha256(
            {"instruction": locked_v2.LOCKED_SYSTEM_INSTRUCTION}
        ),
    }
    engine.atomic_write(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
