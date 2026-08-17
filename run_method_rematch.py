"""Re-run paraphrase, German round-trip and Chinese round-trip on one strong model.

The published four-method comparison ran every method on Qwen 3.5 9B, a model
later shown to manage one of six documents under the anchor contract. Its
ordering therefore describes how a weak model fails at three tasks, not how the
three methods compare. This re-runs the same frozen prompts on one capable
non-signatory model so the only variable left is the method.

Anchor protection is deliberately absent here: the paraphrase arm must get the
same plain treatment as the two translation arms, or the comparison measures
the protection rather than the method.
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
import run_model_canary_terra_locked as anchors_v1
import run_model_canary_terra_locked_v2 as locked_v2
import unmark
from watermark_toy import Document, score_corpus, score_text


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "method-rematch-v1.json"
DEFAULT_MODEL = "minimax/minimax-m3"
MAX_COMPLETION_TOKENS = 32_768
METHODS = ("paraphrase", "roundtrip-de", "roundtrip-zh")
MODELS = (DEFAULT_MODEL,)


def base_url() -> str:
    return os.environ.get(
        "OPENROUTER_BASE_URL", unmark.DEFAULT_OPENROUTER_BASE_URL
    ).rstrip("/")


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise unmark.ConfigurationError("OPENROUTER_API_KEY is required")
    return key


def call(model: str, prompt) -> dict[str, object]:
    messages = (
        prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    )
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "stream": False,
        "reasoning": {"effort": "medium"},
        "provider": {
            "allow_fallbacks": True,
            "data_collection": "deny",
            "require_parameters": False,
        },
    }
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
    last = ""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                payload = json.loads(response.read())
            break
        except urllib.error.HTTPError as error:
            last = f"HTTP {error.code}: {error.read(2048).decode('utf-8', 'replace')[:300]}"
            if error.code != 429 or attempt == 4:
                return {"error": last}
            time.sleep(5 * (attempt + 1))
        except Exception as error:
            return {"error": f"{type(error).__name__}: {str(error)[:300]}"}
    if payload is None:
        return {"error": last or "no response"}
    choice = payload["choices"][0]
    usage = payload.get("usage") or {}
    return {
        "content": choice["message"].get("content") or "",
        "cost": str(usage.get("cost", "0")),
        "finishReason": choice.get("finish_reason"),
        "provider": payload.get("provider"),
    }


def anchor_retention(source: str, output: str) -> dict[str, object]:
    expected = sorted(
        m.group(0).lower() for m in anchors_v1._ANCHOR_RE.finditer(source)
    )
    pool = sorted(m.group(0).lower() for m in anchors_v1._ANCHOR_RE.finditer(output))
    kept = 0
    for value in expected:
        if value in pool:
            pool.remove(value)
            kept += 1
    return {
        "expected": len(expected),
        "kept": kept,
        "retention": (kept / len(expected)) if expected else None,
    }


def run_document(model: str, method: str, document_id: str, source: str) -> dict:
    protected = unmark.protect_tokens(source)
    calls: list[dict[str, object]] = []
    spent = Decimal(0)
    if method == "paraphrase-anchored":
        # The v2 arm: anchors stay visible in copy-verbatim markers, and the
        # markers are stripped before scoring so the comparison is like for like.
        protected = locked_v2.protect_visible_anchors(source)
        request = locked_v2.build_visible_locked_request(protected.masked)
        # One request carrying both messages, not one request per message.
        prompts = [[dict(row) for row in locked_v2.locked_request_messages(request)]]
    elif method == "paraphrase":
        prompts = [unmark.build_paraphrase_prompt(protected.masked)]
    else:
        pivot = "de" if method.endswith("de") else "zh"
        prompts = [unmark.build_forward_prompt(protected.masked, pivot)]
    text = ""
    for index, prompt in enumerate(prompts):
        response = call(model, prompt)
        if "error" in response:
            return {
                "documentId": document_id,
                "method": method,
                "outcome": "error",
                "detail": response["error"],
            }
        spent += Decimal(str(response["cost"]))
        calls.append({"provider": response.get("provider"), "stage": index})
        text = str(response["content"])
    if method.startswith("roundtrip"):
        pivot = "de" if method.endswith("de") else "zh"
        try:
            unmark.validate_intermediate(text, pivot, protected.tokens)
        except Exception as error:
            return {
                "documentId": document_id,
                "method": method,
                "outcome": "pivot_rejected",
                "detail": f"{type(error).__name__}: {str(error)[:200]}",
                "costCredits": format(spent, "f"),
            }
        response = call(model, unmark.build_backward_prompt(text, pivot))
        if "error" in response:
            return {
                "documentId": document_id,
                "method": method,
                "outcome": "error",
                "detail": response["error"],
                "costCredits": format(spent, "f"),
            }
        spent += Decimal(str(response["cost"]))
        calls.append({"provider": response.get("provider"), "stage": len(calls)})
        text = str(response["content"])
    issues: list[dict[str, str]] = []
    restored: str | None = None
    if method == "paraphrase-anchored":
        issues.extend(
            locked_v2.anchor_alignment_issues(
                unmark.restore_tokens(protected.masked, protected.tokens),
                locked_v2.strip_markers(text)
                if "\u27ea" not in text
                else unmark.restore_tokens(
                    unmark.canonicalize_placeholders(text, protected.tokens),
                    protected.tokens,
                ),
            )
        )
        text = locked_v2.strip_markers(text)
        protected = unmark.protect_tokens(source)
    try:
        normalized = unmark.canonicalize_placeholders(text, protected.tokens)
        issues.extend(
            unmark.result_validation_issues(protected.masked, normalized, None)
        )
        restored = unmark.restore_tokens(normalized, protected.tokens)
    except Exception as error:
        issues.append(
            {
                "code": "placeholder_contract",
                "message": f"{type(error).__name__}: {str(error)[:200]}",
            }
        )
    evaluated = restored if restored is not None else text
    fidelity = fidelity_metrics(source, evaluated)
    if (
        engine.require_mapping(fidelity.get("protectedTokens"), "protected").get(
            "exactlyRestored"
        )
        is not True
    ):
        issues.append(
            {"code": "protected_values", "message": "protected values changed"}
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
        "anchorRetention": anchor_retention(source, evaluated),
        "callCount": len(calls),
        "costCredits": format(spent, "f"),
        "detector": detector,
        "documentId": document_id,
        "evaluatedOutputText": evaluated,
        "method": method,
        "outcome": "completed",
        "pipelineIssues": issues,
        "providers": [c["provider"] for c in calls],
        "wordDistance": float(
            engine.require_mapping(fidelity.get("wordLevenshtein"), "distance").get(
                "normalizedDistance", 0
            )
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    sources = engine.load_sources()
    jobs = [
        (f"{model}::{method}", model, method, document_id)
        for model in args.models
        for method in args.methods
        for document_id in engine.DOCUMENT_IDS
    ]
    print(json.dumps({"event": "start", "calls": len(jobs)}), flush=True)
    collected: dict[str, list[dict]] = {key: [] for key, _, _, _ in jobs}
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(
                run_document, model, method, document_id, sources[document_id]
            ): (key, document_id)
            for key, model, method, document_id in jobs
        }
        done = 0
        for future in futures.as_completed(pending):
            key, document_id = pending[future]
            row = future.result()
            collected[key].append(row)
            done += 1
            print(
                json.dumps(
                    {
                        "done": f"{done}/{len(jobs)}",
                        "method": key,
                        "document": document_id,
                        "outcome": row.get("outcome"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = {}
    for method, rows in collected.items():
        order = {v: i for i, v in enumerate(engine.DOCUMENT_IDS)}
        rows.sort(key=lambda r: order[r["documentId"]])
        produced = [r for r in rows if r.get("outcome") == "completed"]
        outputs = [
            Document(r["documentId"], str(r["evaluatedOutputText"])) for r in produced
        ]
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
        retentions = [
            r["anchorRetention"]["retention"]
            for r in produced
            if r["anchorRetention"]["retention"] is not None
        ]
        kept = sum(r["anchorRetention"]["kept"] for r in produced)
        expected = sum(r["anchorRetention"]["expected"] for r in produced)
        cost = sum(
            (Decimal(r["costCredits"]) for r in rows if "costCredits" in r), Decimal(0)
        )
        summary[method] = {
            "cleanDocuments": len([r for r in produced if not r["pipelineIssues"]]),
            "documents": rows,
            "meanWordDistance": (
                sum(r["wordDistance"] for r in produced) / len(produced)
            )
            if produced
            else None,
            "pooledAnchorRetention": (kept / expected) if expected else None,
            "pooledDetector": pooled,
            "producedDocuments": len(produced),
            "totalCostCredits": format(cost, "f"),
            "worstAnchorRetention": min(retentions) if retentions else None,
        }
        pooled_status = (pooled or {}).get("status")
        print(
            json.dumps(
                {
                    "result": method,
                    "clean": f"{summary[method]['cleanDocuments']}/{len(produced)}",
                    "anchorRetention": summary[method]["pooledAnchorRetention"],
                    "detector": pooled_status,
                    "cost": summary[method]["totalCostCredits"],
                    "wordDistance": summary[method]["meanWordDistance"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    engine.atomic_write(
        args.output,
        {
            "methods": summary,
            "models": list(args.models),
            "note": "Plain frozen prompts, no anchor protection, one strong model.",
            "schemaVersion": 1,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
