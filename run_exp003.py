"""Experiment 003: four removal methods against a denser keyed watermark.

The first experiment marked less than one percent of words, so the check had
few positions to work with and lost significance as soon as a rewrite removed
a handful of them. This version widens the lexicon and marks every eligible
position instead of a tenth of them, which puts the signal in roughly an
eighth of all words. The check then keeps enough positions to stay meaningful
after a rewrite, which is the regime a real generation-time watermark lives in.

Marking, scoring and every gate here are local. Only the rewrite itself costs
money, and each document is sent once per method with no retry.
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
import watermark_toy as toy


ROOT = Path(__file__).resolve().parent
LEXICON_PATH = ROOT / "fixtures" / "synonym_pairs-v2.json"
DENSITY_BPS = 10_000
CONTEXT_WIDTH = 4
MIN_ACTIVE = 20
MARKED_PATH = ROOT / "results" / "exp003-marked-corpus-v1.json"
MAX_COMPLETION_TOKENS = 32_768
METHODS = (
    "synonyms",
    "roundtrip-de",
    "roundtrip-zh",
    "paraphrase",
    "paraphrase-anchored",
)


def marker_key() -> bytes:
    base, _ = engine.load_detector()
    return getattr(base, "key")


def lexicon() -> toy.SynonymLexicon:
    return toy.load_lexicon(LEXICON_PATH)


def score(documents, *, details: bool = False) -> dict[str, object]:
    return toy.score_corpus(
        tuple(documents),
        key=marker_key(),
        density_bps=DENSITY_BPS,
        lexicon=lexicon(),
        context_width=CONTEXT_WIDTH,
        min_active_positions=MIN_ACTIVE,
    ).to_dict(include_documents=details)


def score_one(document_id: str, text: str) -> dict[str, object]:
    return toy.score_text(
        text,
        key=marker_key(),
        density_bps=DENSITY_BPS,
        lexicon=lexicon(),
        document_id=document_id,
        context_width=CONTEXT_WIDTH,
        min_active_positions=MIN_ACTIVE,
    ).to_dict()


def mark_corpus() -> dict[str, object]:
    """Put the watermark into every eligible position of the six documents."""
    sources = engine.load_sources()
    key = marker_key()
    lex = lexicon()
    rows = []
    for document_id in engine.DOCUMENT_IDS:
        result = toy.encode_text(
            sources[document_id],
            key=key,
            document_id=document_id,
            density_bps=DENSITY_BPS,
            lexicon=lex,
            context_width=CONTEXT_WIDTH,
        )
        rows.append(
            {
                "documentId": document_id,
                "markedText": result.text,
                "originalText": sources[document_id],
                "changedWords": result.changed_positions
                if hasattr(result, "changed_positions")
                else None,
            }
        )
    marked_score = score(
        [toy.Document(r["documentId"], str(r["markedText"])) for r in rows],
        details=True,
    )
    unmarked_score = score(
        [toy.Document(r["documentId"], str(r["originalText"])) for r in rows]
    )
    payload = {
        "contextWidth": CONTEXT_WIDTH,
        "densityBps": DENSITY_BPS,
        "documents": rows,
        "lexiconPath": "fixtures/synonym_pairs-v2.json",
        "lexiconSha256": lex.sha256,
        "markedScore": marked_score,
        "schemaVersion": 1,
        "unmarkedScore": unmarked_score,
    }
    engine.atomic_write(MARKED_PATH, payload)
    return payload


def load_marked() -> dict[str, str]:
    payload = json.loads(MARKED_PATH.read_text(encoding="utf-8"))
    return {row["documentId"]: row["markedText"] for row in payload["documents"]}


def base_url() -> str:
    return os.environ.get(
        "OPENROUTER_BASE_URL", unmark.DEFAULT_OPENROUTER_BASE_URL
    ).rstrip("/")


def api_key() -> str:
    value = os.environ.get("OPENROUTER_API_KEY")
    if not value:
        raise unmark.ConfigurationError("OPENROUTER_API_KEY is required")
    return value


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
    last = ""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                payload = json.loads(response.read())
            break
        except urllib.error.HTTPError as error:
            last = f"HTTP {error.code}: {error.read(1024).decode('utf-8', 'replace')[:200]}"
            if error.code != 429 or attempt == 4:
                return {"error": last}
            time.sleep(5 * (attempt + 1))
        except Exception as error:
            return {"error": f"{type(error).__name__}: {str(error)[:200]}"}
    else:
        return {"error": last or "no response"}
    choice = payload["choices"][0]
    usage = payload.get("usage") or {}
    return {
        "content": choice["message"].get("content") or "",
        "cost": str(usage.get("cost", "0")),
        "finishReason": choice.get("finish_reason"),
        "provider": payload.get("provider"),
    }


def run_document(model: str, method: str, document_id: str, source: str) -> dict:
    protected = unmark.protect_tokens(source)
    spent = Decimal(0)
    anchored = method == "paraphrase-anchored"
    if anchored:
        protected = locked_v2.protect_visible_anchors(source)
        request = locked_v2.build_visible_locked_request(protected.masked)
        prompt = [dict(row) for row in locked_v2.locked_request_messages(request)]
    elif method == "synonyms":
        prompt = unmark.build_synonym_prompt(protected.masked)
    elif method == "paraphrase":
        prompt = unmark.build_paraphrase_prompt(protected.masked)
    else:
        pivot = "de" if method.endswith("de") else "zh"
        prompt = unmark.build_forward_prompt(protected.masked, pivot)

    response = call(model, prompt)
    if "error" in response:
        return {
            "documentId": document_id,
            "method": method,
            "outcome": "error",
            "detail": response["error"],
        }
    spent += Decimal(str(response["cost"]))
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
                "costCredits": format(spent, "f"),
                "detail": f"{type(error).__name__}: {str(error)[:160]}",
            }
        response = call(model, unmark.build_backward_prompt(text, pivot))
        if "error" in response:
            return {
                "documentId": document_id,
                "method": method,
                "outcome": "error",
                "costCredits": format(spent, "f"),
                "detail": response["error"],
            }
        spent += Decimal(str(response["cost"]))
        text = str(response["content"])

    issues: list[dict[str, str]] = []
    if anchored:
        try:
            restored_marked = unmark.restore_tokens(
                unmark.canonicalize_placeholders(text, protected.tokens),
                protected.tokens,
            )
            issues.extend(
                locked_v2.anchor_alignment_issues(
                    unmark.restore_tokens(protected.masked, protected.tokens),
                    restored_marked,
                )
            )
        except Exception:
            issues.extend(locked_v2.anchor_alignment_issues(protected.masked, text))
        text = locked_v2.strip_markers(text)
        protected = unmark.protect_tokens(source)

    restored: str | None = None
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
                "message": f"{type(error).__name__}: {str(error)[:160]}",
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
    return {
        "costCredits": format(spent, "f"),
        "detector": score_one(document_id, evaluated),
        "documentId": document_id,
        "evaluatedOutputText": evaluated,
        "method": method,
        "outcome": "completed",
        "pipelineIssues": issues,
        "provider": response.get("provider"),
        "wordDistance": float(
            engine.require_mapping(fidelity.get("wordLevenshtein"), "distance").get(
                "normalizedDistance", 0
            )
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mark", action="store_true")
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.mark:
        payload = mark_corpus()
        marked = payload["markedScore"]
        unmarked = payload["unmarkedScore"]
        print(
            json.dumps(
                {
                    "marked": {
                        k: marked[k]
                        for k in ("hits", "activePositions", "status", "zScore")
                    },
                    "unmarked": {
                        k: unmarked[k] for k in ("hits", "activePositions", "status")
                    },
                    "path": str(MARKED_PATH),
                },
                sort_keys=True,
            )
        )
        return 0

    if not args.models or args.output is None:
        raise SystemExit("--models and --output are required unless --mark is used")

    marked = load_marked()
    jobs = [
        (f"{model}::{method}", model, method, document_id)
        for model in args.models
        for method in args.methods
        for document_id in engine.DOCUMENT_IDS
    ]
    print(json.dumps({"event": "start", "calls": len(jobs)}), flush=True)
    collected: dict[str, list[dict]] = {key: [] for key, _, _, _ in jobs}
    done = 0
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(
                run_document, model, method, document_id, marked[document_id]
            ): (key, document_id)
            for key, model, method, document_id in jobs
        }
        for future in futures.as_completed(pending):
            key, document_id = pending[future]
            row = future.result()
            collected[key].append(row)
            done += 1
            print(
                json.dumps(
                    {
                        "done": f"{done}/{len(jobs)}",
                        "arm": key,
                        "document": document_id,
                        "outcome": row.get("outcome"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = {}
    order = {value: index for index, value in enumerate(engine.DOCUMENT_IDS)}
    for key, rows in collected.items():
        rows.sort(key=lambda r: order[r["documentId"]])
        produced = [r for r in rows if r.get("outcome") == "completed"]
        pooled = None
        if len(produced) == len(engine.DOCUMENT_IDS):
            pooled = score(
                [
                    toy.Document(r["documentId"], str(r["evaluatedOutputText"]))
                    for r in produced
                ]
            )
        cost = sum(
            (Decimal(r["costCredits"]) for r in rows if "costCredits" in r), Decimal(0)
        )
        summary[key] = {
            "cleanDocuments": len([r for r in produced if not r["pipelineIssues"]]),
            "documents": rows,
            "meanWordDistance": (
                sum(r["wordDistance"] for r in produced) / len(produced)
            )
            if produced
            else None,
            "pooledDetector": pooled,
            "producedDocuments": len(produced),
            "totalCostCredits": format(cost, "f"),
        }
        status = (pooled or {}).get("status")
        hits = (pooled or {}).get("hits")
        active = (pooled or {}).get("activePositions")
        print(
            json.dumps(
                {
                    "result": key,
                    "clean": f"{summary[key]['cleanDocuments']}/{len(produced)}",
                    "detector": f"{hits}/{active} {status}",
                    "cost": summary[key]["totalCostCredits"],
                    "wordDistance": summary[key]["meanWordDistance"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    engine.atomic_write(
        args.output,
        {
            "contextWidth": CONTEXT_WIDTH,
            "densityBps": DENSITY_BPS,
            "lexiconSha256": lexicon().sha256,
            "methods": summary,
            "schemaVersion": 1,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
