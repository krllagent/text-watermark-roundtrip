"""Experiment 004: four removal methods against the real SynthID watermark.

Everything here runs on one corpus. The documents were written by a small model
with Google's reference watermarking turned on, so every generated token
carries signal rather than an eighth of the words. The same documents are then
rewritten by six models with four instructions, and both questions are asked of
the same text: did the marker go, and did the meaning survive.

Detection is the mean g-value statistic from the paper. It needs the text and
the keys, never the original document and never the model. The threshold is
calibrated on unmarked generations from the same prompts and seeds, which ship
alongside the marked ones in the corpus file.
"""

from __future__ import annotations

import argparse
from concurrent import futures
from decimal import Decimal
import json
import os
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request

import torch
from transformers import AutoTokenizer, SynthIDTextWatermarkLogitsProcessor

from run_experiment import fidelity_metrics
import run_model_canary_luna as engine
import unmark


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "results" / "synthid-corpus-v1.json"
MAX_COMPLETION_TOKENS = 32_768
METHODS = ("synonyms", "roundtrip-de", "roundtrip-zh", "paraphrase")
_DETECTOR: dict[str, object] = {}


def detector():
    """Mean g-value detector: reference processor, no training, no model."""
    if not _DETECTOR:
        payload = json.loads(CORPUS.read_text(encoding="utf-8"))
        _DETECTOR["tokenizer"] = AutoTokenizer.from_pretrained(payload["model"])
        _DETECTOR["processor"] = SynthIDTextWatermarkLogitsProcessor(
            ngram_len=payload["ngramLen"],
            keys=payload["keys"],
            sampling_table_size=65_536,
            sampling_table_seed=0,
            context_history_size=payload["contextHistorySize"],
            device=torch.device("cpu"),
        )
        _DETECTOR["ngram"] = payload["ngramLen"]
    return _DETECTOR["processor"], _DETECTOR["tokenizer"], _DETECTOR["ngram"]


def mean_g(text: str) -> dict[str, object]:
    processor, tokenizer, ngram = detector()
    ids = tokenizer([text], return_tensors="pt", add_special_tokens=False)["input_ids"]
    if ids.shape[1] <= ngram:
        return {"meanG": None, "scoredTokens": 0}
    g = processor.compute_g_values(input_ids=ids)
    mask = processor.compute_context_repetition_mask(input_ids=ids)
    valid = mask.unsqueeze(-1).expand_as(g)
    total = valid.sum().item()
    if total == 0:
        return {"meanG": None, "scoredTokens": 0}
    return {
        "meanG": float((g * valid).sum().item() / total),
        "scoredTokens": int(mask.sum().item()),
        "tokens": int(ids.shape[1]),
    }


def load_corpus() -> dict[str, object]:
    return json.loads(CORPUS.read_text(encoding="utf-8"))


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
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                payload = json.loads(response.read())
            break
        except urllib.error.HTTPError as error:
            last = f"HTTP {error.code}: {error.read(512).decode('utf-8', 'replace')[:160]}"
            if error.code not in (429, 502, 503) or attempt == 5:
                return {"error": last}
            time.sleep(6 * (attempt + 1))
        except Exception as error:
            return {"error": f"{type(error).__name__}: {str(error)[:160]}"}
    else:
        return {"error": last or "no response"}
    choice = payload["choices"][0]
    usage = payload.get("usage") or {}
    return {
        "content": choice["message"].get("content") or "",
        "cost": str(usage.get("cost", "0")),
        "provider": payload.get("provider"),
    }


def run_document(model: str, method: str, document_id: str, source: str) -> dict:
    protected = unmark.protect_tokens(source)
    spent = Decimal(0)
    if method == "synonyms":
        prompt = unmark.build_synonym_prompt(protected.masked)
    elif method == "paraphrase":
        prompt = unmark.build_paraphrase_prompt(protected.masked)
    else:
        pivot = "de" if method.endswith("de") else "zh"
        prompt = unmark.build_forward_prompt(protected.masked, pivot)

    response = call(model, prompt)
    if "error" in response:
        return {"documentId": document_id, "method": method, "outcome": "error",
                "detail": response["error"]}
    spent += Decimal(str(response["cost"]))
    text = str(response["content"])

    if method.startswith("roundtrip"):
        pivot = "de" if method.endswith("de") else "zh"
        try:
            unmark.validate_intermediate(text, pivot, protected.tokens)
        except Exception as error:
            return {"documentId": document_id, "method": method,
                    "outcome": "pivot_rejected", "costCredits": format(spent, "f"),
                    "detail": f"{type(error).__name__}: {str(error)[:160]}"}
        response = call(model, unmark.build_backward_prompt(text, pivot))
        if "error" in response:
            return {"documentId": document_id, "method": method, "outcome": "error",
                    "costCredits": format(spent, "f"), "detail": response["error"]}
        spent += Decimal(str(response["cost"]))
        text = str(response["content"])

    issues: list[dict[str, str]] = []
    restored: str | None = None
    try:
        normalized = unmark.canonicalize_placeholders(text, protected.tokens)
        issues.extend(unmark.result_validation_issues(protected.masked, normalized, None))
        restored = unmark.restore_tokens(normalized, protected.tokens)
    except Exception as error:
        issues.append({"code": "placeholder_contract",
                       "message": f"{type(error).__name__}: {str(error)[:160]}"})
    evaluated = restored if restored is not None else text
    fidelity = fidelity_metrics(source, evaluated)
    if (
        engine.require_mapping(fidelity.get("protectedTokens"), "protected").get(
            "exactlyRestored"
        )
        is not True
    ):
        issues.append({"code": "protected_values", "message": "protected values changed"})
    return {
        "costCredits": format(spent, "f"),
        "detector": mean_g(evaluated),
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
    parser.add_argument("--models", nargs="*", required=True)
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    corpus = load_corpus()
    marked = {r["documentId"]: r["markedText"] for r in corpus["documents"]}
    detector()  # warm the tokenizer before threads start

    jobs = [
        (f"{model}::{method}", model, method, document_id)
        for model in args.models
        for method in args.methods
        for document_id in sorted(marked)
    ]
    print(json.dumps({"event": "start", "calls": len(jobs)}), flush=True)
    collected: dict[str, list[dict]] = {key: [] for key, _, _, _ in jobs}
    done = 0
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(run_document, model, method, document_id, marked[document_id]):
            (key, document_id)
            for key, model, method, document_id in jobs
        }
        for future in futures.as_completed(pending):
            key, document_id = pending[future]
            row = future.result()
            collected[key].append(row)
            done += 1
            print(json.dumps({"done": f"{done}/{len(jobs)}", "arm": key,
                              "document": document_id, "outcome": row.get("outcome")},
                             sort_keys=True), flush=True)

    summary = {}
    for key, rows in collected.items():
        rows.sort(key=lambda r: r["documentId"])
        produced = [r for r in rows if r.get("outcome") == "completed"]
        gs = [r["detector"]["meanG"] for r in produced if r["detector"]["meanG"] is not None]
        cost = sum((Decimal(r["costCredits"]) for r in rows if "costCredits" in r), Decimal(0))
        summary[key] = {
            "cleanDocuments": len([r for r in produced if not r["pipelineIssues"]]),
            "documents": rows,
            "meanG": statistics.mean(gs) if gs else None,
            "meanWordDistance": (sum(r["wordDistance"] for r in produced) / len(produced))
            if produced else None,
            "producedDocuments": len(produced),
            "totalCostCredits": format(cost, "f"),
        }
        print(json.dumps({
            "result": key,
            "meanG": round(summary[key]["meanG"], 4) if gs else None,
            "clean": f"{summary[key]['cleanDocuments']}/{len(produced)}",
            "cost": summary[key]["totalCostCredits"],
            "wordDistance": summary[key]["meanWordDistance"],
        }, sort_keys=True), flush=True)

    engine.atomic_write(args.output, {
        "corpusMarkedMeanG": corpus["markedMeanG"],
        "corpusUnmarkedMeanG": corpus["unmarkedMeanG"],
        "methods": summary,
        "schemaVersion": 1,
        "watermark": "synthid-text-reference",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
