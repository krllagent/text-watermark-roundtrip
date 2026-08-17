"""Score rewritten texts with a panel of large judges and take the median.

Three earlier evaluations disagreed with each other: one automatic judge, one
strict reader protocol, one lenient one. Quality judgements of this kind are
inherently subjective, so a single judge produces a number that mostly reflects
the judge. A panel of independent large models, aggregated by median, is more
useful: the median is robust to one outlier judge, and the resulting figure is
read as a level rather than a precise score.

Judges are told only that a text was rewritten. They are never told how, so the
same prompt is valid for a paraphrase, a synonym edit, and a round-trip
translation, and no judge can favour or penalise a method it recognises.

A judge being a watermark signatory does not matter here. That constraint
applies to the model that rewrites a user's text, not to a model that reads two
texts and counts differences.
"""

from __future__ import annotations

import argparse
from concurrent import futures
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import statistics
import time
import urllib.error
import urllib.request

import run_model_canary_luna as engine
import unmark


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "judge-panel-v1.json"
JUDGES = (
    "openai/gpt-5.6-sol",
    "anthropic/claude-fable-5",
    "x-ai/grok-4.6",
)
MAX_COMPLETION_TOKENS = 8_192

INSTRUCTION = (
    "You compare an original text with a rewritten version of it and count "
    "GROSS meaning errors.\n\n"
    "A GROSS error is a place where the rewrite states something materially "
    "different from the original: a changed or lost fact, a changed number, "
    "quantity, date or name, a reversed or dropped negation, a changed "
    "requirement or recommendation, a changed threshold or deadline, a reversed "
    "cause and effect, or a claim the original never made. A gross error would "
    "mislead someone who relied on the rewrite without seeing the original.\n\n"
    "Do NOT count: different word choice, different sentence structure, "
    "different register or tone, sentences merged or split, or synonyms that "
    "leave the meaning intact. Do NOT count stray bracket characters or "
    "formatting artefacts.\n\n"
    "You are not told how the rewrite was produced. Do not speculate about it.\n\n"
    "Return ONLY a JSON object of this exact shape:\n"
    '{"grossErrorCount": <integer>, "errors": [{"original": "<exact quote from '
    'the original>", "rewrite": "<exact quote from the rewrite>", "why": "<one '
    'short sentence>"}]}\n'
    "The length of errors must equal grossErrorCount. Return 0 and an empty "
    "list when the rewrite carries the same meaning throughout."
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


def build_prompt(source: str, candidate: str) -> str:
    return (
        f"{INSTRUCTION}\n\n"
        f"--- ORIGINAL ---\n{source}\n--- END ORIGINAL ---\n\n"
        f"--- REWRITE ---\n{candidate}\n--- END REWRITE ---"
    )


def extract_json(text: str) -> dict[str, object] | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = fenced.group(1) if fenced else None
    if raw is None:
        start = text.find("{")
        end = text.rfind("}")
        raw = text[start : end + 1] if start != -1 and end > start else None
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def judge_one(judge: str, source: str, candidate: str) -> dict[str, object]:
    body = {
        "model": judge,
        "messages": [{"role": "user", "content": build_prompt(source, candidate)}],
        "max_tokens": MAX_COMPLETION_TOKENS,
        "stream": False,
        "provider": {"allow_fallbacks": True, "data_collection": "deny"},
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
            last = f"HTTP {error.code}: {error.read(1024).decode('utf-8', 'replace')[:200]}"
            if error.code != 429 or attempt == 4:
                return {"error": last}
            time.sleep(5 * (attempt + 1))
        except Exception as error:
            return {"error": f"{type(error).__name__}: {str(error)[:200]}"}
    if payload is None:
        return {"error": last or "no response"}
    choice = payload["choices"][0]
    usage = payload.get("usage") or {}
    parsed = extract_json(str(choice["message"].get("content") or ""))
    if parsed is None or not isinstance(parsed.get("grossErrorCount"), int):
        return {
            "error": "judge did not return the required JSON",
            "costCredits": str(usage.get("cost", "0")),
        }
    errors = parsed.get("errors")
    return {
        "costCredits": str(usage.get("cost", "0")),
        "errors": errors if isinstance(errors, list) else [],
        "grossErrorCount": int(parsed["grossErrorCount"]),
        "judge": judge,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--judges", nargs="*", default=list(JUDGES))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    sources = engine.load_sources()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    items: list[tuple[str, str, str]] = []
    for method, data in payload["methods"].items():
        for row in data["documents"]:
            if row.get("outcome") != "completed":
                continue
            items.append((method, row["documentId"], row["evaluatedOutputText"]))

    jobs = [(m, d, t, j) for (m, d, t) in items for j in args.judges]
    print(json.dumps({"event": "start", "calls": len(jobs)}), flush=True)
    verdicts: dict[tuple[str, str], list[dict[str, object]]] = {}
    spent = Decimal(0)
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(judge_one, judge, sources[document_id], text): (
                method,
                document_id,
                judge,
            )
            for method, document_id, text, judge in jobs
        }
        done = 0
        for future in futures.as_completed(pending):
            method, document_id, judge = pending[future]
            row = future.result()
            row["judge"] = judge
            spent += Decimal(str(row.get("costCredits", "0")))
            verdicts.setdefault((method, document_id), []).append(row)
            done += 1
            if done % 9 == 0 or done == len(jobs):
                print(
                    json.dumps({"done": f"{done}/{len(jobs)}"}, sort_keys=True),
                    flush=True,
                )

    per_document = []
    per_method: dict[str, list[float]] = {}
    for (method, document_id), rows in sorted(verdicts.items()):
        counts = [int(r["grossErrorCount"]) for r in rows if "grossErrorCount" in r]
        median = statistics.median(counts) if counts else None
        if median is not None:
            per_method.setdefault(method, []).append(median)
        per_document.append(
            {
                "counts": {
                    str(r["judge"]): r.get("grossErrorCount", None) for r in rows
                },
                "documentId": document_id,
                "errors": {
                    str(r["judge"]): r.get("errors", []) for r in rows if "errors" in r
                },
                "medianGrossErrors": median,
                "method": method,
            }
        )
    summary = {
        method: {
            "documents": len(values),
            "meanOfMedians": sum(values) / len(values),
            "medianOfMedians": statistics.median(values),
            "totalMedianErrors": sum(values),
            "worstDocument": max(values),
        }
        for method, values in per_method.items()
    }
    engine.atomic_write(
        args.output,
        {
            "instructionSha256": engine.object_sha256({"instruction": INSTRUCTION}),
            "judges": list(args.judges),
            "perDocument": per_document,
            "schemaVersion": 1,
            "summary": summary,
            "totalCostCredits": format(spent, "f"),
        },
    )
    print()
    print(
        f"{'method':16} {'docs':>5} {'median errors/doc':>18} {'total':>7} {'worst':>6}"
    )
    for method, data in sorted(summary.items(), key=lambda kv: kv[1]["meanOfMedians"]):
        print(
            f"{method:16} {data['documents']:>5} {data['meanOfMedians']:>18.2f} "
            f"{data['totalMedianErrors']:>7.0f} {data['worstDocument']:>6.0f}"
        )
    print(f"judging cost: ${spent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
