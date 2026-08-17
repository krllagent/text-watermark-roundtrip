"""Re-measure the published four-method comparison with a reader-free metric.

The published semantic failure counts came from one automatic judge, and that
judge was later shown to miss major errors. Anchor retention needs no judge: a
logical anchor is a word whose replacement changes meaning, so the multiset of
anchors in the output should equal the multiset in the source. Comparing that
multiset is objective and identical across methods.

This reads the stored outputs of the frozen experiment and reports, per method,
how many of the source anchors survived. No provider call is made.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

from run_model_canary_terra_locked import _ANCHOR_RE
import run_model_canary_luna as engine


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "results" / "experiment-raw-v1.json"
OUTPUT = ROOT / "results" / "anchor-retention-reanalysis-v1.json"
_WORD = re.compile(r"[A-Za-z]+")


def anchors(text: str) -> list[str]:
    """Anchor occurrences in a plain text, lowercased for comparison."""
    return sorted(match.group(0).lower() for match in _ANCHOR_RE.finditer(text))


def multiset_retention(source: str, output: str) -> dict[str, object]:
    expected = anchors(source)
    observed = anchors(output)
    pool = list(observed)
    kept = 0
    missing: list[str] = []
    for value in expected:
        if value in pool:
            pool.remove(value)
            kept += 1
        else:
            missing.append(value)
    return {
        "added": len(pool),
        "expected": len(expected),
        "kept": kept,
        "missingSample": sorted(set(missing))[:8],
        "retention": (kept / len(expected)) if expected else None,
    }


def main() -> int:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    report: dict[str, object] = {}
    for method in raw["methods"]:
        method_id = method["methodId"]
        rows = []
        for document in method["documents"]:
            source = document.get("markedInputText") or document.get("originalText")
            output = document.get("outputText")
            if not isinstance(source, str) or not isinstance(output, str) or not output:
                continue
            row = multiset_retention(source, output)
            row["documentId"] = document["documentId"]
            row["sourceWords"] = len(_WORD.findall(source))
            rows.append(row)
        retentions = [r["retention"] for r in rows if r["retention"] is not None]
        total_expected = sum(int(r["expected"]) for r in rows)
        total_kept = sum(int(r["kept"]) for r in rows)
        report[method_id] = {
            "documents": rows,
            "documentsMeasured": len(rows),
            "meanRetention": (sum(retentions) / len(retentions))
            if retentions
            else None,
            "minRetention": min(retentions) if retentions else None,
            "pooledRetention": (total_kept / total_expected)
            if total_expected
            else None,
            "totalAnchors": total_expected,
        }
    engine.atomic_write(OUTPUT, {"methods": report, "schemaVersion": 1})
    print(
        f"{'method':16} {'docs':>5} {'anchors':>8} {'pooled':>8} {'mean':>7} {'worst':>7}"
    )
    for method_id, data in report.items():
        print(
            f"{method_id:16} {data['documentsMeasured']:>5} {data['totalAnchors']:>8} "
            f"{data['pooledRetention'] * 100:>7.1f}% {data['meanRetention'] * 100:>6.1f}% "
            f"{data['minRetention'] * 100:>6.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
