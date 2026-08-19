"""Combine the removal axis and the meaning axis for experiment 004.

Both come from the same documents: the mean g-value the reference detector
reports for a rewritten text, and the median gross error count four judges
assign to the same rewrite. Chance for the detector is one half, so the
distance from one half is what a removal claim rests on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--judge", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    run = json.loads(args.run.read_text(encoding="utf-8"))
    judged: dict[str, dict] = {}
    for path in args.judge:
        judged.update(json.loads(path.read_text(encoding="utf-8"))["summary"])

    order = {"synonyms": 0, "roundtrip-de": 1, "roundtrip-zh": 2, "paraphrase": 3}
    rows = []
    for key, arm in run["methods"].items():
        model, method = key.split("::")
        errors = judged.get(key, {}).get("meanOfMedians")
        rows.append(
            {
                "cleanDocuments": arm["cleanDocuments"],
                "costPer1000DocumentsUsd": round(
                    float(arm["totalCostCredits"])
                    / max(arm["producedDocuments"], 1)
                    * 1000,
                    3,
                ),
                "grossErrorsPerDocument": errors,
                "meanG": arm["meanG"],
                "method": method,
                "model": model,
                "producedDocuments": arm["producedDocuments"],
                "wordChangeFraction": arm["meanWordDistance"],
            }
        )

    print(
        f"corpus marked meanG {run['corpusMarkedMeanG']:.4f}, "
        f"unmarked {run['corpusUnmarkedMeanG']:.4f}, chance 0.5"
    )
    print()
    print(
        f"{'method':14} {'model':26} {'meanG':>7} {'errors':>7} {'clean':>7} "
        f"{'wordD':>6} {'$/1k':>7}"
    )
    for row in sorted(rows, key=lambda r: (order[r["method"]], r["model"])):
        g = row["meanG"]
        e = row["grossErrorsPerDocument"]
        wd = row["wordChangeFraction"]
        print(
            f"{row['method']:14} {row['model']:26} "
            f"{(f'{g:.4f}' if g is not None else '-'):>7} "
            f"{(f'{e:.2f}' if e is not None else '-'):>7} "
            f"{row['cleanDocuments']:>3}/{row['producedDocuments']:<3} "
            f"{(f'{wd * 100:.0f}%' if wd else '-'):>6} "
            f"{row['costPer1000DocumentsUsd']:>7.2f}"
        )

    print()
    print(
        f"{'method':14} {'meanG':>7} {'errors':>7} {'wordD':>6} {'$/1k':>7} {'models':>7}"
    )
    aggregate = {}
    for method in sorted(order, key=order.get):
        arms = [r for r in rows if r["method"] == method]
        gs = [r["meanG"] for r in arms if r["meanG"] is not None]
        es = [
            r["grossErrorsPerDocument"]
            for r in arms
            if r["grossErrorsPerDocument"] is not None
        ]
        wds = [r["wordChangeFraction"] for r in arms if r["wordChangeFraction"]]
        cs = [r["costPer1000DocumentsUsd"] for r in arms]
        if not gs:
            continue
        aggregate[method] = {
            "meanG": round(statistics.mean(gs), 4),
            "grossErrorsPerDocument": round(statistics.mean(es), 3) if es else None,
            "wordChangeFraction": round(statistics.mean(wds), 4) if wds else None,
            "costPer1000DocumentsUsd": round(statistics.mean(cs), 3),
            "modelCount": len(arms),
        }
        a = aggregate[method]
        err = a["grossErrorsPerDocument"]
        wd = a["wordChangeFraction"]
        err_text = f"{err:.2f}" if err is not None else "-"
        wd_text = f"{wd * 100:.0f}%" if wd else "-"
        print(
            f"{method:14} {a['meanG']:>7.4f} {err_text:>7} {wd_text:>6} "
            f"{a['costPer1000DocumentsUsd']:>7.2f} {a['modelCount']:>7}"
        )

    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "aggregate": aggregate,
                    "arms": rows,
                    "corpusMarkedMeanG": run["corpusMarkedMeanG"],
                    "corpusUnmarkedMeanG": run["corpusUnmarkedMeanG"],
                    "schemaVersion": 1,
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
