"""Compute official P-SP for curated method outputs on CPU."""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Callable, Mapping, Sequence

from curated_percent_eval import ROOT, sha256_file, sha256_text, utc_now, write_json_atomic


DEFAULT_CORPUS = ROOT / "results" / "quality-synthid-corpus-curated-v1.json"
DEFAULT_METHODS = ROOT / "results" / "curated-methods-v1.json"
DEFAULT_DIPPER = ROOT / "results" / "curated-dipper-v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "curated-psp-v1.json"


def score_or_failure(
    scorer: Callable[[str, str], float] | None, source: str, candidate: str
) -> float:
    if not candidate.strip():
        return 0.0
    if scorer is None:
        raise ValueError("nonempty candidates require a P-SP scorer")
    return float(scorer(source, candidate))


def summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_method: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(float(row["psp"]))
    return {
        method: {
            "meanPspPercent": 100.0 * statistics.mean(values),
            "pairCount": len(values),
        }
        for method, values in sorted(by_method.items())
    }


def _load_model(psp_root: Path):
    old_cwd = Path.cwd()
    sys.path.insert(0, str(psp_root))
    os.chdir(psp_root)
    try:
        module = importlib.import_module("src.models.psp_model")
        model = module.PspModel()
    finally:
        os.chdir(old_cwd)
    return model, old_cwd


def _candidate_rows(corpus, methods, dipper):
    sources = {
        document["documentId"]: document["marked"]["text"]
        for document in corpus["documents"]
    }
    for row in methods["documents"]:
        completed = row.get("outcome") == "completed"
        yield {
            "candidate": str(row["evaluatedOutputText"]) if completed else "",
            "documentId": row["documentId"],
            "method": row["method"],
            "pipelineCompleted": completed,
            "source": sources[row["documentId"]],
        }
    if dipper is not None:
        for row in dipper["documents"]:
            if row.get("inputKind") != "marked-source":
                continue
            yield {
                "candidate": str(row["dipperOutputText"]),
                "documentId": row["documentId"],
                "method": "dipper",
                "pipelineCompleted": True,
                "source": sources[row["documentId"]],
            }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--methods", type=Path, default=DEFAULT_METHODS)
    parser.add_argument("--dipper", type=Path, default=DEFAULT_DIPPER)
    parser.add_argument("--psp-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--without-dipper", action="store_true")
    args = parser.parse_args(argv)
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    methods = json.loads(args.methods.read_text(encoding="utf-8"))
    dipper = None
    if not args.without_dipper:
        dipper = json.loads(args.dipper.read_text(encoding="utf-8"))
    model, _ = _load_model(args.psp_root.resolve())
    rows = []
    for index, row in enumerate(_candidate_rows(corpus, methods, dipper), start=1):
        source = str(row.pop("source"))
        candidate = str(row.pop("candidate"))
        value = score_or_failure(model.get_psp, source, candidate)
        rows.append(
            {
                **row,
                "candidateTextSha256": sha256_text(candidate),
                "pairId": f"{row['documentId']}::{row['method']}",
                "psp": value,
                "pspPercent": 100.0 * value,
                "sourceTextSha256": sha256_text(source),
            }
        )
        print(
            json.dumps(
                {"event": "psp", "index": index, "pairId": rows[-1]["pairId"], "psp": value},
                sort_keys=True,
            ),
            flush=True,
        )
    weight_path = args.psp_root / "src" / "models" / "psp" / "model.para.lc.100.pt"
    sp_path = args.psp_root / "src" / "models" / "psp" / "paranmt.model"
    now = utc_now()
    artifact = {
        "createdAt": now,
        "inputs": {
            "corpusSha256": sha256_file(args.corpus),
            "dipperSha256": None if dipper is None else sha256_file(args.dipper),
            "methodsSha256": sha256_file(args.methods),
        },
        "methodology": (
            "Run the official P-SP model from the ETH watermark-stealing tree locally "
            "on CPU. Score source and candidate directly; assign zero to a pipeline "
            "failure that produced no candidate."
        ),
        "model": {
            "paranmtModelSha256": sha256_file(sp_path),
            "weightsSha256": sha256_file(weight_path),
        },
        "rows": rows,
        "schemaVersion": 1,
        "sources": [
            "https://github.com/eth-sri/watermark-stealing",
            "https://www.cs.cmu.edu/~jwieting/paraphrase-at-scale-english.zip",
        ],
        "summary": summarize(rows),
        "verifiedAt": now,
    }
    write_json_atomic(args.output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
