"""Locate the residual SynthID signal in Experiment 004 rewrites.

For every transformed document, this analysis splits detector positions into
exact tokenizer n-grams copied from the marked source and novel n-grams.  The
split uses the same n-gram width and context-repetition mask as the detector.
Exact integer g-value counts are stored alongside display means.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "results" / "synthid-corpus-v1.json"
DEFAULT_METHODS = ROOT / "results" / "exp004-methods-v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "exp004-ngram-retention-v1.json"
DEFAULT_THRESHOLD = 0.5067

COUNT_FIELDS = (
    "validPositions",
    "reusedPositions",
    "novelPositions",
    "gValueCount",
    "gOneCount",
    "reusedGValueCount",
    "reusedGOneCount",
    "novelGValueCount",
    "novelGOneCount",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def fit_line(points: Iterable[tuple[float, float]]) -> dict[str, float]:
    pairs = list(points)
    if len(pairs) < 2:
        raise ValueError("at least two points are required")
    xs = [point[0] for point in pairs]
    ys = [point[1] for point in pairs]
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    variance_y = sum((y - mean_y) ** 2 for y in ys)
    if variance_x == 0 or variance_y == 0:
        raise ValueError("both coordinates must vary")
    slope = covariance / variance_x
    intercept = mean_y - slope * mean_x
    residual_sum_squares = sum(
        (y - (intercept + slope * x)) ** 2 for x, y in pairs
    )
    return {
        "intercept": intercept,
        "pearsonR": covariance / math.sqrt(variance_x * variance_y),
        "rSquared": 1 - residual_sum_squares / variance_y,
        "slope": slope,
    }


def aggregate_counts(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    materialized = list(rows)
    totals = {
        field: sum(int(row[field]) for row in materialized)
        for field in COUNT_FIELDS
    }
    mean_g = ratio(totals["gOneCount"], totals["gValueCount"])
    reused_mean_g = ratio(
        totals["reusedGOneCount"], totals["reusedGValueCount"]
    )
    novel_mean_g = ratio(totals["novelGOneCount"], totals["novelGValueCount"])
    observed_excess = (
        totals["gOneCount"] - 0.5 * totals["gValueCount"]
    )
    reused_excess = (
        totals["reusedGOneCount"] - 0.5 * totals["reusedGValueCount"]
    )
    return {
        "documentCount": len(materialized),
        **totals,
        "exactNgramReuseFraction": ratio(
            totals["reusedPositions"], totals["validPositions"]
        ),
        "meanG": mean_g,
        "novelMeanG": novel_mean_g,
        "reusedContributionToObservedWatermarkExcess": (
            reused_excess / observed_excess if observed_excess else None
        ),
        "reusedMeanG": reused_mean_g,
    }


def analyze(
    corpus_path: Path,
    methods_path: Path,
    *,
    threshold: float,
) -> dict[str, object]:
    try:
        import torch
        from transformers import AutoTokenizer, SynthIDTextWatermarkLogitsProcessor
    except ImportError as error:
        raise SystemExit(
            "This analysis requires the repository's .venv-wm environment"
        ) from error

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    experiment = json.loads(methods_path.read_text(encoding="utf-8"))
    ngram_len = int(corpus["ngramLen"])
    tokenizer = AutoTokenizer.from_pretrained(corpus["model"])
    processor = SynthIDTextWatermarkLogitsProcessor(
        ngram_len=ngram_len,
        keys=corpus["keys"],
        sampling_table_size=65_536,
        sampling_table_seed=0,
        context_history_size=corpus["contextHistorySize"],
        device=torch.device("cpu"),
    )
    source_token_ids = {
        document["documentId"]: tokenizer(
            [document["markedText"]],
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0].tolist()
        for document in corpus["documents"]
    }
    source_ngrams = {
        document_id: {
            tuple(token_ids[index : index + ngram_len])
            for index in range(len(token_ids) - ngram_len + 1)
        }
        for document_id, token_ids in source_token_ids.items()
    }

    rows: list[dict[str, object]] = []
    excluded = []
    maximum_recalculation_delta = 0.0
    for arm_id, arm in sorted(experiment["methods"].items()):
        model, method = arm_id.split("::", 1)
        for document in arm["documents"]:
            if document.get("outcome") != "completed":
                continue
            reported_mean = (document.get("detector") or {}).get("meanG")
            text = document.get("evaluatedOutputText") or ""
            ids = tokenizer(
                [text], return_tensors="pt", add_special_tokens=False
            )["input_ids"]
            if reported_mean is None or ids.shape[1] < ngram_len:
                excluded.append(
                    {
                        "armId": arm_id,
                        "documentId": document["documentId"],
                        "reason": "not_scorable",
                    }
                )
                continue

            g_values = processor.compute_g_values(ids)[0]
            context_mask = processor.compute_context_repetition_mask(ids)[0].bool()
            output_ids = ids[0].tolist()
            copied = torch.tensor(
                [
                    tuple(output_ids[index : index + ngram_len])
                    in source_ngrams[document["documentId"]]
                    for index in range(len(output_ids) - ngram_len + 1)
                ],
                dtype=torch.bool,
            )
            reused_mask = context_mask & copied
            novel_mask = context_mask & ~copied
            valid_positions = int(context_mask.sum().item())
            reused_positions = int(reused_mask.sum().item())
            novel_positions = int(novel_mask.sum().item())
            depth = int(g_values.shape[-1])
            counts = {
                "validPositions": valid_positions,
                "reusedPositions": reused_positions,
                "novelPositions": novel_positions,
                "gValueCount": valid_positions * depth,
                "gOneCount": int(g_values[context_mask].sum().item()),
                "reusedGValueCount": reused_positions * depth,
                "reusedGOneCount": int(g_values[reused_mask].sum().item()),
                "novelGValueCount": novel_positions * depth,
                "novelGOneCount": int(g_values[novel_mask].sum().item()),
            }
            summary = aggregate_counts([counts])
            recomputed_mean = float(summary["meanG"])
            maximum_recalculation_delta = max(
                maximum_recalculation_delta,
                abs(recomputed_mean - float(reported_mean)),
            )
            rows.append(
                {
                    "armId": arm_id,
                    "detectedAtPublishedThreshold": recomputed_mean >= threshold,
                    "documentId": document["documentId"],
                    "method": method,
                    "model": model,
                    "outputTokenCount": int(ids.shape[1]),
                    "reportedMeanG": reported_mean,
                    **counts,
                    "exactNgramReuseFraction": summary["exactNgramReuseFraction"],
                    "meanG": recomputed_mean,
                    "novelMeanG": summary["novelMeanG"],
                    "reusedMeanG": summary["reusedMeanG"],
                }
            )

    overall = aggregate_counts(rows)
    regression = fit_line(
        (float(row["exactNgramReuseFraction"]), float(row["meanG"]))
        for row in rows
    )
    regression["thresholdCrossingReuseFraction"] = (
        (threshold - regression["intercept"]) / regression["slope"]
    )
    pooled_threshold_reuse = (
        (threshold - float(overall["novelMeanG"]))
        / (float(overall["reusedMeanG"]) - float(overall["novelMeanG"]))
    )
    method_ids = sorted({str(row["method"]) for row in rows})
    arm_ids = sorted({str(row["armId"]) for row in rows})

    def grouped(group_rows: list[dict[str, object]]) -> dict[str, object]:
        result = aggregate_counts(group_rows)
        result["detectedDocumentCount"] = sum(
            bool(row["detectedAtPublishedThreshold"]) for row in group_rows
        )
        return result

    return {
        "analysisVersion": "exact-tokenizer-ngram-retention-v1",
        "documents": rows,
        "excludedDocuments": excluded,
        "finding": {
            "detectedDocumentCount": sum(
                bool(row["detectedAtPublishedThreshold"]) for row in rows
            ),
            "exactNgramReuseMeanGRegression": regression,
            "maxDetectorRecalculationAbsDelta": maximum_recalculation_delta,
            "pooledReuseFractionAtPublishedThreshold": pooled_threshold_reuse,
            **overall,
        },
        "inputs": {
            "corpus": {
                "path": str(corpus_path.relative_to(ROOT)),
                "sha256": sha256_file(corpus_path),
            },
            "methods": {
                "path": str(methods_path.relative_to(ROOT)),
                "sha256": sha256_file(methods_path),
            },
        },
        "methodAggregates": {
            method: grouped([row for row in rows if row["method"] == method])
            for method in method_ids
        },
        "armAggregates": {
            arm_id: grouped([row for row in rows if row["armId"] == arm_id])
            for arm_id in arm_ids
        },
        "methodology": (
            "Re-tokenize each completed Experiment 004 output with the source "
            "model tokenizer. At every non-repeated detector position, classify "
            "the full SynthID n-gram as reused when the identical token-id tuple "
            "appears anywhere in that document's marked source; otherwise classify "
            "it as novel. Sum binary g-values separately for both partitions and "
            "fit per-document meanG against exact n-gram reuse fraction."
        ),
        "publishedDetectionThreshold": threshold,
        "schemaVersion": 1,
        "sources": [
            {
                "title": "SynthID-Text: watermarking large language model output",
                "url": "https://doi.org/10.1038/s41586-024-08025-4",
            },
            {
                "title": "Transformers SynthID text reference implementation",
                "url": "https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/generation/logits_process.py",
            },
            {
                "title": "Experiment repository",
                "url": "https://github.com/krllagent/text-watermark-roundtrip",
            },
        ],
        "verifiedAt": utc_now(),
        "watermark": {
            "contextHistorySize": corpus["contextHistorySize"],
            "depth": len(corpus["keys"]),
            "model": corpus["model"],
            "ngramLen": ngram_len,
            "samplingTableSeed": 0,
            "samplingTableSize": 65_536,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--methods", type=Path, default=DEFAULT_METHODS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)
    payload = analyze(
        args.corpus.resolve(),
        args.methods.resolve(),
        threshold=args.threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    finding = payload["finding"]
    print(
        json.dumps(
            {
                "detected": finding["detectedDocumentCount"],
                "documents": finding["documentCount"],
                "novelMeanG": finding["novelMeanG"],
                "output": str(args.output),
                "pearsonR": finding["exactNgramReuseMeanGRegression"]["pearsonR"],
                "reusedMeanG": finding["reusedMeanG"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
