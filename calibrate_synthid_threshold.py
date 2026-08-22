"""Calibrate SynthID mean-g detection with an exact random-table null.

For a fixed token sequence, repeated n-gram hashes can address the same entry
of SynthID's binary sampling table.  Treating every g-value as independent
therefore understates variance.  This runner counts how many times each table
entry contributes, groups entries by their integer weight, and samples the
resulting weighted Bernoulli sum.  It is model-free and requires no paid API.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "results" / "quality-synthid-corpus-gpu-v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "quality-synthid-calibration-v1.json"
DEFAULT_REPLICATES = 100_000
DEFAULT_SEED = 20260821
DEFAULT_ALPHA = 0.01
LEGACY_THRESHOLD = 0.5067


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def nearest_rank(values: np.ndarray, quantile: float) -> float:
    if values.size == 0 or not 0 < quantile <= 1:
        raise ValueError("nearest-rank quantile requires values and 0 < q <= 1")
    ordered = np.sort(values)
    index = math.ceil(quantile * ordered.size) - 1
    return float(ordered[index])


def simulate_weighted_bernoulli_null(
    frequencies: dict[int, int],
    *,
    total_weight: int,
    replicates: int,
    seed: int,
) -> np.ndarray:
    if total_weight <= 0 or replicates <= 0:
        raise ValueError("positive total weight and replicate count are required")
    if sum(weight * count for weight, count in frequencies.items()) != total_weight:
        raise ValueError("weight frequencies do not sum to total weight")
    rng = np.random.default_rng(seed)
    totals = np.zeros(replicates, dtype=np.int64)
    for weight, count in sorted(frequencies.items()):
        if weight <= 0 or count <= 0:
            raise ValueError("weight frequencies must be positive")
        totals += weight * rng.binomial(count, 0.5, size=replicates)
    return totals.astype(np.float64) / total_weight


def _weights_for_trace(corpus: dict[str, object], trace: dict[str, object]):
    import torch
    from transformers import SynthIDTextWatermarkLogitsProcessor

    processor = SynthIDTextWatermarkLogitsProcessor(
        ngram_len=int(corpus["ngramLen"]),
        keys=corpus["keys"],
        sampling_table_size=len(corpus["samplingTable"]),
        sampling_table_seed=0,
        context_history_size=int(corpus["contextHistorySize"]),
        device=torch.device("cpu"),
    )
    ids = torch.tensor([trace["detectorTokenIds"]], dtype=torch.long)
    ngrams = ids.unfold(1, int(corpus["ngramLen"]), 1)
    indices = processor.compute_ngram_keys(ngrams)[0] % len(corpus["samplingTable"])
    mask = torch.tensor(trace["contextRepetitionMask"], dtype=torch.bool)
    if mask.shape[0] != indices.shape[0]:
        raise ValueError("stored context mask does not match detector token sequence")
    flat = indices[mask].flatten()
    table_weights = torch.bincount(
        flat, minlength=len(corpus["samplingTable"])
    ).to(torch.int64)
    nonzero = table_weights[table_weights > 0]
    frequency_tensor = torch.bincount(nonzero)
    frequencies = {
        weight: int(frequency_tensor[weight].item())
        for weight in range(1, int(frequency_tensor.shape[0]))
        if int(frequency_tensor[weight].item())
    }
    return frequencies, int(flat.numel())


def _side_calibration(
    corpus: dict[str, object],
    trace: dict[str, object],
    *,
    alpha: float,
    replicates: int,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    frequencies, total_weight = _weights_for_trace(corpus, trace)
    if total_weight != int(trace["validGValueCount"]):
        raise ValueError("null weights disagree with stored valid g-value count")
    values = simulate_weighted_bernoulli_null(
        frequencies,
        total_weight=total_weight,
        replicates=replicates,
        seed=seed,
    )
    observed = float(trace["meanG"])
    threshold = nearest_rank(values, 1 - alpha)
    p_value = (1 + int(np.count_nonzero(values >= observed))) / (replicates + 1)
    summary = {
        "detectedAtLengthAwareAlpha": observed >= threshold,
        "empiricalPValue": p_value,
        "fprAtLegacyThreshold": float(np.mean(values >= LEGACY_THRESHOLD)),
        "legacyThreshold": LEGACY_THRESHOLD,
        "lengthAwareThreshold": threshold,
        "nullMean": float(values.mean()),
        "nullStandardDeviation": float(values.std(ddof=1)),
        "observedMeanG": observed,
        "samplingTableEntriesUsed": sum(frequencies.values()),
        "validGValueCount": total_weight,
        "weightFrequencies": {str(key): value for key, value in frequencies.items()},
    }
    return summary, values


def calibrate(
    corpus: dict[str, object],
    *,
    corpus_path: Path,
    alpha: float,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    rows = []
    unmarked_nulls = []
    for index, document in enumerate(corpus["documents"]):
        marked, _ = _side_calibration(
            corpus,
            document["marked"],
            alpha=alpha,
            replicates=replicates,
            seed=seed + index * 2,
        )
        unmarked, null_values = _side_calibration(
            corpus,
            document["unmarked"],
            alpha=alpha,
            replicates=replicates,
            seed=seed + index * 2 + 1,
        )
        unmarked_nulls.append(null_values)
        rows.append(
            {
                "documentId": document["documentId"],
                "marked": marked,
                "unmarked": unmarked,
            }
        )
    mixture = np.concatenate(unmarked_nulls)
    single_threshold = nearest_rank(mixture, 1 - alpha)
    marked_scores = [float(row["marked"]["observedMeanG"]) for row in rows]
    unmarked_scores = [float(row["unmarked"]["observedMeanG"]) for row in rows]
    return {
        "alpha": alpha,
        "corpus": {
            "path": str(corpus_path),
            "sha256": sha256_file(corpus_path),
        },
        "createdAt": utc_now(),
        "documents": rows,
        "legacyThresholdAudit": {
            "claimedFpr": 0.01,
            "empiricalMixtureFpr": float(np.mean(mixture >= LEGACY_THRESHOLD)),
            "threshold": LEGACY_THRESHOLD,
        },
        "methodology": (
            "For every exact marked and unmarked detector token sequence, count all "
            "lookups of each binary SynthID sampling-table entry after the stored "
            "context-repetition mask. Draw an exact weighted Bernoulli null in which "
            "each table entry is independently random, preserving hash collisions and "
            "within-text dependencies. Use the nearest-rank upper 1-alpha quantile. "
            "The single threshold is the same quantile of the equal-document mixture "
            "of all unmarked conditional nulls. No model or paid API is used."
        ),
        "replicatesPerSequence": replicates,
        "schemaVersion": 1,
        "seed": seed,
        "singleThreshold": {
            "markedDetectedCount": sum(score >= single_threshold for score in marked_scores),
            "markedDocumentCount": len(marked_scores),
            "observedUnmarkedFalsePositiveCount": sum(
                score >= single_threshold for score in unmarked_scores
            ),
            "threshold": single_threshold,
            "unmarkedDocumentCount": len(unmarked_scores),
        },
        "sources": [
            "https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.SynthIDTextWatermarkLogitsProcessor",
            "https://doi.org/10.1038/s41586-024-08025-4",
        ],
        "verifiedAt": utc_now(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args(argv)
    if not 0 < args.alpha < 0.5:
        raise SystemExit("alpha must be between zero and one half")
    corpus = json.loads(args.input.read_text(encoding="utf-8"))
    artifact = calibrate(
        corpus,
        corpus_path=args.input,
        alpha=args.alpha,
        replicates=args.replicates,
        seed=args.seed,
    )
    write_json_atomic(args.output, artifact)
    print(
        json.dumps(
            {
                "legacyFpr": artifact["legacyThresholdAudit"]["empiricalMixtureFpr"],
                "markedDetected": artifact["singleThreshold"]["markedDetectedCount"],
                "output": str(args.output),
                "threshold": artifact["singleThreshold"]["threshold"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
