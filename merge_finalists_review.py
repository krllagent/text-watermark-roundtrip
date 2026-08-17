"""Merge the blind reviews of the two finalists and unblind them.

Readers judged twelve interleaved pairs without model labels. This merges
their parts, validates every finding against the committed packet, and only
then joins the verdicts to the model each pair came from.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_model_canary_luna as engine


ROOT = Path(__file__).resolve().parent
PACKET = ROOT / "results" / "finalists-blind-packet-v1.json"
MAPPING = ROOT / "results" / "finalists-blind-mapping-v1.json"
PARTS = sorted((ROOT / "results").glob(".finalists-review-part*.json"))
OUTPUT = ROOT / "results" / "finalists-blind-review-v1.json"
FIELDS = {"criterion", "sourceQuote", "candidateQuote", "explanation"}


def main() -> int:
    packet = json.loads(PACKET.read_text(encoding="utf-8"))
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    pair_by_id = {row["pairId"]: row for row in packet["pairs"]}
    reviews: dict[str, dict[str, object]] = {}
    for part in PARTS:
        for row in json.loads(part.read_text(encoding="utf-8"))["reviews"]:
            pair_id = row["pairId"]
            if pair_id in reviews:
                raise SystemExit(f"duplicate review for {pair_id}")
            if pair_id not in pair_by_id:
                raise SystemExit(f"review references an unknown pair {pair_id}")
            verdict = row["verdict"]
            findings = row["findings"]
            if verdict not in {"pass", "minor", "major"}:
                raise SystemExit(f"invalid verdict for {pair_id}")
            if (verdict == "pass") != (not findings):
                raise SystemExit(f"verdict and findings disagree for {pair_id}")
            for finding in findings:
                if set(finding) != FIELDS:
                    raise SystemExit(f"finding fields changed for {pair_id}")
                if finding["sourceQuote"] not in pair_by_id[pair_id]["sourceText"]:
                    raise SystemExit(f"sourceQuote is not in the source for {pair_id}")
                if (
                    finding["candidateQuote"]
                    not in pair_by_id[pair_id]["candidateText"]
                ):
                    raise SystemExit(
                        f"candidateQuote is not in the candidate for {pair_id}"
                    )
            reviews[pair_id] = row
    missing = set(pair_by_id) - set(reviews)
    if missing:
        raise SystemExit(f"missing reviews for {sorted(missing)}")

    per_model: dict[str, dict[str, object]] = {}
    joined = []
    for row in mapping["mapping"]:
        review = reviews[row["pairId"]]
        model = row["model"]
        bucket = per_model.setdefault(
            model, {"pass": 0, "minor": 0, "major": 0, "findings": 0}
        )
        bucket[review["verdict"]] += 1
        bucket["findings"] += len(review["findings"])
        joined.append(
            {
                "automaticIssues": row["automaticIssues"],
                "documentId": row["documentId"],
                "findings": review["findings"],
                "model": model,
                "pairId": row["pairId"],
                "verdict": review["verdict"],
            }
        )
    result = {
        "packetSha256": packet["packetSha256"],
        "perModel": per_model,
        "results": sorted(joined, key=lambda r: (r["model"], r["documentId"])),
        "reviewers": "three independent blind readers, four interleaved pairs each",
        "schemaVersion": 1,
    }
    engine.atomic_write(OUTPUT, result)
    for model, bucket in sorted(per_model.items()):
        total = bucket["pass"] + bucket["minor"] + bucket["major"]
        print(
            f"{model:22} pass {bucket['pass']}/{total}  minor {bucket['minor']}  "
            f"major {bucket['major']}  findings {bucket['findings']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
