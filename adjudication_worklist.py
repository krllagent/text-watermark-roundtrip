"""Build the manual-adjudication worklist for a single-candidate panel.

Lists every non-unanimous claim vote and every material error with the claim
text, the relevant source sentence(s), and the candidate sentence(s), so a human
can confirm or overturn each panel decision against the actual texts. Performs
no provider calls.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from curated_percent_eval import utc_now, write_json_atomic


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def keywords(claim_text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'-]{3,}|\$?\d[\d,.%]*", claim_text)
    stop = {"that", "with", "from", "were", "have", "this", "their", "which", "than", "into", "after", "before", "would", "about"}
    return {w.lower() for w in words if w.lower() not in stop}


def relevant(text: str, claim_text: str, limit: int = 3) -> list[str]:
    keys = keywords(claim_text)
    scored = []
    for sentence in sentences(text):
        low = sentence.lower()
        score = sum(1 for k in keys if k in low)
        if score:
            scored.append((score, sentence))
    scored.sort(key=lambda row: -row[0])
    return [s for _, s in scored[:limit]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel_input = json.loads(args.input.read_text(encoding="utf-8"))
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    blind = {k: v.split("::") for k, v in panel_input["blindMap"].items()}
    votes: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    errors: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    scores: dict[tuple[str, str], list[tuple[str, int, int]]] = defaultdict(list)
    for call in panel["calls"].values():
        judge = call["judge"]
        for cand in call["candidates"]:
            doc, method = blind[f"{call['batchId']}::{cand['candidateId']}"]
            for row in cand["claims"]:
                votes[(doc, method, row["id"])].append((judge, row["status"]))
            for err in cand.get("materialErrors", []):
                errors[(doc, method)].append((judge, err))
            scores[(doc, method)].append((judge, cand["readabilityPercent"], cand["usabilityPercent"]))
    items = []
    for (doc, method, claim_id), rows in sorted(votes.items()):
        statuses = [s for _, s in rows]
        preserved = statuses.count("preserved")
        if len(set(statuses)) == 1:
            continue
        batch = next(b for b in panel_input["batches"] if b["documentId"] == doc and blind[f"{b['batchId']}::{b['candidates'][0]['candidateId']}"][1] == method)
        claim_text = next(c["text"] for c in batch["claims"] if c["id"] == claim_id)
        items.append(
            {
                "claimId": claim_id,
                "claimText": claim_text,
                "documentId": doc,
                "method": method,
                "panelMajorityPreserved": preserved >= 3,
                "preservedVotes": preserved,
                "votes": dict(rows),
                "candidateSentences": relevant(batch["candidates"][0]["text"], claim_text),
                "sourceSentences": relevant(batch["sourceText"], claim_text),
                "candidateEmpty": not batch["candidates"][0]["text"].strip(),
                "adjudication": None,
                "adjudicationNote": None,
            }
        )
    error_items = []
    for (doc, method), rows in sorted(errors.items()):
        error_items.append({"documentId": doc, "method": method, "errors": [{"judge": j, "error": e} for j, e in rows]})
    artifact = {
        "createdAt": utc_now(),
        "methodology": (
            "Every non-unanimous claim vote and every judge-reported material error from "
            "the single-candidate panel, with keyword-matched source and candidate "
            "sentences for manual review. Adjudication fields are filled by a human "
            "reviewer against the full texts; the panel majority remains the published "
            "number and adjudication is reported separately."
        ),
        "nonUnanimousClaims": items,
        "materialErrors": error_items,
        "scores": {f"{d}::{m}": rows for (d, m), rows in sorted(scores.items())},
        "schemaVersion": 1,
        "sources": panel_input["sources"],
        "verifiedAt": utc_now(),
    }
    write_json_atomic(args.output, artifact)
    print(json.dumps({"nonUnanimous": len(items), "candidatesWithErrors": len(error_items)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
