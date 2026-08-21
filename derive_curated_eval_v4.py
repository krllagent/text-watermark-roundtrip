"""Derive panel v4 with the canary-qualified xAI fallback judge."""

from __future__ import annotations

import hashlib
import json

from curated_percent_eval import ROOT, utc_now, write_json_atomic


SOURCE = ROOT / "configs" / "curated-percent-eval-v3.json"
OUTPUT = ROOT / "configs" / "curated-percent-eval-v4.json"


def derive(source: dict[str, object], source_sha256: str) -> dict[str, object]:
    value = json.loads(json.dumps(source))
    matches = [judge for judge in value["judges"] if judge["vendor"] == "xAI"]
    if len(matches) != 1 or matches[0]["model"] != "x-ai/grok-4.20":
        raise ValueError("v3 xAI judge differs from the malformed-response endpoint")
    matches[0].update(
        {
            "completionUsdPerToken": "0.000006",
            "model": "x-ai/grok-4.6",
            "promptUsdPerToken": "0.000002",
        }
    )
    value["parentDerivation"] = value.get("derivedFrom")
    value["derivedFrom"] = {
        "path": "configs/curated-percent-eval-v3.json",
        "reason": (
            "Grok 4.20 returned malformed JSON in two full-batch attempts at different "
            "token caps. Replace only the xAI judge with Grok 4.6, which had a prior "
            "10/10 structured-output record, subject to a fresh frozen canary."
        ),
        "sha256": source_sha256,
    }
    value["verifiedAt"] = utc_now()
    return value


def main() -> int:
    raw = SOURCE.read_bytes()
    write_json_atomic(OUTPUT, derive(json.loads(raw), hashlib.sha256(raw).hexdigest()))
    print(hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
