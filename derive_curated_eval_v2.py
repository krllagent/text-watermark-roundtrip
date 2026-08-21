"""Derive the frozen curated evaluation v2 after the Google canary rejection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from curated_percent_eval import ROOT, utc_now, write_json_atomic


SOURCE = ROOT / "configs" / "curated-percent-eval-v1.json"
OUTPUT = ROOT / "configs" / "curated-percent-eval-v2.json"


def derive(source: dict[str, object], *, source_sha256: str) -> dict[str, object]:
    value = json.loads(json.dumps(source))
    matches = [
        judge for judge in value["judges"] if judge["vendor"] == "Google"
    ]
    if len(matches) != 1 or matches[0]["model"] != "google/gemini-3.7-flash":
        raise ValueError("v1 Google judge differs from the rejected canary contract")
    matches[0].update(
        {
            "completionUsdPerToken": "0.00000034",
            "model": "google/gemma-4-31b-it",
            "promptUsdPerToken": "0.0000001",
        }
    )
    value["derivedFrom"] = {
        "path": "configs/curated-percent-eval-v1.json",
        "reason": (
            "Gemini 3.7 Flash rejected the canary because reasoning is mandatory; "
            "the experiment requires reasoning effort none for bounded cost."
        ),
        "sha256": source_sha256,
    }
    value["verifiedAt"] = utc_now()
    return value


def main() -> int:
    raw = SOURCE.read_bytes()
    source = json.loads(raw)
    output = derive(source, source_sha256=hashlib.sha256(raw).hexdigest())
    write_json_atomic(OUTPUT, output)
    print(hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
