"""Derive panel v3 with enough completion room for five-candidate batches."""

from __future__ import annotations

import hashlib
import json

from curated_percent_eval import ROOT, utc_now, write_json_atomic


SOURCE = ROOT / "configs" / "curated-percent-eval-v2.json"
OUTPUT = ROOT / "configs" / "curated-percent-eval-v3.json"


def derive(source: dict[str, object], source_sha256: str) -> dict[str, object]:
    value = json.loads(json.dumps(source))
    if value["panel"]["maxCompletionTokensPerCall"] != 900:
        raise ValueError("v2 panel token cap differs from rejected full batch")
    value["panel"]["maxCompletionTokensPerCall"] = 1400
    value["parentDerivation"] = value.get("derivedFrom")
    value["derivedFrom"] = {
        "path": "configs/curated-percent-eval-v2.json",
        "reason": (
            "The first five-candidate full batch produced two locally malformed or "
            "truncated JSON responses at the 900-token cap. Increase only the output "
            "cap; judges, prompt, claims, schema, scoring, and budgets remain frozen."
        ),
        "sha256": source_sha256,
    }
    value["verifiedAt"] = utc_now()
    return value


def main() -> int:
    raw = SOURCE.read_bytes()
    write_json_atomic(
        OUTPUT,
        derive(json.loads(raw), hashlib.sha256(raw).hexdigest()),
    )
    print(hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
