"""Derive panel v5: single-candidate prompts with canary-qualified judges.

The v3 panel let three judges see five candidates of one document in one
prompt, and at least one verdict carried a neighbouring candidate's errors
across. v5 judges every candidate alone. The single-candidate canary (identical,
tampered, and empty prompts) disqualified Claude Haiku 4.5 (awarded all ten
claims and 100% usability to an empty text, detected 2/5 tampers) and Grok 4.20
(awarded all ten claims to an empty text), so the Anthropic and xAI seats move
to Claude Sonnet 5 and Grok 4.6, subject to a fresh frozen canary. Grok 4.6
rejects reasoning_effort="none" (HTTP 400), so its seat carries a frozen
per-judge reasoningEffort="low"; Grok 4.3 also rewarded an empty text (9/10
preserved) and Grok 4.5 returned HTTP 400 in the probe.
"""

from __future__ import annotations

import hashlib
import json

from curated_percent_eval import ROOT, utc_now, write_json_atomic


SOURCE = ROOT / "configs" / "curated-percent-eval-v3.json"
OUTPUT = ROOT / "configs" / "curated-percent-eval-v5.json"

PANEL_BUDGET_USD = "1.60"
REPLACEMENTS = {
    "Anthropic": {
        "expected": "anthropic/claude-haiku-4.5",
        "model": "anthropic/claude-sonnet-5",
        "promptUsdPerToken": "0.000002",
        "completionUsdPerToken": "0.00001",
    },
    "xAI": {
        "expected": "x-ai/grok-4.20",
        "model": "x-ai/grok-4.6",
        "promptUsdPerToken": "0.000002",
        "completionUsdPerToken": "0.000006",
        "reasoningEffort": "low",
    },
}


def derive(source: dict[str, object], source_sha256: str) -> dict[str, object]:
    value = json.loads(json.dumps(source))
    for vendor, spec in REPLACEMENTS.items():
        matches = [judge for judge in value["judges"] if judge["vendor"] == vendor]
        if len(matches) != 1 or matches[0]["model"] != spec["expected"]:
            raise ValueError(f"v3 {vendor} judge differs from the disqualified endpoint")
        matches[0].update(
            {
                "completionUsdPerToken": spec["completionUsdPerToken"],
                "model": spec["model"],
                "promptUsdPerToken": spec["promptUsdPerToken"],
            }
        )
        if "reasoningEffort" in spec:
            matches[0]["reasoningEffort"] = spec["reasoningEffort"]
    budgets = value["budgetsUsd"]
    others = sum(
        float(v) for k, v in budgets.items() if k not in ("panel", "totalAdditional")
    )
    budgets["panel"] = PANEL_BUDGET_USD
    budgets["totalAdditional"] = f"{others + float(PANEL_BUDGET_USD):.2f}"
    panel = value.setdefault("panel", {})
    panel["candidatesPerPrompt"] = 1
    panel["canaryCalls"] = 12
    value["parentDerivation"] = value.get("derivedFrom")
    value["derivedFrom"] = {
        "path": "configs/curated-percent-eval-v3.json",
        "reason": (
            "Five-candidate prompts let judges transfer one candidate's errors to a "
            "neighbour. Judge every candidate alone. The single-candidate canary with "
            "an empty text disqualified Claude Haiku 4.5 and Grok 4.20; replace them "
            "with Claude Sonnet 5 and Grok 4.6 and raise the panel budget for the "
            "larger number of prompts."
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
