"""Run the blinded four-vendor curated panel with ten-percent score steps.

The script prepares a frozen two-candidate canary, then executes either canary
or full batches through the normal OpenRouter environment contract. Every call
is durably reserved before dispatch, provider-reported cost is checkpointed,
and unresolved in-flight calls block any automatic redispatch.
"""

from __future__ import annotations

import argparse
from concurrent import futures
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import threading
from typing import Mapping, Sequence

from curated_percent_eval import (
    ALLOWED_PERCENTAGES,
    DEFAULT_CONFIG,
    ROOT,
    load_design,
    sha256_file,
    utc_now,
    validate_panel_candidate,
    write_json_atomic,
)
from run_synthid_smoke import ExplicitNonZdrTransport
import unmark


DEFAULT_CANARY_INPUT = ROOT / "results" / "curated-panel-canary-input-v1.json"
DEFAULT_CANARY_OUTPUT = ROOT / "results" / "curated-panel-canary-v1.json"
DEFAULT_FULL_INPUT = ROOT / "results" / "curated-panel-input-v1.json"
DEFAULT_FULL_OUTPUT = ROOT / "results" / "curated-panel-v1.json"

INSTRUCTION = """You are one member of a blinded text-quality panel. Compare one
authoritative source with every candidate. You are not told how any candidate
was made and must not speculate about its method.

For each of the ten fixed claims, return exactly one status:
- preserved: the complete material meaning is retained;
- changed: a fact, number, date, name, scope, negation, cause, requirement, or
  recommendation is materially different or an unsupported replacement appears;
- missing: the claim is omitted.

Also score two properties. You may use ONLY 0, 10, 20, 30, 40, 50, 60, 70,
80, 90, or 100. Never return 25, 50-based quarters, a range, or a decimal.

readabilityPercent measures grammatical, coherent, natural prose. 100 means no
readability defect; 0 means unusable word salad.

usabilityPercent asks whether someone could rely on the candidate without the
source. It includes preserved details, absence of invented material claims,
coherence, and readability. 100 means ready to use; 90 means only trivial
copy-editing; 70 means noticeably damaged but still usable; below 70 is not
independently reliable.

List at most five material errors as short self-contained strings. Return only
the schema-conforming JSON object."""


def _sha256_json(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_prompt(batch: Mapping[str, object]) -> str:
    data = {
        "candidates": [
            {"candidateId": row["candidateId"], "text": row["text"]}
            for row in batch["candidates"]
        ],
        "claims": batch["claims"],
        "sourceText": batch["sourceText"],
    }
    return (
        INSTRUCTION
        + "\n\n--- BEGIN UNTRUSTED DATA ---\n"
        + json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n--- END UNTRUSTED DATA ---"
    )


def response_format(batch: Mapping[str, object]) -> dict[str, object]:
    candidate_ids = [row["candidateId"] for row in batch["candidates"]]
    claim_ids = [row["id"] for row in batch["claims"]]
    claim_schema = {
        "additionalProperties": False,
        "properties": {
            "id": {"enum": claim_ids, "type": "string"},
            "status": {
                "enum": ["preserved", "changed", "missing"],
                "type": "string",
            },
        },
        "required": ["id", "status"],
        "type": "object",
    }
    candidate_schema = {
        "additionalProperties": False,
        "properties": {
            "candidateId": {"enum": candidate_ids, "type": "string"},
            "claims": {
                "items": claim_schema,
                "maxItems": len(claim_ids),
                "minItems": len(claim_ids),
                "type": "array",
            },
            "materialErrors": {
                "items": {"type": "string"},
                "maxItems": 5,
                "type": "array",
            },
            "readabilityPercent": {"enum": list(ALLOWED_PERCENTAGES), "type": "integer"},
            "usabilityPercent": {"enum": list(ALLOWED_PERCENTAGES), "type": "integer"},
        },
        "required": [
            "candidateId",
            "claims",
            "materialErrors",
            "readabilityPercent",
            "usabilityPercent",
        ],
        "type": "object",
    }
    schema = {
        "additionalProperties": False,
        "properties": {
            "candidates": {
                "items": candidate_schema,
                "maxItems": len(candidate_ids),
                "minItems": len(candidate_ids),
                "type": "array",
            }
        },
        "required": ["candidates"],
        "type": "object",
    }
    return {
        "json_schema": {
            "name": "curated_text_quality_panel",
            "schema": schema,
            "strict": True,
        },
        "type": "json_schema",
    }


def _extract_json(text: str) -> object:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return json.loads(stripped)


def parse_response(text: str, batch: Mapping[str, object]) -> list[dict[str, object]]:
    value = _extract_json(text)
    if not isinstance(value, Mapping) or not isinstance(value.get("candidates"), list):
        raise ValueError("panel response must contain a candidates array")
    expected_candidate_ids = [row["candidateId"] for row in batch["candidates"]]
    claim_ids = [row["id"] for row in batch["claims"]]
    rows = value["candidates"]
    if [row.get("candidateId") for row in rows] != expected_candidate_ids:
        raise ValueError("panel response candidate IDs or order mismatch")
    return [
        validate_panel_candidate(
            row,
            expected_candidate_id=candidate_id,
            expected_claim_ids=claim_ids,
        )
        for row, candidate_id in zip(rows, expected_candidate_ids, strict=True)
    ]


def build_canary_batch(
    *, source: str, claims: Sequence[Mapping[str, object]]
) -> tuple[dict[str, object], dict[str, object]]:
    tampered = source
    replacements = (
        ("twelve library employees", "two library employees"),
        ("On May 16th, 2023", "On June 16th, 2023"),
        ("$7,500", "$75,000"),
        ("from May 16th to June 30th", "from July 16th to June 30th"),
        ("continue utilizing the digital visitor log", "discontinue the digital visitor log"),
    )
    for old, new in replacements:
        if tampered.count(old) != 1:
            raise ValueError(f"canary tamper phrase must occur once: {old}")
        tampered = tampered.replace(old, new, 1)
    batch = {
        "batchId": "doc-01-canary",
        "candidates": [
            {
                "candidateId": "candidate-01",
                "text": source,
                "textSha256": hashlib.sha256(source.encode()).hexdigest(),
            },
            {
                "candidateId": "candidate-02",
                "text": tampered,
                "textSha256": hashlib.sha256(tampered.encode()).hexdigest(),
            },
        ],
        "claims": list(claims),
        "documentId": "doc-01",
        "sourceText": source,
        "sourceTextSha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    expected = {
        "identicalCandidateId": "candidate-01",
        "tamperedCandidateId": "candidate-02",
        "tamperedClaimIds": ["c03", "c05", "c07", "c08", "c10"],
    }
    return batch, expected


def build_single_canary_batches(
    *, source: str, claims: Sequence[Mapping[str, object]]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """One-candidate canary: identical, tampered, and empty texts in separate prompts.

    Single-candidate prompts remove cross-candidate contamination, so the canary
    must show that each judge still (a) preserves a byte-identical text, (b)
    detects the five deliberate material changes, and (c) does not award claims
    or usability to an empty candidate.
    """
    paired, expected_pair = build_canary_batch(source=source, claims=claims)
    identical_text = paired["candidates"][0]["text"]
    tampered_text = paired["candidates"][1]["text"]

    def batch(batch_id: str, text: str) -> dict[str, object]:
        return {
            "batchId": batch_id,
            "candidates": [
                {
                    "candidateId": "candidate-01",
                    "text": text,
                    "textSha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            ],
            "claims": list(claims),
            "documentId": "doc-01",
            "sourceText": source,
            "sourceTextSha256": hashlib.sha256(source.encode()).hexdigest(),
        }

    batches = [
        batch("doc-01-canary-identical", identical_text),
        batch("doc-01-canary-tampered", tampered_text),
        batch("doc-01-canary-empty", ""),
    ]
    expected = {
        "candidateId": "candidate-01",
        "emptyBatchId": "doc-01-canary-empty",
        "identicalBatchId": "doc-01-canary-identical",
        "tamperedBatchId": "doc-01-canary-tampered",
        "tamperedClaimIds": list(expected_pair["tamperedClaimIds"]),
    }
    return batches, expected


def _prepare_canary(args: argparse.Namespace) -> int:
    config, corpus, _, claims = load_design(args.config)
    source = corpus["documents"][0]["marked"]["text"]
    if getattr(args, "single", False):
        batches, expected = build_single_canary_batches(
            source=source, claims=claims["documents"]["doc-01"]
        )
        methodology = (
            "Three one-candidate prompts test every judge before a single-candidate "
            "panel: a byte-identical positive control, a readable negative control "
            "with five deliberate material changes, and an empty candidate that must "
            "receive no preserved claims and no usability."
        )
    else:
        batch, expected = build_canary_batch(
            source=source, claims=claims["documents"]["doc-01"]
        )
        batches = [batch]
        methodology = (
            "One byte-identical positive control and one readable negative control with "
            "five deliberate material changes test every judge before the full panel."
        )
    now = utc_now()
    artifact = {
        "batches": batches,
        "createdAt": now,
        "expected": expected,
        "methodology": methodology,
        "schemaVersion": 1,
        "sources": config["sources"],
        "verifiedAt": now,
    }
    write_json_atomic(args.output, artifact)
    return 0


def judge_reasoning_effort(
    config: Mapping[str, object], judge: Mapping[str, object]
) -> str:
    """Per-judge reasoning override, defaulting to the frozen panel setting.

    Some endpoints (Grok 4.6) reject reasoning_effort="none"; the override is
    part of the frozen config so the exact request shape stays reproducible.
    """
    panel = config.get("panel", {})
    return str(judge.get("reasoningEffort", panel.get("reasoningEffort", "none")))


class PanelRunner:
    def __init__(
        self,
        *,
        config_path: Path,
        panel_input_path: Path,
        output_path: Path,
        mode: str,
        resume_valid_from: Path | None = None,
        prior_spend_from: Path | None = None,
        selected_models: Sequence[str] | None = None,
    ) -> None:
        self.config, _, _, _ = load_design(config_path)
        self.config_path = config_path
        self.panel_input_path = panel_input_path
        self.panel_input = json.loads(panel_input_path.read_text(encoding="utf-8"))
        self.output_path = output_path
        self.mode = mode
        self.resume_valid_from = resume_valid_from
        self.prior_spend_from = prior_spend_from
        configured_models = [judge["model"] for judge in self.config["judges"]]
        self.selected_models = list(selected_models or configured_models)
        if not self.selected_models or not set(self.selected_models).issubset(
            configured_models
        ):
            raise ValueError("selected panel models must be a nonempty configured subset")
        self.lock = threading.Lock()
        self.budget = Decimal(str(self.config["budgetsUsd"]["panel"]))
        self.state = self._load_state()

    def _new_state(self) -> dict[str, object]:
        now = utc_now()
        prior_cost = Decimal(0)
        prior_evidence = None
        if self.prior_spend_from is not None:
            previous = json.loads(self.prior_spend_from.read_text(encoding="utf-8"))
            prior_cost = Decimal(str(previous["totalCostUsd"]))
            prior_evidence = {
                "path": str(self.prior_spend_from),
                "sha256": sha256_file(self.prior_spend_from),
            }
        return {
            "budgetUsd": format(self.budget, "f"),
            "calls": {},
            "configSha256": sha256_file(self.config_path),
            "createdAt": now,
            "inFlight": {},
            "methodology": (
                "Durable four-vendor panel checkpoint. Reserve each request before "
                "dispatch, never redispatch an unresolved call, blind method identity, "
                "and reject any percentage outside 0..100 in ten-point increments."
            ),
            "mode": self.mode,
            "panelInputSha256": sha256_file(self.panel_input_path),
            "selectedModels": self.selected_models,
            "priorCostUsd": format(prior_cost, "f"),
            "priorSpendEvidence": prior_evidence,
            "schemaVersion": 1,
            "sources": self.config["sources"],
            "status": "running",
            "totalCostUsd": format(prior_cost, "f"),
            "verifiedAt": now,
        }

    def _load_state(self) -> dict[str, object]:
        if not self.output_path.exists():
            state = self._new_state()
            if self.resume_valid_from is not None:
                previous = json.loads(
                    self.resume_valid_from.read_text(encoding="utf-8")
                )
                if previous.get("panelInputSha256") != state["panelInputSha256"]:
                    raise ValueError("resume panel input hash mismatch")
                if previous.get("mode") != self.mode:
                    raise ValueError("resume panel mode mismatch")
                state["calls"] = {
                    key: value
                    for key, value in previous.get("calls", {}).items()
                    if isinstance(value, Mapping)
                    and "candidates" in value
                    and value.get("judge")
                    in set(self.selected_models)
                }
                state["resumedValidCallsFrom"] = {
                    "path": str(self.resume_valid_from),
                    "sha256": sha256_file(self.resume_valid_from),
                }
                state["totalCostUsd"] = format(
                    Decimal(str(state.get("priorCostUsd", "0")))
                    + sum(
                        (
                            Decimal(str(row["costUsd"]))
                            for row in state["calls"].values()
                        ),
                        Decimal(0),
                    ),
                    "f",
                )
            write_json_atomic(self.output_path, state)
            return state
        state = json.loads(self.output_path.read_text(encoding="utf-8"))
        expected = {
            "budgetUsd": format(self.budget, "f"),
            "configSha256": sha256_file(self.config_path),
            "mode": self.mode,
            "panelInputSha256": sha256_file(self.panel_input_path),
            "selectedModels": self.selected_models,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"panel checkpoint {key} mismatch")
        if state.get("inFlight"):
            raise RuntimeError("panel checkpoint has unresolved in-flight calls")
        return state

    def _spent(self) -> Decimal:
        return Decimal(str(self.state.get("priorCostUsd", "0"))) + sum(
            (Decimal(str(row["costUsd"])) for row in self.state["calls"].values()),
            Decimal(0),
        )

    def _reserve(self, key: str, judge: Mapping[str, object], batch: Mapping[str, object]):
        prompt = build_prompt(batch)
        schema = response_format(batch)
        prompt_token_bound = math.ceil(
            (len(prompt.encode("utf-8")) + len(json.dumps(schema).encode("utf-8"))) / 2
        )
        ceiling = (
            Decimal(prompt_token_bound) * Decimal(str(judge["promptUsdPerToken"]))
            + Decimal(self.config["panel"]["maxCompletionTokensPerCall"])
            * Decimal(str(judge["completionUsdPerToken"]))
        )
        with self.lock:
            if key in self.state["calls"]:
                return None
            pending = sum(
                (Decimal(str(row["ceilingUsd"])) for row in self.state["inFlight"].values()),
                Decimal(0),
            )
            if self._spent() + pending + ceiling > self.budget:
                raise RuntimeError("panel cost ceiling would exceed frozen budget")
            self.state["inFlight"][key] = {
                "ceilingUsd": format(ceiling, "f"),
                "requestSha256": _sha256_json(
                    {"model": judge["model"], "prompt": prompt, "schema": schema}
                ),
            }
            self.state["verifiedAt"] = utc_now()
            write_json_atomic(self.output_path, self.state)
        return prompt, schema

    def _client(self, judge: Mapping[str, object]):
        return unmark.OpenRouterClient.from_env(
            transport=ExplicitNonZdrTransport(),
            timeout=900,
            allow_fallbacks=True,
            require_parameters=False,
            reasoning_effort=judge_reasoning_effort(self.config, judge),
            temperature=None,
            max_tokens=int(self.config["panel"]["maxCompletionTokensPerCall"]),
            max_prompt_price=float(Decimal(str(judge["promptUsdPerToken"])) * 1_000_000),
            max_completion_price=float(
                Decimal(str(judge["completionUsdPerToken"])) * 1_000_000
            ),
        )

    def _call(self, judge: Mapping[str, object], batch: Mapping[str, object]):
        key = f"{batch['batchId']}::{judge['model']}"
        reserved = self._reserve(key, judge, batch)
        if reserved is None:
            return key, None
        prompt, schema = reserved
        try:
            completion = self._client(judge).complete(
                prompt,
                model=str(judge["model"]),
                max_tokens=int(self.config["panel"]["maxCompletionTokensPerCall"]),
                response_format=schema,
            )
        except unmark.ProviderHTTPError as error:
            if 400 <= error.status < 500:
                return key, {
                    "batchId": batch["batchId"],
                    "costUsd": "0",
                    "judge": judge["model"],
                    "terminalError": error.to_dict(),
                }
            raise
        result = {
            "batchId": batch["batchId"],
            "completionSha256": hashlib.sha256(completion.content.encode()).hexdigest(),
            "costUsd": str(completion.usage.cost),
            "judge": judge["model"],
            "provider": completion.provider,
            "usage": completion.usage.to_dict(),
        }
        try:
            result["candidates"] = parse_response(completion.content, batch)
        except (ValueError, json.JSONDecodeError) as error:
            # The paid provider response is authoritative evidence even when local
            # parsing fails. Persist it before making any decision; never redispatch
            # this exact request automatically.
            result["completionContent"] = completion.content
            result["parseError"] = f"{type(error).__name__}: {str(error)[:240]}"
        return key, result

    def _save_result(self, key: str, result: Mapping[str, object] | None) -> None:
        if result is None:
            return
        with self.lock:
            self.state["calls"][key] = dict(result)
            self.state["inFlight"].pop(key, None)
            self.state["totalCostUsd"] = format(self._spent(), "f")
            self.state["verifiedAt"] = utc_now()
            write_json_atomic(self.output_path, self.state)

    def run(self) -> None:
        judges = [
            judge
            for judge in self.config["judges"]
            if judge["model"] in self.selected_models
        ]
        for batch in self.panel_input["batches"]:
            work = [
                (judge, batch)
                for judge in judges
                if f"{batch['batchId']}::{judge['model']}" not in self.state["calls"]
            ]
            errors = []
            with futures.ThreadPoolExecutor(max_workers=4) as executor:
                pending = {
                    executor.submit(self._call, judge, batch): judge["model"]
                    for judge, batch in work
                }
                for future in futures.as_completed(pending):
                    try:
                        key, result = future.result()
                        self._save_result(key, result)
                    except Exception as error:
                        errors.append(f"{pending[future]}: {type(error).__name__}: {error}")
            if errors:
                raise RuntimeError("panel call failed closed: " + " | ".join(errors))
            terminal = [
                key
                for key, value in self.state["calls"].items()
                if value.get("batchId") == batch["batchId"]
                and ("terminalError" in value or "parseError" in value)
            ]
            if terminal:
                raise RuntimeError(
                    "panel provider rejected terminal requests: " + ", ".join(terminal)
                )
            print(
                json.dumps(
                    {
                        "batchId": batch["batchId"],
                        "event": "panel_batch_complete",
                        "spentUsd": self.state["totalCostUsd"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        self.state["status"] = "complete"
        self.state["totalCostUsd"] = format(self._spent(), "f")
        self.state["verifiedAt"] = utc_now()
        if self.mode == "canary":
            if len(judges) != 4:
                raise ValueError("canary requires all four configured judges")
            self.state["canary"] = validate_canary(
                self.panel_input, self.state["calls"], self.config["judges"]
            )
            if not self.state["canary"]["passed"]:
                self.state["status"] = "canary_failed"
        write_json_atomic(self.output_path, self.state)


def validate_single_canary(
    panel_input: Mapping[str, object],
    calls: Mapping[str, object],
    judges: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    expected = panel_input["expected"]
    batches = {batch["batchId"]: batch for batch in panel_input["batches"]}
    claim_ids = [row["id"] for row in batches[expected["identicalBatchId"]]["claims"]]
    target = set(expected["tamperedClaimIds"])
    candidate_id = expected["candidateId"]

    def verdict(batch_id: str, judge: Mapping[str, object]) -> Mapping[str, object]:
        call = calls[f"{batch_id}::{judge['model']}"]
        by_candidate = {row["candidateId"]: row for row in call["candidates"]}
        return by_candidate[candidate_id]

    rows = []
    for judge in judges:
        identical = verdict(expected["identicalBatchId"], judge)
        tampered = verdict(expected["tamperedBatchId"], judge)
        empty = verdict(expected["emptyBatchId"], judge)
        identical_preserved = sum(
            row["status"] == "preserved" for row in identical["claims"]
        )
        tampered_status = {row["id"]: row["status"] for row in tampered["claims"]}
        detected_targets = sum(tampered_status[claim_id] != "preserved" for claim_id in target)
        preserved_controls = sum(
            tampered_status[claim_id] == "preserved"
            for claim_id in claim_ids
            if claim_id not in target
        )
        empty_preserved = sum(row["status"] == "preserved" for row in empty["claims"])
        passed = (
            identical_preserved == 10
            and identical["readabilityPercent"] >= 90
            and identical["usabilityPercent"] >= 90
            and detected_targets >= 4
            and preserved_controls >= 4
            and empty_preserved == 0
            and empty["usabilityPercent"] <= 10
        )
        rows.append(
            {
                "detectedTargetCount": detected_targets,
                "emptyPreservedCount": empty_preserved,
                "emptyUsabilityPercent": empty["usabilityPercent"],
                "identicalPreservedCount": identical_preserved,
                "judge": judge["model"],
                "passed": passed,
                "preservedControlCount": preserved_controls,
            }
        )
    return {"judges": rows, "passed": all(row["passed"] for row in rows)}


def validate_canary(
    panel_input: Mapping[str, object],
    calls: Mapping[str, object],
    judges: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    expected = panel_input["expected"]
    if "emptyBatchId" in expected:
        return validate_single_canary(panel_input, calls, judges)
    batch = panel_input["batches"][0]
    all_claim_ids = [row["id"] for row in batch["claims"]]
    target = set(expected["tamperedClaimIds"])
    rows = []
    for judge in judges:
        call = calls[f"{batch['batchId']}::{judge['model']}"]
        by_candidate = {row["candidateId"]: row for row in call["candidates"]}
        identical = by_candidate[expected["identicalCandidateId"]]
        tampered = by_candidate[expected["tamperedCandidateId"]]
        identical_preserved = sum(
            row["status"] == "preserved" for row in identical["claims"]
        )
        tampered_status = {row["id"]: row["status"] for row in tampered["claims"]}
        detected_targets = sum(tampered_status[claim_id] != "preserved" for claim_id in target)
        preserved_controls = sum(
            tampered_status[claim_id] == "preserved"
            for claim_id in all_claim_ids
            if claim_id not in target
        )
        passed = (
            identical_preserved == 10
            and identical["readabilityPercent"] >= 90
            and identical["usabilityPercent"] >= 90
            and detected_targets >= 4
            and preserved_controls >= 4
        )
        rows.append(
            {
                "detectedTargetCount": detected_targets,
                "identicalPreservedCount": identical_preserved,
                "judge": judge["model"],
                "passed": passed,
                "preservedControlCount": preserved_controls,
            }
        )
    return {"judges": rows, "passed": all(row["passed"] for row in rows)}


def resolve_lost_paid_responses(state: dict[str, object]) -> dict[str, object]:
    """Close parse-time losses at their reserved upper cost without redispatch."""
    inflight = dict(state.get("inFlight") or {})
    if not inflight:
        raise ValueError("panel checkpoint has no unresolved calls")
    calls = state["calls"]
    for key, reservation in inflight.items():
        batch_id, judge = key.split("::", 1)
        calls[key] = {
            "batchId": batch_id,
            "costIsUpperBound": True,
            "costUsd": reservation["ceilingUsd"],
            "judge": judge,
            "terminalError": {
                "code": "paid_response_lost_before_parse_checkpoint",
                "message": (
                    "Provider returned content, but local JSON parsing failed before "
                    "the completion was checkpointed; exact request must not be retried."
                ),
            },
        }
    state["inFlight"] = {}
    state["status"] = "batch_failed"
    state["totalCostUsd"] = format(
        Decimal(str(state.get("priorCostUsd", "0")))
        + sum((Decimal(str(row["costUsd"])) for row in calls.values()), Decimal(0)),
        "f",
    )
    state["verifiedAt"] = utc_now()
    return state


def _run(args: argparse.Namespace) -> int:
    runner = PanelRunner(
        config_path=args.config,
        panel_input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        resume_valid_from=args.resume_valid_from,
        prior_spend_from=args.prior_spend_from,
        selected_models=args.models,
    )
    runner.run()
    if runner.state["status"] != "complete":
        raise RuntimeError(f"panel ended with status {runner.state['status']}")
    return 0


def _resolve_lost(args: argparse.Namespace) -> int:
    state = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    resolve_lost_paid_responses(state)
    write_json_atomic(args.checkpoint, state)
    return 0


def make_prior_ledger(
    checkpoint: Mapping[str, object], *, reusable_models: Sequence[str]
) -> dict[str, object]:
    reusable = sum(
        (
            Decimal(str(row["costUsd"]))
            for row in checkpoint["calls"].values()
            if "candidates" in row and row.get("judge") in set(reusable_models)
        ),
        Decimal(0),
    )
    total = Decimal(str(checkpoint["totalCostUsd"]))
    if reusable > total:
        raise ValueError("reusable panel cost exceeds checkpoint total")
    now = utc_now()
    return {
        "createdAt": now,
        "methodology": (
            "Subtract valid calls that will be resumed from the cumulative panel cost; "
            "carry every abandoned, failed, canary, and non-resumed call as prior spend."
        ),
        "reusableCostUsd": format(reusable, "f"),
        "reusableModels": list(reusable_models),
        "schemaVersion": 1,
        "sources": checkpoint["sources"],
        "totalCostUsd": format(total - reusable, "f"),
        "verifiedAt": now,
    }


def _make_prior_ledger(args: argparse.Namespace) -> int:
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    artifact = make_prior_ledger(checkpoint, reusable_models=args.reusable_models)
    artifact["checkpointSha256"] = sha256_file(args.checkpoint)
    write_json_atomic(args.output, artifact)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-canary")
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare.add_argument("--output", type=Path, default=DEFAULT_CANARY_INPUT)
    prepare.add_argument(
        "--single",
        action="store_true",
        help="one-candidate canary: identical, tampered, and empty prompts",
    )
    prepare.set_defaults(handler=_prepare_canary)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--mode", choices=("canary", "full"), required=True)
    run.add_argument("--resume-valid-from", type=Path)
    run.add_argument("--prior-spend-from", type=Path)
    run.add_argument("--models", nargs="+")
    run.set_defaults(handler=_run)
    resolve = commands.add_parser("resolve-lost")
    resolve.add_argument("--checkpoint", type=Path, required=True)
    resolve.set_defaults(handler=_resolve_lost)
    ledger = commands.add_parser("make-prior-ledger")
    ledger.add_argument("--checkpoint", type=Path, required=True)
    ledger.add_argument("--reusable-models", nargs="+", required=True)
    ledger.add_argument("--output", type=Path, required=True)
    ledger.set_defaults(handler=_make_prior_ledger)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
