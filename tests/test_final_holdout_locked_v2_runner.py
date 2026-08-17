from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def run_in_subprocess(code: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


class FinalHoldoutLockedV2RunnerTests(unittest.TestCase):
    def test_import_does_not_configure_any_engine(self) -> None:
        result = run_in_subprocess(
            """
import json
import run_model_canary_luna as luna
import run_final_holdout_v9_luna as base
before = {
    'model': luna.MODEL,
    'prompt': str(luna.PROMPT_PRICE),
    'candidate': base.CANDIDATE.model,
    'prefix': base.CALL_ID_PREFIX,
}
import run_final_holdout_v9_terra_locked_v2
after = {
    'model': luna.MODEL,
    'prompt': str(luna.PROMPT_PRICE),
    'candidate': base.CANDIDATE.model,
    'prefix': base.CALL_ID_PREFIX,
}
print(json.dumps({'before': before, 'after': after}))
"""
        )

        self.assertEqual(result["before"], result["after"])
        self.assertEqual(result["before"]["candidate"], "openai/gpt-5.6-luna")

    def test_configure_binds_terra_prices_and_v2_request_builder(self) -> None:
        result = run_in_subprocess(
            """
import json
import run_final_holdout_v9_luna as base
import run_model_canary_luna as engine
import run_model_canary_terra as terra
import run_model_canary_terra_locked_v2 as locked
import run_final_holdout_v9_terra_locked_v2 as v2
v2.configure()
payload = engine.expected_payload(v2.request_for("Orders by noon must ship."))
print(json.dumps({
    'engineModel': engine.MODEL,
    'candidateModel': base.CANDIDATE.model,
    'prompt': str(engine.PROMPT_PRICE),
    'terraPrompt': str(terra.PROMPT_PRICE),
    'payloadModel': payload['model'],
    'systemPrompt': payload['messages'][0]['content'] == locked.LOCKED_SYSTEM_INSTRUCTION,
    'prefix': base.CALL_ID_PREFIX,
    'analyzeIsV2': base.analyze_output is v2.analyze_output,
}))
"""
        )

        self.assertEqual(result["engineModel"], "openai/gpt-5.6-terra")
        self.assertEqual(result["candidateModel"], "openai/gpt-5.6-terra")
        self.assertEqual(result["prompt"], result["terraPrompt"])
        self.assertEqual(result["payloadModel"], "openai/gpt-5.6-terra")
        self.assertTrue(result["systemPrompt"])
        self.assertEqual(result["prefix"], "locked-v2-final")
        self.assertTrue(result["analyzeIsV2"])

    def test_engine_contract_verification_rejects_wrong_candidate(self) -> None:
        result = run_in_subprocess(
            """
import json
import run_model_canary_luna as engine
import run_final_holdout_v9_terra_locked_v2 as v2
v2.configure()
engine.MODEL = "openai/gpt-5.6-luna"
try:
    v2.verify_engine_contract()
    outcome = "accepted"
except Exception as error:
    outcome = type(error).__name__
print(json.dumps({'outcome': outcome}))
"""
        )

        self.assertEqual(result["outcome"], "CanaryError")

    def test_analysis_records_anchor_and_sentence_failures(self) -> None:
        result = run_in_subprocess(
            """
import json
import run_final_holdout_v9_terra_locked_v2 as v2
v2.configure()
protocol = __import__('run_final_holdout_v9_luna').load_protocol()
document_id = 'holdout-01'
source = protocol['sources'][document_id]
masked = v2.protect_visible_anchors(source).masked
# A faithful response keeps every anchor but must not echo the input verbatim.
reworded = masked.replace(' a ', ' one single ', 1)
assert reworded != masked
faithful = v2.analyze_output(protocol, document_id, source, reworded)
unwrapped = reworded.replace('\\u27ea', '', 1).replace('\\u27eb', '', 1)
dropped = v2.analyze_output(protocol, document_id, source, unwrapped)
print(json.dumps({
    'faithfulIssues': faithful['pipelineIssues'],
    'faithfulAnchors': faithful['visibleAnchorCount'],
    'droppedIssueCodes': sorted({row['code'] for row in dropped['pipelineIssues']}),
}))
"""
        )

        self.assertEqual(result["faithfulIssues"], [])
        self.assertGreater(result["faithfulAnchors"], 0)
        self.assertIn("anchor_alignment", result["droppedIssueCodes"])

    def test_dry_run_binds_frozen_baseline_and_v2_method(self) -> None:
        completed = subprocess.run(
            [sys.executable, "run_final_holdout_v9_terra_locked_v2.py", "--dry-run"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)

        self.assertEqual(result["calls"], 20)
        self.assertEqual(result["model"], "openai/gpt-5.6-terra")
        self.assertEqual(result["method"], "visible_anchor_sentence_aligned_v2")
        self.assertEqual(result["sourceScore"]["hits"], 38)
        self.assertEqual(result["sourceScore"]["activePositions"], 42)
        self.assertEqual(result["sourceScore"]["status"], "detected")


if __name__ == "__main__":
    unittest.main()
