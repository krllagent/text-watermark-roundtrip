from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys
import unittest

from run_model_canary_terra_locked import (
    LOCKED_SYSTEM_INSTRUCTION,
    build_logic_locked_request,
    locked_request_messages,
    protect_logic_anchors,
    sentence_count,
)
from unmark import restore_tokens


ROOT = Path(__file__).resolve().parents[1]


class LogicLockedTerraTests(unittest.TestCase):
    def test_logic_anchors_round_trip_exactly(self) -> None:
        source = (
            "Orders confirmed by noon should leave today. "
            "Success requires at least 95% and recovery within three minutes. "
            "Niko must end the test because continuing it may change the result."
        )

        protected = protect_logic_anchors(source)
        originals = {token.original.lower() for token in protected.tokens}

        self.assertIn("by noon", originals)
        self.assertIn("should", originals)
        self.assertIn("at least", originals)
        self.assertIn("within three minutes", originals)
        self.assertIn("must", originals)
        self.assertIn("end", originals)
        self.assertIn("because", originals)
        self.assertIn("may", originals)
        self.assertEqual(restore_tokens(protected.masked, protected.tokens), source)

    def test_request_keeps_instruction_outside_untrusted_json(self) -> None:
        request = build_logic_locked_request("Text with ⟦T1⟧.")
        messages = locked_request_messages(request)

        self.assertEqual(
            messages[0], {"content": LOCKED_SYSTEM_INSTRUCTION, "role": "system"}
        )
        self.assertEqual(
            json.loads(messages[1]["content"]),
            {"sourceText": "Text with ⟦T1⟧."},
        )

    def test_sentence_counter_detects_merges(self) -> None:
        self.assertEqual(sentence_count("First. Second? Third!"), 3)
        self.assertEqual(sentence_count("First and second. Third!"), 2)

    def test_dry_run_is_bound_and_below_forty_three_cents(self) -> None:
        completed = subprocess.run(
            [sys.executable, "run_model_canary_terra_locked.py", "--dry-run"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)

        self.assertEqual(result["model"], "openai/gpt-5.6-terra")
        self.assertEqual(result["calls"], 6)
        self.assertEqual(result["sourceScore"]["hits"], 33)
        self.assertEqual(result["sourceScore"]["activePositions"], 33)
        self.assertEqual(
            result["prerequisiteBindings"]["plainTerraFinalSha256"],
            "101d1acd893a5cc51f49c847a6f851b7115a9baac11c23f660b4dab04d2d836f",
        )
        self.assertLess(
            Decimal(result["maximumConservativeCostCredits"]), Decimal("0.43")
        )

    def test_import_does_not_mutate_luna_engine(self) -> None:
        code = """
import json
import run_model_canary_luna as luna
before = {
    'model': luna.MODEL,
    'prompt': str(luna.PROMPT_PRICE),
    'completion': str(luna.COMPLETION_PRICE),
    'label': luna.CANDIDATE_LABEL,
    'version': luna.SCRIPT_VERSION,
}
import run_model_canary_terra
import run_model_canary_terra_locked
after = {
    'model': luna.MODEL,
    'prompt': str(luna.PROMPT_PRICE),
    'completion': str(luna.COMPLETION_PRICE),
    'label': luna.CANDIDATE_LABEL,
    'version': luna.SCRIPT_VERSION,
}
print(json.dumps({'before': before, 'after': after}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)

        self.assertEqual(result["before"], result["after"])


if __name__ == "__main__":
    unittest.main()
