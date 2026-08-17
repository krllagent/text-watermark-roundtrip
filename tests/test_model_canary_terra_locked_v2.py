from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys
import unittest

from run_model_canary_terra_locked_v2 import (
    LOCKED_SYSTEM_INSTRUCTION,
    anchor_alignment_issues,
    build_visible_locked_request,
    locked_request_messages,
    protect_visible_anchors,
    split_sentences,
    strip_markers,
)
from unmark import restore_tokens


ROOT = Path(__file__).resolve().parents[1]


class VisibleAnchorTests(unittest.TestCase):
    def test_visible_anchors_round_trip_exactly(self) -> None:
        source = (
            "Orders confirmed by noon should leave today. "
            "Success requires at least 95% and recovery within three minutes. "
            "No result is offered as a claim about real clay, and all names "
            "are fictional."
        )

        protected = protect_visible_anchors(source)

        self.assertIn("⟪by noon⟫", protected.masked)
        self.assertIn("⟪should⟫", protected.masked)
        self.assertIn("⟪at least⟫", protected.masked)
        self.assertIn("⟪No⟫ result", protected.masked)
        self.assertIn("⟪all⟫ names", protected.masked)
        self.assertEqual(
            restore_tokens(strip_markers(protected.masked), protected.tokens),
            source,
        )

    def test_anchor_containing_placeholder_is_wrapped_without_corruption(self) -> None:
        source = "Recovery must finish within 3 minutes or fail."

        protected = protect_visible_anchors(source)

        self.assertEqual(
            restore_tokens(strip_markers(protected.masked), protected.tokens),
            source,
        )
        # every exact token from the plain masking survives in the ordered tuple
        self.assertTrue(
            all(token.placeholder in protected.masked for token in protected.tokens)
        )

    def test_alignment_passes_on_verbatim_anchor_copy(self) -> None:
        masked = "⟪No⟫ result is offered as a claim. ⟪All⟫ names are fictional."
        output = "⟪No⟫ result is presented as an assertion. ⟪All⟫ names are invented."

        self.assertEqual(anchor_alignment_issues(masked, output), [])

    def test_alignment_catches_dropped_anchor(self) -> None:
        masked = "⟪No⟫ result is offered as a claim. ⟪All⟫ names are fictional."
        output = "The result is presented as an assertion. ⟪All⟫ names are invented."

        issues = anchor_alignment_issues(masked, output)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["code"], "anchor_alignment")
        self.assertIn("'No'", issues[0]["message"])

    def test_alignment_catches_reworded_anchor(self) -> None:
        masked = "Orders ⟪by noon⟫ ship today."
        output = "Orders ⟪before noon⟫ ship today."

        issues = anchor_alignment_issues(masked, output)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["code"], "anchor_alignment")

    def test_alignment_catches_anchor_moved_across_sentences(self) -> None:
        masked = "The pilot ⟪must⟫ stop. The report is ready."
        output = "The pilot stops. The report ⟪must⟫ be ready."

        issues = anchor_alignment_issues(masked, output)

        self.assertEqual(
            sorted(issue["code"] for issue in issues),
            ["anchor_alignment", "anchor_alignment"],
        )

    def test_alignment_fails_closed_on_sentence_misalignment(self) -> None:
        # Global anchor multiset is intact, but the model split one source
        # sentence into two, moving an anchor into the added piece.
        masked = "The pilot ⟪must⟫ stop today."
        output = "The pilot stops. This ⟪must⟫ happen today."

        issues = anchor_alignment_issues(masked, output)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["code"], "anchor_alignment")
        self.assertIn("misaligned", issues[0]["message"])

    def test_alignment_catches_unbalanced_markers(self) -> None:
        masked = "The pilot ⟪must⟫ stop."
        output = "The pilot ⟪must stop."

        issues = anchor_alignment_issues(masked, output)

        self.assertIn("anchor_markers", [issue["code"] for issue in issues])

    def test_split_sentences_keeps_tail_without_terminator(self) -> None:
        self.assertEqual(
            split_sentences("First. Second? Trailing fragment"),
            ["First.", "Second?", "Trailing fragment"],
        )

    def test_request_keeps_instruction_outside_untrusted_json(self) -> None:
        request = build_visible_locked_request("Text with ⟪no⟫ change and ⟦T1⟧.")
        messages = locked_request_messages(request)

        self.assertEqual(
            messages[0], {"content": LOCKED_SYSTEM_INSTRUCTION, "role": "system"}
        )
        self.assertEqual(
            json.loads(messages[1]["content"]),
            {"sourceText": "Text with ⟪no⟫ change and ⟦T1⟧."},
        )

    def test_dry_run_is_bound_to_terra_and_frozen_payloads(self) -> None:
        completed = subprocess.run(
            [sys.executable, "run_model_canary_terra_locked_v2.py", "--dry-run"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)

        self.assertEqual(result["model"], "openai/gpt-5.6-terra")
        self.assertEqual(result["calls"], 6)
        self.assertEqual(
            result["prerequisiteBindings"]["lockedV1FinalSha256"],
            "252d1b1f00fd5a42f715424250b7a97aa3db958cb1a747a0bb0e212e00fa6f84",
        )
        self.assertLess(
            Decimal(result["maximumConservativeCostCredits"]), Decimal("0.50")
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
    'protect': luna.protect_tokens.__module__,
}
import run_model_canary_terra_locked_v2
after = {
    'model': luna.MODEL,
    'prompt': str(luna.PROMPT_PRICE),
    'completion': str(luna.COMPLETION_PRICE),
    'label': luna.CANDIDATE_LABEL,
    'version': luna.SCRIPT_VERSION,
    'protect': luna.protect_tokens.__module__,
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
