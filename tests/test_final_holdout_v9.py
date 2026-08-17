from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import unittest

from freeze_final_holdout_v9 import (
    ARTIFACT_PATH,
    KEY_PATH,
    PLAN_PATH,
    build_final_holdout_v9_package,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE_A_COMMIT = "559a1d2749a1f0468f12bf3a51d81eca644164fe"


class FinalHoldoutV9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package = build_final_holdout_v9_package(ROOT / PLAN_PATH, root=ROOT)

    def test_key_was_drawn_once_after_review_mask_commit(self) -> None:
        key_artifact = self.package.key_artifact
        self.assertEqual(key_artifact["phaseA"]["commit"], PHASE_A_COMMIT)
        self.assertEqual(key_artifact["draw"]["drawCount"], 1)
        self.assertEqual(key_artifact["draw"]["byteCount"], 32)
        self.assertFalse(key_artifact["draw"]["redrawAllowed"])
        self.assertFalse(key_artifact["draw"]["outcomeKnownAtDraw"])
        self.assertEqual(len(bytes.fromhex(key_artifact["keyHex"])), 32)

        # Phase A contains the exact review/mask but cannot contain the later key file.
        for binding_name in ("inventory", "review", "allowlist"):
            binding = key_artifact["phaseA"][binding_name]
            committed = subprocess.run(
                ["git", "show", f"{PHASE_A_COMMIT}:{binding['path']}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
            ).stdout
            self.assertEqual(hashlib.sha256(committed).hexdigest(), binding["sha256"])
        absent = subprocess.run(
            ["git", "cat-file", "-e", f"{PHASE_A_COMMIT}:{KEY_PATH}"],
            cwd=ROOT,
            capture_output=True,
        )
        self.assertNotEqual(absent.returncode, 0)

        old_key = json.loads((ROOT / "fixtures/experiment-config-v1.json").read_text())[
            "marker"
        ]["keyHex"]
        self.assertNotEqual(key_artifact["keyHex"], old_key)

    def test_new_key_controls_pass_without_redraw(self) -> None:
        artifact = self.package.artifact
        gate = artifact["prepaidGate"]
        encoding = artifact["encoding"]

        self.assertEqual(gate["status"], "passed")
        self.assertEqual(gate["unmarked"]["status"], "not_detected")
        self.assertEqual(gate["unmarked"]["activePositions"], 42)
        self.assertEqual(gate["unmarked"]["hits"], 24)
        self.assertEqual(gate["marked"]["status"], "detected")
        self.assertEqual(gate["marked"]["activePositions"], 42)
        self.assertEqual(gate["marked"]["hits"], 38)
        self.assertEqual(
            gate["marked"]["pValueExact"],
            {"denominator": 2199023255552, "numerator": 62157},
        )
        self.assertEqual(encoding["approvedActiveOpportunities"], 29)
        self.assertEqual(encoding["approvedActiveFavored"], 29)
        self.assertEqual(encoding["rejectedActiveSkipped"], 13)
        self.assertEqual(encoding["physicalChanges"], 14)
        self.assertEqual(encoding["rejectedPhysicalChanges"], 0)

    def test_old_key_controls_are_explicitly_developmental(self) -> None:
        superseded = self.package.artifact["developmentalPredecessor"]
        self.assertEqual(superseded["classification"], "developmental_not_confirmatory")
        self.assertEqual(
            superseded["phaseACommit"],
            PHASE_A_COMMIT,
        )
        self.assertEqual(self.package.artifact["providerCalls"], 0)
        self.assertEqual(
            self.package.artifact["providerExecution"]["status"],
            "blocked_pending_committed_v7_winner_binding",
        )

    def test_1000_wrong_keys_and_all_outputs_have_exact_parity(self) -> None:
        controls = self.package.artifact["wrongKeyControlsOnMarked"]
        self.assertEqual(controls["count"], 1000)
        self.assertEqual(
            controls["sufficientCount"] + controls["insufficientCount"],
            1000,
        )
        for relative_path, expected_bytes in self.package.files.items():
            self.assertEqual((ROOT / relative_path).read_bytes(), expected_bytes)
        self.assertEqual(
            json.loads((ROOT / ARTIFACT_PATH).read_text()),
            self.package.artifact,
        )


if __name__ == "__main__":
    unittest.main()
