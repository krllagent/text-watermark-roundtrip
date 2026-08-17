from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TerraCanaryTests(unittest.TestCase):
    def test_dry_run_is_exact_and_local(self) -> None:
        completed = subprocess.run(
            [sys.executable, "run_model_canary_terra.py", "--dry-run"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)

        self.assertEqual(result["model"], "openai/gpt-5.6-terra")
        self.assertEqual(result["route"], "azure/eu")
        self.assertEqual(result["calls"], 6)
        self.assertEqual(result["sourceScore"]["status"], "detected")
        self.assertEqual(result["sourceScore"]["hits"], 33)
        self.assertEqual(result["sourceScore"]["activePositions"], 33)
        self.assertEqual(
            set(result["payloadSha256s"]),
            {"doc-11", "doc-12", "doc-15", "doc-20", "doc-03", "doc-19"},
        )
        self.assertLess(
            Decimal(result["maximumConservativeCostCredits"]), Decimal("0.42")
        )


if __name__ == "__main__":
    unittest.main()
