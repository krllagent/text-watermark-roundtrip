from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
KEY_HEX = (
    "00112233445566778899aabbccddeeff"
    "102132435465768798a9babbdcddedef"
)


class CliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ROOT / "watermark_toy.py"), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_encode_prints_canonical_json_without_echoing_key(self) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as source:
            source.write("Big methods help small teams start and show clear results. " * 5)
            source.flush()
            completed = self.run_cli(
                "encode",
                "--key-hex",
                KEY_HEX,
                "--document-id",
                "cli-golden",
                "--density-bps",
                "10000",
                "--input",
                source.name,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        parsed = json.loads(completed.stdout)
        self.assertEqual(parsed["documentId"], "cli-golden")
        self.assertEqual(parsed["activePositions"], parsed["eligiblePositions"])
        self.assertNotIn(KEY_HEX, completed.stdout)
        canonical = json.dumps(
            parsed,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self.assertEqual(completed.stdout, canonical + "\n")

    def test_detect_reports_a_document_diagnostic(self) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as source:
            source.write("Big methods help small teams start and show clear results. " * 5)
            source.flush()
            completed = self.run_cli(
                "detect",
                "--key-hex",
                KEY_HEX,
                "--document-id",
                "cli-detect",
                "--density-bps",
                "10000",
                "--input",
                source.name,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        parsed = json.loads(completed.stdout)
        self.assertEqual(parsed["documentId"], "cli-detect")
        self.assertEqual(parsed["scoringUnit"], "document_diagnostic")
        self.assertEqual(parsed["activePositions"], parsed["eligiblePositions"])

    def test_invalid_key_and_density_fail_with_cli_errors(self) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as source:
            source.write("big start")
            source.flush()
            invalid_key = self.run_cli(
                "detect",
                "--key-hex",
                "not-hex",
                "--document-id",
                "cli-error",
                "--input",
                source.name,
            )
            invalid_density = self.run_cli(
                "detect",
                "--key-hex",
                KEY_HEX,
                "--document-id",
                "cli-error",
                "--density-bps",
                "0",
                "--input",
                source.name,
            )

        self.assertEqual(invalid_key.returncode, 2)
        self.assertIn("non-hexadecimal", invalid_key.stderr)
        self.assertNotIn("Traceback", invalid_key.stderr)
        self.assertEqual(invalid_density.returncode, 2)
        self.assertIn("between 1 and 10000", invalid_density.stderr)
        self.assertNotIn("Traceback", invalid_density.stderr)


if __name__ == "__main__":
    unittest.main()
