from __future__ import annotations

from pathlib import Path
import unittest

from recalculate_fidelity_v2 import (
    build_fidelity_recalculation,
    load_bound_sources,
)


ROOT = Path(__file__).resolve().parents[1]


class FidelityRecalculationTests(unittest.TestCase):
    def test_corrected_counts_separate_semantics_from_pipeline_defects(self) -> None:
        transform, audit = load_bound_sources(ROOT)

        artifact = build_fidelity_recalculation(transform, audit)

        counts = {
            method["methodId"]: (
                method["semanticFailureCount"],
                method["pipelineDefectCount"],
                method["fidelityFailureCount"],
            )
            for method in artifact["methods"]
        }
        self.assertEqual(
            counts,
            {
                "synonyms": (0, 0, 0),
                "roundtrip-de": (10, 3, 13),
                "roundtrip-zh": (10, 5, 12),
                "paraphrase": (2, 4, 6),
            },
        )

    def test_chinese_overlap_keeps_union_denominator_at_twelve(self) -> None:
        transform, audit = load_bound_sources(ROOT)
        artifact = build_fidelity_recalculation(transform, audit)
        chinese = next(
            method
            for method in artifact["methods"]
            if method["methodId"] == "roundtrip-zh"
        )
        overlap = [
            row
            for row in chinese["documents"]
            if row["semanticFailure"] and row["pipelineDefect"]
        ]

        self.assertEqual(len(overlap), 3)
        self.assertEqual(chinese["fidelityFailureCount"], 12)


if __name__ == "__main__":
    unittest.main()
