import unittest

import dipper_smoke


def _marked(document_id, text, score):
    return {
        "documentId": document_id,
        "marked": {
            "meanG": score,
            "prompt": f"Prompt for {document_id}",
            "text": text,
        },
    }


class DipperSmokeTests(unittest.TestCase):
    def test_analysis_methodology_records_dynamic_counts_and_threshold(self):
        text = dipper_smoke.analyzed_methodology(
            direct_count=10,
            second_stage_count=0,
            threshold=0.5095383054287164,
        )

        self.assertIn("10 direct marked-source cases", text)
        self.assertIn("0 two-stage diagnostic cases", text)
        self.assertIn("0.509538305429", text)

    def test_build_inputs_contains_two_direct_and_eight_second_stage_cases(self):
        corpus = {
            "documents": [
                _marked("doc-01", "Marked one.", 0.68),
                _marked("doc-04", "Marked four.", 0.69),
            ]
        }
        smoke_rows = []
        for document_id, source in (
            ("doc-01", "Marked one."),
            ("doc-04", "Marked four."),
        ):
            for method in dipper_smoke.PRIOR_METHODS:
                smoke_rows.append(
                    {
                        "documentId": document_id,
                        "evaluatedOutputText": f"{document_id} after {method}.",
                        "method": method,
                        "sourceText": source,
                        "transformedDetector": {"meanG": 0.55},
                    }
                )

        artifact = dipper_smoke.build_input_artifact(
            corpus=corpus,
            corpus_sha256="a" * 64,
            smoke={"documents": smoke_rows},
            smoke_sha256="b" * 64,
            verified_at="2026-08-20T12:00:00Z",
        )

        self.assertEqual(len(artifact["cases"]), 10)
        self.assertEqual(
            [case["caseId"] for case in artifact["cases"][:2]],
            ["doc-01::marked-source", "doc-04::marked-source"],
        )
        self.assertEqual(
            sum(case["inputKind"] == "marked-source" for case in artifact["cases"]),
            2,
        )
        self.assertEqual(
            sum(case["inputKind"] == "prior-smoke-output" for case in artifact["cases"]),
            8,
        )
        self.assertEqual(artifact["attack"]["lexicalDiversity"], 60)
        self.assertEqual(artifact["attack"]["orderDiversity"], 20)
        self.assertEqual(artifact["attack"]["sentenceChunkSize"], 3)
        self.assertEqual(artifact["attack"]["seed"], 123)

    def test_build_inputs_rejects_smoke_row_with_wrong_marked_source(self):
        corpus = {"documents": [_marked("doc-01", "Marked one.", 0.68)]}
        smoke = {
            "documents": [
                {
                    "documentId": "doc-01",
                    "evaluatedOutputText": "Rewrite.",
                    "method": method,
                    "sourceText": "A different source.",
                    "transformedDetector": {"meanG": 0.55},
                }
                for method in dipper_smoke.PRIOR_METHODS
            ]
        }

        with self.assertRaisesRegex(ValueError, "source does not match"):
            dipper_smoke.build_input_artifact(
                corpus=corpus,
                corpus_sha256="a" * 64,
                smoke=smoke,
                smoke_sha256="b" * 64,
                document_ids=("doc-01",),
                verified_at="2026-08-20T12:00:00Z",
            )

    def test_dipper_prompt_uses_similarity_codes_and_accumulated_prefix(self):
        prompt = dipper_smoke.dipper_prompt(
            prefix="Original request. Rewritten first window.",
            sentence_window="Second sentence. Third sentence.",
            lexical_diversity=60,
            order_diversity=20,
        )

        self.assertEqual(
            prompt,
            "lexical = 40, order = 80 Original request. Rewritten first window. "
            "<sent> Second sentence. Third sentence. </sent>",
        )

    def test_summary_separates_direct_test_from_second_stage_diagnostic(self):
        rows = [
            {
                "inputKind": "marked-source",
                "methodBeforeDipper": None,
                "removedAfterDipper": True,
                "beforeDetector": {"meanG": 0.68},
                "afterDetector": {"meanG": 0.49},
                "wordDistanceFromDipperInput": 0.7,
            },
            {
                "inputKind": "marked-source",
                "methodBeforeDipper": None,
                "removedAfterDipper": False,
                "beforeDetector": {"meanG": 0.69},
                "afterDetector": {"meanG": 0.52},
                "wordDistanceFromDipperInput": 0.6,
            },
            {
                "inputKind": "prior-smoke-output",
                "methodBeforeDipper": "paraphrase",
                "removedAfterDipper": True,
                "beforeDetector": {"meanG": 0.57},
                "afterDetector": {"meanG": 0.48},
                "wordDistanceFromDipperInput": 0.65,
            },
        ]

        summary = dipper_smoke.summarize_analyzed_cases(rows)

        self.assertEqual(summary["directMarkedSources"]["caseCount"], 2)
        self.assertEqual(summary["directMarkedSources"]["removedCount"], 1)
        self.assertEqual(summary["priorSmokeOutputs"]["removedCount"], 1)
        self.assertEqual(summary["byPriorMethod"]["paraphrase"]["removedCount"], 1)
        self.assertAlmostEqual(summary["directMarkedSources"]["meanAfterG"], 0.505)


if __name__ == "__main__":
    unittest.main()
