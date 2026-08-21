import unittest

import run_quality_corpus_on_runpod as runner


class QualityCorpusRunPodTests(unittest.TestCase):
    def test_payload_requests_ephemeral_48gb_gpu_without_provider_secret(self):
        payload = runner.build_pod_payload(
            job_source_b64="c291cmNl",
            config_b64="e30=",
        )

        self.assertEqual(payload["gpuCount"], 1)
        self.assertEqual(
            payload["gpuTypeIds"],
            ["NVIDIA A40", "NVIDIA RTX A6000"],
        )
        self.assertEqual(payload["cloudType"], "SECURE")
        self.assertFalse(payload["interruptible"])
        self.assertEqual(payload["ports"], ["8000/http"])
        self.assertNotIn("RUNPOD_API_KEY", payload["env"])
        self.assertIn("CORPUS_JOB_B64", payload["env"])
        self.assertIn("CORPUS_CONFIG_B64", payload["env"])

    def test_budget_deadline_stays_below_two_dollars_at_rate_cap(self):
        estimate = runner.cost_upper_bound_usd(
            compute_per_hour=runner.MAX_COMPUTE_PER_HOUR_USD,
            elapsed_seconds=runner.WATCHDOG_MINUTES * 60,
            container_disk_gb=runner.CONTAINER_DISK_GB,
            volume_disk_gb=runner.VOLUME_DISK_GB,
        )

        self.assertLess(estimate, runner.HARD_COST_CAP_USD)


if __name__ == "__main__":
    unittest.main()
