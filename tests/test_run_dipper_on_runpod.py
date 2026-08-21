import os
import unittest
from unittest import mock
import urllib.error

import run_dipper_on_runpod as runner


class RunDipperOnRunPodTests(unittest.TestCase):
    def test_base_url_uses_configured_guard_endpoint(self):
        with mock.patch.dict(
            os.environ,
            {
                "RUNPOD_API_KEY": "fake-runpod-key",
                "RUNPOD_API_BASE_URL": "http://127.0.0.1:8787/runpod/v1/",
            },
            clear=True,
        ):
            config = runner.resolve_runpod_config()

        self.assertEqual(config.api_key, "fake-runpod-key")
        self.assertEqual(config.base_url, "http://127.0.0.1:8787/runpod/v1")

    def test_base_url_defaults_to_official_endpoint_without_override(self):
        with mock.patch.dict(
            os.environ,
            {"RUNPOD_API_KEY": "direct-provider-key"},
            clear=True,
        ):
            config = runner.resolve_runpod_config()

        self.assertEqual(config.base_url, "https://rest.runpod.io/v1")

    def test_failed_configured_endpoint_does_not_fall_back(self):
        requested = []

        def failing_open(request, timeout):
            requested.append(request.full_url)
            raise urllib.error.URLError("configured endpoint unavailable")

        client = runner.RunPodClient(
            runner.RunPodConfig(
                api_key="fake-runpod-key",
                base_url="http://guard.invalid/runpod/v1",
            ),
            opener=failing_open,
        )

        with self.assertRaisesRegex(urllib.error.URLError, "unavailable"):
            client.request("GET", "/pods")

        self.assertEqual(requested, ["http://guard.invalid/runpod/v1/pods"])

    def test_pod_payload_is_ephemeral_a100_and_contains_no_provider_secret(self):
        payload = runner.build_pod_payload(
            job_source_b64="c291cmNl",
            input_b64="aW5wdXQ=",
        )

        self.assertEqual(payload["gpuCount"], 1)
        self.assertEqual(payload["containerDiskInGb"], 120)
        self.assertEqual(payload["volumeInGb"], 10)
        self.assertEqual(
            payload["gpuTypeIds"],
            ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"],
        )
        self.assertFalse(payload["interruptible"])
        self.assertEqual(payload["ports"], ["8000/http"])
        self.assertNotIn("RUNPOD_API_KEY", payload["env"])
        self.assertIn("DIPPER_JOB_B64", payload["env"])
        self.assertIn("DIPPER_INPUT_B64", payload["env"])

    def test_community_cloud_is_an_explicit_non_secret_payload_choice(self):
        payload = runner.build_pod_payload(
            job_source_b64="c291cmNl",
            input_b64="aW5wdXQ=",
            cloud_type="COMMUNITY",
        )

        self.assertEqual(payload["cloudType"], "COMMUNITY")
        self.assertNotIn("RUNPOD_API_KEY", payload["env"])

    def test_h100_fallback_is_an_explicit_gpu_list(self):
        payload = runner.build_pod_payload(
            job_source_b64="c291cmNl",
            input_b64="aW5wdXQ=",
            gpu_type_ids=("NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3"),
        )

        self.assertEqual(
            payload["gpuTypeIds"],
            ["NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3"],
        )

    def test_curated_run_can_use_a_new_unique_pod_name(self):
        payload = runner.build_pod_payload(
            job_source_b64="c291cmNl",
            input_b64="aW5wdXQ=",
            pod_name="dipper-curated-20260821-a",
        )

        self.assertEqual(payload["name"], "dipper-curated-20260821-a")

    def test_cost_upper_bound_includes_compute_and_ephemeral_storage(self):
        estimate = runner.cost_upper_bound_usd(
            compute_per_hour=1.39,
            elapsed_seconds=3_600,
            container_disk_gb=120,
            volume_disk_gb=10,
        )

        self.assertGreater(estimate, 1.39)
        self.assertLess(estimate, 1.42)

    def test_curated_twelve_minute_watchdog_stays_below_thirty_five_cents(self):
        estimate = runner.cost_upper_bound_usd(
            compute_per_hour=1.59,
            elapsed_seconds=12 * 60,
            container_disk_gb=120,
            volume_disk_gb=10,
        )

        self.assertLess(estimate, 0.35)


if __name__ == "__main__":
    unittest.main()
