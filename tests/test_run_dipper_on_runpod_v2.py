import os
import unittest
from unittest import mock

import run_dipper_on_runpod_v2 as runner


class RunDipperOnRunPodV2Tests(unittest.TestCase):
    def test_client_sends_non_default_user_agent_required_by_v2_edge(self):
        captured = {}

        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *_): return None
            def read(self): return b'{"pods":[]}'

        def opener(request, timeout):
            captured["userAgent"] = request.get_header("User-agent")
            return Response()

        with mock.patch("urllib.request.urlopen", opener):
            client = runner.Client(runner.Config("fake", "http://guard.invalid/v2"))
            client.request("GET", "/pods")

        self.assertEqual(captured["userAgent"], "dipper-curated/1.0")

    def test_configured_base_and_official_default_are_deterministic(self):
        with mock.patch.dict(
            os.environ,
            {
                "RUNPOD_API_KEY": "fake",
                "RUNPOD_API_BASE_URL": "http://127.0.0.1:8787/runpod-v2/v2/",
            },
            clear=True,
        ):
            guarded = runner.resolve_config()
        with mock.patch.dict(
            os.environ, {"RUNPOD_API_KEY": "direct"}, clear=True
        ):
            direct = runner.resolve_config()

        self.assertEqual(guarded.base_url, "http://127.0.0.1:8787/runpod-v2/v2")
        self.assertEqual(direct.base_url, "https://api.runpod.io/v2")

    def test_payload_is_nested_v2_and_contains_no_provider_key(self):
        payload = runner.build_payload(
            job_source_b64="c291cmNl",
            input_b64="aW5wdXQ=",
            gpu_type="NVIDIA A100 80GB PCIe",
        )

        self.assertEqual(payload["gpu"]["id"], "NVIDIA A100 80GB PCIe")
        self.assertEqual(payload["gpu"]["count"], 1)
        self.assertNotIn("gpuTypeIds", payload)
        self.assertNotIn("mounts", payload)
        self.assertNotIn("RUNPOD_API_KEY", payload["env"])
        self.assertIn("DIPPER_JOB_GZ_B64", payload["env"])
        self.assertIn("gzip -d", payload["args"])
        self.assertIn("bash -lc", payload["args"])

    def test_list_response_is_wrapped(self):
        self.assertEqual(runner.pods_from_response({"pods": [{"id": "a"}]}), [{"id": "a"}])
        with self.assertRaisesRegex(ValueError, "wrapped pods"):
            runner.pods_from_response([])

    def test_create_payload_name_is_stable_for_lost_response_adoption(self):
        payload = runner.build_payload(
            job_source_b64="c291cmNl",
            input_b64="aW5wdXQ=",
            gpu_type="NVIDIA A100 80GB PCIe",
        )

        self.assertEqual(payload["name"], "dipper-curated-v2-20260821")


if __name__ == "__main__":
    unittest.main()
