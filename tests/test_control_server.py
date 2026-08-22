import http.client
import os
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

import control_server
import run_dipper_on_runpod as runner
import run_dipper_on_runpod_v2 as runner_v2
import run_quality_corpus_on_runpod as runner_quality


class ControlServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "control-ready.json").write_text('{"service":"x"}', encoding="utf-8")
        (root / "sub").mkdir()
        self.token = "t" * 40
        handler = type("H", (control_server.TokenProtectedHandler,), {"token": self.token})
        directory = str(root)

        def factory(*args, **kwargs):
            return handler(*args, directory=directory, **kwargs)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), factory)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def _get(self, path, token=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=5)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        conn.request("GET", path, headers=headers)
        response = conn.getresponse()
        body = response.read()
        conn.close()
        return response.status, body

    def test_rejects_missing_or_wrong_token_and_listing(self):
        self.assertEqual(self._get("/control-ready.json")[0], 401)
        self.assertEqual(self._get("/control-ready.json", "wrong" * 8)[0], 401)
        self.assertEqual(self._get("/", self.token)[0], 404)
        self.assertEqual(self._get("/sub/", self.token)[0], 404)

    def test_serves_file_with_correct_token(self):
        status, body = self._get("/control-ready.json", self.token)
        self.assertEqual(status, 200)
        self.assertEqual(body, b'{"service":"x"}')

    def test_main_refuses_short_token(self):
        os.environ["CONTROL_TOKEN_TEST"] = "short"
        try:
            self.assertEqual(
                control_server.main(["--directory", self.tmp.name, "--token-env", "CONTROL_TOKEN_TEST"]),
                2,
            )
        finally:
            del os.environ["CONTROL_TOKEN_TEST"]


class RunnerHardeningTests(unittest.TestCase):
    def test_payloads_carry_token_and_never_start_public_http_server(self):
        token = "k" * 43
        payloads = [
            runner.build_pod_payload(job_source_b64="am9i", input_b64="aW4=", control_token=token, control_server_source_b64="c3J2"),
            runner_v2.build_payload(job_source_b64="am9i", input_b64="aW4=", gpu_type="NVIDIA A100 80GB PCIe", control_token=token, control_server_source_b64="c3J2"),
            runner_quality.build_pod_payload(job_source_b64="am9i", config_b64="Y2Zn", control_token=token, control_server_source_b64="c3J2"),
        ]
        for payload in payloads:
            env = payload["env"]
            self.assertEqual(env["CONTROL_TOKEN"], token)
            self.assertEqual(env["CONTROL_SERVER_B64"], "c3J2")
            command = payload.get("dockerStartCmd") or [payload["args"]]
            joined = " ".join(command)
            self.assertNotIn("python3 -m http.server", joined)
            self.assertIn("control_server.py --directory /workspace --port 8000", joined)

    def test_payload_rejects_short_token(self):
        with self.assertRaises(ValueError):
            runner.build_pod_payload(job_source_b64="am9i", input_b64="aW4=", control_token="short")

    def test_http_get_sends_bearer_token(self):
        seen = {}

        class FakeResponse:
            status = 200

            def read(self):
                return b"ok"

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(request, timeout):
            seen["auth"] = request.get_header("Authorization")
            return FakeResponse()

        original = runner.urllib.request.urlopen
        runner.urllib.request.urlopen = fake_urlopen
        try:
            status, body = runner._http_get("https://example.test/x", timeout=1, token="z" * 40)
        finally:
            runner.urllib.request.urlopen = original
        self.assertEqual((status, body), (200, b"ok"))
        self.assertEqual(seen["auth"], "Bearer " + "z" * 40)

    def test_watchdog_keeps_key_out_of_argv_and_writes_0600_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["XDG_RUNTIME_DIR"] = tmp
            try:
                env_file = runner.watchdog_env_file("unit-1", api_key="SECRETKEY123", base_url="https://api.example")
            finally:
                del os.environ["XDG_RUNTIME_DIR"]
            self.assertEqual(env_file.stat().st_mode & 0o777, 0o600)
            content = env_file.read_text(encoding="utf-8")
            self.assertIn('RUNPOD_API_KEY="SECRETKEY123"', content)
            command = runner.watchdog_command(unit="unit-1", minutes=12, env_file=env_file, script=Path("/x/run.py"), pod_id="pod1")
            self.assertNotIn("SECRETKEY123", " ".join(command))
            self.assertIn(f"--property=EnvironmentFile={env_file}", command)
            self.assertIn("--env-file", command)


if __name__ == "__main__":
    unittest.main()
