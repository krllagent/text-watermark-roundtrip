import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "merge_runpod_v2_guard.py"
PROFILE = ROOT / "configs" / "runpod-v2-guard-profile.yaml"
SETUP = ROOT / "scripts" / "setup-runpod-v2-guard.sh"


def _module():
    spec = importlib.util.spec_from_file_location("merge_runpod_v2_guard", HELPER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RunPodV2GuardSetupTests(unittest.TestCase):
    def test_profile_is_pod_only_on_new_host(self):
        profile = yaml.safe_load(PROFILE.read_text())
        service = profile["services"]["runpod_v2"]

        self.assertEqual(service["upstream"], "https://api.runpod.io")
        self.assertEqual(service["allow"]["path_prefixes"], ["/v2/pods"])
        self.assertEqual(service["allow"]["methods"], ["GET", "POST", "DELETE"])
        self.assertEqual(service["auth"]["token_ref"], "runpod_api_key")

    def test_merge_is_add_only_and_idempotent(self):
        module = _module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("services:\n  existing:\n    local_prefix: /existing\n")
            self.assertTrue(module.merge(path, PROFILE))
            first = path.read_bytes()
            self.assertFalse(module.merge(path, PROFILE))
            self.assertEqual(path.read_bytes(), first)

    def test_setup_has_backup_validation_rollback_and_narrow_restart(self):
        subprocess.run(["bash", "-n", str(SETUP)], check=True)
        text = SETUP.read_text()

        self.assertIn("config.yaml.before-$TS", text)
        self.assertIn("rollback_on_error", text)
        self.assertIn('validate --config "$CONFIG_PATH" --secrets "$SECRETS_PATH"', text)
        self.assertIn("systemctl restart agent-api-guard.service", text)
        self.assertNotIn("agent-api-guard-tls-proxy.service", text)


if __name__ == "__main__":
    unittest.main()
