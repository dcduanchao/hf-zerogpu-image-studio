"""Offline authorization tests; no HF network calls or real Space changes."""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui-space"))
import admin_switch

OWNER = "duan0320"
GPU_ID = "duan0320/my-zerogpu-engine"


class TestSwitchPermissions(unittest.TestCase):
    def setUp(self):
        self.environ = patch.dict(os.environ, {
            "HF_ADMIN_USERNAME": OWNER,
            "GPU_SPACE_ID": GPU_ID,
        })
        self.environ.start()
        self.addCleanup(self.environ.stop)

    def test_anonymous_denied(self):
        with self.assertRaises(admin_switch.SwitchDenied):
            admin_switch.switch_active_model("moss-image", "")
    
    def test_invalid_model_denied(self):
        with self.assertRaises(admin_switch.SwitchDenied):
            admin_switch.switch_active_model("injected-model", "hf_fake123")

    def test_different_user_denied_before_mutation(self):
        fake = SimpleNamespace(
            whoami=lambda: {"name": "other-user"},
            add_space_variable=lambda **kw: self.fail("unauthorized mutation"),
        )
        with patch.object(admin_switch, "HfApi", return_value=fake):
            with self.assertRaises(admin_switch.SwitchDenied):
                admin_switch.switch_active_model("moss-image", "hf_fake123")

    def test_owner_can_switch_only_registered_model(self):
        mutations = []
        fake = SimpleNamespace(
            whoami=lambda: {"name": OWNER},
            add_space_variable=lambda **kw: mutations.append(kw),
        )
        with patch.object(admin_switch, "HfApi", return_value=fake):
            result = admin_switch.switch_active_model("moss-image", "hf_valid123")
        self.assertIn("moss-image", result)
        self.assertEqual(mutations, [{
            "repo_id": GPU_ID, "key": "ACTIVE_GPU_MODEL", "value": "moss-image"
        }])

    def test_wrong_namespace_denied(self):
        with patch.dict(os.environ, {"HF_ADMIN_USERNAME": "someone-else"}):
            with self.assertRaises(admin_switch.SwitchDenied):
                admin_switch.switch_active_model("z-image-turbo", "hf_valid123")


if __name__ == "__main__":
    unittest.main()
