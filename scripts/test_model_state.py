"""Offline checks for read-only model status and strict request routing."""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui-space"))
import model_state
import model_router


class ActiveModelTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"GPU_SPACE_ID": "duan0320/my-zerogpu-engine"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def fake_client(self, result):
        return SimpleNamespace(predict=lambda **kwargs: result)

    def test_matching_model_allowed(self):
        with patch.object(model_state, "Client", return_value=self.fake_client("moss-image")):
            self.assertEqual(model_state.require_active_model("moss-image"), "moss-image")

    def test_mismatched_model_denied_with_admin_hint(self):
        with patch.object(model_state, "Client", return_value=self.fake_client("moss-image")):
            with self.assertRaises(model_state.ModelMismatch) as found:
                model_state.require_active_model("z-image-turbo")
        self.assertIn("联系管理员", str(found.exception))

    def test_unreachable_backend_fails_closed(self):
        with patch.object(model_state, "Client", side_effect=RuntimeError("backend restarting")):
            with self.assertRaises(model_state.ModelUnavailable):
                model_state.require_active_model("z-image-turbo")

    def test_invalid_backend_status_fails_closed(self):
        with patch.object(model_state, "Client", return_value=self.fake_client("unknown")):
            with self.assertRaises(model_state.ModelUnavailable):
                model_state.active_model_id()

    def test_router_never_invokes_inactive_model(self):
        with patch.object(model_state, "Client", return_value=self.fake_client("z-image-turbo")):
            with patch.object(model_router, "_client", side_effect=AssertionError("inference forbidden")):
                with self.assertRaises(model_state.ModelMismatch):
                    model_router.generate("moss-image", "test", 512, 512)

    def test_status_endpoint_is_read_only(self):
        with patch.object(model_state, "Client", return_value=self.fake_client("z-image-turbo")) as client:
            self.assertEqual(model_state.active_model_id(), "z-image-turbo")
            self.assertEqual(client.call_args.kwargs["token"], False)


if __name__ == "__main__":
    unittest.main()
