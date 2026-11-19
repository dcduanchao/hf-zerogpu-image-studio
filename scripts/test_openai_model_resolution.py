"""OpenAI compatibility tests; mock GPU and file generation (no GPU calls)."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui-space"))
import openai_api
from fastapi import HTTPException


class ModelResolutionTests(unittest.TestCase):
    def test_stale_model_uses_current_active_model_and_reports_it(self):
        with patch.object(openai_api, "active_model_id", return_value="moss-image"), patch.object(
            openai_api, "route_generate", return_value=("remote-image", "ok")
        ) as generate, patch.object(openai_api, "image_url", return_value="https://example.com/img.png"):
            response = openai_api.predict("z-image-turbo", "cat", "512x512", 1, None)
        self.assertEqual(response["model"], "moss-image")
        self.assertEqual(response["size"], "512x512")
        self.assertEqual(generate.call_args.args[0], "moss-image")

    def test_unknown_model_and_missing_model_are_advisory(self):
        with patch.object(openai_api, "active_model_id", return_value="z-image-turbo"):
            self.assertEqual(openai_api.validate("not-real", "512x512", 1)[0], "z-image-turbo")
            self.assertEqual(openai_api.validate(None, "512x512", 1)[0], "z-image-turbo")

    def test_incompatible_size_uses_current_model_default(self):
        with patch.object(openai_api, "active_model_id", return_value="moss-image"), patch.object(
            openai_api, "route_generate", return_value=("remote-image", "ok")
        ) as generate, patch.object(openai_api, "image_url", return_value="https://example.com/img.png"):
            response = openai_api.predict("z-image-turbo", "cat", "123x456", 1, None)
        self.assertEqual(response["model"], "moss-image")
        self.assertEqual(response["size"], "1024x1024")
        self.assertEqual(generate.call_args.args[2:4], (1024, 1024))

    def test_edits_reject_active_model_without_edit_support(self):
        with patch.object(openai_api, "active_model_id", return_value="moss-image"):
            with self.assertRaises(HTTPException) as found:
                openai_api.validate("z-image-turbo", "512x512", 1, edit=True)
        self.assertEqual(found.exception.status_code, 400)
        self.assertEqual(found.exception.detail["error"]["code"], "unsupported_edit")

    def test_unavailable_gpu_fails_closed(self):
        with patch.object(openai_api, "active_model_id", side_effect=openai_api.ModelUnavailable("offline")):
            with self.assertRaises(HTTPException) as found:
                openai_api.validate("moss-image", "512x512", 1)
        self.assertEqual(found.exception.status_code, 503)

    def test_response_model_uses_active_even_when_input_model_missing(self):
        with patch.object(openai_api, "active_model_id", return_value="z-image-turbo"), patch.object(
            openai_api, "route_generate", return_value=("remote-image", "ok")
        ), patch.object(openai_api, "image_url", return_value="https://example.com/img.png"):
            response = openai_api.predict(None, "cat", "512x512", 1, None)
        self.assertEqual(response["model"], "z-image-turbo")
        self.assertEqual(len(response["data"]), 1)


if __name__ == "__main__":
    unittest.main()
