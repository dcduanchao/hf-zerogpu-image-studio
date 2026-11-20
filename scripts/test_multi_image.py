"""Offline multi-image API tests; no GPU and no remote downloads."""
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui-space"))
import input_images
import openai_api
from fastapi.testclient import TestClient
from PIL import Image

client = TestClient(openai_api.app)


def png_data(color="red"):
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=color).save(buf, "PNG")
    return buf.getvalue()


class MultiReferenceTests(unittest.TestCase):
    def test_multiple_files_into_collage(self):
        with patch.object(openai_api, "active_model_id", return_value="z-image-turbo"), patch.object(
            openai_api, "predict", return_value={"model": "z-image-turbo", "edit_mode": "experimental_reference_collage"}
        ) as predict:
            response = client.post("/v1/images/edits", data={"prompt": "combine"},
                files=[("image", ("a.png", png_data(), "image/png")),
                       ("image", ("b.png", png_data("blue"), "image/png"))])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(predict.call_args.kwargs["input_image_count"], 2)
        self.assertEqual(predict.call_args.kwargs["edit_mode"], "experimental_reference_collage")

    def test_public_url_input_and_mixed_order(self):
        with patch.object(openai_api, "active_model_id", return_value="z-image-turbo"), patch.object(
            input_images, "fetch_public_image", return_value=png_data("blue")
        ), patch.object(openai_api, "predict", return_value={"model": "z-image-turbo"}) as predict:
            response = client.post("/v1/images/edits", data={"prompt": "combine",
                   "image_urls": '["https://example.com/ref.png"]'},
                files=[("image", ("a.png", png_data(), "image/png"))])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(predict.call_args.kwargs["input_image_count"], 2)

    def test_missing_images(self):
        with patch.object(openai_api, "active_model_id", return_value="z-image-turbo"):
            response = client.post("/v1/images/edits", data={"prompt": "combine"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "missing_image")

    def test_too_many_images(self):
        with patch.object(openai_api, "active_model_id", return_value="z-image-turbo"):
            response = client.post("/v1/images/edits", data={"prompt": "combine"},
                files=[("image", (f"{i}.png", png_data(), "image/png")) for i in range(5)])
        self.assertEqual(response.status_code, 400)

    def test_moss_edits_rejected_before_url_fetch(self):
        with patch.object(openai_api, "active_model_id", return_value="moss-image"), patch.object(
            input_images, "fetch_public_image", side_effect=AssertionError("must not fetch")
        ):
            response = client.post("/v1/images/edits",
                data={"prompt": "combine", "image_urls": '["https://example.com/a.png"]'})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "unsupported_edit")

    def test_localhost_and_private_address_rejected(self):
        for url in ("http://127.0.0.1/secret", "http://localhost/secret", "http://10.0.0.1/x"):
            with self.assertRaises(input_images.ImageInputError):
                input_images.fetch_public_image(url)

    def test_collage_output(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = []
            for i in range(3):
                path = Path(temp) / f"{i}.png"
                path.write_bytes(png_data(["red", "green", "blue"][i]))
                paths.append(str(path))
            output, mode = input_images.collage(paths, Path(temp) / "merged.png")
            self.assertEqual(mode, "experimental_reference_collage")
            with Image.open(output) as img:
                self.assertGreater(img.width, 512)

    def test_models_include_capabilities_and_notes(self):
        data = client.get("/v1/models").json()["data"]
        z = next(x for x in data if x["id"] == "z-image-turbo")
        m = next(x for x in data if x["id"] == "moss-image")
        self.assertTrue(z["supports_multi_reference"])
        self.assertTrue(z["supports_public_url"])
        self.assertTrue(z["usage_notes"])
        self.assertFalse(m["supports_edits"])


if __name__ == "__main__":
    unittest.main()
