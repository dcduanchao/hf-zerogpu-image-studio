"""Offline API smoke tests; do not require GPU or HF credentials."""
import importlib.util
from pathlib import Path
from fastapi.testclient import TestClient

entry = Path(__file__).resolve().parents[1] / "gateway" / "main.py"
spec = importlib.util.spec_from_file_location("gateway_main", entry)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
client = TestClient(mod.app)

def test_health():
    assert client.get("/health").status_code == 200

def test_models():
    assert {x["id"] for x in client.get("/v1/models").json()["data"]} == {"z-image-turbo", "sdxl-turbo"}

def test_invalid_auth():
    r = client.post("/v1/images/generations",
        headers={"Authorization": "Bearer not-a-token"},
        json={"model": "z-image-turbo", "prompt": "test"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_token"

def test_anonymous():
    assert mod.credentials(None, None) is None

def test_token():
    assert mod.credentials("Bearer hf_abcdef12345", None) == "hf_abcdef12345"

def test_no_fallback():
    assert mod.credentials(None, "hf_abcdef12345") == "hf_abcdef12345"
