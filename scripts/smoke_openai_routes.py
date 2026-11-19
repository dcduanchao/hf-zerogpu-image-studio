"""Read-only HTTP tests for the deployed CPU Space. Never invokes GPU inference."""
import json
import os
import time
import urllib.request
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
namespace = os.getenv("HF_NAMESPACE", "").strip() or api.whoami()["name"]
name = os.getenv("UI_SPACE_NAME", "my-image-studio")
base = f"https://{namespace}-{name}.hf.space"
for path in ("/health", "/v1/models"):
    url = base + path
    success = False
    for attempt in range(12):
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                payload = json.load(response)
                status = response.status
            if status == 200 and isinstance(payload, dict):
                if path == "/v1/models" and not payload.get("data"):
                    raise ValueError("Empty models array")
                print(f"PASS {path}: HTTP {status} {str(payload)[:200]}")
                success = True
                break
        except Exception as exc:
            code = getattr(exc, "code", None)
            print(f"{path} try {attempt+1}: {type(exc).__name__} HTTP={code}", flush=True)
            if attempt < 11:
                time.sleep(10)
    if not success:
        raise SystemExit(f"FAIL: {url} is not responding as expected")
