"""Read-only deployment status check. Never prints secrets or triggers GPU jobs."""
import os
import json
import time
import urllib.request
import urllib.error

from huggingface_hub import HfApi

api = HfApi(token=os.environ.get("HF_TOKEN"))
namespace = os.getenv("HF_NAMESPACE", "").strip() or api.whoami()["name"]
spaces = [
    (namespace, os.getenv("GPU_SPACE_NAME", "my-zerogpu-engine")),
    (namespace, os.getenv("UI_SPACE_NAME", "my-image-studio")),
]

for namespace, name in spaces:
    repo_id = namespace + "/" + name
    url = "https://huggingface.co/api/spaces/" + repo_id
    request = urllib.request.Request(
        url, headers={"Authorization": "Bearer " + os.environ["HF_TOKEN"]}
    )
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            data = json.load(response)
        runtime = data.get("runtime") or {}
        card = {
            "space": repo_id,
            "stage": runtime.get("stage"),
            "hardware": runtime.get("hardware"),
            "requested_hardware": runtime.get("requestedHardware"),
            "sdk": data.get("sdk"),
            "sha": data.get("sha"),
        }
        print(json.dumps(card, ensure_ascii=False))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"{repo_id}: unable to read public runtime metadata: {type(exc).__name__}")
