"""Model dispatch shared by Gradio UI and FastAPI."""
import importlib
import os
from pathlib import Path
from urllib.parse import urlparse

from gradio_client import Client

from model_registry import get_model
from model_state import require_active_model

ADAPTERS = {
    "z-image-turbo": "model_adapters.z_image",
    "moss-image": "model_adapters.moss_image",
}

def _client(model_id, token=None, *, download_files=True, headers=None):
    model = get_model(model_id)
    # No owner HF_TOKEN fallback. Caller identity is always explicit.
    kwargs = {"token": token if token else False}
    if headers:
        kwargs["headers"] = headers
    if not download_files:
        kwargs["download_files"] = False
    return Client(model.space_id, **kwargs)

def generate(model_id, prompt, width, height, seed=-1, token=None,
             image_path=None, strength=0.6, download_files=True, headers=None):
    model = get_model(model_id)
    require_active_model(model_id)
    if image_path is not None and not model.can_edit:
        raise ValueError(f"{model.name} 当前不支持图片编辑")
    adapter = importlib.import_module(ADAPTERS[model_id])
    client = _client(model_id, token, download_files=download_files, headers=headers)
    return adapter.generate(client, prompt, width, height, seed, image_path, strength)

def image_url(value, model_id):
    """Return upstream Gradio file URL; never convert a local path to a guessed URL."""
    model = get_model(model_id)
    if isinstance(value, dict):
        url = value.get("url")
    else:
        url = getattr(value, "url", None)
        if not url and isinstance(value, str) and value.startswith("https://"):
            url = value
    expected_host = model.space_id.replace("/", "-").lower() + ".hf.space"
    parsed = urlparse(url or "")
    if not url or parsed.scheme != "https" or parsed.hostname != expected_host:
        raise ValueError("上游未返回可信的 Gradio 临时文件 URL")
    return url

def local_image_path(value):
    if isinstance(value, dict):
        path = value.get("path")
    else:
        path = value if isinstance(value, (str, os.PathLike)) else getattr(value, "path", None)
    if not path or not Path(path).is_file():
        raise ValueError("上游未返回已下载的图片文件")
    return str(path)
