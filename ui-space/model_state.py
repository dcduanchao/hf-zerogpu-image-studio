"""Read-only authoritative status of the model loaded in the shared GPU Space.

The GPU Space /active_model endpoint returns the model actually initialized in
its running process. Do not use Hub's desired ACTIVE_GPU_MODEL environment
variable as a readiness signal during restarts.
"""
import os

from gradio_client import Client
from model_registry import MODELS


class ModelUnavailable(Exception):
    """GPU Space has no verifiable active model (restart, build or outage)."""


class ModelMismatch(Exception):
    """The requested model differs from the currently loaded model."""


def active_model_id() -> str:
    space_id = os.environ.get("GPU_SPACE_ID", "").strip()
    if not space_id or "/" not in space_id:
        raise ModelUnavailable("GPU Space 未配置，请联系管理员")
    try:
        result = Client(space_id, token=False, download_files=False).predict(
            api_name="/active_model"
        )
    except Exception:
        raise ModelUnavailable("GPU 模型正在启动或暂时不可用，请稍后重试") from None
    if not isinstance(result, str) or result not in MODELS:
        raise ModelUnavailable("无法确认当前已加载的 GPU 模型，请联系管理员")
    return result


def require_active_model(requested_model: str) -> str:
    active = active_model_id()
    if requested_model != active:
        name = MODELS[active].name
        raise ModelMismatch(
            f"当前 GPU 仅加载 {name}（{active}）；请求的 {requested_model} 不可用。"
            "请联系管理员切换模型。"
        )
    return active
