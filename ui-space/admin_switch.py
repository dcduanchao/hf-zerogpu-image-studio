"""Owner-only model switching for the existing GPU Space.

No owner token is stored by this application. A privileged caller presents
their own HF token and is verified against the configured HF username.
No public/anonymous call can mutate the active GPU model.
"""
import os
import re
from threading import Lock

from huggingface_hub import HfApi

from model_registry import MODELS

_SWITCH_LOCK = Lock()
_TOKEN_FORMAT = re.compile(r"hf_[A-Za-z0-9]+")


class SwitchDenied(Exception):
    """Switching is not authorized or not configured."""


def switch_active_model(model_id: str, admin_token: str) -> str:
    if model_id not in MODELS:
        raise SwitchDenied("未知模型")
    token = (admin_token or "").strip()
    if not _TOKEN_FORMAT.fullmatch(token):
        raise SwitchDenied("需要有效的 Hugging Face 管理员 Token")

    owner = os.environ.get("HF_ADMIN_USERNAME", "").strip()
    gpu_space = os.environ.get("GPU_SPACE_ID", "").strip()
    if not owner or not gpu_space or "/" not in gpu_space:
        raise SwitchDenied("管理员配置缺失，请检查 HF_ADMIN_USERNAME 和 GPU_SPACE_ID")
    if gpu_space.split("/", 1)[0] != owner:
        raise SwitchDenied("管理员用户名与 GPU Space 所有者不匹配")

    # A user cannot claim another identity by changing UI fields.
    # The Hub validates the token and returns the authenticated username.
    try:
        api = HfApi(token=token)
        who = api.whoami()
        verified_name = who.get("name", "")
    except Exception:
        raise SwitchDenied("无法验证管理员 HF Token") from None
    if verified_name != owner:
        raise SwitchDenied("权限不足：只有 GPU Space 所有者可以切换模型")

    # Serialize switch operations within the CPU process. HF itself verifies
    # write permission when updating the Space variable.
    if not _SWITCH_LOCK.acquire(blocking=False):
        raise SwitchDenied("已有模型切换操作正在进行，请稍后重试")
    try:
        try:
            # Updating Space configuration triggers a restart automatically.
            # Do not call restart_space() again to avoid double restarts.
            api.add_space_variable(
                repo_id=gpu_space, key="ACTIVE_GPU_MODEL", value=model_id,
            )
        except Exception:
            raise SwitchDenied("Space 配置更新失败；确认 Token 具有该 Space 的写入权限") from None
    finally:
        _SWITCH_LOCK.release()
    return f"已提交模型切换：{model_id}。GPU Space 将重启并重新加载模型；就绪前请勿提交生图请求。"
