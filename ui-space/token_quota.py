"""Validate visitor HF identity and show official ZeroGPU quota without logging tokens.

Gradio 6.15.1 requires huggingface_hub < 2.0, while the ZeroGPU quota
SDK helper was introduced in huggingface_hub 2.1. Query the same official
read-only REST endpoint directly with httpx instead.
"""
import re
from datetime import datetime, timezone

import httpx
from huggingface_hub import HfApi

TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]+")
QUOTA_ENDPOINT = "https://huggingface.co/api/spaces/zero-gpu/quota"


def gpu_time(seconds) -> str:
    total = max(0, int(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours} 小时 {minutes} 分 {secs} 秒" if hours else f"{minutes} 分 {secs} 秒"


def fetch_quota(token: str):
    """Read quota with the user's token; never use the owner's deployment token."""
    response = httpx.get(
        QUOTA_ENDPOINT,
        headers={"Authorization": f"Bearer {token}"},
        timeout=12.0,
        follow_redirects=False,
    )
    response.raise_for_status()
    quota = response.json()
    if not isinstance(quota, dict):
        raise ValueError("Unexpected ZeroGPU quota response")
    remaining = quota["current"]
    base = quota["base"]
    reset_value = quota.get("resetsAt")
    if reset_value:
        reset = datetime.fromisoformat(reset_value.replace("Z", "+00:00"))
        reset_text = reset.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        reset_text = "暂未提供（可能是本周期尚未使用 ZeroGPU）"
    return remaining, base, reset_text


def token_quota_info(token: str) -> str:
    token = (token or "").strip()
    if not token:
        return "匿名模式：使用共享匿名额度；无法查询个人 ZeroGPU 剩余时间。"
    if not TOKEN_PATTERN.fullmatch(token):
        return "Token 格式无效；请检查输入。"

    try:
        identity = HfApi(token=token).whoami()
    except Exception:
        return "Token 验证失败；请检查权限和内容。"

    username = identity.get("name") or "HF 用户"
    prefix = f"**Token 有效：{username}**（将按该账号配额调度）"
    try:
        remaining, base, reset_text = fetch_quota(token)
        return (
            f"{prefix}\n\n"
            f"**ZeroGPU 剩余：{gpu_time(remaining)}**\n\n"
            f"基础额度：{gpu_time(base)}\n\n"
            f"下次重置：{reset_text}\n\n"
            "上述时间为 Hugging Face 当前报告的 GPU 配额，不代表单次任务可连续运行的时间。"
        )
    except Exception:
        return (
            f"{prefix}\n\n"
            "ZeroGPU 剩余额度暂时无法查询。Fine-grained Token 需要启用 "
            "**Billing → Read billing usage and payment method status** 权限；"
            "也可能是 Hugging Face 配额服务暂时不可用。"
        )
