"""Validate visitor identity and show official ZeroGPU quota without logging tokens."""
import re
from datetime import timezone

from huggingface_hub import HfApi

TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]+")


def gpu_time(seconds) -> str:
    total = max(0, int(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours} 小时 {minutes} 分 {secs} 秒" if hours else f"{minutes} 分 {secs} 秒"


def token_quota_info(token: str) -> str:
    token = (token or "").strip()
    if not token:
        return "匿名模式：使用共享匿名额度；无法查询个人 ZeroGPU 剩余时间。"
    if not TOKEN_PATTERN.fullmatch(token):
        return "Token 格式无效；请检查输入。"

    api = HfApi(token=token)
    try:
        identity = api.whoami()
    except Exception:
        return "Token 验证失败；请检查权限和内容。"

    username = identity.get("name") or "HF 用户"
    prefix = f"**Token 有效：{username}**（将按该账号配额调度）"
    try:
        quota = api.get_zero_gpu_quota()
        remaining = gpu_time(quota.remaining)
        base = gpu_time(quota.base)
        reset = quota.resets_at
        reset_text = (reset.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                      if reset is not None else "暂未提供（通常表示本周期尚未使用 ZeroGPU）")
        return (
            f"{prefix}\n\n"
            f"**ZeroGPU 剩余：{remaining}**\n\n"
            f"基础额度：{base}\n\n"
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
