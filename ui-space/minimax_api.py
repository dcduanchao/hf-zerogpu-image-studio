"""MiniMax-H3-Image official API backend; uses caller's MiniMax API key.

This path does not use Hugging Face Inference Providers or ZeroGPU.
Reference: https://github.com/MiniMax-AI/MiniMax-H3-Image/tree/main/examples
"""
import base64
import io
import os
import tempfile
from pathlib import Path

import httpx
from PIL import Image, UnidentifiedImageError

MODEL_ID = "minimax-h3-image"
UPSTREAM_MODEL = "MiniMax-H3-Image"
# Official global and China endpoints. Host is fixed; never derived from user input.
API_BASE = {
    "global": "https://api.minimax.io",
    "cn": "https://api.minimax.cn",
}
SIZES = ("1024x1024", "1024x768", "768x1024", "1536x1024",
         "1024x1536", "2048x2048", "2048x1536", "1536x2048")
MAX_IMAGE_BYTES = 30 * 1024 * 1024


class MiniMaxError(Exception):
    def __init__(self, message, status=502):
        super().__init__(message)
        self.status = status


def generate(prompt, width, height, key, *, seed=-1, images=None):
    if not (key or "").strip():
        raise MiniMaxError("需要访问者自己的 MiniMax API Key（不是 HF Token）", 401)
    if (f"{int(width)}x{int(height)}") not in SIZES:
        raise MiniMaxError("不支持的 MiniMax 尺寸", 400)
    if not prompt or not prompt.strip():
        raise MiniMaxError("提示词不能为空", 400)
    if len(prompt) > 4000:
        raise MiniMaxError("提示词不能超过 4000 字", 400)
    if images is not None and not (1 <= len(images) <= 4):
        raise MiniMaxError("MiniMax 本界面最多 4 张参考图", 400)
    region = os.getenv("MINIMAX_API_REGION", "global").strip().lower()
    if region not in API_BASE:
        raise MiniMaxError("MINIMAX_API_REGION 配置无效", 500)
    body = {
        "model": UPSTREAM_MODEL,
        "prompt": prompt,
        "size": f"{int(width)}x{int(height)}",
        "n": 1,
        "response_format": "b64_json",
        "output_format": "png",
    }
    if int(seed) >= 0:
        body["seed"] = int(seed)
    if images:
        encoded = []
        for path in images:
            with Image.open(path) as im:
                raw = io.BytesIO()
                im.convert("RGB").save(raw, "PNG")
            encoded.append({"image_url": "data:image/png;base64," + base64.b64encode(raw.getvalue()).decode("ascii")})
        body["images"] = encoded
    endpoint = "/v1/images/edits" if images else "/v1/images/generations"
    try:
        with httpx.Client(timeout=300, follow_redirects=False, trust_env=False) as client:
            res = client.post(
                API_BASE[region] + endpoint,
                headers={"Authorization": "Bearer " + key.strip(), "Content-Type": "application/json"},
                json=body,
            )
        if res.status_code in (401, 403):
            raise MiniMaxError("MiniMax API Key 无效或没有权限", 401)
        if res.status_code == 429:
            raise MiniMaxError("MiniMax API 请求超出额度或频率限制", 429)
        if not res.is_success:
            raise MiniMaxError(f"MiniMax API 请求失败（HTTP {res.status_code}），请检查输入和官方控制台", 502)
        payload = res.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list) or not payload["data"]:
            raise MiniMaxError("MiniMax 未返回图片数据")
        item = payload["data"][0]
        if item.get("error"):
            raise MiniMaxError("MiniMax 图片生成失败或触发服务端内容审核", 422)
        if not isinstance(item.get("b64_json"), str):
            raise MiniMaxError("MiniMax 未返回 b64_json")
        binary = base64.b64decode(item["b64_json"], validate=True)
        if len(binary) > MAX_IMAGE_BYTES:
            raise MiniMaxError("MiniMax 返回图片过大", 502)
        with Image.open(io.BytesIO(binary)) as picture:
            picture.load()
            target = Path(tempfile.mkdtemp(prefix="minimax_h3_")) / "image.png"
            picture.convert("RGB").save(target, "PNG")
        return str(target), "MiniMax-H3-Image | 官方 API | 独立计费 | " + str(payload.get("request_id", ""))
    except MiniMaxError:
        raise
    except (httpx.HTTPError, ValueError, KeyError, OSError, UnidentifiedImageError):
        raise MiniMaxError("MiniMax 请求失败或图片响应无效，请检查官方 API 状态") from None
