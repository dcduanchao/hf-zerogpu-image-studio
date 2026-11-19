"""OpenAI Images-compatible subset; deploy on trusted CPU host (NOT ZeroGPU)."""
import base64
import io
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Literal

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

SPACE_ID = os.getenv("GPU_SPACE_ID", "").strip()
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 16_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
SIZE_MAP = {
    "512x512": (512, 512), "768x768": (768, 768),
    "1024x1024": (1024, 1024), "1024x768": (1024, 768),
    "768x1024": (768, 1024),
}
MODELS = {
    "z-image-turbo": {"edit": True},
    "sdxl-turbo": {"edit": False},
}

app = FastAPI(title="ZeroGPU OpenAI-Compatible Images API")


class GenerateRequest(BaseModel):
    model: str = "z-image-turbo"
    prompt: str = Field(min_length=1, max_length=4000)
    size: str = "512x512"
    n: int = Field(default=1, ge=1, le=4)
    response_format: Literal["url", "b64_json"] = "url"
    seed: int = -1


def api_error(status, code, message):
    return HTTPException(status, detail={"error": {"type": "api_error", "code": code, "message": message}})


@app.exception_handler(HTTPException)
async def handle_http_exception(_, exc):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": {
        "type": "api_error", "code": "bad_request", "message": str(exc.detail)}})


def credentials(authorization: str | None, x_hf_token: str | None):
    """Use either Authorization: Bearer hf_... or X-HF-Token, never a server fallback."""
    if authorization and x_hf_token:
        raise api_error(400, "ambiguous_credentials", "只选择一种 HF Token 传递方式")
    if authorization:
        match = re.fullmatch(r"Bearer (hf_[A-Za-z0-9]+)", authorization.strip())
        if not match:
            raise api_error(401, "invalid_token", "请提供 Bearer hf_... Token")
        return match.group(1)
    if x_hf_token:
        if not re.fullmatch(r"hf_[A-Za-z0-9]+", x_hf_token.strip()):
            raise api_error(401, "invalid_token", "HF Token 格式无效")
        return x_hf_token.strip()
    return None


def validate(model, size, n, edit=False):
    if model not in MODELS:
        raise api_error(400, "model_not_found", "不支持该模型")
    if edit and not MODELS[model]["edit"]:
        raise api_error(400, "unsupported_edit", "该模型不支持图片编辑")
    if size not in SIZE_MAP:
        raise api_error(400, "invalid_size", f"尺寸仅支持：{', '.join(SIZE_MAP)}")
    if not 1 <= int(n) <= 4:
        raise api_error(400, "invalid_n", "n 必须为 1~4")
    return SIZE_MAP[size]


def classify_error(exc):
    msg = str(exc).lower()
    if any(s in msg for s in ["quota", "insufficient", "duration", "gpu limit", "exceeded"]):
        return api_error(429, "zero_gpu_quota_exceeded", "ZeroGPU 额度不足或任务执行时长不允许")
    if any(s in msg for s in ["401", "403", "unauthorized", "forbidden", "invalid token"]):
        return api_error(401, "hf_auth_failed", "HF Token 无效或权限不足；不会回退匿名")
    if any(s in msg for s in ["timeout", "timed out"]):
        return api_error(504, "upstream_timeout", "ZeroGPU 推理超时")
    return api_error(502, "upstream_error", "ZeroGPU 服务调用失败，请检查 GPU Space 日志")


def to_base64(path):
    path = Path(path)
    if not path.is_file():
        raise api_error(502, "bad_upstream_response", "没有收到图片文件")
    with Image.open(path) as source:
        out = io.BytesIO()
        source.convert("RGB").save(out, "PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


def get_remote_image_url(result):
    """Extract Gradio's existing temporary file URL without downloading or rehosting."""
    if isinstance(result, dict):
        url = result.get("url")
    else:
        url = getattr(result, "url", None)
    if not isinstance(url, str) or not url:
        raise api_error(502, "missing_image_url", "GPU Space 未返回可访问的图片 URL")
    parsed = urlparse(url)
    # Only return direct URLs hosted by the configured Space.
    expected_host = SPACE_ID.replace("/", "-").lower() + ".hf.space"
    if parsed.scheme != "https" or (parsed.hostname or "").lower() != expected_host:
        raise api_error(502, "invalid_image_url", "GPU Space 返回的图片地址不安全或不受支持")
    return url


def predict(model, prompt, size, n, token, image_path=None, strength=0.6,
            seed=-1, response_format="url"):
    if not SPACE_ID or "/" not in SPACE_ID:
        raise api_error(503, "not_configured", "请设置 GPU_SPACE_ID 环境变量")
    width, height = validate(model, size, n, edit=image_path is not None)
    if response_format not in ("url", "b64_json"):
        raise api_error(400, "unsupported_format", "仅支持 url 或 b64_json")
    try:
        client = Client(
            SPACE_ID,
            token=token if token else False,
            download_files=False if response_format == "url" else "/tmp/gradio",
        )
        data = []
        for _ in range(n):
            result, _meta = client.predict(
                model, prompt, width, height,
                handle_file(str(image_path)) if image_path else None,
                float(strength), int(seed), api_name="/infer",
            )
            if response_format == "url":
                data.append({"url": get_remote_image_url(result)})
            else:
                data.append({"b64_json": to_base64(result)})
    except HTTPException:
        raise
    except Exception as exc:
        raise classify_error(exc) from None
    return {"created": int(time.time()), "data": data}


@app.get("/health")
def health():
    return {"ok": True, "configured": bool(SPACE_ID)}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [
        {"id": name, "object": "model", "owned_by": "huggingface"} for name in MODELS
    ]}


@app.post("/v1/images/generations")
def generate_images(req: GenerateRequest,
                    authorization: str | None = Header(default=None),
                    x_hf_token: str | None = Header(default=None)):
    token = credentials(authorization, x_hf_token)
    return predict(req.model, req.prompt, req.size, req.n, token, seed=req.seed,
                   response_format=req.response_format)


@app.post("/v1/images/edits")
async def edit_images(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    model: str = Form("z-image-turbo"),
    size: str = Form("512x512"),
    n: int = Form(1),
    strength: float = Form(0.6),
    response_format: Literal["url", "b64_json"] = Form("url"),
    authorization: str | None = Header(default=None),
    x_hf_token: str | None = Header(default=None),
):
    token = credentials(authorization, x_hf_token)
    validate(model, size, n, edit=True)
    if not prompt.strip() or len(prompt) > 4000:
        raise api_error(400, "invalid_prompt", "提示词不能为空且不能超出 4000 字")
    if not 0.05 <= strength <= 1.0:
        raise api_error(400, "invalid_strength", "strength 必须在 0.05~1")
    body = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(body) > MAX_UPLOAD_BYTES:
        raise api_error(413, "image_too_large", "最大上传 12 MB")
    try:
        with Image.open(io.BytesIO(body)) as img:
            img.load()
            if img.width * img.height > MAX_PIXELS:
                raise api_error(413, "too_many_pixels", "输入图片像素过多")
            import tempfile
            with tempfile.TemporaryDirectory(prefix="image_gateway_") as folder:
                filename = Path(folder) / "source.png"
                img.convert("RGB").save(filename, "PNG")
                return predict(model, prompt, size, n, token, image_path=filename,
                               strength=strength, response_format=response_format)
    except (UnidentifiedImageError, ValueError, OSError):
        raise api_error(400, "invalid_image", "上传的图片不能读取") from None
