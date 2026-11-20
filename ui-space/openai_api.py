"""OpenAI Images-compatible subset; deploy on trusted CPU host (NOT ZeroGPU)."""
import base64
import re
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from model_registry import MODELS, get_model
from model_router import generate as route_generate, image_url, local_image_path
from model_state import active_model_id, ModelMismatch, ModelUnavailable
from input_images import ImageInputError, MAX_BYTES, MAX_IMAGES, parse_urls, prepare_images, collage
from pydantic import BaseModel, Field

app = FastAPI(title="ZeroGPU OpenAI-Compatible Images API")


class GenerateRequest(BaseModel):
    model: str | None = None
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
    """Resolve the actually loaded model; client-provided model is advisory.

    For text-to-image, a size unsupported by the active model falls back to its
    default size. Editing requires an edit-capable active model.
    """
    try:
        active = active_model_id()
        spec = get_model(active)
    except ModelUnavailable as exc:
        raise api_error(503, "gpu_model_unavailable", str(exc)) from None
    except (ValueError, RuntimeError):
        raise api_error(503, "model_not_configured", "当前 GPU 模型未配置") from None

    if edit and not spec.can_edit:
        raise api_error(400, "unsupported_edit", "当前激活模型不支持图片编辑，请联系管理员切换模型")
    if not 1 <= int(n) <= 4:
        raise api_error(400, "invalid_n", "n 必须为 1~4")
    # Unknown, omitted, or stale model identifiers are accepted for
    # compatibility; the GPU model is never changed by this endpoint.
    resolved_size = size if size in spec.sizes else spec.default_size
    width, height = map(int, resolved_size.split("x"))
    return active, resolved_size, width, height


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


def predict(model, prompt, size, n, token, image_path=None, strength=0.6,
            seed=-1, response_format="url", input_image_count=0, edit_mode=None):
    active, resolved_size, width, height = validate(model, size, n, edit=image_path is not None)
    if response_format not in ("url", "b64_json"):
        raise api_error(400, "unsupported_format", "仅支持 url 或 b64_json")
    try:
        data = []
        for _ in range(n):
            image, _meta = route_generate(
                active, prompt, width, height, seed=seed, token=token,
                image_path=image_path, strength=strength,
                download_files=response_format == "b64_json",
            )
            if response_format == "url":
                data.append({"url": image_url(image, active)})
            else:
                data.append({"b64_json": to_base64(local_image_path(image))})
    except HTTPException:
        raise
    except ModelMismatch as exc:
        raise api_error(409, "model_not_active", str(exc)) from None
    except ModelUnavailable as exc:
        raise api_error(503, "gpu_model_unavailable", str(exc)) from None
    except Exception as exc:
        raise classify_error(exc) from None
    result = {"created": int(time.time()), "model": active, "size": resolved_size, "data": data}
    if input_image_count:
        result["input_image_count"] = input_image_count
        result["edit_mode"] = edit_mode or "single_image"
    return result


@app.get("/health")
def health():
    return {"ok": True, "configured": all(bool(m.space_id) for m in MODELS.values())}


@app.get("/v1/models/active")
def active_model():
    try:
        active = active_model_id()
    except ModelUnavailable as exc:
        raise api_error(503, "gpu_model_unavailable", str(exc)) from None
    return {"model": active, "name": MODELS[active].name, "status": "ready"}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [
        {"id": name, "object": "model", "owned_by": spec.space_id, "name": spec.name,
         "sizes": list(spec.sizes), "supports_edits": spec.can_edit,
         "supports_multi_reference": spec.multi_reference,
         "supports_public_url": spec.public_url, "usage_notes": spec.usage_notes}
        for name, spec in MODELS.items()
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
    image: list[UploadFile] | None = File(None),
    image_urls: str | None = Form(None),
    image_url: list[str] | None = Form(None),
    prompt: str = Form(...),
    model: str | None = Form(None),
    size: str = Form("512x512"),
    n: int = Form(1),
    strength: float = Form(0.6),
    response_format: Literal["url", "b64_json"] = Form("url"),
    authorization: str | None = Header(default=None),
    x_hf_token: str | None = Header(default=None),
):
    token = credentials(authorization, x_hf_token)
    if not prompt.strip() or len(prompt) > 4000:
        raise api_error(400, "invalid_prompt", "提示词不能为空且不能超出 4000 字")
    if not 0.05 <= strength <= 1.0:
        raise api_error(400, "invalid_strength", "strength 必须在 0.05~1")
    # Reject edits on text-only models before downloading any supplied URL.
    validate(model, size, n, edit=True)
    try:
        urls = parse_urls(image_urls or "", image_url)
        uploads = image or []
        if len(uploads) + len(urls) > MAX_IMAGES:
            raise ImageInputError("最多允许 4 张输入图片（含主图）", code="too_many_images")
        contents = []
        for upload in uploads:
            payload = await upload.read(MAX_BYTES + 1)
            contents.append(payload)
        import tempfile
        with tempfile.TemporaryDirectory(prefix="image_gateway_") as folder:
            paths = prepare_images(contents, urls, folder)
            input_path, edit_mode = collage(paths, Path(folder) / "references.png")
            return predict(
                model, prompt, size, n, token, image_path=input_path,
                strength=strength, response_format=response_format,
                input_image_count=len(paths), edit_mode=edit_mode,
            )
    except ImageInputError as exc:
        raise api_error(exc.status, exc.code, str(exc)) from None
