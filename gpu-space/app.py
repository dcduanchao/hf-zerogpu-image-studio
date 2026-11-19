"""Shared ZeroGPU inference service: load only one model per Space process.

The CPU UI exposes model selection and admin switching. This GPU Space
retains stable /infer, /infer_moss and /active_model endpoints.
"""
import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone

import spaces  # Must be imported before torch/diffusers in ZeroGPU.
import gradio as gr
import torch
from PIL import Image

from models.moss_image import DEFAULT_NEGATIVE, MossImageModel, gpu_duration
from models.z_image import ZImageModel

ACTIVE_GPU_MODEL = os.getenv("ACTIVE_GPU_MODEL", "z-image-turbo").strip().lower()
SUPPORTED_MODELS = ("z-image-turbo", "moss-image")
Z_SIZES = {(512, 512), (768, 768), (1024, 1024), (1024, 768), (768, 1024)}

if ACTIVE_GPU_MODEL not in SUPPORTED_MODELS:
    raise RuntimeError(f"Unsupported ACTIVE_GPU_MODEL: {ACTIVE_GPU_MODEL}")

# ZeroGPU moves registered tensors into the worker when a GPU job starts.
# Do not instantiate both large pipelines at the same time.
engine = ZImageModel() if ACTIVE_GPU_MODEL == "z-image-turbo" else MossImageModel()
engine.load()
print(f"Active ZeroGPU model: {ACTIVE_GPU_MODEL}", flush=True)


def require_active(model_id: str):
    if model_id != ACTIVE_GPU_MODEL:
        raise gr.Error(
            f"当前仅加载 {ACTIVE_GPU_MODEL}；请求模型 {model_id} 不可用，请联系管理员切换模型"
        )


def save_image(image, prefix: str) -> str:
    path = os.path.join(tempfile.mkdtemp(prefix=prefix), "image.png")
    image.save(path, format="PNG")
    return path


def log_inference(model, request_id, started, gpu_seconds, success, error_type=None):
    record = {
        "event": "gpu_infer",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "model": model,
        "status": "success" if success else "error",
        "gpu_infer_seconds": round(gpu_seconds, 3) if gpu_seconds is not None else None,
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    if error_type is not None:
        record["error_type"] = error_type
    # Never include uploaded images, full prompts, or HF credentials in logs.
    print(json.dumps(record, ensure_ascii=False), flush=True)


@spaces.GPU(duration=60)
def infer(model: str, prompt: str, width: int, height: int,
          source_image: Image.Image | None, strength: float, seed: int):
    """Stable Z-Image /infer signature consumed by the CPU UI and OpenAI API."""
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    gpu_seconds = None
    success = False
    error_type = None
    try:
        require_active(model)
        prompt = str(prompt or "").strip()
        width, height = int(width), int(height)
        strength = float(strength)
        if not prompt or len(prompt) > 4000:
            raise gr.Error("提示词不能为空，最多 4000 字")
        if (width, height) not in Z_SIZES:
            raise gr.Error("不支持的 Z-Image 尺寸")
        if not 0.05 <= strength <= 1.0:
            raise gr.Error("strength 必须在 0.05 至 1.0")
        seed = int(seed)
        if seed < 0:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        torch.cuda.synchronize()
        gpu_start = time.perf_counter()
        try:
            image = engine.generate(
                prompt, width, height, seed, source_image, strength
            )
        finally:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            gpu_seconds = time.perf_counter() - gpu_start

        path = save_image(image, "zimage_")
        success = True
        return path, (
            f"request_id={request_id} | model={model} | "
            f"seed={seed} | gpu_infer={gpu_seconds:.2f}s | "
            f"total={time.perf_counter() - started:.2f}s"
        )
    except Exception as exc:
        error_type = type(exc).__name__
        raise
    finally:
        log_inference(model, request_id, started, gpu_seconds, success, error_type)


@spaces.GPU(duration=gpu_duration)
def infer_moss(prompt: str, negative_prompt: str, width: int, height: int,
               steps: int, guidance_scale: float, lora_scale: float,
               realistic_scale: float, add_trigger: bool, seed: int):
    """Stable MOSS /infer_moss signature consumed by the CPU UI and OpenAI API."""
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    gpu_seconds = None
    success = False
    error_type = None
    try:
        require_active("moss-image")
        prompt = str(prompt or "").strip()
        width, height = int(width), int(height)
        if not prompt or len(prompt) > 4000:
            raise gr.Error("提示词不能为空，最多 4000 字")
        if not (512 <= width <= 1536 and 512 <= height <= 1536):
            raise gr.Error("MOSS 尺寸必须在 512 到 1536 之间")
        seed = int(seed)
        if seed < 0:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        torch.cuda.synchronize()
        gpu_start = time.perf_counter()
        try:
            image = engine.generate(
                prompt, negative_prompt or "", width, height, steps,
                guidance_scale, lora_scale, realistic_scale,
                bool(add_trigger), seed,
            )
        finally:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            gpu_seconds = time.perf_counter() - gpu_start

        success = True
        return save_image(image, "moss_"), (
            f"model=moss-image | seed={seed} | gpu_infer={gpu_seconds:.2f}s"
        )
    except Exception as exc:
        error_type = type(exc).__name__
        raise
    finally:
        log_inference("moss-image", request_id, started, gpu_seconds,
                      success, error_type)


def active_model():
    """CPU UI reads the model actually loaded by this running process."""
    return ACTIVE_GPU_MODEL


with gr.Blocks(title="ZeroGPU Image Engine") as demo:
    gr.Markdown("# ZeroGPU Image Engine\n模型由管理员在 AI Image Studio 中切换。")
    prompt = gr.Textbox(label="提示词", value="一只可爱的宇航员猫在月球上")
    with gr.Row():
        width = gr.Number(value=512, precision=0, label="宽")
        height = gr.Number(value=512, precision=0, label="高")
        seed = gr.Number(value=-1, precision=0, label="种子（-1 随机）")
    output_image = gr.Image(type="filepath", label="生成图片")
    output_info = gr.Textbox(label="运行信息")

    # Register both endpoints regardless of active model; inactive requests
    # are rejected by require_active(), without loading another pipeline.
    z_enabled = ACTIVE_GPU_MODEL == "z-image-turbo"
    z_model = gr.Dropdown(choices=["z-image-turbo"], value="z-image-turbo", visible=False)
    strength = gr.Slider(0.05, 1.0, value=0.6, step=0.05,
                         label="编辑强度", visible=z_enabled)
    source_image = gr.Image(type="pil", label="原图（可选）", visible=z_enabled)
    gr.Button("Z-Image 生成", visible=z_enabled, variant="primary").click(
        infer,
        inputs=[z_model, prompt, width, height, source_image, strength, seed],
        outputs=[output_image, output_info],
        api_name="infer", concurrency_limit=1,
    )

    moss_enabled = ACTIVE_GPU_MODEL == "moss-image"
    with gr.Column(visible=moss_enabled):
        negative_prompt = gr.Textbox(value=DEFAULT_NEGATIVE, label="Negative prompt")
        with gr.Accordion("MOSS 参数", open=False):
            steps = gr.Slider(1, 50, value=30, step=1, label="Steps")
            guidance = gr.Slider(1, 7, value=3, step=0.1, label="CFG")
            lora = gr.Slider(0, 1.5, value=0.7, step=0.05, label="LoRA")
            realistic = gr.Slider(0, 2, value=1, step=0.05, label="Realistic")
            trigger = gr.Checkbox(value=True, label="Add moss_xvstyle trigger")
        gr.Button("MOSS 生成", variant="primary").click(
            infer_moss,
            inputs=[prompt, negative_prompt, width, height, steps,
                    guidance, lora, realistic, trigger, seed],
            outputs=[output_image, output_info],
            api_name="infer_moss", concurrency_limit=1,
        )

    # The status endpoint requires neither GPU reservation nor admin credentials.
    gr.Button(visible=False).click(
        active_model, inputs=[], outputs=gr.Textbox(visible=False),
        api_name="active_model", concurrency_limit=1,
    )

demo.queue(default_concurrency_limit=1)
demo.launch()
