"""ZeroGPU engine. Do not accept HF tokens as prediction inputs."""
import os
import time
import tempfile
import json
import uuid
from datetime import datetime, timezone

import gradio as gr
import spaces
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image
from models.z_image import ZImageModel
from models.moss_image import MossImageModel, DEFAULT_NEGATIVE, gpu_duration

Z_MODEL_ID = os.getenv("Z_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
SDXL_MODEL_ID = os.getenv("SDXL_MODEL_ID", "stabilityai/sdxl-turbo")
ENABLE_SDXL = os.getenv("ENABLE_SDXL", "0") == "1"

# Keep Z-Image available even if optional MOSS initialization fails.
z_image = ZImageModel()
z_image.load()
moss_image = None
moss_error = None
if os.getenv("ENABLE_MOSS", "1") == "1":
    try:
        moss_image = MossImageModel()
        moss_image.load()
        print("MOSS plugin loaded successfully", flush=True)
    except Exception as exc:
        moss_image = None
        moss_error = type(exc).__name__
        print(f"MOSS plugin unavailable: {moss_error}; Z-Image remains enabled", flush=True)

sdxl_txt = None
if ENABLE_SDXL:
    sdxl_txt = AutoPipelineForText2Image.from_pretrained(
        SDXL_MODEL_ID, torch_dtype=torch.float16, variant="fp16"
    )
    sdxl_txt.to("cuda")

ALLOWED_SIZES = {(512, 512), (768, 768), (1024, 1024), (1024, 768), (768, 1024)}


@spaces.GPU(duration=60)
def infer(model: str, prompt: str, width: int, height: int,
          source_image: Image.Image | None, strength: float, seed: int):
    """Return image and timing info; log every GPU invocation (including failures).

    These are measured wall-clock durations, not Hugging Face billed GPU seconds.
    ZeroGPU scheduling/queue wait happens before entering this function.
    """
    request_id = uuid.uuid4().hex[:12]
    total_start = time.perf_counter()
    gpu_infer_seconds = None
    status = "error"
    error_type = None
    mode = "img2img" if source_image is not None else "txt2img"
    width_for_log, height_for_log = width, height
    seed_for_log = None

    try:
        prompt = str(prompt).strip()
        width, height = int(width), int(height)
        width_for_log, height_for_log = width, height

        if not prompt or len(prompt) > 4000:
            raise gr.Error("提示词不能为空，最多 4000 字")
        if (width, height) not in ALLOWED_SIZES:
            raise gr.Error("不支持的图片尺寸")
        if not 0.05 <= float(strength) <= 1.0:
            raise gr.Error("strength 必须在 0.05 至 1.0")

        seed = int(seed)
        if seed < 0:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        seed_for_log = seed
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # CUDA is asynchronous; synchronize on both sides for reliable inference timing.
        torch.cuda.synchronize()
        gpu_start = time.perf_counter()
        try:
            with torch.inference_mode():
                if model == "z-image-turbo":
                    options = dict(
                        prompt=prompt, num_inference_steps=8,
                        guidance_scale=0.0, generator=generator,
                    )
                    if source_image is None:
                        image = z_image.generate(prompt, width, height, seed)
                    else:
                        image = z_image.generate(prompt, width, height, seed, source_image, strength)
                elif model == "sdxl-turbo" and sdxl_txt is not None:
                    if source_image is not None:
                        raise gr.Error("SDXL-Turbo 当前仅支持文生图")
                    image = sdxl_txt(
                        prompt=prompt, width=width, height=height,
                        num_inference_steps=2, guidance_scale=0.0,
                        generator=generator,
                    ).images[0]
                else:
                    raise gr.Error("模型不存在或未启用")
        finally:
            # If CUDA itself failed, leave the original inference error intact.
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            gpu_infer_seconds = time.perf_counter() - gpu_start

        path = os.path.join(tempfile.mkdtemp(prefix="zimage_"), "image.png")
        image.save(path, format="PNG")
        status = "success"
        return (
            path,
            f"request_id={request_id} | model={model} | mode={mode} "
            f"| seed={seed} | gpu_infer={gpu_infer_seconds:.2f}s "
            f"| total={time.perf_counter() - total_start:.2f}s",
        )
    except Exception as exc:
        error_type = type(exc).__name__
        raise
    finally:
        record = {
            "event": "gpu_infer",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "status": status,
            "model": str(model),
            "mode": mode,
            "width": width_for_log,
            "height": height_for_log,
            "seed": seed_for_log,
            "gpu_infer_seconds": round(gpu_infer_seconds, 3) if gpu_infer_seconds is not None else None,
            "total_seconds": round(time.perf_counter() - total_start, 3),
        }
        if error_type is not None:
            record["error_type"] = error_type
        # No HF tokens, uploaded image contents or full prompts in logs.
        print(json.dumps(record, ensure_ascii=False, default=str), flush=True)

@spaces.GPU(duration=gpu_duration)
def infer_moss(prompt: str, negative_prompt: str, width: int, height: int,
               steps: int, guidance_scale: float, lora_scale: float,
               realistic_scale: float, add_trigger: bool, seed: int):
    """Separate decorated endpoint allows per-model reservation duration."""
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    gpu_seconds = None
    error_type = None
    status = "error"
    try:
        if moss_image is None:
            raise gr.Error(f"MOSS 模型未加载成功：{moss_error or 'disabled'}")
        prompt = str(prompt or "").strip()
        if not prompt or len(prompt) > 4000:
            raise gr.Error("请输入 1～4000 字的提示词")
        width, height = int(width), int(height)
        if not (512 <= width <= 1536 and 512 <= height <= 1536):
            raise gr.Error("MOSS 尺寸必须在 512 到 1536 之间")
        seed = int(seed)
        if seed < 0:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        torch.cuda.synchronize()
        gpu_start = time.perf_counter()
        try:
            image = moss_image.generate(
                prompt, negative_prompt, width, height, steps,
                guidance_scale, lora_scale, realistic_scale, add_trigger, seed,
            )
        finally:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            gpu_seconds = time.perf_counter() - gpu_start
        path = os.path.join(tempfile.mkdtemp(prefix="moss_"), "image.png")
        image.save(path, format="PNG")
        status = "success"
        return path, f"model=moss-image | seed={seed} | gpu_infer={gpu_seconds:.2f}s"
    except Exception as exc:
        error_type = type(exc).__name__
        raise
    finally:
        record = {
            "event": "gpu_infer",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "model": "moss-image",
            "status": status,
            "gpu_infer_seconds": round(gpu_seconds, 3) if gpu_seconds is not None else None,
            "total_seconds": round(time.perf_counter() - started, 3),
        }
        if error_type:
            record["error_type"] = error_type
        print(json.dumps(record, ensure_ascii=False), flush=True)


with gr.Blocks(title="ZeroGPU Image Engine") as demo:
    gr.Markdown("# ZeroGPU Image Engine\n此 Space 提供 GPU 推理 API。请从独立 UI Space 调用。")
    with gr.Row():
        model = gr.Dropdown(
            choices=["z-image-turbo"] + (["sdxl-turbo"] if ENABLE_SDXL else []),
            value="z-image-turbo", label="模型",
        )
        prompt = gr.Textbox(label="提示词", value="一只可爱的宇航员猫在月球上")
    with gr.Row():
        width = gr.Number(value=512, precision=0, label="宽")
        height = gr.Number(value=512, precision=0, label="高")
        strength = gr.Slider(0.05, 1.0, 0.6, step=0.05, label="图生图强度")
        seed = gr.Number(value=-1, precision=0, label="随机种子（-1随机）")
    source_image = gr.Image(type="pil", label="输入原图（可选）")
    run = gr.Button("生成", variant="primary")
    output_image = gr.Image(type="filepath", label="生成结果")
    output_info = gr.Textbox(label="运行信息")
    with gr.Accordion("MOSS-Image 参数（MOSS API）", open=False):
        moss_negative = gr.Textbox(value=DEFAULT_NEGATIVE, label="Negative prompt")
        moss_steps = gr.Slider(1, 50, value=30, step=1, label="Steps")
        moss_cfg = gr.Slider(1, 7, value=3, label="CFG")
        moss_lora = gr.Slider(0, 1.5, value=0.7, label="LoRA scale")
        moss_realistic = gr.Slider(0, 2, value=1, label="Realistic scale")
        moss_trigger = gr.Checkbox(value=True, label="Add moss_xvstyle trigger")
    run.click(
        fn=infer,
        inputs=[model, prompt, width, height, source_image, strength, seed],
        outputs=[output_image, output_info],
        api_name="infer",
        concurrency_limit=1,
    )
    moss_run = gr.Button("MOSS 生成", variant="secondary")
    moss_run.click(
        fn=infer_moss,
        inputs=[prompt, moss_negative, width, height, moss_steps, moss_cfg,
                moss_lora, moss_realistic, moss_trigger, seed],
        outputs=[output_image, output_info],
        api_name="infer_moss",
        concurrency_limit=1,
    )

demo.queue(default_concurrency_limit=1)
demo.launch()
