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
from diffusers import ZImagePipeline, ZImageImg2ImgPipeline, AutoPipelineForText2Image

Z_MODEL_ID = os.getenv("Z_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
SDXL_MODEL_ID = os.getenv("SDXL_MODEL_ID", "stabilityai/sdxl-turbo")
ENABLE_SDXL = os.getenv("ENABLE_SDXL", "0") == "1"

# ZeroGPU handles CUDA through its GPU decorator and module-load placement.
z_txt = ZImagePipeline.from_pretrained(Z_MODEL_ID, dtype=torch.bfloat16)
z_txt.to("cuda")
z_edit = ZImageImg2ImgPipeline.from_pipe(z_txt)

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
                        image = z_txt(width=width, height=height, **options).images[0]
                    else:
                        image = z_edit(
                            image=source_image.convert("RGB").resize((width, height)),
                            strength=float(strength), **options,
                        ).images[0]
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
    run.click(
        fn=infer,
        inputs=[model, prompt, width, height, source_image, strength, seed],
        outputs=[output_image, output_info],
        api_name="infer",
        concurrency_limit=1,
    )

demo.queue(default_concurrency_limit=1)
demo.launch()
