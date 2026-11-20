"""Experimental Qwen-Image-2.1 GGUF image generation on HF ZeroGPU large.

Model loading is deliberately lazy: existing CUDA probes remain available
even if Qwen model dependencies or weights are inaccessible.
"""
import gc
import os
import secrets
import tempfile
import time
from pathlib import Path

import spaces  # import before torch
import gradio as gr

GGUF_REPO = "KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF"
GGUF_FILE = "qwen-image-2.1-UC-Q4_K_M.gguf"
BASE_REPO = "Qwen/Qwen-Image-2.1"
BASE_REV = "790c92633540aa0cb11d9abf19eb46d861714758"
PIPE = None


def build_pipeline():
    """Create once; ZeroGPU manages worker GPU access during inference."""
    global PIPE
    if PIPE is not None:
        return PIPE

    import torch
    from huggingface_hub import hf_hub_download
    from diffusers import (
        QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
        GGUFQuantizationConfig,
    )

    path = hf_hub_download(
        repo_id=GGUF_REPO,
        filename=GGUF_FILE,
        token=os.environ.get("HF_TOKEN") or None,
    )
    transformer = QwenImage21Transformer2DModel.from_single_file(
        path,
        config=BASE_REPO,
        subfolder="transformer",
        revision=BASE_REV,
        quantization_config=GGUFQuantizationConfig(
            compute_dtype=torch.bfloat16
        ),
        torch_dtype=torch.bfloat16,
    )
    pipeline = QwenImage21Pipeline.from_pretrained(
        BASE_REPO,
        revision=BASE_REV,
        transformer=transformer,
        dtype=torch.bfloat16,
    )
    # Keep large text encoder on CPU when it is not in use.
    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)
    PIPE = pipeline
    return PIPE


@spaces.GPU(duration=300, size="large")
def generate_qwen_gguf(prompt, size, steps, seed, guidance):
    """Qwen GGUF image generation. Run as direct Gradio event target."""
    import torch

    prompt = (prompt or "").strip()
    if not prompt or len(prompt) > 4000:
        raise gr.Error("Prompt 不能为空，最多 4000 字")
    if size not in ("512x512", "768x768", "1024x1024"):
        raise gr.Error("不支持的图像尺寸")
    width, height = map(int, size.split("x"))
    steps = int(steps)
    if steps not in range(1, 41):
        raise gr.Error("Steps 必须为 1-40")
    guidance = float(guidance)
    if not 1 <= guidance <= 6:
        raise gr.Error("Guidance 必须为 1-6")
    seed = int(seed)
    if seed < 0:
        seed = secrets.randbelow(2**31)
    if seed >= 2**31:
        raise gr.Error("Seed 超出范围")

    started = time.perf_counter()
    try:
        pipe = build_pipeline()
        result = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=torch.Generator("cpu").manual_seed(seed),
        )
        output = Path(tempfile.mkdtemp(prefix="qwen_gguf_")) / "image.png"
        result.images[0].save(output)
        stats = {
            "model": GGUF_REPO,
            "file": GGUF_FILE,
            "base": BASE_REPO,
            "seed": seed,
            "size": size,
            "steps": steps,
            "seconds": round(time.perf_counter() - started, 2),
            "cuda": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            stats["max_gpu_allocated_gb"] = round(
                torch.cuda.max_memory_allocated() / 2**30, 2
            )
        return str(output), stats
    except Exception as exc:
        import traceback
        # Log exception class and a capped traceback; never log tokens.
        print("Qwen GGUF generation failure:", type(exc).__name__, flush=True)
        traceback.print_exc(limit=7)
        raise gr.Error(
            "Qwen GGUF 测试失败："
            + type(exc).__name__ + ": " + str(exc)[:450]
        ) from None


def mount_qwen_section():
    """Call from inside the parent gr.Blocks context."""
    gr.Markdown("## 实验：Qwen-Image-2.1 GGUF · ZeroGPU large")
    gr.Markdown(
        "Q4_K_M 图像 Transformer + 官方基础模型组件；"
        "未经过 ZeroGPU 生图验证。首次加载需要下载模型，可能超过 GPU 时长。"
    )
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Qwen Prompt",
                value="An astronaut exploring a jungle, cinematic lighting",
                lines=3,
            )
            size = gr.Dropdown(
                choices=["512x512", "768x768", "1024x1024"],
                value="512x512",
                label="Size",
            )
            steps = gr.Slider(1, 40, value=10, step=1, label="Steps")
            guidance = gr.Slider(1, 6, value=4, step=0.5, label="Guidance")
            seed = gr.Number(value=42, precision=0, label="Seed (-1 随机)")
            button = gr.Button("测试 Qwen GGUF 生图", variant="primary")
        with gr.Column():
            image = gr.Image(type="filepath", label="Qwen GGUF Result")
            result = gr.JSON(label="Qwen GGUF Stats")
    button.click(
        generate_qwen_gguf,
        inputs=[prompt, size, steps, seed, guidance],
        outputs=[image, result],
        api_name="qwen_gguf_generate",
        concurrency_limit=1,
    )
