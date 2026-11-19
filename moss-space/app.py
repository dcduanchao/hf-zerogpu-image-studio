"""Self-hosted MOSS-Image SFT with two optional LoRA adapters.

Original independent integration based on the public Apache-2.0 example:
https://huggingface.co/spaces/asrarkhann112/moss-image-nsfw-lora
Weights remain hosted by their respective model authors.
"""
import spaces  # MUST precede importing torch/diffusers for ZeroGPU.

import json
import logging
import os
import random
import tempfile
import time
import uuid
from datetime import datetime, timezone

import gradio as gr
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from peft.tuners.lora.layer import LoraLayer
from safetensors.torch import load_file

from models.moss_image import (
    DEFAULT_NEGATIVE,
    MAX_SEED,
    MossImageModel,
    gpu_duration,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moss-engine")
model = MossImageModel()
model.load()

@spaces.GPU(duration=gpu_duration)
def generate(prompt, negative_prompt=DEFAULT_NEGATIVE, width=1024, height=1024,
             steps=30, guidance_scale=3.0, lora_scale=0.7, realistic_scale=1.0,
             add_trigger=True, seed=-1):
    """Run one GPU reservation. Return the PNG path and log duration."""
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    gpu_seconds = None
    status = "error"
    error_type = None
    try:
        if not (prompt or "").strip():
            raise gr.Error("请输入提示词")
        if len(prompt) > 4000:
            raise gr.Error("提示词不能超过 4000 字")
        actual_seed = int(seed) if int(seed) >= 0 else random.randint(0, MAX_SEED)
        torch.cuda.synchronize()
        gpu_start = time.perf_counter()
        try:
            image = model.generate(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                width=width, height=height,
                steps=steps, guidance_scale=guidance_scale,
                lora_scale=lora_scale, realistic_scale=realistic_scale,
                add_trigger=bool(add_trigger), seed=actual_seed,
            )
        finally:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            gpu_seconds = time.perf_counter() - gpu_start
        filepath = os.path.join(tempfile.mkdtemp(prefix="moss_"), "image.png")
        image.save(filepath, "PNG")
        status = "success"
        return filepath
    except Exception as exc:
        error_type = type(exc).__name__
        raise
    finally:
        info = {
            "event": "gpu_infer",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "status": status,
            "model": "moss-image",
            "width": width, "height": height,
            "gpu_infer_seconds": round(gpu_seconds, 3) if gpu_seconds is not None else None,
            "total_seconds": round(time.perf_counter() - start, 3),
        }
        if error_type:
            info["error_type"] = error_type
        logger.info(json.dumps(info, ensure_ascii=False, default=str))

with gr.Blocks(title="MOSS Image Engine") as demo:
    gr.Markdown("# MOSS-Image Engine\n独立 ZeroGPU 推理服务，供 AI Image Studio 调用。")
    prompt = gr.Textbox(label="Prompt", lines=3)
    negative_prompt = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE)
    with gr.Row():
        width = gr.Slider(512, 1536, value=1024, step=16, label="Width")
        height = gr.Slider(512, 1536, value=1024, step=16, label="Height")
    with gr.Row():
        steps = gr.Slider(1, 50, value=30, step=1, label="Steps")
        guidance_scale = gr.Slider(1, 7, value=3.0, step=0.1, label="Guidance")
    with gr.Row():
        lora_scale = gr.Slider(0, 1.5, value=0.7, step=0.05, label="LoRA")
        realistic_scale = gr.Slider(0, 2.0, value=1.0, step=0.05, label="Realistic")
    add_trigger = gr.Checkbox(value=True, label="Add moss_xvstyle trigger")
    seed = gr.Number(-1, precision=0, label="Seed (-1 for random)")
    button = gr.Button("生成", variant="primary")
    output = gr.Image(type="filepath", label="图片")
    button.click(
        generate,
        inputs=[prompt, negative_prompt, width, height, steps, guidance_scale,
                lora_scale, realistic_scale, add_trigger, seed],
        outputs=output, api_name="generate", concurrency_limit=1,
    )

demo.queue(default_concurrency_limit=1)
demo.launch()
