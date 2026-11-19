"""ZeroGPU engine. Do not accept HF tokens as prediction inputs."""
import os
import time
import tempfile

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
    """Return (PNG path, execution metadata); one image per GPU reservation."""
    start = time.perf_counter()
    prompt = str(prompt).strip()
    width, height = int(width), int(height)
    if not prompt or len(prompt) > 4000:
        raise gr.Error("提示词不能为空，最多 4000 字")
    if (width, height) not in ALLOWED_SIZES:
        raise gr.Error("不支持的图片尺寸")
    if not 0.05 <= float(strength) <= 1.0:
        raise gr.Error("strength 必须在 0.05 至 1.0")

    seed = int(seed)
    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    generator = torch.Generator(device="cuda").manual_seed(seed)

    with torch.inference_mode():
        if model == "z-image-turbo":
            options = dict(prompt=prompt, num_inference_steps=8,
                           guidance_scale=0.0, generator=generator)
            if source_image is None:
                image = z_txt(width=width, height=height, **options).images[0]
            else:
                image = z_edit(
                    image=source_image.convert("RGB").resize((width, height)),
                    strength=float(strength), **options
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

    path = os.path.join(tempfile.mkdtemp(prefix="zimage_"), "image.png")
    image.save(path, format="PNG")
    return path, f"model={model} | seed={seed} | gpu_function={time.perf_counter() - start:.1f}s"


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
