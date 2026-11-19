import os
import re
from pathlib import Path

import gradio as gr
from gradio_client import Client, handle_file
from huggingface_hub import HfApi

SPACE_ID = os.environ.get("GPU_SPACE_ID", "").strip()
MODEL_NAMES = {"Z-Image-Turbo": "z-image-turbo", "SDXL-Turbo": "sdxl-turbo"}
SIZE_MAP = {"512 × 512": (512, 512), "768 × 768": (768, 768),
            "1024 × 1024": (1024, 1024), "1024 × 768": (1024, 768),
            "768 × 1024": (768, 1024)}


def friendly_error(exc):
    msg = str(exc)
    low = msg.lower()
    if any(x in low for x in ["quota", "duration", "gpu limit", "exceeded", "insufficient"]):
        return "ZeroGPU 额度不足或本次任务超出可用时长，请稍后再试。"
    if any(x in low for x in ["401", "403", "invalid token", "unauthorized", "forbidden"]):
        return "HF Token 无效或权限不足；不会自动改为匿名。"
    if any(x in low for x in ["timeout", "timed out"]):
        return "ZeroGPU 请求超时，请稍后重试。"
    return "生成失败，请查看 GPU Space 的 Logs（不展示可能包含敏感数据的原始异常）。"


def generate(mode, model_label, prompt, source, size_label, count, strength, seed, token,
             request: gr.Request):
    if not SPACE_ID or "/" not in SPACE_ID:
        raise gr.Error("请先设置 UI Space 变量 GPU_SPACE_ID=用户名/GPU-Space名")
    prompt = (prompt or "").strip()
    if not prompt or len(prompt) > 4000:
        raise gr.Error("请输入 1～4000 字的提示词")
    model = MODEL_NAMES[model_label]
    if mode == "图片编辑" and not source:
        raise gr.Error("图片编辑模式需要上传原图")
    if mode == "图片编辑" and model != "z-image-turbo":
        raise gr.Error("当前只有 Z-Image-Turbo 支持图片编辑")
    width, height = SIZE_MAP[size_label]
    count = int(count)
    if count not in [1, 2, 3, 4]:
        raise gr.Error("最多生成 4 张")
    token = (token or "").strip()
    if token and not re.fullmatch(r"hf_[A-Za-z0-9]+", token):
        raise gr.Error("HF Token 格式不正确")

    headers = {}
    if not token and request is not None:
        xip = request.headers.get("x-ip-token", "")
        if xip:
            headers["x-ip-token"] = xip

    images = []
    messages = []
    try:
        client = Client(SPACE_ID, token=token if token else False, headers=headers)
        for idx in range(count):
            img_arg = handle_file(source) if mode == "图片编辑" else None
            result_path, meta = client.predict(
                model, prompt, width, height, img_arg,
                float(strength), int(seed) if int(seed) >= 0 else -1,
                api_name="/infer",
            )
            if not Path(result_path).is_file():
                raise RuntimeError("GPU Space 未返回本地图片文件")
            images.append((result_path, f"第 {idx+1} 张 · {model}"))
            messages.append(meta)
    except Exception as exc:
        if not images:
            raise gr.Error(friendly_error(exc)) from None
        return images, "已生成部分图片，后续任务未完成：" + friendly_error(exc)

    return images, "\n".join(messages)


def token_info(token):
    token = (token or "").strip()
    if not token:
        return "匿名模式：使用共享匿名额度（不保证独立额度）。"
    try:
        identity = HfApi(token=token).whoami()
        return f"Token 有效：{identity.get('name', 'HF 用户')}；将按该账号配额调度。"
    except Exception:
        return "Token 验证失败；请检查权限和内容。"


CSS = """
.gradio-container {max-width: 980px!important; margin: auto!important;}
#hero {text-align: center; padding: 8px 0 14px;}
"""
with gr.Blocks(title="AI 图片工作室", css=CSS) as demo:
    gr.Markdown("# 🎨 AI 图片工作室\n文生图 · 图片编辑 · Hugging Face ZeroGPU", elem_id="hero")
    with gr.Row():
        with gr.Column(scale=5):
            mode = gr.Radio(["文生图", "图片编辑"], value="文生图", label="生成模式")
            prompt = gr.Textbox(label="图片描述", lines=5, max_lines=8,
                                placeholder="描述你想生成的画面，或者说明如何修改原图……")
            source = gr.Image(type="filepath", label="上传原图（图片编辑使用）", visible=False)
            with gr.Row():
                model = gr.Dropdown(list(MODEL_NAMES), value="Z-Image-Turbo", label="模型")
                size = gr.Dropdown(list(SIZE_MAP), value="512 × 512", label="尺寸")
            with gr.Row():
                count = gr.Slider(1, 4, 1, step=1, label="生成数量")
                seed = gr.Number(-1, precision=0, label="Seed（-1 随机）")
            strength = gr.Slider(0.05, 1.0, value=0.6, step=0.05,
                                 label="图片编辑强度", visible=False)
            with gr.Accordion("Hugging Face Token（可选）", open=True):
                token = gr.Textbox(label="HF Token", type="password",
                                   placeholder="留空匿名；填写 hf_... 计入你的账号额度")
                check = gr.Button("验证 Token", size="sm")
                status = gr.Markdown("匿名模式：共享额度；耗尽则直接返回错误。")
                check.click(token_info, inputs=token, outputs=status, api_visibility="private")
                gr.Markdown("Token 仅用于本次请求，禁止记录到日志；使用你信任的 Space。")
            run = gr.Button("开始生成", variant="primary")
        with gr.Column(scale=5):
            gallery = gr.Gallery(label="结果", columns=2, height=500,
                                 show_label=True, preview=True)
            info = gr.Textbox(label="执行信息", lines=3)
    mode.change(
        lambda value: (gr.update(visible=value == "图片编辑"),
                       gr.update(visible=value == "图片编辑")),
        inputs=mode, outputs=[source, strength], api_visibility="private",
    )
    run.click(
        generate,
        inputs=[mode, model, prompt, source, size, count, strength, seed, token],
        outputs=[gallery, info],
        api_visibility="private", concurrency_limit=4,
    )

demo.queue(default_concurrency_limit=4)

# Serve both the Gradio UI (/) and OpenAI-style Images API (/v1/images/*)
# from the same CPU Basic Space, on a single port.
import uvicorn
from openai_api import app as api_app

application = gr.mount_gradio_app(api_app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(application, host="0.0.0.0", port=7860, workers=1)

