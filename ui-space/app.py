import re

import gradio as gr
from huggingface_hub import HfApi
from token_quota import token_quota_info
from model_registry import MODELS, display_names, get_model
from model_router import generate as route_generate, local_image_path
from admin_switch import SwitchDenied, switch_active_model
from model_state import active_model_id, require_active_model, ModelMismatch, ModelUnavailable

MODEL_NAMES = display_names()
SIZE_LABEL = lambda size: size.replace("x", " × ")
SIZE_MAP = {SIZE_LABEL(size): size for spec in MODELS.values() for size in spec.sizes}


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
    prompt = (prompt or "").strip()
    if not prompt or len(prompt) > 4000:
        raise gr.Error("请输入 1～4000 字的提示词")
    model = MODEL_NAMES[model_label]
    try:
        require_active_model(model)
    except (ModelMismatch, ModelUnavailable) as exc:
        raise gr.Error(str(exc)) from None
    spec = get_model(model)
    if mode == "图片编辑" and not source:
        raise gr.Error("图片编辑模式需要上传原图")
    if mode == "图片编辑" and not spec.can_edit:
        raise gr.Error(f"{spec.name} 当前不支持图片编辑")
    size = SIZE_MAP[size_label]
    if size not in spec.sizes:
        raise gr.Error(f"{spec.name} 不支持尺寸 {size}")
    width, height = map(int, size.split("x"))
    count = int(count)
    if count < 1 or count > 4:
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
        for idx in range(count):
            result, meta = route_generate(
                model, prompt, width, height,
                seed=int(seed) if seed is not None else -1, token=token or None,
                image_path=source if mode == "图片编辑" else None,
                strength=strength, download_files=True, headers=headers,
            )
            images.append((local_image_path(result), f"第 {idx+1} 张 · {spec.name}"))
            messages.append(meta)
    except (ModelMismatch, ModelUnavailable) as exc:
        raise gr.Error(str(exc)) from None
    except Exception as exc:
        if not images:
            raise gr.Error(friendly_error(exc)) from None
        return images, "部分图片已生成，后续失败：" + friendly_error(exc)

    return images, "\n".join(messages)


def refresh_active_model():
    """Refresh model choices from the GPU process actually serving inference."""
    try:
        active = active_model_id()
    except ModelUnavailable as exc:
        return (
            gr.update(choices=[], value=None, interactive=False),
            gr.update(choices=[], value=None, interactive=False),
            gr.update(value="文生图"),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(value=f"GPU 暂时不可用：{exc}"),
            gr.update(interactive=False),
        )
    spec = MODELS[active]
    return (
        gr.update(choices=[spec.name], value=spec.name, interactive=False),
        gr.update(choices=[SIZE_LABEL(v) for v in spec.sizes],
                  value=SIZE_LABEL(spec.default_size), interactive=True),
        gr.update(value="文生图"),
        gr.update(visible=False), gr.update(visible=False),
        gr.update(value=f"当前已加载：{spec.name}（{active}）。如需其他模型，请联系管理员切换。"),
        gr.update(interactive=True),
    )


def token_info(token):
    return token_quota_info(token)


def admin_switch_model(model_label, admin_token):
    try:
        model_id = MODEL_NAMES[model_label]
        return switch_active_model(model_id, admin_token)
    except (SwitchDenied, KeyError) as exc:
        return f"切换失败：{exc}"


CSS = """
.gradio-container {max-width: 980px!important; margin: auto!important;}
#hero {text-align: center; padding: 8px 0 14px;}
"""
with gr.Blocks(title="AI 图片工作室") as demo:
    gr.Markdown("# 🎨 AI 图片工作室\n文生图 · 图片编辑 · Hugging Face ZeroGPU", elem_id="hero")
    with gr.Row():
        with gr.Column(scale=5):
            mode = gr.Radio(["文生图", "图片编辑"], value="文生图", label="生成模式")
            prompt = gr.Textbox(label="图片描述", lines=5, max_lines=8,
                                placeholder="描述你想生成的画面，或者说明如何修改原图……")
            source = gr.Image(type="filepath", label="上传原图（图片编辑使用）", visible=False)
            with gr.Row():
                model = gr.Dropdown([MODELS["z-image-turbo"].name], value=MODELS["z-image-turbo"].name,
                                    label="当前可用模型", interactive=False)
                size = gr.Dropdown([SIZE_LABEL(v) for v in MODELS["z-image-turbo"].sizes],
                                   value="512 × 512", label="尺寸")
            with gr.Row():
                count = gr.Slider(1, 4, 1, step=1, label="生成数量")
                seed = gr.Number(-1, precision=0, label="Seed（-1 随机）")
            strength = gr.Slider(0.05, 1.0, value=0.6, step=0.05,
                                 label="图片编辑强度", visible=False)
            with gr.Accordion("Hugging Face Token（可选）", open=True):
                token = gr.Textbox(label="HF Token", type="password",
                                   placeholder="留空匿名；填写 hf_... 计入你的账号额度")
                check = gr.Button("验证 Token / 查询 ZeroGPU 剩余时间", size="sm")
                status = gr.Markdown("匿名模式：共享额度；耗尽则直接返回错误。")
                check.click(token_info, inputs=token, outputs=status, api_visibility="private")
                gr.Markdown("Token 仅用于本次请求，禁止记录到日志；使用你信任的 Space。")
            model_info = gr.Markdown("正在检查 GPU 当前模型……")
            refresh_button = gr.Button("刷新当前模型状态", size="sm")
            run = gr.Button("开始生成", variant="primary", interactive=False)
            with gr.Accordion("管理员 · 切换 GPU 模型（仅所有者）", open=False):
                gr.Markdown("切换后 GPU Space 将重启，正在执行的生成任务可能中断。只有经过 HF 账号验证的 Space 所有者有权操作。")
                admin_model = gr.Dropdown(list(MODEL_NAMES), value=MODELS["z-image-turbo"].name, label="要加载的 GPU 模型")
                admin_token = gr.Textbox(label="管理员 HF Token（需具有 Space 写权限）",
                                         type="password", placeholder="hf_...")
                admin_run = gr.Button("验证身份并切换模型", variant="secondary")
                admin_status = gr.Markdown("普通用户无需使用此功能。")
                admin_run.click(admin_switch_model, inputs=[admin_model, admin_token],
                                outputs=admin_status, api_visibility="private", concurrency_limit=1)
        with gr.Column(scale=5):
            gallery = gr.Gallery(label="结果", columns=2, height=500,
                                 show_label=True, preview=True)
            info = gr.Textbox(label="执行信息", lines=3)
    demo.load(refresh_active_model, inputs=[],
              outputs=[model, size, mode, source, strength, model_info, run],
              api_visibility="private")
    refresh_button.click(refresh_active_model, inputs=[],
                         outputs=[model, size, mode, source, strength, model_info, run],
                         api_visibility="private")
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

