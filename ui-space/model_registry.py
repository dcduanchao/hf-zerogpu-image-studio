"""Single source of truth for UI, OpenAI model listing, and routing."""
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    space_id: str
    sizes: tuple[str, ...]
    default_size: str
    can_edit: bool = False
    multi_reference: bool = False
    public_url: bool = True
    usage_notes: str = ""

Z_SPACE = os.getenv("GPU_SPACE_ID", "duan0320/my-zerogpu-engine").strip()

MODELS = {
    "z-image-turbo": ModelSpec(
        "z-image-turbo", "Z-Image Turbo", Z_SPACE,
        ("512x512", "768x768", "1024x1024", "1024x768", "768x1024"),
        "512x512", True, True, True,
        "文生图及单图 img2img；多图通过参考拼图进行实验性编辑（非原生多图融合）。第1张是主图，后续为参考；公共 HTTPS/HTTP 图片 URL 可用。",
    ),
    "minimax-h3-image": ModelSpec(
        "minimax-h3-image", "MiniMax H3-Image · 官方 API（自带 Key）", "MiniMax 官方 API",
        ("1024x1024", "1024x768", "768x1024", "1536x1024", "1024x1536",
         "2048x2048", "2048x1536", "1536x2048"),
        "1024x1024", True, True, True,
        "通过 MiniMax 官方 API 生图/多图编辑；需要独立 MiniMax API Key（不是 HF Token），"
        "费用属于 MiniMax。此选项不在 ZeroGPU 加载 68GB 级大模型；支持最多 4 张图片或公共 URL。",
    ),
    "moss-image": ModelSpec(
        "moss-image", "MOSS-Image (Multi-LoRA)", Z_SPACE,
        ("512x512", "768x768", "1024x1024", "1024x768", "768x1024",
         "576x1024", "1024x576", "864x1536", "1536x864"),
        "1024x1024", False, False, False,
        "当前 MOSS-Image SFT 插件仅支持文生图，不支持图片编辑、参考图融合或图片 URL 输入。",
    ),
}

def get_model(model_id: str) -> ModelSpec:
    try:
        model = MODELS[model_id]
    except KeyError:
        raise ValueError(f"未知模型：{model_id}") from None
    if model_id != "minimax-h3-image" and (not model.space_id or "/" not in model.space_id):
        raise RuntimeError(f"模型 {model_id} 的 Space ID 未配置")
    return model

def display_names():
    return {spec.name: spec.id for spec in MODELS.values()}
