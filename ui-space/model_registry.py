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

Z_SPACE = os.getenv("GPU_SPACE_ID", "duan0320/my-zerogpu-engine").strip()
MOSS_SPACE = os.getenv("MOSS_SPACE_ID", "asrarkhann112/moss-image-nsfw-lora").strip()

MODELS = {
    "z-image-turbo": ModelSpec(
        "z-image-turbo", "Z-Image Turbo", Z_SPACE,
        ("512x512", "768x768", "1024x1024", "1024x768", "768x1024"),
        "512x512", True,
    ),
    "moss-image": ModelSpec(
        "moss-image", "MOSS-Image (Multi-LoRA)", MOSS_SPACE,
        ("512x512", "768x768", "1024x1024", "1024x768", "768x1024",
         "576x1024", "1024x576", "864x1536", "1536x864"),
        "1024x1024", False,
    ),
}

def get_model(model_id: str) -> ModelSpec:
    try:
        model = MODELS[model_id]
    except KeyError:
        raise ValueError(f"未知模型：{model_id}") from None
    if not model.space_id or "/" not in model.space_id:
        raise RuntimeError(f"模型 {model_id} 的 Space ID 未配置")
    return model

def display_names():
    return {spec.name: spec.id for spec in MODELS.values()}
