"""MOSS-Image-SFT inference and LoRA initialization.

Based on the public Apache-2.0 example at:
https://huggingface.co/spaces/asrarkhann112/moss-image-nsfw-lora
The base model and adapter weights are fetched from the original Hub repos.
"""
import json
import logging
import os

import torch
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from peft.tuners.lora.layer import LoraLayer
from safetensors.torch import load_file

logger = logging.getLogger(__name__)
MODEL_REPO = os.getenv("MOSS_MODEL_ID", "OpenMOSS-Team/MOSS-Image-SFT")
LORA_REPO = os.getenv("MOSS_LORA_ID", "axiomofmind/Moss-image-nsfw-uncensored-lora")
REALISTIC_REPO = os.getenv("MOSS_REALISTIC_LORA_ID", "asrarkhann112/moss-realistic-snapshot-lora")
REALISTIC_FILENAME = "MOSSRealisticSnapshotV1.safetensors"
TRIGGER = "moss_xvstyle"
MAX_SEED = 2**32 - 1
DEFAULT_NEGATIVE = (
    "blurry, low quality, distorted, bad anatomy, deformed fingers, extra limbs, "
    "CGI, render, 3d, cartoon, airbrushed, plastic skin, oversaturated, watermark"
)

def snap(value):
    value = int(value)
    return min(1536, max(512, round(value / 16) * 16))

def gpu_duration(prompt, negative_prompt, width, height, steps, *args, **kwargs):
    pixels = snap(width) * snap(height) / (1024 * 1024)
    return int(min(180, 20 + int(steps) * 2.2 * max(pixels, 0.5)))

class SnapshotLoKR:
    def __init__(self, deltas):
        self.deltas = deltas
        self.scale = 0.0

    @torch.no_grad()
    def set_scale(self, value):
        new_scale = max(0.0, min(2.0, float(value)))
        change = new_scale - self.scale
        if abs(change) < 1e-6:
            return
        for layer, delta in self.deltas:
            layer.weight.add_(
                delta.to(device=layer.weight.device, dtype=layer.weight.dtype),
                alpha=change,
            )
        self.scale = new_scale

def load_snapshot(transformer):
    filename = hf_hub_download(REALISTIC_REPO, REALISTIC_FILENAME)
    tensors = load_file(filename, device="cpu")
    loaded = []
    # The LoKR checkpoint factors the projection matrices as Kronecker products.
    # Do not change mapping without checking the upstream weight tensor shapes.
    for i, layer in enumerate(transformer.layers):
        key = f"diffusion_model.layers.{i}.proj"
        matrix_targets = (
            ("ffn_down", ("ffn_down",)),
            ("ffn_up_gate", ("ffn_up_gate",)),
            ("o", ("o",)),
            ("qkv", ("q", "k", "v")),
        )
        for checkpoint_name, module_names in matrix_targets:
            k1, k2 = f"{key}.{checkpoint_name}.lokr_w1", f"{key}.{checkpoint_name}.lokr_w2"
            if k1 not in tensors or k2 not in tensors:
                continue
            matrix = torch.kron(tensors[k1].float(), tensors[k2].float()).to(torch.bfloat16)
            if len(module_names) == 1:
                target = layer.proj[module_names[0]]
                if matrix.shape != target.weight.shape:
                    raise RuntimeError(f"LoKR shape mismatch at {k1}")
                loaded.append((target, matrix))
            else:
                chunks = torch.split(matrix, layer.proj["q"].weight.shape[0], dim=0)
                for name, chunk in zip(module_names, chunks):
                    target = layer.proj[name]
                    if chunk.shape != target.weight.shape:
                        raise RuntimeError(f"LoKR shape mismatch at {k1} -> {name}")
                    loaded.append((target, chunk))
    del tensors
    if not loaded:
        raise RuntimeError("Realistic Snapshot LoKR tensors were not loaded")
    logger.info("Loaded %d LoKR matrices", len(loaded))
    return SnapshotLoKR(loaded)

class MossImageModel:
    def __init__(self):
        self.pipe = None
        self.realistic = None
        self.lora_layers = []

    def load(self):
        logger.info("Loading MOSS base model: %s", MODEL_REPO)
        pipeline = DiffusionPipeline.from_pretrained(
            MODEL_REPO,
            custom_pipeline="pipeline_moss_image",
            trust_remote_code=True,
            torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
        )
        pipeline.scheduler = type(pipeline.scheduler).from_config(pipeline.scheduler.config)
        config_path = hf_hub_download(LORA_REPO, "adapter_config.json")
        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)
        inject_adapter_in_model(
            LoraConfig(
                r=config["r"],
                lora_alpha=config["lora_alpha"],
                target_modules=config["target_modules"],
            ), pipeline.transformer, adapter_name="default",
        )
        weights_path = hf_hub_download(LORA_REPO, "pytorch_lora_weights.safetensors")
        state = {
            k.removeprefix("transformer."): v
            for k, v in load_file(weights_path, device="cpu").items()
        }
        _, unexpected = set_peft_model_state_dict(
            pipeline.transformer, state, adapter_name="default",
        )
        if unexpected:
            logger.warning("Unexpected LoRA keys (first 5): %s", unexpected[:5])
        del state

        self.realistic = load_snapshot(pipeline.transformer)
        pipeline.transformer.to(torch.bfloat16)
        pipeline.to("cuda")
        self.lora_layers = [layer for layer in pipeline.transformer.modules()
                            if isinstance(layer, LoraLayer)]
        self.pipe = pipeline
        logger.info("MOSS-Image model loaded")

    @torch.inference_mode()
    def generate(self, prompt, negative_prompt, width, height, steps,
                 guidance_scale, lora_scale, realistic_scale, add_trigger, seed):
        if self.pipe is None:
            raise RuntimeError("MOSS model not loaded")
        if add_trigger and not prompt.lower().startswith(TRIGGER):
            prompt = f"{TRIGGER}, {prompt}"
        for layer in self.lora_layers:
            layer.set_scale("default", float(lora_scale))
        self.realistic.set_scale(realistic_scale)
        image = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            width=snap(width), height=snap(height),
            num_inference_steps=max(1, min(50, int(steps))),
            guidance_scale=float(guidance_scale),
            uncond_skip_layers=[22],
            generator=torch.Generator("cpu").manual_seed(int(seed) % (MAX_SEED + 1)),
        ).images[0]
        return image
