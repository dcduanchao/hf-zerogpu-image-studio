"""Z-Image Turbo implementation within the shared ZeroGPU Space."""
import os
import torch
from diffusers import ZImagePipeline, ZImageImg2ImgPipeline

class ZImageModel:
    def __init__(self):
        self.txt = None
        self.edit = None

    def load(self):
        self.txt = ZImagePipeline.from_pretrained(
            os.getenv("Z_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo"), dtype=torch.bfloat16
        )
        self.txt.to("cuda")
        self.edit = ZImageImg2ImgPipeline.from_pipe(self.txt)

    @torch.inference_mode()
    def generate(self, prompt, width, height, seed, source_image=None, strength=0.6):
        options = dict(
            prompt=prompt,
            num_inference_steps=8,
            guidance_scale=0.0,
            generator=torch.Generator(device="cuda").manual_seed(int(seed)),
        )
        if source_image is None:
            return self.txt(width=width, height=height, **options).images[0]
        return self.edit(
            image=source_image.convert("RGB").resize((width, height)),
            strength=float(strength), **options,
        ).images[0]
