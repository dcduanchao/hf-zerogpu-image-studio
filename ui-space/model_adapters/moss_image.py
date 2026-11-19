"""MOSS adapter for the unified ZeroGPU /infer_moss endpoint."""
import secrets

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, bad anatomy, deformed fingers, "
    "extra limbs, CGI, render, 3d, cartoon, airbrushed, plastic skin, "
    "oversaturated, watermark"
)

def generate(client, prompt, width, height, seed, image_path=None, strength=0.6):
    if image_path is not None:
        raise ValueError("MOSS-Image Space 当前仅支持文生图")
    actual_seed = int(seed) if int(seed) >= 0 else secrets.randbelow(2**31 - 1)
    result = client.predict(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=int(width),
        height=int(height),
        steps=30,
        guidance_scale=3.0,
        lora_scale=0.7,
        realistic_scale=1.0,
        add_trigger=True,
        seed=actual_seed,
        api_name="/infer_moss",
    )
    if not isinstance(result, (list, tuple)) or len(result) != 2:
        raise RuntimeError("MOSS 统一 GPU 接口响应格式异常")
    return result[0], str(result[1])
