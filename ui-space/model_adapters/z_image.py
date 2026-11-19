"""Adapter for our existing ZeroGPU engine (/infer)."""
from gradio_client import handle_file

def generate(client, prompt, width, height, seed, image_path=None, strength=0.6):
    result = client.predict(
        "z-image-turbo", prompt, width, height,
        handle_file(str(image_path)) if image_path else None,
        float(strength), int(seed), api_name="/infer",
    )
    if not isinstance(result, (list, tuple)) or len(result) < 2:
        raise RuntimeError("Z-Image Space 返回格式异常")
    return result[0], str(result[1])
