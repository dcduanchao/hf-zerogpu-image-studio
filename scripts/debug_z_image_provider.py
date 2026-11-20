"""Standalone local Z-Image-Turbo test using Hugging Face Inference Providers.

No dependency on either project Space or its Gradio endpoint.
Run from a separate venv:
    pip install -U huggingface_hub Pillow
    export HF_TOKEN=hf_your_token
    python scripts/debug_z_image_provider.py --prompt "a cat on the moon"

Inference Providers may charge separately from ZeroGPU. This script does not
modify gpu-space/models/z_image.py or deploy the Space.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import InferenceClient

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"


def generate(
    *,
    prompt: str,
    token: str,
    provider: str = "auto",
    width: int = 512,
    height: int = 512,
    seed: int = -1,
    timeout: float = 300,
):
    """Return a PIL image for local visual evaluation.

    Similar parameter names to ZImageModel.generate(), but this is a remote
    hosted model: seed/outputs and pipeline settings may not match Diffusers.
    """
    if not prompt.strip():
        raise ValueError("prompt 不能为空")
    client = InferenceClient(
        model=MODEL_ID,
        token=token,
        provider=provider,
        timeout=timeout,
    )
    options = {
        "prompt": prompt,
        "model": MODEL_ID,
        "width": width,
        "height": height,
    }
    if seed >= 0:
        options["seed"] = seed
    return client.text_to_image(**options)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Z-Image-Turbo Inference Providers debug")
    parser.add_argument("--prompt", default="一只戴着宇航员头盔的小猫站在月球上，电影感光影")
    parser.add_argument("--provider", default="auto", help="HF Inference Provider name, default auto")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=-1, help="-1 means provider-selected random seed")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", default="z_image_provider_output.png")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        parser.error("请先设置环境变量 HF_TOKEN=hf_...，不要把 Token 写入代码")
    if min(args.width, args.height) <= 0 or args.timeout <= 0:
        parser.error("width、height 和 timeout 必须大于 0")

    print(f"model={MODEL_ID} provider={args.provider} size={args.width}x{args.height} seed={args.seed}")
    started = time.perf_counter()
    try:
        image = generate(
            prompt=args.prompt,
            token=token,
            provider=args.provider,
            width=args.width,
            height=args.height,
            seed=args.seed,
            timeout=args.timeout,
        )
    except Exception as exc:
        # Do not print exception details that could contain authenticated URLs.
        print(
            f"推理失败：{type(exc).__name__}。请检查当前 Provider 是否支持该模型、"
            "HF Token 权限与 Inference Providers 额度，或尝试其他 provider。",
            file=sys.stderr,
        )
        return 1

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")
    print(f"SUCCESS elapsed={time.perf_counter()-started:.2f}s output={destination.resolve()}")
    print("注意：这里验证的是远程 Provider 推理，而非 ZeroGPU Diffusers 的内部参数/显存。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
