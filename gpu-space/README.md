---
title: ZeroGPU Image Engine
emoji: 🖼️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.15.1
app_file: app.py
python_version: 3.12
short_description: ZeroGPU image generation and img2img endpoint
---

# ZeroGPU Image Engine

Shared ZeroGPU Space. Only one plugin is loaded at process startup via `ACTIVE_GPU_MODEL` (`z-image-turbo` by default, or `moss-image`). The CPU Studio calls `/infer` (Z-Image) and `/infer_moss` (MOSS), and reads `/active_model` to verify the actual loaded model. Inactive-model requests fail rather than loading both models. Configure ZeroGPU hardware manually. Model weights are downloaded from their upstream Hub repositories. See the repository README for deployment, security and API usage.
