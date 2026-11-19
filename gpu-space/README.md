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

Deploy as Gradio Space and select ZeroGPU hardware in Settings > Hardware. The /infer endpoint is called by the UI or gateway. Supports one-ZeroGPU-Space multi-model plugins: z-image-turbo and moss-image. Each decorated endpoint returns one image. MOSS is loaded with ENABLE_MOSS=1 (default) and has a separate /infer_moss endpoint; Z retains /infer. Model weights may require substantial CPU RAM. Usage attribution is based on the caller's HF authentication, not a model input token.
