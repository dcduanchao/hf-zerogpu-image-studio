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

Deploy as Gradio Space and select ZeroGPU hardware in Settings > Hardware. The /infer endpoint is called by the UI or gateway. Each GPU request returns one image. Usage attribution is based on the caller's HF authentication, not a model input token.
