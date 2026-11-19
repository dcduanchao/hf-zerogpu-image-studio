---
title: HF Image Studio
emoji: 🎨
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
short_description: Image UI with optional HF token, ZeroGPU backend
---

# HF Image Studio

Run on CPU Basic with Docker SDK. One Uvicorn process serves Gradio and the OpenAI-compatible /v1 images API on port 7860. Configure the GPU_SPACE_ID variable to username/my-zerogpu-engine. Token is optional: absent token means unauthenticated/shared anonymous quota. Present token uses the caller's identity. Auth failures never fall back to anonymous or the Space owner's token. Do not log user tokens.
