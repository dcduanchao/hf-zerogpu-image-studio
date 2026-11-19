---
title: HF Image Studio
emoji: 🎨
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.15.1
app_file: app.py
python_version: 3.12
short_description: Image UI with optional HF token, ZeroGPU backend
---

# HF Image Studio

Run on CPU Basic. Configure the GPU_SPACE_ID variable to username/my-zerogpu-engine. Token is optional: absent token means unauthenticated/shared anonymous quota. Present token uses the caller's identity. Auth failures never fall back to anonymous or the Space owner's token. Do not log user tokens.
