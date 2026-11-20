---
title: MiniMax H3 GGUF ZeroGPU Probe
emoji: 🧪
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.15.1
app_file: app.py
python_version: 3.12
short_description: MiniMax H3 GGUF ZeroGPU CUDA test
models:
  - Hearthscale/MiniMax-H3-Image-GGUF
---

# MiniMax H3 GGUF ZeroGPU Compatibility Probe

**This is a test-only ZeroGPU Space, not a live image-generation demo yet.**

- Probe primary Torch CUDA process inside `@spaces.GPU(duration=60, size="large")`.
- Probe `nvidia-smi` and fresh Torch CUDA child processes from inside the same task.
- Report diagnostic data without sensitive tokens or full environment dumps.
- No weights downloaded, no official MiniMax API, no dedicated GPU.
- **Not** proof that stable-diffusion.cpp works: native compiled backend still requires an integration test.

This is deliberately separate from the production
`duan0320/my-zerogpu-engine`, whose loaded model and settings stay unchanged.
