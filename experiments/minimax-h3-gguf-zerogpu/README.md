---
title: GGUF ZeroGPU Lab
emoji: 🧪
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.15.1
app_file: app.py
python_version: 3.12
short_description: Qwen GGUF and MiniMax ZeroGPU tests
models:
  - Hearthscale/MiniMax-H3-Image-GGUF
  - KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF
---

# MiniMax H3 GGUF ZeroGPU Compatibility Probe

**This is a ZeroGPU experimental Space.** It retains the MiniMax CUDA probes and adds a Qwen-Image-2.1 GGUF image-generation test. Inference has not been validated.

- Probe primary Torch CUDA process inside `@spaces.GPU(duration=60, size="large")`.
- Probe `nvidia-smi` and fresh Torch CUDA child processes from inside the same task.
- Report diagnostic data without sensitive tokens or full environment dumps.
- No weights downloaded, no official MiniMax API, no dedicated GPU.
- **Not** proof that stable-diffusion.cpp works: native compiled backend still requires an integration test.

This is deliberately separate from the production
`duan0320/my-zerogpu-engine`, whose loaded model and settings stay unchanged.
