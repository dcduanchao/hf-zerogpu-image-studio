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

CPU Basic / Docker Space. One Uvicorn process exposes the Gradio UI and the OpenAI-compatible Images API on port 7860. Set `GPU_SPACE_ID` to the GPU Space and `HF_ADMIN_USERNAME` to its owner (the deployment workflow configures both). The public UI permits inference only with the currently loaded GPU model. Only the verified HF owner may update `ACTIVE_GPU_MODEL` using a write-enabled HF Token; switching restarts the GPU Space. Visitor tokens are optional and never fall back to the deployment token. See the repository README for complete documentation.
