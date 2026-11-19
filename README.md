# HF ZeroGPU Image Studio

A Hugging Face ZeroGPU image-generation project with a Gradio GPU engine, CPU UI, and an OpenAI-compatible API gateway.

## Layout

- `gpu-space/` — image generation and img2img inference on ZeroGPU
- `ui-space/` — Gradio user interface with optional visitor HF token
- `gateway/` — FastAPI OpenAI-style images API
- `.github/workflows/` — GitHub Actions deployment

## Deployment

1. Add `HF_TOKEN` in GitHub Actions repository secrets, with Hugging Face Space write permission.
2. Optionally set Actions variable `HF_NAMESPACE` to your HF username; by default it is inferred from token.
3. Run the `Deploy Hugging Face Spaces` workflow.
4. In the GPU Space settings, manually select **ZeroGPU** hardware; do not turn on paid GPU.

Never store visitor HF tokens in a repository, or fall back to the owner's token when user quota is depleted.
