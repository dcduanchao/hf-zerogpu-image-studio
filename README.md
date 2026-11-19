# Hugging Face ZeroGPU Image Studio

A multi-model image generation and editing demo with a **Gradio ZeroGPU worker**, separate **CPU Basic UI**, and an optional **FastAPI OpenAI Images-compatible gateway**.

## Components

| Directory | Purpose | Deployment |
| --- | --- | --- |
| `gpu-space/` | Z-Image-Turbo generation and image-to-image; optional SDXL-Turbo | Hugging Face Gradio Space with ZeroGPU hardware |
| `ui-space/` | Chinese-language generation and editing UI, optional per-user HF token | Hugging Face CPU Basic Gradio Space |
| `gateway/` | OpenAI-compatible `/v1/images/generations` and `/v1/images/edits` subset | Separate CPU host/container |
| `scripts/deploy_spaces.py` | Creates or updates GPU/UI Spaces | GitHub Actions |
| `.github/workflows/deploy-spaces.yml` | Automatic deployment workflow | GitHub Actions |

## Deploy both Spaces using GitHub Actions

1. Go to **Settings → Secrets and variables → Actions → Secrets → New repository secret** in this GitHub repository.
2. Enter `HF_TOKEN` as the name. Generate a token with suitable Space write/create permissions at <https://huggingface.co/settings/tokens>. **Never paste tokens in the repository or chat.**
3. If the HF user account differs from the GitHub username, you may set Actions **Variables** `HF_NAMESPACE` to your Hugging Face user name. Otherwise it is inferred from the token.
4. Go to **Actions → Deploy Hugging Face Spaces → Run workflow**. Follow the workflow logs until both repositories are created and files uploaded.
5. Go to your new **GPU Space → Settings → Hardware** and manually select **ZeroGPU (large)**. The script does not enable paid hardware or change billing.
6. Leave the UI Space on **CPU Basic**. The script sets its `GPU_SPACE_ID` Space variable automatically.
7. Test a small 512×512 image in the GPU Space, then the UI Space. To enable the optional SDXL model, set the GPU Space variable `ENABLE_SDXL=1` and restart/rebuild; it is off by default to reduce memory requirements.

Expected Spaces (assuming `HF_NAMESPACE` is your HF username):
- `https://huggingface.co/spaces/<HF_NAMESPACE>/my-zerogpu-engine`
- `https://huggingface.co/spaces/<HF_NAMESPACE>/my-image-studio`

## Authentication and quota

The public UI accepts an optional **visitor's HF token**. When provided, it is sent as client authentication to the GPU Space; when absent, the UI tries anonymous access. The project never uses the deploy `HF_TOKEN` as a fallback for inference. Authentication errors or quota exhaustion are returned as errors; anonymous quotas can be shared/rate-limited.

## Run the gateway separately

```bash
cd gateway
pip install -r requirements.txt
export GPU_SPACE_ID=your-hf-username/my-zerogpu-engine
uvicorn main:app --host 0.0.0.0 --port 8000
```

- `GET /health`
- `GET /v1/models`
- `POST /v1/images/generations` with JSON `model`, `prompt`, `size`, `n`
- `POST /v1/images/edits` with image multipart upload

Returns `data[].url` by default, pointing to a temporary GPU Space image file. Use `response_format=b64_json` to request Base64 instead. This is an OpenAI-compatible subset, **not complete OpenAI Images API compatibility**.

## Limitations and safety

- GPU inference and ZeroGPU queue behavior must be validated in your actual HF Space after deployment. Cold starts and GPU duration can impact results.
- Only Z-Image-Turbo img2img is enabled in this MVP; complex instruction editing may not faithfully preserve subject identity.
- Public gateways should be protected with TLS, rate limits, upload constraints and abuse controls.
- Visitor tokens pass through your CPU UI server process. Do not log requests, capture credentials, or store tokens.
- Your Hugging Face subscription and any prepaid credit settings determine possible charges; no assumption that GPU use always stops automatically at quota.
