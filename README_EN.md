# Hugging Face ZeroGPU Image Studio

[中文](README.md) | [English](README_EN.md)

A multi-model image generation project built on **Hugging Face Spaces, Gradio, FastAPI, and ZeroGPU**. It uses exactly **two deployed Spaces**: a CPU Space for the web UI and OpenAI-compatible Images API, and one ZeroGPU Space containing model plugins. **Only the Space owner can switch the active model**; visitors can generate images only with the currently loaded model.

> Integrated models: `z-image-turbo` (text-to-image and img2img) and `moss-image` (text-to-image with LoRA). Only **one** model is loaded into the ZeroGPU process at startup. Switching requires updating a Space variable and restarting the GPU Space. Successful CI and HTTP smoke tests do **not** prove GPU inference works; test actual generation separately.

## 1. Live links

- **Web UI / OpenAI API (CPU Space)**: https://huggingface.co/spaces/duan0320/my-image-studio
- **ZeroGPU inference Space**: https://huggingface.co/spaces/duan0320/my-zerogpu-engine
- **API base URL**: `https://duan0320-my-image-studio.hf.space`

Replace `duan0320` with your own Hugging Face namespace for a custom deployment.

## 2. Architecture and source layout

```text
Browser / OpenAI-compatible client
                |
                v
my-image-studio (Docker, CPU Basic)
  Gradio UI + FastAPI /v1/images/*
  Model registry / adapters / active-model validation
  HF owner identity check for model switching
                |
                v
my-zerogpu-engine (Gradio, ZeroGPU)
  /active_model        currently loaded model
  /infer               Z-Image
  /infer_moss          MOSS
  models/z_image.py and models/moss_image.py
  Loads only ACTIVE_GPU_MODEL on startup
```

| Path | Responsibility |
| --- | --- |
| `gpu-space/app.py` | GPU startup, single-model inference endpoints, logging and model status |
| `gpu-space/models/` | Pluggable model implementations; model weights are downloaded from HF Hub |
| `ui-space/app.py` | Browser UI and owner-only switch controls |
| `ui-space/openai_api.py` | OpenAI Images-compatible API subset |
| `ui-space/model_registry.py` | Supported model IDs, dimensions and edit capabilities |
| `ui-space/model_router.py` / `model_adapters/` | Model-specific client dispatch |
| `ui-space/model_state.py` | Read the *actually loaded* GPU model; the UI rejects mismatches and the OpenAI API resolves to it |
| `ui-space/admin_switch.py` | Verify the HF owner's identity and change the GPU Space variable |
| `scripts/deploy_spaces.py` / `scripts/check_spaces.py` | Publish and inspect the two Spaces |
| `.github/workflows/deploy-spaces.yml` | CI checks and automatic deployment |

Keep `gpu-space/README.md` and `ui-space/README.md`: their YAML front matter is used by Hugging Face to configure the Space SDK and runtime. The root READMEs are the developer and user guides.

## 3. Supported models and access policy

| Model ID | Capabilities | Example sizes |
| --- | --- | --- |
| `z-image-turbo` | Text-to-image and image-to-image | `512x512`, `768x768`, `1024x1024`, `1024x768`, `768x1024` |
| `moss-image` | Text-to-image using MOSS + LoRA | `512x512`, `768x768`, `1024x1024`, `576x1024`, `864x1536` and others returned by `/v1/models` |

- Only the owner of the Hugging Face GPU Space can switch models. The admin control verifies the supplied HF Token with Hugging Face, requires the account name to match `HF_ADMIN_USERNAME`, and requires write permission on the GPU Space.
- Public visitors see only the currently loaded model in the UI, with a manual status-refresh action.
- The OpenAI API accepts an omitted, stale, or unknown `model` value and routes text-to-image to the **actually loaded GPU model**. The response includes the effective `model` and `size`. This does not change the active model; only the owner can do that. An unavailable backend returns **503 / `gpu_model_unavailable`**.
- Updating `ACTIVE_GPU_MODEL` triggers a Space restart; any in-flight inference may be interrupted. After restart, reload the current-model status in the UI.
- `GET /v1/models` enumerates **registered** models, not necessarily live ones. Use `GET /v1/models/active` for the actual loaded model.

## 4. Deployment

1. Sign up for Hugging Face and create an access token with appropriate create/write permissions on the target Spaces. Prefer narrowly scoped permissions.
2. Fork or clone this GitHub repository. Under **GitHub → Settings → Secrets and variables → Actions → Secrets**, create `HF_TOKEN` for CI deployment. **Never commit, log or paste a token into documentation or chat.**
3. Optional GitHub Actions **Variables**: `HF_NAMESPACE` (default: HF token owner's namespace), `GPU_SPACE_NAME` (default `my-zerogpu-engine`), `UI_SPACE_NAME` (default `my-image-studio`).
4. Launch **Actions → Deploy Hugging Face Spaces → Run workflow**. Pushes to `main` that change `gpu-space/**`, `ui-space/**` or deployment scripts also trigger automatic publishing.
5. The workflow publishes a **Gradio GPU Space** and a **Docker CPU Space**, then configures `GPU_SPACE_ID` and `HF_ADMIN_USERNAME` on the CPU Space. It does not create another Space for each model.
6. In **GPU Space → Settings → Hardware**, manually select **ZeroGPU** if eligible. Leave the UI Space on **CPU Basic**. The workflow does not change your hardware or billing settings automatically.
7. In **GPU Space → Settings → Variables**, use `ACTIVE_GPU_MODEL=z-image-turbo` or `moss-image`. Default: Z-Image. Only one model is initialized during startup.
8. Confirm both Spaces reach Running; check the UI, `/health` and `/v1/models/active`, then run a real image generation request for the active model. CI currently verifies syntax, mocked permissions, publication and CPU HTTP endpoints, not actual GPU output.

### Development standards and release workflow

- **Runtime**: Python 3.12 in the Hugging Face Spaces; the GitHub Actions lightweight tests use Python 3.11. Dependencies live in `gpu-space/requirements.txt` and `ui-space/requirements.txt`.
- Develop on branches, run tests and review changes before merging to `main`. Avoid editing deployed Space files directly: the next GitHub deployment overwrites them.
- To add a model: create a plugin under `gpu-space/models/`; register its single-model startup and `@spaces.GPU` entry point in `gpu-space/app.py`; update `ui-space/model_registry.py`, `model_adapters/` and `model_router.py`; add required packages, tests and documentation. Ensure `/active_model` returns a registered ID. **Do not create another Space.**
- Do not weaken the server-side owner check or fall back to CI's `HF_TOKEN` for visitor inference.
- Keep `/infer`, `/infer_moss`, `/active_model` and the CPU-facing API contracts stable, or update all adapters and tests together.
- Suggested pre-merge checks:

```bash
python -m compileall -q gpu-space ui-space scripts
python -m pip install 'huggingface_hub>=0.34,<2' 'gradio_client>=1.13,<3'
python scripts/test_admin_switch.py
python scripts/test_model_state.py
```

These tests use mocks and do not need GPU credits or change an HF Space. Validate model-library compatibility and GPU memory usage separately for changes involving Diffusers, the MOSS custom pipeline or LoRA loading.

### Run the CPU UI/API locally

```bash
cd ui-space
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GPU_SPACE_ID=duan0320/my-zerogpu-engine
export HF_ADMIN_USERNAME=duan0320
python app.py
# Open http://localhost:7860
```

On Windows PowerShell, use the appropriate venv activation command and `$env:GPU_SPACE_ID=...`. Your local server only proxies to the remote GPU Space; it does not load GPU models.

## 5. Browser usage

1. Open the [Image Studio](https://huggingface.co/spaces/duan0320/my-image-studio).
2. The page queries the GPU process's `/active_model` endpoint. It shows only the available model; image generation is disabled if GPU status cannot be verified.
3. Select text-to-image, or image editing if Z-Image is loaded. Enter a prompt and optionally upload an original image for editing.
4. Choose an available resolution, the number of images (1–4) and a seed (`-1` means random).
5. Optionally enter **your own visitor HF Token** for quota attribution; leave it blank to attempt anonymous, shared-allowance inference. Click Generate and inspect the results.
6. **Owner-only switching**: expand **“管理员 · 切换 GPU 模型（仅所有者）”**, select a model, supply the owner's write-enabled HF Token, and submit. When the GPU Space has restarted, click **“刷新当前模型状态”** to refresh the page. Other visitors cannot bypass this server-side authorization by calling Gradio directly.

The owner/admin token is privileged: use it only in a trusted UI. No privileged token is passed as an inference parameter; visitor calls never silently fall back to the deployment token.


### Check your personal ZeroGPU time remaining

In the **Hugging Face Token** panel, enter your own visitor token and select **“验证 Token / 查询 ZeroGPU 剩余时间”** (Verify Token / Check remaining ZeroGPU time). The UI shows the verified username, quota remaining, base quota, and reset time (UTC) as returned by Hugging Face. It calls the official read-only HTTPS endpoint `/api/spaces/zero-gpu/quota` directly, while retaining the `huggingface_hub<2` dependency required by Gradio 6.15.1. Fine-grained tokens require the **Billing → Read billing usage and payment method status** permission. When quota lookup fails, the app keeps the successful identity validation and clearly reports that quota is unavailable. Remaining GPU-seconds are *not* a guarantee of uninterrupted inference time. Anonymous users cannot query personal remaining time.

## 6. OpenAI-compatible Images API

The CPU Space provides an **OpenAI Images API subset**, not the complete official OpenAI API. Its base URL is:

`https://duan0320-my-image-studio.hf.space`

### Readiness and model discovery

```bash
BASE=https://duan0320-my-image-studio.hf.space
curl "$BASE/health"
curl "$BASE/v1/models"          # Registered models
curl "$BASE/v1/models/active"   # Actually loaded model
```

### Text-to-image generation

```bash
curl -sS "$BASE/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "z-image-turbo",
    "prompt": "a tiny astronaut cat on the moon, cinematic lighting",
    "size": "512x512",
    "n": 1,
    "seed": -1,
    "response_format": "url"
  }'
```

The `model` field is optional and advisory: the API always generates with the actual `/v1/models/active` model, even if the caller sends an old or unknown model ID. If the requested `size` is unsupported by that model, the API uses its default size. Read the effective `model` and `size` fields in the response. Editing still fails with `unsupported_edit` if the current model cannot edit images. URL responses contain a `created` timestamp, the effective `model` and `size`, and `data[].url` pointing to a temporary GPU Space image; such URLs can expire.

### Optional HF Token and Base64 response

Choose exactly **one** of `Authorization: Bearer hf_...` and `X-HF-Token: hf_...`. For example, place your visitor token in an environment variable:

```bash
export VISITOR_HF_TOKEN='hf_REPLACE_WITH_YOUR_TOKEN'
curl -sS "$BASE/v1/images/generations" \
  -H "Authorization: Bearer $VISITOR_HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "z-image-turbo",
    "prompt": "a watercolor landscape",
    "size": "512x512",
    "n": 1,
    "response_format": "b64_json"
  }'
```

The PNG image is returned in `data[0].b64_json`. Without authentication headers, the service attempts anonymous inference. Token-based calls use the visitor's identity, with no automatic owner-token fallback.

### Image editing — Z-Image only

```bash
curl -sS "$BASE/v1/images/edits" \
  -F "image=@input.png" \
  -F "prompt=turn the background into a garden" \
  -F "model=z-image-turbo" \
  -F "size=512x512" \
  -F "strength=0.6" \
  -F "response_format=url"
```

This endpoint only works when Z-Image is the active model. Uploads are limited to **12 MB** and **16 million pixels**; `strength` must be between `0.05` and `1.0`. `n` is 1–4. The generation JSON supports `prompt`, `model`, `size`, `n`, `seed` and `response_format`; editing uses multipart form fields.


### Experimental multi-image edits and public URLs

When `z-image-turbo` is active, `/v1/images/edits` accepts **1–4 images** total, using uploaded files and/or public HTTP(S) image URLs. Uploaded files are ordered first, followed by URLs in the given order. **The first image is primary** and the others are references. For more than one image, the CPU service builds a tiled *contact sheet* and passes it as **one** image-to-image input. This is **experimental collage-based editing**, not native multi-image model conditioning; subject identity or reference fidelity is not guaranteed.

```bash
curl -sS "$BASE/v1/images/edits" \
  -F "image=@main.png" \
  -F "image=@reference.png" \
  -F 'image_urls=["https://example.com/reference2.jpg"]' \
  -F "prompt=use the subject from image one with colors from references" \
  -F "size=1024x1024" \
  -F "strength=0.6"
```

In Postman use **Body → form-data**, with repeated `image` fields set to **File**. Specify `image_urls` as a JSON array string or repeat the `image_url` text field. Only publicly accessible HTTP(S) image URLs are allowed, with private addresses and redirects blocked. Maximum **12 MB / 16 million pixels per image**; PNG, JPEG, WebP and GIF are accepted. Responses provide the effective `model`, `size`, `input_image_count` and `edit_mode` (`single_image` or `experimental_reference_collage`). The UI supports multiple uploads and URLs, and the UI and `GET /v1/models` document per-model capabilities and usage notes.

The existing `moss-image` plugin remains **text-to-image only** and returns `unsupported_edit` for edits instead of silently ignoring reference images.

### Error handling

| HTTP | Error code | Meaning / action |
| --- | --- | --- |
| 400 | `model_not_found`, `invalid_size`, etc. | Invalid model ID, resolution or input |
| 401 | `invalid_token` / `hf_auth_failed` | Check your visitor HF Token |
| 400 | `unsupported_edit` | The active model does not support editing; contact the admin |
| 413 | `image_too_large` / `too_many_pixels` | Uploaded image exceeds limits |
| 429 | `zero_gpu_quota_exceeded` | Quota or GPU duration limit |
| 503 | `gpu_model_unavailable` | GPU restarting, building or unreachable |
| 502/504 | `upstream_error` / `upstream_timeout` | Check GPU logs, network and request timeout |

## 7. Troubleshooting, licensing and security

- **GPU OOM**: ensure only `ACTIVE_GPU_MODEL` is initialized. Loading Z-Image and MOSS together may exhaust ZeroGPU memory before inference starts. `torch.cuda.empty_cache()` cannot free weights still referenced by a model.
- **Old model shown after switching**: check `/v1/models/active` after the GPU Space finishes restarting, then refresh the UI.
- **503 or cold-start failure**: inspect the GPU Space Build and Runtime Logs. CI publication does not guarantee that model initialization or inference succeeded.
- **MOSS weights and LoRA**: external model/pipeline/adapter repositories may carry their own licenses and usage restrictions. Review those terms independently; weights are not committed to this project.
- **Public hosting**: add authentication, throttling, abuse protection and data-handling controls as needed. Visitor HF Tokens pass through the CPU server; do not persist or log them.
- **ZeroGPU availability and cost** depend on HF platform policy, your account and hardware quota. Do not assume all requests are free.

## 8. Repository scope

Only `my-image-studio` and `my-zerogpu-engine` are deployed by this repository. The original standalone `gateway/` and `moss-space/` code is obsolete and has been removed from the source tree. Removing repository code does **not** automatically delete any older Space that already exists in your Hugging Face account; review those resources separately.

中文文档：[README.md](README.md).
