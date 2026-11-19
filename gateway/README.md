# OpenAI-compatible Images API Gateway

This CPU-only FastAPI app provides a subset of OpenAI Images API. It calls the GPU Gradio Space by the caller's HF Token, with no owner-token fallback.

Set GPU_SPACE_ID to the namespace and Space name of your GPU worker.

Start locally:
```bash
pip install -r requirements.txt
export GPU_SPACE_ID=your-name/my-zerogpu-engine
uvicorn main:app --host 0.0.0.0 --port 8000
```

Endpoints: GET /health, GET /v1/models, POST /v1/images/generations, POST /v1/images/edits.

The generation endpoint accepts model, prompt, size, n (1-4), and returns OpenAI-style data[].b64_json. The edits endpoint accepts multipart image and prompt; only z-image-turbo supports it in this MVP.

Use Authorization: Bearer hf_... or X-HF-Token, or omit both to attempt anonymous use. No token is stored. When public, put the gateway behind TLS, authentication/rate limiting, and abuse protection. It is not an exact implementation of every OpenAI Images API option.
