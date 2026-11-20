FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py openai_api.py model_registry.py model_router.py model_state.py admin_switch.py token_quota.py input_images.py ./
COPY model_adapters/ ./model_adapters/

EXPOSE 7860

# One ASGI server for both the Gradio UI and /v1 Images API.
CMD ["uvicorn", "app:application", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
