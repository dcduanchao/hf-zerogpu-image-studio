"""Deploy only the MiniMax GGUF test ZeroGPU Space, never production Spaces."""
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE_FILES = ROOT / "experiments" / "minimax-h3-gguf-zerogpu"


def main():
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    owner = os.getenv("HF_NAMESPACE", "").strip() or api.whoami()["name"]
    if "/" in owner:
        raise ValueError("HF_NAMESPACE must be a namespace")
    repo = f"{owner}/minimax-h3-gguf-zerogpu-test"
    print("Deploying test-only ZeroGPU Space:", repo)

    # Never fall back to CPU or a paid dedicated GPU silently.
    try:
        api.create_repo(
            repo_id=repo,
            repo_type="space",
            space_sdk="gradio",
            space_hardware="zero-a10g",
            private=False,
            exist_ok=True,
        )
    except Exception as exc:
        print("ZeroGPU Space creation failed. No CPU/dedicated-GPU fallback.")
        print("Reason:", type(exc).__name__, str(exc)[:400])
        raise

    api.upload_folder(
        repo_id=repo,
        repo_type="space",
        folder_path=str(SPACE_FILES),
        commit_message="Deploy MiniMax H3 GGUF native CUDA ZeroGPU compatibility probe",
    )
    info = api.space_info(repo)
    hardware = (getattr(info, "runtime", None) or {}).get("hardware") if isinstance(getattr(info, "runtime", None), dict) else getattr(getattr(info, "runtime", None), "hardware", None)
    print("Actual Space hardware reported:", hardware)
    print("Space: https://huggingface.co/spaces/" + repo)
    print("The test does not load GGUF weights or claim image-generation readiness.")


if __name__ == "__main__":
    main()
