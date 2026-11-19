"""Create/update the GPU and UI Spaces without changing their hardware."""
import os
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

def required(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required configuration: {name}")
    return value

def main():
    api = HfApi(token=required("HF_TOKEN"))
    namespace = os.environ.get("HF_NAMESPACE", "").strip() or api.whoami()["name"]
    if "/" in namespace:
        raise ValueError("HF_NAMESPACE must be a user/organization name")
    gpu_name = os.environ.get("GPU_SPACE_NAME", "my-zerogpu-engine").strip()
    ui_name = os.environ.get("UI_SPACE_NAME", "my-image-studio").strip()
    if any("/" in v or not v for v in (gpu_name, ui_name)):
        raise ValueError("Space names must be nonempty and have no slash")
    gpu_id = f"{namespace}/{gpu_name}"
    ui_id = f"{namespace}/{ui_name}"
    for repo_id, directory, sdk in ((gpu_id, "gpu-space", "gradio"),
                                    (ui_id, "ui-space", "docker")):
        print(f"Creating/updating {repo_id} from {directory}")
        api.create_repo(repo_id=repo_id, repo_type="space", space_sdk=sdk,
                        private=False, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="space",
                          folder_path=str(ROOT / directory),
                          commit_message="Deploy from GitHub Actions")
    # Never touch existing ZeroGPU hardware settings.
    api.add_space_variable(repo_id=ui_id, key="GPU_SPACE_ID", value=gpu_id)
    print(f"GPU: https://huggingface.co/spaces/{gpu_id}")
    print(f"UI: https://huggingface.co/spaces/{ui_id}")
    print("One CPU Space + one ZeroGPU Space; all model plugins share the GPU Space.")

if __name__ == "__main__":
    main()
