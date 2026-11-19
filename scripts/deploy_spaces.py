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
    moss_name = os.environ.get("MOSS_SPACE_NAME", "my-moss-image-engine").strip()
    if any("/" in v or not v for v in (gpu_name, ui_name, moss_name)):
        raise ValueError("Space names must be nonempty and have no slash")
    gpu_id = f"{namespace}/{gpu_name}"
    ui_id = f"{namespace}/{ui_name}"
    moss_id = f"{namespace}/{moss_name}"
    for repo_id, directory, sdk in ((gpu_id, "gpu-space", "gradio"),
                                    (ui_id, "ui-space", "docker"),
                                    (moss_id, "moss-space", "gradio")):
        print(f"Creating/updating {repo_id} from {directory}")
        api.create_repo(repo_id=repo_id, repo_type="space", space_sdk=sdk,
                        private=False, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="space",
                          folder_path=str(ROOT / directory),
                          commit_message="Deploy from GitHub Actions")
    # Only request ZeroGPU on the new MOSS Space, never modify existing hardware.\n    try:\n        details = api.space_info(moss_id, expand=["runtime"])\n        current = getattr(getattr(details, "runtime", None), "hardware", None)\n        if current != "zero-a10g":\n            api.request_space_hardware(repo_id=moss_id, hardware="zero-a10g")\n            print("Requested ZeroGPU hardware for new MOSS Space")\n    except Exception as exc:\n        print(f"MOSS hardware auto-selection unavailable: {type(exc).__name__}; select ZeroGPU manually")\n    api.add_space_variable(repo_id=ui_id, key="GPU_SPACE_ID", value=gpu_id)
    api.add_space_variable(repo_id=ui_id, key="MOSS_SPACE_ID", value=moss_id)
    print(f"GPU: https://huggingface.co/spaces/{gpu_id}")
    print(f"UI: https://huggingface.co/spaces/{ui_id}")
    print(f"MOSS: https://huggingface.co/spaces/{moss_id}")
    print("Select ZeroGPU for the new MOSS Space manually in Settings > Hardware.")

if __name__ == "__main__":
    main()
