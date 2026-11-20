"""Only upload sd-cli into the preexisting MiniMax test Space, never production."""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: publish_minimax_sd_cli.py /path/to/sd-cli-blackwell.tar.gz")
    artifact = Path(sys.argv[1])
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    namespace = os.environ.get("HF_NAMESPACE", "").strip() or api.whoami()["name"]
    repo = f"{namespace}/minimax-h3-gguf-zerogpu-test"
    if namespace != "duan0320":
        raise ValueError("Refusing to publish to an unexpected namespace")
    # Repository must already exist; no CPU/dedicated hardware fallback.
    api.repo_info(repo_id=repo, repo_type="space")
    api.upload_file(
        path_or_fileobj=str(artifact),
        path_in_repo="assets/sd-cli-blackwell.tar.gz",
        repo_id=repo,
        repo_type="space",
        commit_message="Install CUDA sm_120 stable-diffusion.cpp binary for ZeroGPU test",
    )
    print("Published to https://huggingface.co/spaces/" + repo)
    print("The binary has not yet passed native CUDA execution or GGUF inference tests.")


if __name__ == "__main__":
    main()
