---
title: MOSS Image Engine
emoji: 📸
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
python_version: "3.12"
short_description: Independent MOSS-Image SFT with LoRA text-to-image engine
---

# MOSS-Image Engine

Standalone, self-hosted ZeroGPU inference using:
- `OpenMOSS-Team/MOSS-Image-SFT`
- `axiomofmind/Moss-image-nsfw-uncensored-lora` (PEFT LoRA)
- `asrarkhann112/moss-realistic-snapshot-lora` (LoKR, `MOSSRealisticSnapshotV1.safetensors`)

Adapted independently from the publicly available Apache-2.0-licensed [source Space](https://huggingface.co/spaces/asrarkhann112/moss-image-nsfw-lora). Review each model/LoRA repository's license separately before production use. Model weights are downloaded from original repositories; they are not included in this GitHub repo.

**Hardware:** In HF Space Settings, manually select **ZeroGPU** for this Space. The GitHub deployment does not modify hardware. MOSS model initialization and first inference are resource-intensive; successful build does not guarantee inference fits ZeroGPU constraints.

**API:** Gradio `/generate`, input signature mirrors the source Space. No HF token is accepted as a prediction parameter. Caller credentials are transmitted by the Gradio client only.
