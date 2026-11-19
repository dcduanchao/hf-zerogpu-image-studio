# Hugging Face ZeroGPU Image Studio

[中文](README.md) | [English](README_EN.md)

基于 **Hugging Face Spaces、Gradio、FastAPI 和 ZeroGPU** 的多模型图片生成项目。固定使用 **两个 Space**：一个 CPU Space 提供网页及 OpenAI 兼容接口，一个 ZeroGPU Space 运行模型插件。**管理员决定当前加载的模型**，普通用户只能使用已加载模型。

> 当前集成：`z-image-turbo`（文生图、图生图）与 `moss-image`（文生图，带 LoRA）。在 ZeroGPU 上一次只加载一个模型；切换需修改 GPU Space 变量并重启。代码发布、语法检查和 API 冒烟测试不代表真实 GPU 推理已经验证。

## 1. 在线入口

- **网页 / OpenAI API（CPU）**：https://huggingface.co/spaces/duan0320/my-image-studio
- **GPU 推理服务**：https://huggingface.co/spaces/duan0320/my-zerogpu-engine
- **API 基地址**：`https://duan0320-my-image-studio.hf.space`

将以上 `duan0320` 替换为自己的 Hugging Face 用户名即可复用部署流程。

## 2. 架构及目录

```text
浏览器 / OpenAI API 客户端
             |
             v
my-image-studio (CPU Basic, Docker)
  - Gradio 网页
  - FastAPI /health、/v1/models、/v1/models/active
  - POST /v1/images/generations、/v1/images/edits
  - 模型注册、路由、活动模型校验
  - HF 所有者验证、管理员模型切换
             |
             v
my-zerogpu-engine (ZeroGPU, Gradio)
  - /active_model  当前实际加载模型
  - /infer         Z-Image
  - /infer_moss    MOSS
  - models/z_image.py、models/moss_image.py
  - 启动时只实例化 ACTIVE_GPU_MODEL 指定的模型
```

| 路径 | 职责 |
| --- | --- |
| `gpu-space/app.py` | ZeroGPU 推理入口、单模型加载、状态接口与日志 |
| `gpu-space/models/` | 模型插件；模型权重从 Hugging Face Hub 加载 |
| `ui-space/app.py` | Gradio 网页和管理员操作区 |
| `ui-space/openai_api.py` | OpenAI Images 兼容子集 |
| `ui-space/model_registry.py` | 支持的模型、尺寸、编辑能力 |
| `ui-space/model_router.py`、`model_adapters/` | 模型请求分发 |
| `ui-space/model_state.py` | 查询 GPU 运行进程中的实际活动模型，核对活动模型，网页拒绝不匹配请求；OpenAI API 自动改用活动模型 |
| `ui-space/admin_switch.py` | 校验 HF 账号所有权后更新 GPU Space 变量 |
| `scripts/deploy_spaces.py`、`scripts/check_spaces.py` | 部署及只读状态检查 |
| `.github/workflows/deploy-spaces.yml` | 自动检查及发布两个 Space |

`gpu-space/README.md` 和 `ui-space/README.md` 的 YAML 头信息用于 Hugging Face Space 配置，**不能删除**。根目录的两份 README 是项目使用手册。

## 3. 功能与约束

| 模型 ID | 功能 | 常用尺寸 |
| --- | --- | --- |
| `z-image-turbo` | 文生图、图片编辑（img2img） | `512x512`、`768x768`、`1024x1024`、`1024x768`、`768x1024` |
| `moss-image` | 文生图；自定义 MOSS Pipeline + LoRA | `512x512`、`768x768`、`1024x1024`、`576x1024`、`864x1536` 等，详见 `/v1/models` |

- **只有 HF Space 所有者**能够切换活动模型。管理员必须在页面提供与 `HF_ADMIN_USERNAME` 相同账号的有效 HF Token，且该 Token 对 GPU Space 有写权限。
- 普通用户的模型下拉框只展示当前加载的模型；页面可手动刷新状态。
- OpenAI API 的 `model` 是可选提示：无论填写过时、不存在的模型 ID，还是省略该字段，服务端均使用**当前实际激活模型**执行文生图，并在响应中返回 `model` 和最终 `size`。页面和管理员切换不受此规则影响；状态暂不可用时返回 **503 `gpu_model_unavailable`**。
- 管理员更改 `ACTIVE_GPU_MODEL` 后，GPU Space 需要重启，生成中的任务可能中断。切换完成后点“刷新当前模型状态”。
- `GET /v1/models` 列出**已注册模型**，并不表示所有模型同时加载；以 `GET /v1/models/active` 为准。

## 4. 部署前准备

1. 创建 Hugging Face 账号；准备一个对目标 Spaces 拥有创建/写入权限的 HF Access Token。尽量使用最小必要权限。
2. Fork 或克隆本仓库，并在 GitHub 仓库 **Settings → Secrets and variables → Actions → Secrets** 新增 `HF_TOKEN`（仅 CI/CD 发布使用）。**不得将 Token 写入 Git、README、日志、前端代码或提交到聊天中。**
3. 可选的 GitHub Actions **Variables**：`HF_NAMESPACE`（默认从部署 Token 所有者推断）、`GPU_SPACE_NAME`（默认 `my-zerogpu-engine`）、`UI_SPACE_NAME`（默认 `my-image-studio`）。
4. 到 **Actions → Deploy Hugging Face Spaces → Run workflow** 手动运行；以后修改 `gpu-space/**`、`ui-space/**` 或部署脚本并推送 `main` 会自动运行。
5. 工作流创建/上传 **GPU Gradio Space** 和 **CPU Docker Space**，并给 CPU Space 设置 `GPU_SPACE_ID`、`HF_ADMIN_USERNAME`。工作流不会自动为每个模型创建新 Space。
6. 进入 GPU Space **Settings → Hardware**，手动选择符合账号资格的 **ZeroGPU**；CPU Space 使用 **CPU Basic**。工作流不会擅自修改硬件/计费。
7. GPU Space 的 **Settings → Variables** 可设置 `ACTIVE_GPU_MODEL=z-image-turbo` 或 `moss-image`（未设置时默认 Z-Image）；每次只加载一个。
8. 等待两个 Space 显示 Running，并检查页面、`/health`、`/v1/models/active`，最后分别验证当前活动模型的一次真实生图。CI 仅检查语法、权限测试、发布和 CPU API 连接，不测试 GPU 生图。

### 本地开发与发布规范

- **Python 环境**：Space 使用 Python 3.12；CI 语法/轻量测试使用 Python 3.11。GPU 环境依赖位于 `gpu-space/requirements.txt`，CPU 环境依赖位于 `ui-space/requirements.txt`。
- 功能开发在分支完成，先自检、审查再合并到 `main`。避免直接在 Hugging Face Space 修改代码：下一次 GitHub Actions 发布会覆盖其内容。
- 新增模型时：在 `gpu-space/models/` 添加模型类；在 `gpu-space/app.py` 注册单模型加载逻辑及带 `@spaces.GPU` 的推理入口；更新 `ui-space/model_registry.py`、`model_adapters/`、`model_router.py`，以及依赖、测试和文档。保证 `/active_model` 返回已注册的模型 ID；**不新增 Space**。
- 保持模型切换只接受经过服务端验证的所有者；不要把部署 `HF_TOKEN` 当作普通用户推理的默认 Token。
- 保持 API 稳定：`/infer`、`/infer_moss`、`/active_model` 和 CPU API；改动参数时同步更新 adapters/测试。
- 合并前执行：

```bash
python -m compileall -q gpu-space ui-space scripts
python -m pip install 'huggingface_hub>=0.34,<2,<3' 'gradio_client>=1.13,<3'
python scripts/test_admin_switch.py
python scripts/test_model_state.py
```

上述单元测试使用模拟调用，不需要 GPU，也不会实际修改 Hugging Face Space。运行端到端 GPU 测试需要 ZeroGPU Space 和实际配额。若修改 Hugging Face SDK 依赖或 MOSS 自定义 Pipeline，必须额外核对版本兼容与内存峰值。

### 本地启动 CPU UI / API

```bash
cd ui-space
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GPU_SPACE_ID=duan0320/my-zerogpu-engine
export HF_ADMIN_USERNAME=duan0320
python app.py
# 浏览器打开 http://localhost:7860
```

在 Windows PowerShell 中使用相应的虚拟环境激活方式，并用 `$env:GPU_SPACE_ID=...` 设置环境变量。GPU 模型仍在远程 Space 上运行；本地 CPU 服务只负责 UI/API 转发。

## 5. 页面操作教程

1. 打开 [AI 图片工作室](https://huggingface.co/spaces/duan0320/my-image-studio)。
2. 页面会读取 GPU Space 的 `/active_model`，显示**当前唯一可用模型**；若正在重启，生成按钮会暂时不可用。
3. 选择文生图或（仅 Z-Image 支持的）图片编辑，输入提示词；编辑时上传图片。
4. 选择当前模型支持的尺寸、生成数量（1～4）、随机种子（`-1` 表示随机）。可选填**访问者自己的** Hugging Face Token；留空尝试匿名共享额度。
5. 点击“开始生成”，在结果区查看图片；配额、GPU 任务时长和模型冷启动都可能影响响应。
6. 管理员切换：展开 **“管理员 · 切换 GPU 模型（仅所有者）”**，选择新模型并输入**所有者自己的、有 GPU Space 写权限的 HF Token**，点击验证切换。完成重启后点击 **“刷新当前模型状态”**。其他用户即使尝试调用管理员回调，也会被服务端拒绝。

> 管理员 Token 是高权限凭据，仅在信任的站点输入。GPU Space 不接受“管理员 Token”作为模型推理参数。访客 Token 不会替代部署 Token，且身份错误时不会自动回退匿名。


### 查看访问者的 ZeroGPU 剩余额度

在页面的 **Hugging Face Token** 区域输入你自己的 Token，点击 **“验证 Token / 查询 ZeroGPU 剩余时间”**。页面会分别显示账号身份、官方返回的剩余 GPU 时间、基础额度和重置时间（以 UTC 显示）。配额查询使用 Hugging Face 官方只读 HTTPS 接口 `/api/spaces/zero-gpu/quota`，保持与 Gradio 6.15.1 所需 `huggingface_hub<2` 的兼容性。Fine-grained Token 需要 **Billing → Read billing usage and payment method status** 权限；查询失败时仍然可显示已验证的用户名，不会把配额查询错误误判为无效 Token。剩余时间是 GPU 配额秒数，不等于一次任务能连续运行的时长。匿名模式无法查询个人剩余额度。

## 6. OpenAI Images API 教程

CPU Space 同一端口提供 FastAPI；这是 OpenAI **Images API 的兼容子集**，不是完整的 OpenAI API 或官方 SDK 全功能实现。

### 查询状态和模型

```bash
BASE=https://duan0320-my-image-studio.hf.space
curl "$BASE/health"
curl "$BASE/v1/models"          # 所有已注册模型
curl "$BASE/v1/models/active"   # 当前实际加载模型
```

### 文生图（URL）

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

`model` 可以省略或传入旧模型 ID；API 会自动改用 `/v1/models/active` 返回的当前激活模型。若传入的尺寸不被当前模型支持，使用当前模型的默认尺寸，并在响应 `size` 中说明。若当前模型不支持图片编辑，则编辑请求仍返回 `unsupported_edit`。URL 响应形如 `{"created": 1234567890, "model":"moss-image", "size":"1024x1024", "data":[{"url":"https://..."}]}`，上游临时图片 URL 可能过期。

### 使用自己的 HF Token / Base64

可选认证方式**二选一**：`Authorization: Bearer hf_...` 或 `X-HF-Token: hf_...`。以下示例使用环境变量，避免把 Token 直接写在命令里：

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

结果在 `data[0].b64_json`。省略认证头时尝试匿名 ZeroGPU 配额；有效 Token 的请求按调用者身份处理，不会回退使用 Space 所有者 Token。

### 图片编辑（仅 Z-Image）

```bash
curl -sS "$BASE/v1/images/edits" \
  -F "image=@input.png" \
  -F "prompt=turn the background into a garden" \
  -F "model=z-image-turbo" \
  -F "size=512x512" \
  -F "strength=0.6" \
  -F "response_format=url"
```

仅在当前加载 `z-image-turbo` 时可用。上传图片上限 **12 MB**，最大 **1600 万像素**；`strength` 范围 `0.05`～`1.0`。请求数 `n` 为 1～4。JSON 文生图支持 `prompt`、`model`、`size`、`n`、`seed`、`response_format`；编辑使用 `multipart/form-data`。

### 错误码和运行限制

| HTTP | 代码 | 含义 / 处理 |
| --- | --- | --- |
| 400 | `model_not_found`、`invalid_size` 等 | 模型 ID、尺寸或参数非法 |
| 401 | `invalid_token` / `hf_auth_failed` | 检查访问者 HF Token |
| 400 | `unsupported_edit` | 当前模型不支持图片编辑，联系管理员切换 |
| 413 | `image_too_large` / `too_many_pixels` | 编辑上传过大 |
| 429 | `zero_gpu_quota_exceeded` | ZeroGPU 配额或任务时间受限 |
| 503 | `gpu_model_unavailable` | GPU 正在启动、构建、切换或暂不可用 |
| 502/504 | `upstream_error` / `upstream_timeout` | 检查 GPU Space Logs、网络和超时 |

## 7. 故障排查与安全说明

- **显存 OOM**：确认 GPU Space 仅加载 `ACTIVE_GPU_MODEL` 一个模型，切勿在启动时同时实例化 Z-Image 和 MOSS。ZeroGPU 权重搬运可能在进入推理函数前 OOM，单纯 `torch.cuda.empty_cache()` 无法解决。
- **切换后 UI 仍旧模型**：确认 GPU Space 重启完成，使用 `/v1/models/active` 核对，再在页面点击刷新。
- **显示 503**：先检查 GPU Space 的 Build/Runtime Logs 及 CPU Space 状态；GitHub Actions 发布成功不保证模型成功初始化。
- **MOSS 依赖**：MOSS 自定义 Pipeline、基础权重和 LoRA 可能有独立的许可与使用条件，使用前自行检查各 Hub 仓库。模型权重不纳入本 Git 仓库。
- **公开服务**：生产环境需要考虑身份认证、限流、滥用防护和图片数据保护；访问者 HF Token 会经过 CPU 服务，请避免记录或持久化。
- **计费与配额**：ZeroGPU 可用性、时长和额度受 HF 账号与平台政策限制，切勿假设“免费”或不会产生额外成本。

## 8. 项目状态

本仓库只部署 `my-image-studio` 与 `my-zerogpu-engine`。早期的 `gateway/` 与独立 `moss-space/` 实现已退出当前架构。删除旧代码**不会自动删除 Hugging Face 账号中既有的旧 Space**，线上资源如需清理应单独核对并操作。

English documentation: [README_EN.md](README_EN.md).
