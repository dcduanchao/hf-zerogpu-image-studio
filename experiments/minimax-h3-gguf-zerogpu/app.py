"""ZeroGPU compatibility gate for MiniMax H3 GGUF / native CUDA subprocesses.

No model weights are downloaded. Diagnose native-CUDA process visibility
before attempting the 25+ GB GGUF inference stack.
"""
import json
import os
import subprocess
import sys
import time

import spaces  # Must precede torch on ZeroGPU
import gradio as gr


MODEL_ID = "Hearthscale/MiniMax-H3-Image-GGUF"


@spaces.GPU(duration=60, size="large")
def run_probe():
    import torch
    started = time.monotonic()
    report = {
        "model_to_test": MODEL_ID,
        "zero_gpu": os.environ.get("SPACES_ZERO_GPU", ""),
        "gpu_phase": "inside @spaces.GPU(duration=60, size=large)",
        "main_process": {},
        "native_subprocess": {},
    }
    try:
        report["main_process"]["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["main_process"]["device"] = torch.cuda.get_device_name(0)
            x = torch.ones((128, 128), device="cuda", dtype=torch.float32)
            report["main_process"]["test_sum"] = float(x.sum().item())
            del x
    except Exception as exc:
        report["main_process"]["error"] = type(exc).__name__ + ": " + str(exc)[:300]

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=12, check=False,
        )
        report["native_subprocess"]["nvidia_smi_exit"] = result.returncode
        report["native_subprocess"]["nvidia_smi_output"] = (result.stdout or result.stderr)[:500]
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["native_subprocess"]["nvidia_smi_error"] = type(exc).__name__

    # Spawning a new Python child exercises a fresh CUDA runtime process,
    # as does stable-diffusion.cpp. The test is intentionally non-mutating.
    script = (
        "import torch, json; "
        "d={'cuda_available':torch.cuda.is_available()}; "
        "d['device']=torch.cuda.get_device_name(0) if d['cuda_available'] else None; "
        "print(json.dumps(d))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=25, check=False,
            env={k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")},
        )
        report["native_subprocess"]["child_torch_exit"] = result.returncode
        report["native_subprocess"]["child_torch_output"] = (
            result.stdout.strip() if result.returncode == 0 else result.stderr[-700:]
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["native_subprocess"]["child_torch_error"] = type(exc).__name__

    report["wall_seconds"] = round(time.monotonic() - started, 2)
    report["interpretation"] = (
        "Main GPU + subprocess results are diagnostic only. Even if the child "
        "sees CUDA, a compiled stable-diffusion.cpp binary and real GGUF "
        "inference must still be verified before enabling image generation."
    )
    return json.dumps(report, ensure_ascii=False, indent=2)



# This C program is compiled with a normal C compiler, using dlopen/dlsym.
# It tests the native CUDA Driver API without depending on nvcc or CUDA headers.
NATIVE_C = r"""
#include <stdio.h>
#include <dlfcn.h>

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef unsigned long long CUdeviceptr;

int main(void) {
    void *lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "dlopen libcuda.so.1 failed: %s\n", dlerror());
        return 10;
    }
    CUresult (*cuInit)(unsigned int) = dlsym(lib, "cuInit");
    CUresult (*cuDeviceGet)(CUdevice*, int) = dlsym(lib, "cuDeviceGet");
    CUresult (*cuCtxCreate)(CUcontext*, unsigned int, CUdevice) = dlsym(lib, "cuCtxCreate_v2");
    CUresult (*cuCtxDestroy)(CUcontext) = dlsym(lib, "cuCtxDestroy_v2");
    CUresult (*cuMemAlloc)(CUdeviceptr*, size_t) = dlsym(lib, "cuMemAlloc_v2");
    CUresult (*cuMemFree)(CUdeviceptr) = dlsym(lib, "cuMemFree_v2");
    if (!cuInit || !cuDeviceGet || !cuCtxCreate || !cuCtxDestroy || !cuMemAlloc || !cuMemFree) {
        fprintf(stderr, "required CUDA driver symbols missing\n");
        return 11;
    }
    CUdevice dev = 0;
    CUcontext ctx = NULL;
    CUdeviceptr memory = 0;
    CUresult code;
    code = cuInit(0);
    if (code != 0) { fprintf(stderr, "cuInit=%d\n", code); return 12; }
    code = cuDeviceGet(&dev, 0);
    if (code != 0) { fprintf(stderr, "cuDeviceGet=%d\n", code); return 13; }
    code = cuCtxCreate(&ctx, 0, dev);
    if (code != 0) { fprintf(stderr, "cuCtxCreate=%d\n", code); return 14; }
    code = cuMemAlloc(&memory, 4096);
    if (code != 0) {
        fprintf(stderr, "cuMemAlloc=%d\n", code);
        cuCtxDestroy(ctx);
        return 15;
    }
    printf("native_cuda_driver_success=1 allocated_bytes=4096\n");
    cuMemFree(memory);
    cuCtxDestroy(ctx);
    dlclose(lib);
    return 0;
}
"""


@spaces.GPU(duration=60, size="large")
def run_native_cuda_probe():
    """Compile standalone C and exercise CUDA driver API in a child process."""
    import shutil
    from pathlib import Path
    import tempfile

    report = {
        "stage": "native CUDA Driver API via standalone C executable",
        "zero_gpu": os.environ.get("SPACES_ZERO_GPU", ""),
        "hardware": "ZeroGPU large",
    }
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    report["c_compiler"] = compiler
    if not compiler:
        report["outcome"] = "C compiler unavailable (no conclusion on CUDA)"
        return json.dumps(report, ensure_ascii=False, indent=2)

    try:
        with tempfile.TemporaryDirectory(prefix="zerogpu_native_") as d:
            source = Path(d) / "cuda_probe.c"
            executable = Path(d) / "cuda_probe"
            source.write_text(NATIVE_C, encoding="utf-8")
            build = subprocess.run(
                [compiler, "-O2", str(source), "-o", str(executable), "-ldl"],
                capture_output=True, text=True, timeout=20, check=False,
            )
            report["compile_exit"] = build.returncode
            if build.returncode != 0:
                report["compile_error"] = build.stderr[-1200:]
                report["outcome"] = "Compile failed; native CUDA not tested"
            else:
                run = subprocess.run(
                    [str(executable)], capture_output=True, text=True,
                    timeout=20, check=False,
                )
                report["execute_exit"] = run.returncode
                report["stdout"] = run.stdout.strip()[:500]
                report["stderr"] = run.stderr.strip()[-800:]
                report["outcome"] = (
                    "NATIVE CUDA DRIVER PASS" if run.returncode == 0 and
                    "native_cuda_driver_success=1" in run.stdout
                    else "NATIVE CUDA DRIVER FAIL"
                )
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["outcome"] = type(exc).__name__
        report["error"] = str(exc)[:300]
    report["limitations"] = (
        "This validates an independent native CUDA process and allocation, "
        "not GPU kernels or stable-diffusion.cpp model loading."
    )
    return json.dumps(report, ensure_ascii=False, indent=2)



# Prewritten PTX is loaded via libcuda and JIT-compiled by the GPU driver.
# This does not use nvcc, torch, or model weights. It does execute a real kernel.
CUDA_KERNEL_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef unsigned long long CUdeviceptr;

static const char ptx[] =
".version 7.0\n"
".target sm_80\n"
".address_size 64\n"
".visible .entry increment(.param .u64 ptr) {\n"
"  .reg .b64 %rd<3>;\n"
"  .reg .b32 %r<3>;\n"
"  ld.param.u64 %rd1, [ptr];\n"
"  cvta.to.global.u64 %rd2, %rd1;\n"
"  ld.global.u32 %r1, [%rd2];\n"
"  add.u32 %r2, %r1, 1;\n"
"  st.global.u32 [%rd2], %r2;\n"
"  ret;\n"
"}\n";

int main(void) {
    void *lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib) { fprintf(stderr, "libcuda missing: %s\n", dlerror()); return 10; }
#define SYMBOL(name) dlsym(lib, name)
    CUresult (*cuInit)(unsigned int) = SYMBOL("cuInit");
    CUresult (*cuDeviceGet)(CUdevice*, int) = SYMBOL("cuDeviceGet");
    CUresult (*cuCtxCreate)(CUcontext*, unsigned int, CUdevice) = SYMBOL("cuCtxCreate_v2");
    CUresult (*cuMemAlloc)(CUdeviceptr*, size_t) = SYMBOL("cuMemAlloc_v2");
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t) = SYMBOL("cuMemcpyHtoD_v2");
    CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, size_t) = SYMBOL("cuMemcpyDtoH_v2");
    CUresult (*cuModuleLoadData)(CUmodule*, const void*) = SYMBOL("cuModuleLoadData");
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = SYMBOL("cuModuleGetFunction");
    CUresult (*cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, unsigned int, void*, void**, void**) = SYMBOL("cuLaunchKernel");
    CUresult (*cuCtxSynchronize)(void) = SYMBOL("cuCtxSynchronize");
    CUresult (*cuModuleUnload)(CUmodule) = SYMBOL("cuModuleUnload");
    CUresult (*cuMemFree)(CUdeviceptr) = SYMBOL("cuMemFree_v2");
    CUresult (*cuCtxDestroy)(CUcontext) = SYMBOL("cuCtxDestroy_v2");
    if (!cuInit || !cuDeviceGet || !cuCtxCreate || !cuMemAlloc ||
        !cuMemcpyHtoD || !cuMemcpyDtoH || !cuModuleLoadData || !cuModuleGetFunction ||
        !cuLaunchKernel || !cuCtxSynchronize || !cuModuleUnload || !cuMemFree || !cuCtxDestroy) {
        fprintf(stderr, "CUDA driver API symbols missing\n"); return 11;
    }
#define CHECK(expr, label) do { CUresult error=(expr); if(error) { \
        fprintf(stderr, "%s returned %d\n", label, error); return 12; } } while(0)
    CUdevice dev = 0; CUcontext ctx = NULL; CUmodule module = NULL;
    CUfunction kernel = NULL; CUdeviceptr data = 0;
    unsigned int host = 41;
    CHECK(cuInit(0), "cuInit");
    CHECK(cuDeviceGet(&dev, 0), "cuDeviceGet");
    CHECK(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");
    CHECK(cuMemAlloc(&data, sizeof(host)), "cuMemAlloc");
    CHECK(cuMemcpyHtoD(data, &host, sizeof(host)), "cuMemcpyHtoD");
    CHECK(cuModuleLoadData(&module, ptx), "cuModuleLoadData");
    CHECK(cuModuleGetFunction(&kernel, module, "increment"), "cuModuleGetFunction");
    void *args[] = { &data };
    CHECK(cuLaunchKernel(kernel, 1,1,1, 1,1,1, 0, NULL, args, NULL), "cuLaunchKernel");
    CHECK(cuCtxSynchronize(), "cuCtxSynchronize");
    CHECK(cuMemcpyDtoH(&host, data, sizeof(host)), "cuMemcpyDtoH");
    printf("native_cuda_kernel_result=%u\n", host);
    cuModuleUnload(module);
    cuMemFree(data);
    cuCtxDestroy(ctx);
    dlclose(lib);
    return host == 42 ? 0 : 13;
}
"""


@spaces.GPU(duration=60, size="large")
def run_cuda_kernel_probe():
    """Run one actual native CUDA GPU kernel from a separate binary process."""
    import shutil
    import tempfile
    from pathlib import Path
    report = {
        "stage": "standalone CUDA kernel execution using Driver API + PTX",
        "hardware": "ZeroGPU large",
        "zero_gpu": os.environ.get("SPACES_ZERO_GPU", ""),
    }
    compiler = shutil.which("cc") or shutil.which("gcc")
    if not compiler:
        report["outcome"] = "No C compiler found"
        return json.dumps(report, ensure_ascii=False, indent=2)
    try:
        with tempfile.TemporaryDirectory(prefix="zerogpu_ptx_") as folder:
            source = Path(folder) / "kernel.c"
            exe = Path(folder) / "kernel"
            source.write_text(CUDA_KERNEL_C, encoding="utf-8")
            build = subprocess.run(
                [compiler, "-O2", str(source), "-o", str(exe), "-ldl"],
                capture_output=True, text=True, timeout=20, check=False,
            )
            report["compile_exit"] = build.returncode
            if build.returncode != 0:
                report["compile_error"] = build.stderr[-1000:]
                report["outcome"] = "COMPILATION FAILED"
            else:
                run = subprocess.run(
                    [str(exe)], capture_output=True, text=True, timeout=25,
                    check=False,
                )
                report["execute_exit"] = run.returncode
                report["stdout"] = run.stdout.strip()[:400]
                report["stderr"] = run.stderr.strip()[-700:]
                report["outcome"] = (
                    "NATIVE CUDA KERNEL PASS"
                    if run.returncode == 0 and "native_cuda_kernel_result=42" in run.stdout
                    else "NATIVE CUDA KERNEL FAIL"
                )
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["outcome"] = type(exc).__name__
        report["error"] = str(exc)[:250]
    report["next_step"] = (
        "If PASS, build the actual CUDA-enabled stable-diffusion.cpp CLI "
        "and test its model-loading path with MiniMax GGUF weights."
    )
    return json.dumps(report, ensure_ascii=False, indent=2)


with gr.Blocks(title="MiniMax H3 GGUF — ZeroGPU test") as demo:
    gr.Markdown(
        "# MiniMax H3 GGUF · ZeroGPU 兼容性测试\n"
        "目标模型：[Hearthscale/MiniMax-H3-Image-GGUF]"
        "(https://huggingface.co/Hearthscale/MiniMax-H3-Image-GGUF)\n\n"
        "**测试阶段 1：** 检查 `@spaces.GPU(size='large')` 主进程和原生子进程 "
        "能否访问 CUDA。此版本不下载模型，不生成图片，也不使用独立 GPU。"
    )
    button = gr.Button("执行 ZeroGPU / CUDA 子进程测试", variant="primary")
    output = gr.Code(label="诊断结果（JSON）", language="json", lines=20)
    button.click(run_probe, inputs=[], outputs=output, api_name="probe", concurrency_limit=1)
    gr.Markdown("## 第二阶段 · 原生 CUDA 进程测试")
    gr.Markdown(
        "在 ZeroGPU large GPU 任务里编译独立 C 程序，直接初始化 CUDA Driver API 并分配 4KB 显存。"
    )
    native_button = gr.Button("执行原生 CUDA 测试", variant="secondary")
    native_output = gr.Code(label="原生 CUDA 检测（JSON）", language="json", lines=16)
    native_button.click(
        run_native_cuda_probe, inputs=[], outputs=native_output,
        api_name="native_cuda_probe", concurrency_limit=1,
    )
    gr.Markdown("## 第三阶段 · 独立原生 CUDA Kernel 执行")
    gr.Markdown("用单独 C 子进程加载 PTX 并真正执行 CUDA Kernel（41 → 42），不需要 nvcc 或模型权重。")
    kernel_button = gr.Button("执行 CUDA Kernel 测试")
    kernel_output = gr.Code(label="CUDA Kernel 测试结果", language="json", lines=15)
    kernel_button.click(
        run_cuda_kernel_probe, inputs=[], outputs=kernel_output,
        api_name="cuda_kernel_probe", concurrency_limit=1,
    )
    gr.Markdown(
        "**下一阶段：** 只有本测试成功，才尝试将 "
        "`stable-diffusion.cpp` 和 GGUF 权重接入本 Space。 "
        "GPU 权限受 Hugging Face 硬件设置控制。"
    )

demo.queue(default_concurrency_limit=1)
demo.launch()
