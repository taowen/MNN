# Qwen3-ASR 0.6B on MNN - 实验记录

## 模型文件

- ONNX 源文件: `/tmp/qwen3-asr-mnn/onnx/audio.onnx` (717MB)
- 量化后模型 (本地):
  - `/tmp/qwen3-asr-mnn/audio_q4.mnn` (117MB, 4-bit, **不可用**)
  - `/tmp/qwen3-asr-mnn/audio_q5.mnn` (139MB, 5-bit, CPU OK, OpenCL FAIL)
  - `/tmp/qwen3-asr-mnn/audio_q6.mnn` (162MB, 6-bit, CPU OK, OpenCL FAIL)
  - `/tmp/qwen3-asr-mnn/audio_q8.mnn` (206MB, 8-bit, CPU OK, OpenCL OK)
- 测试音频: `/tmp/test_speech.wav` (375KB, 16kHz mono 32-bit float, ~6s)
- 手机模型目录: `/data/local/tmp/mnn_models/Qwen3-ASR-0.6B-MNN/`
- 手机 prompt 文件: `/data/local/tmp/asr_prompt.txt`

## 编译

需要在 `project/android/build_64` 目录下编译，启用 Vulkan + QNN + OpenCL：

```bash
rm -rf project/android/build_64 && mkdir -p project/android/build_64 && cd project/android/build_64
cmake /home/taowen/MNN \
  -DCMAKE_TOOLCHAIN_FILE=/home/taowen/android-sdk/ndk/27.2.12479018/build/cmake/android.toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DANDROID_ABI="arm64-v8a" -DANDROID_STL=c++_static \
  -DANDROID_NATIVE_API_LEVEL=android-21 \
  -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
  -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true \
  -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true \
  -DMNN_ARM82=true \
  -DMNN_OPENCL=true -DMNN_VULKAN=true \
  -DMNN_QNN=ON -DQNN_SDK_ROOT=/home/taowen/qnn-sdk/qairt/2.43.0.260128 \
  -DMNN_BUILD_OPENCV=true -DMNN_IMGCODECS=true \
  -DMNN_BUILD_AUDIO=true -DMNN_BUILD_DIFFUSION=ON \
  -DMNN_SEP_BUILD=OFF -DMNN_WITH_PLUGIN=ON \
  "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=16384" \
  -DCMAKE_INSTALL_PREFIX=.
make -j$(nproc)
# 产物: libMNN.so (7.9MB), llm_demo (35KB)
```

## 运行命令

```bash
# 推二进制和模型到手机
adb push project/android/build_64/libMNN.so /data/local/tmp/
adb push project/android/build_64/llm_demo /data/local/tmp/
adb push /tmp/qwen3-asr-mnn/audio_q8.mnn /data/local/tmp/mnn_models/Qwen3-ASR-0.6B-MNN/audio.mnn

# 运行推理
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp time ./llm_demo mnn_models/Qwen3-ASR-0.6B-MNN/config.json asr_prompt.txt 100 2>&1"

# asr_prompt.txt 内容:
# <|audio_start|><audio>/data/local/tmp/test_speech.wav</audio><|audio_end|>
```

## Audio Encoder 量化结果

| Bits | 大小 | 压缩比 | CPU | OpenCL |
|------|------|--------|-----|--------|
| fp32 | 716MB | 1x | OK | OK |
| 8-bit | 206MB | 3.5x | OK | OK |
| 6-bit | 162MB | 4.4x | OK | FAIL (空输出) |
| 5-bit | 139MB | 5.1x | OK | FAIL (空输出) |
| 4-bit | 117MB | 6.1x | FAIL (乱码) | - |

- 5/6-bit 在 OpenCL 上失败, GPU kernel 不支持非标准位宽的反量化
- 4-bit 在 CPU 上也完全不可用, audio encoder 对量化敏感

## 性能基准 (test_speech.wav, ~6s 音频, 8-bit audio encoder)

设备: OnePlus PJZ110, Snapdragon 8 Elite (SM8750)

| LLM 后端 | Audio 后端 | 总耗时 | prefill | decode | 备注 |
|----------|-----------|--------|---------|--------|------|
| **OpenCL (cached)** | CPU | **2.3s** | **779 tok/s** | **53 tok/s** | 最快，需 cache |
| CPU | CPU | 2.6s | 272 tok/s | 87 tok/s | 最稳定 |
| OpenCL (首次) | CPU | 10.5s | 13 tok/s | 21 tok/s | 首次编译 kernel 慢 |
| Vulkan (cached) | CPU | 14.8s | 32 tok/s | 6 tok/s | 差 |
| Vulkan (首次) | CPU | 17.6s | 23 tok/s | 5 tok/s | 差 |
| OpenCL | OpenCL | 85s | - | - | audio 上 GPU 灾难性慢 |
| NPU | CPU | CRASH | - | - | 需专门导出 NPU 模型 |

### 关键发现

- **OpenCL (有 cache) 最快**: prefill 779 tok/s, 总 2.3s，比 CPU 快 10%
- **OpenCL 首次运行慢**: kernel 编译需要 ~7s，cache 后 prefill 从 13 → 779 tok/s
- **OpenCL cache 文件**: `tmp/mnn_cachefile.bin` (~2MB), 相对于 CWD
- **CPU decode 更快**: CPU 87 tok/s vs OpenCL 53 tok/s，decode 阶段 CPU 占优
- **Vulkan 全面差于 OpenCL**: prefill/decode 都慢很多
- **NPU (QNN) 需要专门的模型导出**: 标准 MNN 模型直接 SIGSEGV

## config.json (最优配置: OpenCL LLM + CPU audio)

```json
{
    "llm_model": "llm.mnn",
    "llm_weight": "llm.mnn.weight",
    "backend_type": "opencl",
    "thread_num": 4,
    "precision": "low",
    "memory": "low",
    "sampler_type": "penalty",
    "penalty": 1.1,
    "mllm": {
        "backend_type": "cpu",
        "thread_num": 4,
        "precision": "normal",
        "memory": "low"
    }
}
```

备选（无 GPU 依赖）: `"backend_type": "cpu"`

## MNN 支持的后端

`backend_type_convert` in `transformers/llm/engine/src/llm.cpp:49`:
- `"cpu"` → MNN_FORWARD_CPU
- `"metal"` → MNN_FORWARD_METAL (iOS/macOS)
- `"cuda"` → MNN_FORWARD_CUDA
- `"opencl"` → MNN_FORWARD_OPENCL
- `"vulkan"` → MNN_FORWARD_VULKAN
- `"npu"` → MNN_FORWARD_NN (高通 QNN / MTK Neuropilot)

## NPU (QNN) 要求

QNN 不支持直接加载标准 MNN 模型，需要:
1. 导出时加 `--generate_for_npu --seperate_embed --sym` 标志
2. 使用 16-bit 激活 + 4/8-bit 权重
3. QNN SDK: `/home/taowen/qnn-sdk/qairt/2.43.0.260128`
4. 支持 Hexagon HTP v79 (SM8750)
5. 参考文档: `docs/transformers/llm.md` (NPU 推理 LLM 章节)
