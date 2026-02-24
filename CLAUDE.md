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

## NPU (QNN) 调查结果

### 当前状态: 已实现 KV cache 支持，待测试

#### 在线编译 (runtime QNN graph build) — 已修复

原问题:
1. `QNNAttentionCreator::onCreate` 检查 `inputs.size() < 3` 导致 return nullptr（fused 模式下 inputs 为空）
2. `kv_cache=true` 时无处理逻辑，所有 28 层 attention fallback

修复方案: 固定大小 KV 缓冲区 + APP_WRITE 输入
- 新增 `QNNKVCacheAttention` 类（`QNNAttention.cpp/hpp`）
- 预分配 `maxKVLen=2048` 大小的 K/V 缓冲区作为 QNN 图的 APP_WRITE 输入
- 每步 `onExecute` 中 CPU 侧更新缓冲区，图 shape 不变无需重编译
- 扩展的 mask `[1, 1, seqQ, maxKVLen]` 保证 padding 位置被屏蔽（-10000）
- 支持 GQA（Split+Concat 方式）
- 支持 FP16/FP32 两种精度
- 通过 `onClone` 在 prefill/decode 实例间共享 KV cache

代码改动:
- `QNNBackend.hpp/cpp`: 新增 `registerExtraTensor/Input/Output`, `getInputDataContainer`, 修改 `executeGraph`/`clean`
- `QNNAttention.hpp`: 新增 `QNNKVCacheAttention` 类
- `QNNAttention.cpp`: 修改 creator（kv_cache→QNNKVCacheAttention）、输入验证移到 onEncode、完整 KV cache 实现

数据流:
```
onCopyBuffer → 写入 Q, newK, newV, mask 到 APP_WRITE 缓冲区
onExecute    → 读 newK/newV, 更新 CPU cache, 写 fullKey/fullValue/expandedMask
executeGraph → QNN NPU 用固定大小缓冲区执行 attention
```

#### 离线编译 (generate_llm_qnn.py) — 仍不可用

##### 离线编译架构

离线编译**不是把整个模型放 NPU**，而是拆图——只把静态 shape 的部分编译到 NPU，动态部分留 CPU：

```
Transformer Layer:
┌─────────────────────────────────────────────┐
│  RMSNorm → Q/K/V Proj (Linear)             │ ← 静态 shape → NPU ✓
├─────────────────────────────────────────────┤
│  Attention (with KV cache)                  │ ← 动态 shape → CPU（isBreak=true）
├─────────────────────────────────────────────┤
│  RMSNorm → FFN (Gate/Up/Down Linear)       │ ← 静态 shape → NPU ✓
└─────────────────────────────────────────────┘
```

关键代码 `compilefornpu.cpp:89`：
```cpp
static bool _npuSupportOp(const Op* op) {
    if (gMaxKVSize > 0) return true;  // 设了 KVCACHE_SIZE_LIMIT 才把 attention 也放 NPU
    if (op->type() == OpType_Attention) {
        auto attn = op->main_as_AttentionParam();
        if (nullptr != attn && attn->kv_cache()) return false;  // 默认: KV cache attention 不上 NPU
    }
    return true;
}
```

- `_createSubModuleInfo` 遍历所有 op，遇到 `isBreakOp`（含 KV cache 的 Attention）就断开
- 模型被切成交替的子图段：`[NPU子图0] [CPU attention] [NPU子图1] [CPU attention] ...`
- 每个 NPU 子图编译两个版本：`chunk_size`（如 128）用于 prefill，`1` 用于 decode
- 对应 `config_qnn.json` 中 `"chunk_limits": [128, 1]`
- Prefill 长序列分块处理：500 token → 128+128+128+116，每块用 chunk_size=128 的 NPU 图
- Linear 占 LLM ~80% 算力，只加速 Linear 已经足够有效

##### 四步流程 (`generate_llm_qnn.py`)

1. **makeIO**: 调用 `generateLlmIO`，生成两组测试 IO（seqLen=128 和 seqLen=1）
2. **seperate**: 调用 `compilefornpu`，拆图 + 编译 NPU 子图为 QNN context binary
3. **compile_qnn**: 调用 `npu_convert.py`，指定 soc_id 和 dsp_arch 做最终编译
4. **output_qnn**: 将产物移到模型目录，生成 `config_qnn.json`

##### Qwen3-ASR 失败原因

`generateLlmIO.cpp` 硬编码了标准 LLM 输入格式（`generateLlmIO.cpp:62`）：
```cpp
// 1D: shape = [seqLen]
VARP positionIds = _Input({seqLen}, NCHW, halide_type_of<int>());
```

但 Qwen3-ASR 使用 mrope（Multi-modal RoPE），`position_ids` 应为 `[3, seqLen]`（Temporal/Height/Width 三维位置索引）。标准 LLM（Llama、Qwen2）用 1D position_ids 所以没问题。

崩溃链路：
1. `generateLlmIO` 喂入 1D `position_ids` shape `[seqLen]`
2. 模型内部 mrope 逻辑对 tensor 做变换，中间 tensor 维度数与预期不同
3. Permute op 的 `dims` 配置是 3 个元素（`[1,0,2]`），但实际输入不是 3 维
4. `ShapePermute.cpp:27` 检查 `shape->size() != input->dimensions` → 失败
5. 返回 false 后 `Module::onForward` 产出 null VARP → 后续解引用 segfault

| 模型 | RoPE 类型 | position_ids shape | 与 generateLlmIO 兼容 |
|------|----------|-------------------|---------------------|
| Llama | 标准 RoPE | `[seqLen]` | 兼容 |
| Qwen2 | 标准 RoPE | `[seqLen]` | 兼容 |
| Qwen2-VL | mrope | `[3, seqLen]` | **不兼容** |
| Qwen3-ASR | mrope | `[3, seqLen]` | **不兼容** |

修复方向：`generateLlmIO.cpp` 需从 `llm_config.json` 读取 rope 类型，mrope 时生成 `[3, seqLen]` 的 position_ids。

### 相关信息

- QNN SDK: `/home/taowen/qnn-sdk/qairt/2.43.0.260128`
- SM8750 SoC ID: 69, DSP arch: v79

### 测试 NPU

```bash
# 用 npu 后端的 config.json
adb shell "cat > /data/local/tmp/mnn_models/Qwen3-ASR-0.6B-MNN/config.json" << 'EOF'
{
    "llm_model": "llm.mnn",
    "llm_weight": "llm.mnn.weight",
    "backend_type": "npu",
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
EOF

adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp time ./llm_demo mnn_models/Qwen3-ASR-0.6B-MNN/config.json asr_prompt.txt 100 2>&1"
```

### 模型导出 (NPU 版本, 已完成但无法部署)

```bash
cd /home/taowen/MNN/transformers/llm/export && \
  /home/taowen/Qwen3-ASR/.venv/bin/python llmexport.py \
  --path /home/taowen/Qwen3-ASR/Qwen3-ASR-0.6B \
  --type qwen3_asr \
  --export mnn \
  --dst_path /tmp/qwen3-asr-npu \
  --generate_for_npu --seperate_embed --sym \
  --act_bit 16 --quant_bit 4 --quant_block 64
```

注意: qwen3_asr 不在 transformers 上游, 需要:
- `pip install qwen-asr` (requires transformers==4.57.6)
- 或在 MNN config.py 的 `_register_external_model` 中注册 (已实现)

### 已知风险

- QNN 可能不支持 Reshape 作为 Identity（用于 sink 节点消耗 framework inputs）
- 固定 maxKVLen=2048，超过后报错
- 每步复制完整 KV cache 到 APP_WRITE 缓冲区，有优化空间
- 如果 QNN 图编译拒绝某些 op，attention 仍会 fallback 到 CPU

## 代码修改

### 已提交到 fork (taowen/MNN)

- `CLAUDE.md` - 实验记录

### 未提交的修改

- `source/backend/qnn/backend/QNNBackend.cpp` - SM8750 SoC retry patch + extra I/O 支持 + `getInputDataContainer`
- `source/backend/qnn/backend/QNNBackend.hpp` - 添加 QnnHtpDevice.h include + extra I/O 方法和成员
- `source/backend/qnn/execution/QNNAttention.cpp` - KV cache 支持 (`QNNKVCacheAttention`) + creator 修复
- `source/backend/qnn/execution/QNNAttention.hpp` - `QNNKVCacheAttention` 类声明
- `transformers/llm/export/utils/config.py` - qwen3_asr 外部模型注册 + AutoConfig fallback
- `source/shape/ShapePermute.cpp` - 将断言改为 return false (调试用)
