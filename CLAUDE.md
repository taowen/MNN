# MNN Development Notes

## QNN Offline Compilation Pipeline

### Pipeline Overview
The QNN offline compilation (`generate_llm_qnn.py`) has 4 steps:
1. **Step 1 (Make IO)**: `generateLlmIO` generates test input/output data
2. **Step 2 (Separate Model)**: `compilefornpu` splits model into sub-modules, runs each through QNN convert backend to record QNN graph topology
3. **Step 3 (Compile to QNN)**: `npu_convert.py` uses QNN SDK tools (`qnn-model-lib-generator`, `qnn-context-binary-generator`) to compile graphs into device-specific binaries
4. **Step 4 (Move results)**: Copies output to model directory

### Environment Requirements for QNN Pipeline
- `clang++` must be installed (`sudo apt-get install -y clang`)
- `libc++.so.1` must be installed (`sudo apt-get install -y libc++1`)
- Both are needed by QNN SDK's `qnn-model-lib-generator` and `qnn-context-binary-generator`

### Known Warnings (Harmless)
- `Broad cast error, dim1 = 1024, dim2 = 0` / `Compute Shape Error` during Step 2: These come from the `StaticModule` constructor's premature `mSession->resize()` call when `shapeMutable=false`. The sub-module's input tensors have uninitialized shapes (dim=0) at construction time. The error is harmless because:
  - The resize return value is ignored in the constructor
  - `onForward()` later sets correct shapes via `_resizeTensor`, then `mSession->resize()` succeeds
  - `QnnBackend::onResizeBegin()` calls `clean()` + `createContextAndGraph()` ensuring a fresh QNN graph for each resize attempt
  - The QNN graph is built correctly during the second resize in `onForward`

### Important: `shapeMutable` Setting in compilefornpu.cpp
**DO NOT change `mdconfig.shapeMutable = false` to `true`** in `_compileSubModule()` (compilefornpu.cpp:477).

- `shapeMutable=false` → `Session_Input_Inside` → `allocInput=true` → QNN backend's `onAcquire()` is called for input tensors → they are registered as `QNN_TENSOR_TYPE_APP_WRITE` (graph inputs)
- `shapeMutable=true` → `Session_Input_User` → `allocInput=false` → QNN backend never sees input tensors → `qnn-context-binary-generator` fails with "No graph inputs present for graph [N]"

### QNN Backend Architecture Notes
- `QnnBackend::onClearBuffer()` is a no-op (just returns true)
- `QnnBackend::onResizeBegin()` calls `clean()` + `createContextAndGraph()` — this is where graph state is actually reset
- `onResizeBegin` is called in `Pipeline::_allocForTensor()`, which is part of `allocMemory()`, NOT `encode()`
- So a failed `encode()` (from premature resize) leaves no QNN state to clean up — the graph is only created during `allocMemory()`

### QNN Android Runtime Setup

#### Required CMake Flags
QNN offline mode (Plugin ops loading context binaries) requires **both** flags:
```cmake
-DMNN_QNN=ON -DMNN_WITH_PLUGIN=ON
```
Without `MNN_WITH_PLUGIN=ON`, the Plugin op falls through to the `#else` branch and prints "Plugin is not supported."

#### Required Shared Libraries in APK (`jniLibs/arm64-v8a/`)
All from QNN SDK `lib/aarch64-android/` unless noted:
- `libQnnHtp.so` — HTP backend interface
- `libQnnHtpV79Stub.so` — ARM-side stub for HTP V79
- **`libQnnHtpV79Skel.so`** — DSP skeleton library (**from `lib/hexagon-v79/unsigned/`**, NOT `aarch64-android/`)
- `libQnnHtpV79CalculatorStub.so` — calculator stub
- `libQnnHtpPrepare.so` — offline graph preparation (~84MB)
- `libQnnSystem.so` — system interface (needed for `contextCreateFromBinary`)

**CRITICAL**: Missing `libQnnHtpV79Skel.so` causes `deviceCreate` to fail with error **14001** (`QNN_DEVICE_ERROR_INVALID_CONFIG`). This is the #1 cause of QNN HTP initialization failure on Android. The Skel library is in `hexagon-v79/unsigned/`, not `aarch64-android/` — easy to miss.

#### deviceCreate on SM8750
`deviceCreate(nullptr)` fails on SM8750. Must pass explicit SoC config:
```cpp
QnnHtpDevice_CustomConfig_t htpSocConfig = {};
htpSocConfig.option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
htpSocConfig.socModel = 69;  // SM8750
```
The code in `QNNBackend.cpp` has a fallback that tries known SoC IDs (69, 57, 43, 36).

#### ADSP_LIBRARY_PATH
Set in `MnnLlmApplication.kt::setupQnnEnvironment()`. Must include the app's `nativeLibDir` so FastRPC can find the Skel library:
```
nativeLibDir;/vendor/lib64/hw/audio;/vendor/dsp;/vendor/dsp/xdsp
```

#### Gradle Native Lib Caching
After rebuilding `libMNN.so`, Gradle may use stale cached copies. Force refresh:
```bash
rm -rf app/build/intermediates/stripped_native_libs/standardDebug \
       app/build/intermediates/merged_native_libs/standardDebug \
       app/build/outputs/apk/standard/debug/app-standard-debug.apk
./gradlew assembleStandardDebug
```

### QNN Pipeline Command
```bash
cd transformers/llm/export/npu
source ../../../llm/export/.venv/bin/activate  # or wherever venv is
QNN_SDK_ROOT=/path/to/qnn-sdk/qairt/2.43.0.260128 \
python3 generate_llm_qnn.py \
  --model /path/to/model-MNN-q4 \
  --soc_id 69 \        # SM8750 = 69
  --dsp_arch v79 \     # SM8750 HTP = v79
  --mnn_path ../../../../build/
```

## Android Test Automation (TestBackdoorReceiver)

### Overview
`TestBackdoorReceiver` is a BroadcastReceiver that allows headless testing of LLM models on Android via `adb broadcast` + file-based status polling. No UI interaction or logcat parsing needed.

### Key Files
- **Receiver**: `apps/Android/MnnLlmChat/app/src/main/java/com/alibaba/mnnllm/android/test/TestBackdoorReceiver.kt`
- **Python script**: `transformers/llm/export/test_qwen3_asr.py`
- **Manifest**: receiver registered in `AndroidManifest.xml`

### Critical: Android 8+ Implicit Broadcast Restriction
**Must use explicit broadcasts** (`-n component`) instead of implicit (`-a action`). Android 8.0+ silently drops implicit broadcasts to manifest-registered receivers. The broadcast will show `result=0` with no error, but the receiver never fires.

```bash
# WRONG — silently dropped on Android 8+
adb shell am broadcast -a com.alibaba.mnnllm.android.TEST --es command load ...

# CORRECT — explicit broadcast with component name
adb shell am broadcast -n com.alibaba.mnnllm.android/com.alibaba.mnnllm.android.test.TestBackdoorReceiver --es command load ...
```

### Shell Quoting for adb broadcast extras
When passing `--es key value` through `adb shell`, the entire command is interpreted by the device shell. Values with spaces get split into separate arguments. Fix: pass the entire `am broadcast ...` command as a single string argument to `adb shell`, with values single-quoted:

```python
# In Python, pass one string to "adb shell" so device shell handles quoting
adb("shell", "am broadcast -n COMPONENT --es prompt 'hello, what can you do?'")
```

### Usage
```bash
# Build & install
cd apps/Android/MnnLlmChat && ./gradlew assembleStandardDebug
adb install -r app/build/outputs/apk/standard/debug/app-standard-debug.apk

# Run automated test
source transformers/llm/export/.venv/bin/activate
python3 transformers/llm/export/test_qwen3_asr.py --device SERIAL --prompt "your prompt"

# Status polling reads: adb shell run-as com.alibaba.mnnllm.android cat files/test_status.json
# Output reads:         adb shell run-as com.alibaba.mnnllm.android cat files/test_output.txt
```

### Protocol
1. `load` command → writes `test_status.json` with `loading` → `loaded`
2. `generate` command → writes `generating` → `done` (with metrics)
3. `release` command → writes `idle`
4. On error: `{"status":"error","error":"message"}`

### Native Logging (mls_log.h)
`MNN_DEBUG` / `MNN_ERROR` 同时写 logcat 和文件。**不要用 logcat**，直接读文件：
```bash
adb shell run-as com.alibaba.mnnllm.android cat files/mnn_debug.log
```
- 文件路径: `/data/data/com.alibaba.mnnllm.android/files/mnn_debug.log`
- 用 `"w"` 模式打开 + `std::once_flag`，进程重启时会清空
- 每次写入后 `fflush`，不需要等进程结束

### Qwen3-ASR Prompt Format
**必须**用 `<|audio_start|>` 和 `<|audio_end|>` 包裹 `<audio>` 标签：
```
<|audio_start|><audio>/data/local/tmp/test_speech.wav</audio><|audio_end|>
```

### Critical: GenerateProgressListener.onProgress() Return Value
`onProgress()` return value means **"should we stop?"**:
- `return false` → **continue** generating
- `return true` → **stop** generating immediately

This caught us: `TestBackdoorReceiver` initially returned `true` (thinking it meant "success/continue"), causing `stop_requested_=1` after the first token. The model appeared to only generate 1 token (`decode_len=1`). Fix: return `false` to continue.

The underlying mechanism: `llm_session.cpp` line ~166 sets `stop_requested_ = user_stop_requested` where `user_stop_requested` is the return value of the progress callback.

### Force-Stop and Broadcast Delivery
After `am force-stop`, the app enters "stopped state" and won't receive broadcasts until manually relaunched:
```bash
adb shell am force-stop com.alibaba.mnnllm.android
adb shell am start -n com.alibaba.mnnllm.android/.main.MainActivity  # required!
# Now broadcasts work again
```

### ASR Config Notes
- For ASR models, set `"system_prompt": ""` in config.json — the default "You are a helpful assistant." adds unnecessary context
- `response(ChatMessages)` always applies chat template regardless of `use_template` setting
- Only `response(string)` respects `use_template: false`

### Debugging Workflow
先看日志文件确认在工作，再等待 test_status.json：
1. **先看日志文件**：`adb shell run-as com.alibaba.mnnllm.android cat files/mnn_debug.log`（内容和 logcat 一样，但不需要 PID 过滤）
2. **再等状态文件**：`adb shell run-as com.alibaba.mnnllm.android cat files/test_status.json`

### Multimodal Embedding Chunking Fix (llm.cpp)
`Llm::generate(vector<int>)` 会把 input_ids 按 `mBlockSize`（128）分块，每块调一次 `Omni::embedding()`。但 `embedding()` 会在第一次调用后清空 `mAudioEmbeddings`，第二块找不到 audio embeddings → crash。

修复：在 `llm.cpp` 中，当 `hasMultimodalContent()` 为 true 时跳过分块，一次性 embed 所有 token，让 `forwardVec(VARP)` 在 model forward 层面处理分块。

### Qwen3-ASR Streaming 方案（待实现）

**策略：增量推理（非累积重跑）**
- 参考实现（`/home/taowen/Qwen3-ASR/qwen_asr/inference/qwen3_asr.py`）用累积重跑策略：每次把全部音频从头送进模型。依赖 vLLM 的 prefix caching，不适合端侧。
- MNN 方案：利用已有 KV cache 做增量推理，不重跑历史。

**Pipeline（全部跑在 QNN 上）：**
1. **Audio encoder (QNN)**：每次只处理 1 个新 chunk，shape 固定 `[1, 128, 100]`（1 chunk = `n_window*2` = 100 mel frames），输出 13 个 audio token embeddings
2. **LLM prefill (QNN)**：13 个新 audio token 增量 prefill，复用已有 KV cache
3. **LLM decode (QNN)**：1 token 图，KV cache 持续累积

**KV Cache 容量**：
- `chunk_limits: [128, 1]` 是 prefill 分块大小，不是 KV cache 总容量
- KV cache 由 MNN fused Attention op（CPU 侧）管理，按 `KVMeta.block=4096` 分配
- QNN Plugin ops 处理 linear/norm/FFN，attention（含 KV cache）不在 QNN 图内
- 32s 音频 ≈ 416 audio tokens + ~20 prompt + ~100 generated ≈ 550 tokens，远小于 4096

**Audio encoder windowed attention 注意事项**：
- Qwen3-ASR audio encoder 有 18 层 transformer，`n_window_infer=800` 窗口注意力覆盖 ~8 chunks（104 tokens）
- 单 chunk 独立处理时丢失 cross-chunk attention context
- 如果精度不够，可改为滑动窗口：每次送最近 8 chunks 给 encoder，只取最后 1 chunk 的输出

### Audio Processing — Per-Chunk Mode (omni.cpp)
`audioProcess(VARP)` and `pushAudioChunk()` now use per-chunk processing for Qwen3-ASR (`mNWindowInfer > 0`):
- Each chunk processed independently with fixed shapes `[1, 128, 100]` + `[1, 13, 13]`
- Compatible with both CPU and QNN audio encoder
- Uses `_Slice(input, starts, sizes)` — **NOT** `_Slice(input, starts, ends)`. Sizes should be constant `chunk_frames`, not `(c+1)*chunk_frames`

### QNN Audio Encoder Compilation

#### Pipeline Script
`transformers/llm/export/npu/generate_audio_qnn.py` — same 4-step pattern as `generate_llm_qnn.py`:
```bash
cd transformers/llm/export/npu
QNN_SDK_ROOT=/home/taowen/qnn-sdk/qairt/2.43.0.260128 \
python3 generate_audio_qnn.py \
  --model /home/taowen/MNN/Qwen3-ASR-MNN-q4 \
  --soc_id 69 --dsp_arch v79 \
  --mnn_path ../../../../build/
```

#### IO Generator
`generateAudioIO` creates test IO for audio encoder: single chunk `[1, 128, 100]` + `[1, 13, 13]` mask → `[1, 13, 1024]` output.

#### shapeMutable for QNN Audio Module
**QNN Plugin ops models MUST be loaded with `shapeMutable=false`**. In `Omni::load()`, the audio module detects QNN path by checking `audio_path.find("qnn")` and sets `shapeMutable=false`.

Without this: `shapeMutable=true` → `Session_Input_User` → QNN backend never acquires input tensors → `Compute Shape Error for qnn_audio/graph0.bin` → null pointer crash.

The `Can't open file: .../qnn_audio/audio.mnn.weight` warning is harmless — QNN Plugin models have no external weights.

#### Config for QNN Audio
`config_qnn.json` needs:
```json
{
    "audio_model": "qnn_audio/audio.mnn",
    "system_prompt": "",
    "llm_weight": "llm.mnn.weight"
}
```
- `audio_model` points to QNN Plugin ops model
- `system_prompt: ""` required for ASR (same as CPU config)
- No `mllm` section needed — audio module uses `mRuntimeManager` (CPU backend, Plugin ops call QNN internally)

#### Current Status — Keep Audio on CPU
QNN audio encoder compiles and runs on SM8750 HTP, but **output is wrong due to fp16 precision loss**:
- CPU audio: "language English<asr_text>Hello, this is a test of the Vox Talk speech text system." (162ms)
- QNN audio: "language None<asr_text>" (430ms, wrong)

Tested mitigations that didn't help:
- `fp16_relaxed_precision: 0` (strict fp16) — still wrong
- `mllm` section with `precision: normal` — irrelevant (only affects CPU-side RuntimeManager)

**Recommendation**: Keep audio encoder on CPU (`audio_model: "audio.mnn"`). CPU is both faster (162ms vs 430ms) and correct. The QNN compilation pipeline and tools (`generateAudioIO`, `generate_audio_qnn.py`) are ready if future QNN SDK versions support better precision.

### TestBackdoorReceiver — Config File Support
`TestBackdoorReceiver` supports `config_file` extra to specify which config to load:
```python
# Load with QNN config
adb_broadcast("load", model_dir="...", config_file="config_qnn.json")
# Default: config.json
```
Test script: `--config config_qnn.json` flag in `test_qwen3_asr.py`

## Fun-Audio-Chat S2S Pipeline

### Architecture
Fun-Audio-Chat is a Speech-to-Speech model with 4 stages:
1. **Audio Encoder** (Whisper-like, 32 layers) — speech → embeddings
2. **LLM Thinker** (Qwen3-8B, 36 layers) — generates text + hidden states
3. **CRQ Talker** (Qwen3, 28 layers, hidden=1024) — hidden states → codec tokens (group_size=5)
4. **CosyVoice3** (DiT 22 layers + HIFT vocoder) — codec tokens → waveform

### Exported Files
| File | Size | Description |
|------|------|-------------|
| `llm.mnn` + `.weight` | 4.5 GB | Qwen3-8B thinker (q4) |
| `audio.mnn` + `.weight` | 420 MB | Audio encoder |
| `talker.mnn` + `.weight` | 286 MB | CRQ talker (28-layer Qwen3) |
| `predit.mnn` + `.weight` | 5.5 MB | DiT preprocessing (token embed, upsample, RoPE, mask) |
| `dit.mnn` + `.weight` | 373 MB | 22-layer DiT (flow matching, 10 Euler steps) |
| `bigvgan.mnn` + `.weight` | 280 MB | HIFT vocoder (mel → waveform) |
| `spk_dict.mnn` | 583 KB | Speaker embeddings + prompt tokens |
| `embeddings_int8.bin` | 668 MB | LLM token embeddings |
| `talker_embeddings_bf16.bin` | 52 MB | Talker codec embeddings |
| `pre_matching_bf16.bin` | 161 MB | Pre-matching projection weights |

### Export Command
```bash
source transformers/llm/export/.venv/bin/activate
PYTHONPATH=/home/taowen/Fun-Audio-Chat:$PYTHONPATH python3 transformers/llm/export/llmexport.py \
  --path /home/taowen/Fun-Audio-Chat/pretrained_models/Fun-Audio-Chat-8B \
  --dst_path /home/taowen/MNN/Fun-Audio-Chat-MNN \
  --quant_bit 4 --embed_bit 8 --export mnn
```

### Android Deployment (llm_demo CLI)
```bash
# Build for Android ARM64
cd build_android64
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_SDK/ndk/27.2.12479018/build/cmake/android.toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=28 \
  -DMNN_BUILD_LLM=ON -DMNN_BUILD_LLM_OMNI=ON -DMNN_ARM82=ON \
  -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_BF16=ON -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_SEP_BUILD=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
make -j$(nproc)

# Push to device
adb push build_android64/llm_demo /data/local/tmp/
adb push build_android64/OFF/arm64-v8a/libMNN.so /data/local/tmp/
adb push build_android64/OFF/arm64-v8a/libllm.so /data/local/tmp/
adb push build_android64/express/OFF/arm64-v8a/libMNN_Express.so /data/local/tmp/
adb push build_android64/tools/cv/OFF/arm64-v8a/libMNNOpenCV.so /data/local/tmp/
adb push build_android64/tools/audio/OFF/arm64-v8a/libMNNAudio.so /data/local/tmp/
adb shell chmod +x /data/local/tmp/llm_demo
adb push Fun-Audio-Chat-MNN/ /data/local/tmp/Fun-Audio-Chat-MNN/

# Run
adb shell "cd /data/local/tmp/Fun-Audio-Chat-MNN && LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/llm_demo config.json prompt.txt"
```
- Output WAV saved as `output.wav` in the model directory (24kHz, mono, 16-bit PCM)
- Android NDK: `/home/taowen/android-sdk/ndk/27.2.12479018`
- Must use `MNN_SEP_BUILD=ON` for Android (OFF causes CMake POST_BUILD errors with OBJECT libs)

### Key Implementation Details

#### FusedAttention → Native Attention Conversion
ONNX exports use custom `FusedAttention` Extra ops. MNN's `rebuild_attnention()` converts them to native `Attention` ops, but **only runs when `weight_ops is not None`** (main LLM path). For talker/token2wav exports (`weight_ops=None`), added `rebuild_extra_ops()` in `mnn_converter.py` that does JSON round-trip conversion after `onnx2mnn`.

#### CRQ Talker hidden_size ≠ num_heads*head_dim
CRQ has hidden_size=1024, num_heads=16, head_dim=128 → output_dim=2048. Added explicit `.view(bsz, q_len, -1)` before `o_proj` in `transformers.py` to avoid MNN Reshape errors when FusedAttention changes output layout from `[B,S,H*D]` to `[B,H,S,D]`.

#### Predit spk Shape Fix
The predit ONNX export used `spk` dummy as `[1, 1, 192]` (3D) with a `.squeeze(1)` in forward(). This baked a Squeeze op into the graph. C++ runtime provides `[1, 192]` (2D) from spk_dict.mnn, causing `Cannot Squeeze dim[1], 1 is expected, 192 is got`. Fix: changed dummy to `[1, 192]` and removed `.squeeze(1)` in `CosyVoice3DitPreprocess.forward()`.

#### codec_group_size in C++ Runtime
FunAudioChat generates 5 codec tokens per LLM text token (vs Qwen2.5-Omni's 1:1). Key adaptations in `omni.cpp`:
- `mCodecGroupSize` config-driven (from `codec_group_size` in config.json)
- `Talker::forwardRaw()` skips `logitsIndex` input for CRQ (3-input model vs 4-input)
- `Talker::gen_position_ids()` uses 1D positions (not MRoPE) when `mCodecGroupSize > 1`
- Pre-matching: `hidden_states * pre_matching_weight + bias` → reshape to `[1, group_size, hidden]`
- gen_seq_len/all_seq_len manually incremented (CRQ bypasses `Llm::generate()`)

#### Dit Components shapeMutable
predit/dit/bigvgan use `shapeMutable = (mCodecGroupSize > 1)`: `true` for CosyVoice3 (dynamic codec lengths), `false` for Qwen2.5-Omni (preserves original behavior). CRQ talker always uses `shapeMutable=false` for KV cache.

### Config Keys (llmconfig.hpp / config.json)
- `has_talker`: bool — enable/disable talker loading
- `talker_speaker`: string — speaker name in spk_dict (e.g. "中文女")
- `codec_group_size`: int — codec tokens per text token (5 for FunAudioChat, 1 for Qwen2.5-Omni)
- `codec_bos/eos/pad`: int — codec token IDs (6561/6562/6563 for FunAudioChat)
- `talker_layer_nums`: int — CRQ transformer layers (28)
- `dit_steps`: int — flow matching Euler steps (10 for CosyVoice3, 5 for Qwen2.5-Omni)
- `dit_solver`: int — 1 for Euler, 4 for RK4

