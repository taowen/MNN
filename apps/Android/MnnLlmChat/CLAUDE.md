# MnnLlmChat Build Guide

## Prerequisites

- Android SDK with `ANDROID_HOME` set (e.g. `~/android-sdk`)
- NDK 27.2.12479018 (`ndkVersion` pinned in `app/build.gradle`)
- `ANDROID_NDK` environment variable must point to the NDK path (the `build_64.sh` script references `$ANDROID_NDK`):
  ```bash
  export ANDROID_HOME="$HOME/android-sdk"
  export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/27.2.12479018"
  export ANDROID_NDK="$ANDROID_NDK_HOME"
  ```
- CMake and Make (system-level, used by `build_64.sh`)
- Gradle 8.9 (bundled via `gradlew`)

## Build Steps

### 1. Build MNN native libraries

The app's native code links against `libMNN.so` from `project/android/build_64/lib/`. This must be built first.

```bash
cd project/android
mkdir build_64
cd build_64
bash ../build_64.sh "-DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true -DMNN_ARM82=true -DMNN_USE_LOGCAT=true -DMNN_OPENCL=true -DLLM_SUPPORT_VISION=true -DMNN_BUILD_OPENCV=true -DMNN_IMGCODECS=true -DLLM_SUPPORT_AUDIO=true -DMNN_BUILD_AUDIO=true -DMNN_BUILD_DIFFUSION=ON -DMNN_SEP_BUILD=OFF -DCMAKE_SHARED_LINKER_FLAGS='-Wl,-z,max-page-size=16384' -DCMAKE_INSTALL_PREFIX=."
make install
```

Key notes:
- `MNN_SEP_BUILD=OFF` produces a single `libMNN.so` (no separate `.so` for CL, Express, etc.)
- `max-page-size=16384` is required for Android 15+ (16KB page size support)
- The output lands in `project/android/build_64/lib/libMNN.so`

### 2. Build the Android APK

```bash
cd apps/Android/MnnLlmChat
./gradlew assembleStandardDebug
```

APK output: `app/build/outputs/apk/standard/debug/app-standard-debug.apk`

Two product flavors exist: `standard` and `googleplay`.

## Project Structure

- `app/src/main/cpp/` - JNI native code (LLM, Diffusion, Video processing)
- `app/src/main/cpp/CMakeLists.txt` - Native build config, references MNN headers/libs via relative paths back to repo root
- `../../frameworks/mnn_tts/android` - TTS module (included via `settings.gradle`)
- `../../frameworks/model_downloader/android` - Model downloader module
- `downloadSherpaMnn.gradle` - Downloads prebuilt `libsherpa-mnn-jni.so` from CDN on first build
- `build_prebuilt.gradle` - Optional builtin model bundling (activated with `-PADD_BUILTIN=true`)

## Known Issues

### `Diffusion` abstract class instantiation error

`diffusion_session.cpp` uses `std::make_unique<Diffusion>(...)` but `Diffusion` is an abstract class with pure virtual methods (`run`, `load`). Fix: use the factory method instead:

```cpp
// Before (broken):
this->diffusion_ = std::make_unique<Diffusion>(resource_path_, ...);

// After (fixed):
this->diffusion_.reset(Diffusion::createDiffusion(resource_path_, ...));
```

### `ANDROID_NDK` vs `ANDROID_NDK_HOME`

`build_64.sh` uses `$ANDROID_NDK` to locate the CMake toolchain file. If only `ANDROID_NDK_HOME` is set (common in newer SDK setups), the build will fail silently with a missing toolchain. Ensure `ANDROID_NDK` is exported.

## Qwen3-ASR 0.6B MNN Porting

### Overview

Successfully ported Qwen3-ASR 0.6B (speech-to-text) to run on MNN with CPU inference. Tested on OnePlus PJZ110 (Snapdragon 8 Elite, Android 16).

### Performance (OnePlus PJZ110, CPU)

- Audio processing (6s speech): ~0.21s
- Prefill: 426 tok/s (93 tokens)
- Decode: 78 tok/s (26 tokens)

### Model Export & Conversion

1. Export from Huggingface model to ONNX+MNN:
   ```bash
   cd /home/taowen/Qwen3-ASR
   .venv/bin/python -m transformers.llm.export.llm_export \
     --path Qwen3-ASR-0.6B --type qwen3_asr --export mnn \
     --output /tmp/qwen3-asr-mnn
   ```

2. Model files produced:
   - `llm.mnn` — LLM decoder (Qwen2.5-0.5B backbone)
   - `audio.mnn` — Audio encoder (Conv2D + Transformer)
   - `llm_config.json` — Runtime configuration
   - `tokenizer.txt` — Tokenizer vocabulary
   - `embeddings_bf16.bin` — Embedding weights

### Files Modified (MNN repo)

| File | Change |
|------|--------|
| `transformers/llm/export/utils/audio.py` | Added `Qwen3AsrAudio` class (lines ~308-455) for audio encoder export |
| `transformers/llm/export/utils/model_mapper.py` | Added `qwen3_asr` model type mapping |
| `transformers/llm/engine/src/omni.cpp` | Added Qwen3-ASR audio processing path (8x downsample, windowed attention mask) |
| `transformers/llm/engine/src/omni.hpp` | Added `mNWindowInfer` member variable |
| `transformers/llm/engine/src/prompt.cpp` | Removed empty system prompt skip (Qwen3-ASR requires empty system section) |
| `tools/audio/include/audio/audio.hpp` | Added `periodic` field to `SpectrogramParams` |
| `tools/audio/source/audio.cpp` | Pass `periodic` flag through to `hann_window`; set `periodic=true` in `whisper_fbank` |

### Key Config (`llm_config.json`)

```json
{
    "model_type": "qwen3_asr",
    "hidden_size": 1024,
    "attention_mask": "float",
    "attention_type": "full",
    "is_mrope": true,
    "is_audio": true,
    "audio_pad": 151676,
    "n_window": 50,
    "n_window_infer": 800,
    "tie_embeddings": [275779826, 353571058, 19447808, 4, 64]
}
```

### Bugs Fixed

#### 1. Audio encoder reshape (export-side)

The audio encoder splits mel features `[1, 128, T]` into chunks of 100 frames. The WRONG way interleaves mel bins and time:
```python
# WRONG: scrambles data in C-contiguous memory order
input_features.reshape(-1, 128, 100)

# CORRECT: split along time axis, keep mel bins intact
input_features.reshape(1, 128, -1, 100).permute(2, 0, 1, 3)
```
The export code in `audio.py` uses the correct approach. This was the root cause of the model outputting "language None<asr_text>" with no transcription.

#### 2. Empty system prompt skip (prompt.cpp)

`prompt.cpp` had a line `if (input.second == "") continue;` that skipped empty system prompts. Qwen3-ASR's chat template requires `<|im_start|>system\n<|im_end|>\n` (5 tokens) even with empty content. Removing this line fixed the prompt assembly.

#### 3. C++ audio processing path (omni.cpp)

Added Qwen3-ASR-specific audio processing logic:
- **Chunk size**: `chunk_frames = n_window * 2 = 100`
- **3-layer Conv2D downsampling**: each layer has stride 2, so `tokens_per_chunk = (100+1)/2 → (50+1)/2 → (25+1)/2 = 13`
- **Windowed attention mask**: `n_window_tokens = 13 * (800/100) = 104`, attention is block-diagonal with 104-token windows
- **Padding**: input padded to multiple of `chunk_frames` before processing

#### 4. Periodic Hann window in whisper_fbank (audio.cpp)

MNN's `hann_window` defaulted to symmetric mode (`periodic=false`), but Python's WhisperFeatureExtractor uses a periodic window. This caused mel feature max diff of 0.139 vs Python, enough to break transcription.

Fix: Added `periodic` field to `SpectrogramParams` and set `periodic=true` in `whisper_fbank`. This reduced mel feature error from max=0.139/mean=0.0008 to max=0.005/mean=0.00002 (28x improvement).

### Phone Deployment (via adb)

```bash
# Push model files
adb push /tmp/qwen3-asr-mnn/* /data/local/tmp/mnn_models/Qwen3-ASR-0.6B-MNN/

# Push test audio (16kHz WAV)
adb push test_speech.wav /data/local/tmp/test_speech.wav

# Push llm_demo and libMNN.so
adb push project/android/build_64/llm_demo /data/local/tmp/
adb push project/android/build_64/lib/libMNN.so /data/local/tmp/

# Create prompt file
adb shell 'echo "<|audio_start|><audio>/data/local/tmp/test_speech.wav</audio><|audio_end|>" > /data/local/tmp/asr_prompt.txt'

# Run inference
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./llm_demo /data/local/tmp/mnn_models/Qwen3-ASR-0.6B-MNN/llm_config.json /data/local/tmp/asr_prompt.txt"
```

### Verification

Audio embeddings match Python reference closely (max diff ~0.0003):
- MNN:    `0.0223, -0.0144, -0.0348, 0.0212, -0.0018, -0.0055, -0.0083, 0.0064, 0.0446, 0.0234`
- Python: `0.0232, -0.0140, -0.0347, 0.0208, -0.0018, -0.0051, -0.0086, 0.0066, 0.0449, 0.0235`

Transcription output: "language English<asr_text>I have a hard time falling asleep. Is there any type of music that can help me fall asleep faster?"
