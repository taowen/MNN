# Fun-Audio-Chat Debugging Findings

## Root Cause: Main-Thread Hang

### Problem

The Fun-Audio-Chat app hangs on startup when launched via ChatActivity with a local model path like `local//data/local/tmp/mnn_models/Fun-Audio-Chat`.

### Call Chain

1. `ChatActivity.onCreate()` → `ChatInputComponent.setupAttachmentPickerModule()`
2. → `ModelTypeUtils.isVisualModel(currentModelId)`
3. → `ModelListManager.isVisualModel(modelId)` → `getModelTags(modelId)`
4. → `runBlocking { modelListState.filterIsInstance<Success>().first() }` — **HANGS MAIN THREAD**

### Why It Hangs

`ModelListManager.getModelTags()` first checks `modelIdModelMap[modelId]`. For local models with a path-based modelId (e.g. `local//data/local/tmp/mnn_models/Fun-Audio-Chat`), the model is not found in the map because it was never registered through the normal model discovery flow.

The fallback code then calls `runBlocking` on the main thread, waiting for `modelListState` to emit a `Success` state. If the model list hasn't finished loading yet (or will never contain this model), the main thread blocks forever.

The same `getModelTags()` method is called by `isAudioModel()`, `isVideoModel()`, `isVisualModel()`, and `getExtraTags()` — all are affected.

### Fix

**File**: `ModelListManager.kt` — `getModelTags()`

Replaced the `runBlocking` fallback with a non-blocking `return emptyList()`:

```kotlin
fun getModelTags(modelId: String): List<String> {
    val modelItem = modelIdModelMap[modelId]
    if (modelItem != null) {
        return modelItem.getTags()
    }
    // Don't block — return empty if model not yet in memory map
    return emptyList()
}
```

`getExtraTags()` was already safe (no `runBlocking`).

## Debug Test Case

Added an "LLM Test" case to `DebugActivity` that creates an `LlmSession` directly, bypassing `ChatActivity`/`ChatInputComponent`/`ModelListManager`. This allows testing model loading and inference without triggering the full UI initialization chain.

### How to Use

```bash
adb shell am start -n com.alibaba.mnnllm.android/.debug.DebugActivity
```

1. Select "LLM Test" from the spinner
2. Enter config path (default: `/data/local/tmp/mnn_models/Fun-Audio-Chat/config.json`)
3. Click "Load Model" — wait for "Model loaded successfully" in the log
4. Enter a prompt and click "Send"

## Verification Results (2026-02-21)

**Fix confirmed working.** ChatActivity no longer hangs.

### Test Timeline (logcat PID 8341)

| Time | Event |
|------|-------|
| 13:47:17.390 | Activity created |
| 13:47:17.413 | UI rendered (0.4s — no hang!) |
| 13:47:26.691 | LlmSession CREATED on background thread (9.3s model load) |
| 13:47:26.692 | `chatSession.load() completed successfully` |
| 13:47:47.775 | User sent "hello" |
| 13:47:47.789 | `submitNative` called with prompt |
| 13:47:51.133 | Response: "Hi there! I'm here to help..." (3.3s inference) |

**Before fix**: Main thread hung permanently at `getModelTags()` → `runBlocking`.
**After fix**: `getModelTags()` returns `emptyList()` immediately, UI renders in 0.4s.

### Commands Used

```bash
# Build and install
./gradlew assembleStandardDebug && adb install -r app/build/outputs/apk/standard/debug/app-standard-debug.apk

# Test ChatActivity (fix verified)
adb shell am force-stop com.alibaba.mnnllm.android
adb shell am start -n com.alibaba.mnnllm.android/com.alibaba.mnnllm.android.chat.ChatActivity \
  --es configFilePath "/data/local/tmp/mnn_models/Fun-Audio-Chat/config.json" \
  --es modelId "local//data/local/tmp/mnn_models/Fun-Audio-Chat" \
  --es modelName "Fun-Audio-Chat"

# Note: DebugActivity is android:exported="false", cannot launch via adb directly
# Must navigate to it from within the app (Settings → Debug Mode)
```

### Current Status

- Text chat (s2t mode): Working. Model responds to text prompts.
- Fun-Audio-Chat is loaded as a text LLM. It uses the `llm.mnn` + `audio.mnn` weights.

## Fix 2: Enable Omni Mode for Fun-Audio-Chat

### Problem

`ModelTypeUtils.isOmni(modelName)` only checks for "omni" in the model name. "Fun-Audio-Chat" doesn't contain "omni", so `supportOmni = false`, and `setupOmni()` (which sets up AudioChunksPlayer + AudioDataListener for real-time audio output) is never called.

### Fix

**File**: `ModelTypeUtils.kt` — `isOmni()`

```kotlin
fun isOmni(modelName: String): Boolean {
    val lower = modelName.lowercase(Locale.getDefault())
    return lower.contains("omni") || lower.contains("audio-chat")
}
```

### Verification (2026-02-21)

After fix, `setupOmni()` is called and AudioChunksPlayer is created at 24000Hz:

```
13:53:42.523 ChatPresenter: load: chatSession.load() completed successfully
13:53:42.526 AudioChunksPlayer: start play audio samperate: 24000
13:53:42.558 AudioTrack: set(): sampleRate 24000, format 0x1, channelMask 0x1
```

### Omni Mode Architecture

1. `ChatService.createSession()` sets `llmSession.supportOmni = ModelTypeUtils.isOmni(modelName)`
2. After model load, `ChatActivity.onLoadingChanged(false)` calls `setupOmni()` if `supportOmni == true`
3. `setupOmni()` creates `AudioChunksPlayer(sampleRate=24000)` and sets `AudioDataListener` on the LlmSession
4. When the model generates audio data via native code, `onAudioData(FloatArray, isEnd)` streams it to the speaker
5. Voice Chat (menu item) uses separate ASR + TTS pipeline, independent of omni mode
