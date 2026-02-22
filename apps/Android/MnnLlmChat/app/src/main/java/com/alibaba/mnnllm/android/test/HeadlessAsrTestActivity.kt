package com.alibaba.mnnllm.android.test

import android.os.Bundle
import android.system.Os
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * Headless Activity for testing ASR models via adb.
 *
 * Launch:
 *   adb shell am start -n com.alibaba.mnnllm.android/.test.HeadlessAsrTestActivity \
 *     --es configFilePath "/data/local/tmp/Qwen3-ASR-MNN-q4/config.json" \
 *     --es audioFilePath "/data/local/tmp/test.wav"
 *
 * Monitor:
 *   adb logcat -s MNN_ASR_TEST
 */
class HeadlessAsrTestActivity : AppCompatActivity() {

    companion object {
        const val TAG = "MNN_ASR_TEST"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val configFilePath = intent.getStringExtra("configFilePath")
        val audioFilePath = intent.getStringExtra("audioFilePath")
        val backendType = intent.getStringExtra("backendType")

        if (configFilePath == null || audioFilePath == null) {
            Log.e(TAG, "Missing required extras: configFilePath and audioFilePath")
            Log.e(TAG, "Usage: adb shell am start -n com.alibaba.mnnllm.android/.test.HeadlessAsrTestActivity " +
                    "--es configFilePath \"/path/to/config.json\" --es audioFilePath \"/path/to/audio.wav\"")
            finish()
            return
        }

        Log.i(TAG, "=== ASR Test Starting ===")
        Log.i(TAG, "Config: $configFilePath")
        Log.i(TAG, "Audio:  $audioFilePath")
        if (backendType != null) {
            Log.i(TAG, "Backend: $backendType")
        }

        // Set up QNN environment (ADSP_LIBRARY_PATH) so HTP Skel libs can be found
        setupQnnEnvironment()

        CoroutineScope(Dispatchers.IO).launch {
            runAsrTest(configFilePath, audioFilePath, backendType)
            runOnUiThread { finish() }
        }
    }

    private fun setupQnnEnvironment() {
        try {
            val nativeLibDir = applicationInfo.nativeLibraryDir
            // ADSP_LIBRARY_PATH tells FastRPC where to find DSP skeleton libs.
            // Include vendor paths where libQnnHtpV79Skel.so may reside.
            val adspPath = listOf(
                nativeLibDir,
                "/vendor/lib64/hw/audio",
                "/vendor/dsp",
                "/vendor/dsp/xdsp"
            ).joinToString(";")
            Log.i(TAG, "Setting ADSP_LIBRARY_PATH=$adspPath")
            Os.setenv("ADSP_LIBRARY_PATH", adspPath, true)
            Os.setenv("LD_LIBRARY_PATH", nativeLibDir, true)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to set QNN environment: ${e.message}")
        }
    }

    private fun runAsrTest(configFilePath: String, audioFilePath: String, backendType: String?) {
        try {
            val sessionId = System.currentTimeMillis().toString()
            val session = LlmSession(
                modelId = "headless_asr_test",
                sessionId = sessionId,
                configPath = configFilePath,
                savedHistory = null,
                backendType = backendType
            )

            // Load model
            Log.i(TAG, "Loading model...")
            val loadStart = System.currentTimeMillis()
            session.load()
            val loadTime = System.currentTimeMillis() - loadStart
            Log.i(TAG, "Model loaded in ${loadTime}ms")

            // Build prompt with audio tag
            val prompt = "<audio>$audioFilePath</audio>"
            Log.i(TAG, "Submitting prompt: $prompt")

            val genStart = System.currentTimeMillis()
            val transcription = StringBuilder()
            val result = session.generate(
                prompt = prompt,
                params = emptyMap(),
                progressListener = object : GenerateProgressListener {
                    // Return false = continue generating, true = stop
                    override fun onProgress(progress: String?): Boolean {
                        if (progress != null) {
                            Log.d(TAG, "Token: $progress")
                            transcription.append(progress)
                        }
                        return false
                    }
                }
            )
            val genTime = System.currentTimeMillis() - genStart

            // Log results
            val response = transcription.toString()
            Log.i(TAG, "=== ASR Result ===")
            Log.i(TAG, "Transcription: $response")
            Log.i(TAG, "Generation time: ${genTime}ms")

            // Log performance stats
            val prefillTime = result["prefill_time"] as? Long ?: 0L
            val decodeTime = result["decode_time"] as? Long ?: 0L
            val audioTime = result["audio_time"] as? Long ?: 0L
            val promptLen = result["prompt_len"] as? Long ?: 0L
            val decodeLen = result["decode_len"] as? Long ?: 0L
            Log.i(TAG, "  prompt_len: $promptLen tokens")
            Log.i(TAG, "  decode_len: $decodeLen tokens")
            Log.i(TAG, "  audio_time: ${audioTime / 1000}ms")
            Log.i(TAG, "  prefill: ${prefillTime / 1000}ms (${if (prefillTime > 0) "%.1f".format(promptLen * 1e6 / prefillTime) else "N/A"} tok/s)")
            Log.i(TAG, "  decode: ${decodeTime / 1000}ms (${if (decodeTime > 0) "%.1f".format(decodeLen * 1e6 / decodeTime) else "N/A"} tok/s)")

            Log.i(TAG, "=== ASR Test Complete ===")

            // Clean up
            session.release()

        } catch (e: Exception) {
            Log.e(TAG, "ASR test failed", e)
            Log.e(TAG, "=== ASR Test FAILED ===")
        }
    }
}
