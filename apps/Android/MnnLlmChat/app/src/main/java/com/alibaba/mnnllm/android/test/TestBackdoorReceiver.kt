package com.alibaba.mnnllm.android.test

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.android.llm.LlmSession
import com.google.gson.Gson
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

class TestBackdoorReceiver : BroadcastReceiver() {

    companion object {
        private const val TAG = "TestBackdoor"
        private var session: LlmSession? = null
    }

    override fun onReceive(context: Context, intent: Intent) {
        val command = intent.getStringExtra("command") ?: return
        Log.d(TAG, "Received command: $command")

        when (command) {
            "load" -> handleLoad(context, intent)
            "generate" -> handleGenerate(context, intent)
            "streaming_generate" -> handleStreamingGenerate(context, intent)
            "release" -> handleRelease(context)
            else -> Log.w(TAG, "Unknown command: $command")
        }
    }

    private fun handleLoad(context: Context, intent: Intent) {
        val modelDir = intent.getStringExtra("model_dir") ?: run {
            writeStatus(context, mapOf("status" to "error", "error" to "missing model_dir"))
            return
        }
        writeStatus(context, mapOf("status" to "loading"))

        Thread {
            try {
                val configFile = intent.getStringExtra("config_file") ?: "config.json"
                val configPath = "$modelDir/$configFile"
                val modelId = "local/$modelDir"
                val sessionId = System.currentTimeMillis().toString()

                val llmSession = LlmSession(
                    modelId = modelId,
                    sessionId = sessionId,
                    configPath = configPath,
                    savedHistory = null
                )
                llmSession.load()
                session = llmSession
                writeStatus(context, mapOf("status" to "loaded"))
                Log.d(TAG, "Model loaded successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Load failed", e)
                writeStatus(context, mapOf("status" to "error", "error" to (e.message ?: "unknown")))
            }
        }.start()
    }

    private fun handleGenerate(context: Context, intent: Intent) {
        val prompt = intent.getStringExtra("prompt") ?: run {
            writeStatus(context, mapOf("status" to "error", "error" to "missing prompt"))
            return
        }
        val currentSession = session ?: run {
            writeStatus(context, mapOf("status" to "error", "error" to "no session loaded"))
            return
        }
        writeStatus(context, mapOf("status" to "generating"))
        // Clear previous output
        File(context.filesDir, "test_output.txt").writeText("")

        Thread {
            try {
                val outputFile = File(context.filesDir, "test_output.txt")
                val startTime = System.currentTimeMillis()

                val result = currentSession.generate(prompt, emptyMap(), object : GenerateProgressListener {
                    override fun onProgress(progress: String?): Boolean {
                        if (progress != null) {
                            outputFile.appendText(progress)
                        }
                        return false // false = continue, true = stop
                    }
                })

                val totalTime = System.currentTimeMillis() - startTime
                val statusMap = mutableMapOf<String, Any>(
                    "status" to "done",
                    "total_time_ms" to totalTime
                )
                // Include any metrics returned by native code
                for ((key, value) in result) {
                    statusMap[key] = value
                }
                writeStatus(context, statusMap)
                Log.d(TAG, "Generate completed")
            } catch (e: Exception) {
                Log.e(TAG, "Generate failed", e)
                writeStatus(context, mapOf("status" to "error", "error" to (e.message ?: "unknown")))
            }
        }.start()
    }

    private fun handleStreamingGenerate(context: Context, intent: Intent) {
        val audioFile = intent.getStringExtra("audio_file") ?: run {
            writeStatus(context, mapOf("status" to "error", "error" to "missing audio_file"))
            return
        }
        val promptPrefix = intent.getStringExtra("prompt_prefix") ?: run {
            writeStatus(context, mapOf("status" to "error", "error" to "missing prompt_prefix"))
            return
        }
        val promptSuffix = intent.getStringExtra("prompt_suffix") ?: run {
            writeStatus(context, mapOf("status" to "error", "error" to "missing prompt_suffix"))
            return
        }
        val chunkSeconds = intent.getStringExtra("chunk_seconds")?.toFloatOrNull() ?: 1.0f
        val currentSession = session ?: run {
            writeStatus(context, mapOf("status" to "error", "error" to "no session loaded"))
            return
        }

        writeStatus(context, mapOf("status" to "streaming"))
        File(context.filesDir, "test_output.txt").writeText("")

        Thread {
            try {
                val startTime = System.currentTimeMillis()

                // 1. Start streaming (prefill text prefix)
                currentSession.streamingStart(promptPrefix)

                // 2. Split audio into chunks and push each
                val chunkFiles = splitAudioToChunks(audioFile, chunkSeconds, context.cacheDir)
                for ((i, chunkFile) in chunkFiles.withIndex()) {
                    Log.d(TAG, "Pushing chunk $i/${chunkFiles.size}: ${chunkFile.absolutePath}")
                    currentSession.pushAudioChunk(chunkFile.absolutePath)
                    Log.d(TAG, "Chunk $i done")
                }

                // 3. Finish streaming (prefill suffix + decode)
                val outputFile = File(context.filesDir, "test_output.txt")
                val result = currentSession.streamingFinish(promptSuffix, object : GenerateProgressListener {
                    override fun onProgress(progress: String?): Boolean {
                        if (progress != null) {
                            outputFile.appendText(progress)
                        }
                        return false // false = continue
                    }
                })

                val totalTime = System.currentTimeMillis() - startTime
                val statusMap = mutableMapOf<String, Any>(
                    "status" to "done",
                    "total_time_ms" to totalTime,
                    "num_chunks" to chunkFiles.size
                )
                for ((key, value) in result) {
                    statusMap[key] = value
                }
                writeStatus(context, statusMap)

                // Cleanup temp chunk files
                chunkFiles.forEach { it.delete() }
            } catch (e: Exception) {
                Log.e(TAG, "Streaming generate failed", e)
                writeStatus(context, mapOf("status" to "error", "error" to (e.message ?: "unknown")))
            }
        }.start()
    }

    /**
     * Split a WAV file into chunks of chunkSeconds duration.
     * Properly parses WAV chunks to find the "data" chunk (handles FLLR/LIST/etc).
     */
    private fun splitAudioToChunks(audioPath: String, chunkSeconds: Float, cacheDir: File): List<File> {
        val raf = RandomAccessFile(audioPath, "r")

        // Read RIFF header (12 bytes)
        val riffHeader = ByteArray(12)
        raf.readFully(riffHeader)

        // Parse fmt chunk to get audio format info
        var numChannels = 1
        var sampleRate = 16000
        var bitsPerSample = 16
        var dataOffset = -1L
        var dataSize = 0

        // Scan chunks to find "fmt " and "data"
        while (raf.filePointer < raf.length()) {
            val chunkId = ByteArray(4)
            if (raf.read(chunkId) < 4) break
            val chunkSizeBuf = ByteArray(4)
            if (raf.read(chunkSizeBuf) < 4) break
            val chunkSize = ByteBuffer.wrap(chunkSizeBuf).order(ByteOrder.LITTLE_ENDIAN).getInt()
            val id = String(chunkId)

            if (id == "fmt ") {
                val fmtData = ByteArray(chunkSize)
                raf.readFully(fmtData)
                val fmt = ByteBuffer.wrap(fmtData).order(ByteOrder.LITTLE_ENDIAN)
                numChannels = fmt.getShort(2).toInt()
                sampleRate = fmt.getInt(4)
                bitsPerSample = fmt.getShort(14).toInt()
            } else if (id == "data") {
                dataSize = chunkSize
                dataOffset = raf.filePointer
                break
            } else {
                // Skip unknown chunks (FLLR, LIST, etc.)
                raf.skipBytes(chunkSize)
            }
        }

        if (dataOffset < 0) {
            raf.close()
            Log.e(TAG, "No 'data' chunk found in WAV: $audioPath")
            return emptyList()
        }

        val bytesPerSample = bitsPerSample / 8
        val blockAlign = numChannels * bytesPerSample
        val chunkSamples = (sampleRate * chunkSeconds).toInt()
        val chunkBytes = chunkSamples * blockAlign
        val totalSamples = dataSize / blockAlign
        val numChunks = (totalSamples + chunkSamples - 1) / chunkSamples

        Log.d(TAG, "WAV: ${sampleRate}Hz, ${numChannels}ch, ${bitsPerSample}bit, ${totalSamples} samples (${totalSamples.toFloat()/sampleRate}s), dataOffset=$dataOffset")

        raf.seek(dataOffset)
        val chunkFiles = mutableListOf<File>()
        var remaining = dataSize

        for (i in 0 until numChunks) {
            val thisChunkBytes = minOf(chunkBytes, remaining)
            val pcmData = ByteArray(thisChunkBytes)
            raf.readFully(pcmData)
            remaining -= thisChunkBytes

            // Write chunk WAV file with standard 44-byte header
            val chunkFile = File(cacheDir, "chunk_${i}.wav")
            val chunkDataSize = thisChunkBytes
            val chunkFileSize = 36 + chunkDataSize

            val chunkHeader = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
            chunkHeader.put("RIFF".toByteArray())
            chunkHeader.putInt(chunkFileSize)
            chunkHeader.put("WAVE".toByteArray())
            chunkHeader.put("fmt ".toByteArray())
            chunkHeader.putInt(16) // fmt chunk size
            chunkHeader.putShort(1) // PCM format
            chunkHeader.putShort(numChannels.toShort())
            chunkHeader.putInt(sampleRate)
            chunkHeader.putInt(sampleRate * blockAlign) // byte rate
            chunkHeader.putShort(blockAlign.toShort())
            chunkHeader.putShort(bitsPerSample.toShort())
            chunkHeader.put("data".toByteArray())
            chunkHeader.putInt(chunkDataSize)

            chunkFile.outputStream().use { out ->
                out.write(chunkHeader.array())
                out.write(pcmData)
            }
            chunkFiles.add(chunkFile)
        }
        raf.close()
        Log.d(TAG, "Split $audioPath into ${chunkFiles.size} chunks of ${chunkSeconds}s each")
        return chunkFiles
    }

    private fun handleRelease(context: Context) {
        Thread {
            try {
                session?.release()
                session = null
                writeStatus(context, mapOf("status" to "idle"))
                Log.d(TAG, "Session released")
            } catch (e: Exception) {
                Log.e(TAG, "Release failed", e)
                writeStatus(context, mapOf("status" to "error", "error" to (e.message ?: "unknown")))
            }
        }.start()
    }

    private fun writeStatus(context: Context, status: Map<String, Any>) {
        val json = Gson().toJson(status)
        File(context.filesDir, "test_status.json").writeText(json)
        Log.d(TAG, "Status: $json")
    }
}
