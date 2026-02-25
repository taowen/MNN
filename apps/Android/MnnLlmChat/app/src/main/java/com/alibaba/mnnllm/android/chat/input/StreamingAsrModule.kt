// Created by claude on 2025/02/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat.input

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Raw audio capture via AudioRecord (16kHz mono PCM16).
 * Produces WAV chunk files every ~1 second, calling [onChunkReady] on the recording thread.
 */
class StreamingAsrModule {

    var onChunkReady: ((String) -> Unit)? = null

    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val numChannels = 1
    private val bitsPerSample = 16
    private val bytesPerSample = bitsPerSample / 8
    private val blockAlign = numChannels * bytesPerSample

    // 1 second of audio = sampleRate * blockAlign bytes of PCM data
    private val chunkDurationMs = 1000
    private val chunkSamples = sampleRate * chunkDurationMs / 1000
    private val chunkPcmBytes = chunkSamples * blockAlign

    // 100ms read buffer (same as AsrService)
    private val readIntervalMs = 100
    private val readBufferSamples = sampleRate * readIntervalMs / 1000 // 1600 samples
    private val readBuffer = ShortArray(readBufferSamples)

    private var audioRecord: AudioRecord? = null
    private var recordingThread: Thread? = null
    @Volatile
    private var isRecording = false
    private var cacheDir: File? = null
    private var chunkIndex = 0
    private var startTimeMs = 0L

    fun start(cacheDir: File) {
        this.cacheDir = cacheDir
        this.chunkIndex = 0
        this.startTimeMs = System.currentTimeMillis()

        val minBufSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channelConfig,
            audioFormat,
            minBufSize * 2
        )
        audioRecord!!.startRecording()
        isRecording = true

        recordingThread = Thread({
            recordLoop()
        }, "StreamingAsrRecord")
        recordingThread!!.start()
    }

    /**
     * Stop recording and write the final partial chunk.
     * Returns null if total recording was <0.5s (treated as cancel).
     */
    fun stop(): String? {
        isRecording = false
        recordingThread?.join(2000)
        recordingThread = null

        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null

        val durationMs = System.currentTimeMillis() - startTimeMs
        if (durationMs < 500) {
            cleanup()
            return null
        }
        return "ok"
    }

    fun cleanup() {
        cacheDir?.listFiles()?.filter { it.name.startsWith("asr_chunk_") }?.forEach { it.delete() }
    }

    private fun recordLoop() {
        val accumulator = ByteArrayOutputStream()

        while (isRecording) {
            val ret = audioRecord?.read(readBuffer, 0, readBuffer.size) ?: -1
            if (ret > 0) {
                // Convert shorts to bytes (little-endian)
                val bytes = ByteArray(ret * bytesPerSample)
                ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().put(readBuffer, 0, ret)
                accumulator.write(bytes)

                if (accumulator.size() >= chunkPcmBytes) {
                    val pcmData = accumulator.toByteArray()
                    accumulator.reset()

                    val chunkPath = writeWavChunk(pcmData, chunkIndex)
                    chunkIndex++
                    onChunkReady?.invoke(chunkPath)
                }
            }
        }

        // Write final partial chunk if there's remaining data (>= 0.1s worth)
        val remaining = accumulator.toByteArray()
        val minFinalBytes = sampleRate * blockAlign / 10 // 0.1s minimum
        if (remaining.size >= minFinalBytes) {
            val chunkPath = writeWavChunk(remaining, chunkIndex)
            chunkIndex++
            onChunkReady?.invoke(chunkPath)
        }
    }

    private fun writeWavChunk(pcmData: ByteArray, index: Int): String {
        val chunkFile = File(cacheDir, "asr_chunk_${index}.wav")
        val dataSize = pcmData.size
        val fileSize = 36 + dataSize

        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray())
        header.putInt(fileSize)
        header.put("WAVE".toByteArray())
        header.put("fmt ".toByteArray())
        header.putInt(16) // fmt chunk size
        header.putShort(1) // PCM format
        header.putShort(numChannels.toShort())
        header.putInt(sampleRate)
        header.putInt(sampleRate * blockAlign) // byte rate
        header.putShort(blockAlign.toShort())
        header.putShort(bitsPerSample.toShort())
        header.put("data".toByteArray())
        header.putInt(dataSize)

        chunkFile.outputStream().use { out ->
            out.write(header.array())
            out.write(pcmData)
        }

        Log.d(TAG, "Wrote chunk $index: ${chunkFile.absolutePath} (${dataSize} bytes, ${dataSize.toFloat() / (sampleRate * blockAlign)}s)")
        return chunkFile.absolutePath
    }

    companion object {
        private const val TAG = "StreamingAsrModule"
    }
}

private class ByteArrayOutputStream {
    private var buf = ByteArray(65536)
    private var count = 0

    fun write(data: ByteArray) {
        val needed = count + data.size
        if (needed > buf.size) {
            buf = buf.copyOf(maxOf(buf.size * 2, needed))
        }
        System.arraycopy(data, 0, buf, count, data.size)
        count += data.size
    }

    fun size(): Int = count

    fun toByteArray(): ByteArray = buf.copyOf(count)

    fun reset() {
        count = 0
    }
}
