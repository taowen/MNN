package com.alibaba.mnnllm.android.debug

import android.content.Context
import android.os.Build
import com.alibaba.mnnllm.android.BuildConfig
import com.alibaba.mnnllm.android.benchmark.BenchmarkCallback
import com.alibaba.mnnllm.android.benchmark.BenchmarkProgress
import com.alibaba.mnnllm.android.benchmark.BenchmarkResult
import com.alibaba.mnnllm.android.benchmark.BenchmarkService
import com.alibaba.mnnllm.android.benchmark.RuntimeParameters
import com.alibaba.mnnllm.android.benchmark.TestParameters
import com.alibaba.mnnllm.android.modelist.ModelListManager
import com.alibaba.mnnllm.android.utils.TimberConfig
import io.ktor.http.*
import io.ktor.server.application.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.serialization.Serializable
import timber.log.Timber
import java.io.File

private const val TAG = "DebugRoutes"

// ============ Request / Response Data Classes ============

@Serializable
data class PingResponse(
    val status: String = "ok",
    val appVersion: String,
    val device: String,
    val androidVersion: String,
    val sdkInt: Int
)

@Serializable
data class ModelInfo(
    val modelId: String,
    val localPath: String?,
    val isLocal: Boolean,
    val displayName: String,
    val sizeBytes: Long
)

@Serializable
data class ModelsResponse(
    val models: List<ModelInfo>
)

@Serializable
data class BenchRequest(
    val modelId: String,
    val backend: String = "cpu",
    val nPrompt: Int = 128,
    val nGenerate: Int = 128,
    val nRepeat: Int = 3,
    val threads: Int = 4
)

@Serializable
data class SpeedStat(
    val avg: Double,
    val stdev: Double
)

@Serializable
data class BenchResponse(
    val status: String,
    val modelId: String,
    val backend: String,
    val threads: Int,
    val nPrompt: Int,
    val nGenerate: Int,
    val nRepeat: Int,
    val prefillSpeed: SpeedStat? = null,
    val decodeSpeed: SpeedStat? = null,
    val maxMemoryKb: Long = 0,
    val rawPrefillUs: List<Long> = emptyList(),
    val rawDecodeUs: List<Long> = emptyList(),
    val error: String? = null
)

@Serializable
data class BenchStatusResponse(
    val status: String,
    val modelId: String? = null,
    val lastResult: BenchResponse? = null
)

@Serializable
data class LogsResponse(
    val lines: List<String>,
    val fileLoggingEnabled: Boolean
)

@Serializable
data class LogEnableResponse(
    val status: String,
    val fileLoggingEnabled: Boolean
)

@Serializable
data class ErrorResponse(
    val error: String
)

// ============ Benchmark State ============

private object BenchState {
    @Volatile var status: String = "idle"
    @Volatile var modelId: String? = null
    @Volatile var lastResult: BenchResponse? = null
}

// ============ Routes ============

fun Application.debugRoutes(appContext: Context) {
    routing {
        route("/debug") {

            get("/ping") {
                call.respond(PingResponse(
                    appVersion = BuildConfig.VERSION_NAME,
                    device = "${Build.MANUFACTURER} ${Build.MODEL}",
                    androidVersion = Build.VERSION.RELEASE,
                    sdkInt = Build.VERSION.SDK_INT
                ))
            }

            get("/models") {
                val models = ModelListManager.getCurrentModels()
                if (models == null) {
                    call.respond(
                        HttpStatusCode.ServiceUnavailable,
                        ErrorResponse("Model list not yet initialized")
                    )
                    return@get
                }
                val modelInfos = models.map { wrapper ->
                    ModelInfo(
                        modelId = wrapper.modelItem.modelId ?: "unknown",
                        localPath = wrapper.modelItem.localPath,
                        isLocal = wrapper.isLocal,
                        displayName = wrapper.displayName,
                        sizeBytes = wrapper.downloadSize
                    )
                }
                call.respond(ModelsResponse(models = modelInfos))
            }

            post("/bench/start") {
                try {
                    val request = call.receive<BenchRequest>()

                    if (BenchState.status == "running") {
                        call.respond(
                            HttpStatusCode.Conflict,
                            ErrorResponse("Benchmark already running")
                        )
                        return@post
                    }

                    BenchState.status = "running"
                    BenchState.modelId = request.modelId

                    val result = executeBenchmark(appContext, request)

                    BenchState.lastResult = result
                    BenchState.status = if (result.status == "completed") "completed" else "idle"

                    call.respond(result)
                } catch (e: Exception) {
                    Timber.tag(TAG).e(e, "Benchmark failed")
                    BenchState.status = "idle"
                    val errorResponse = BenchResponse(
                        status = "error",
                        modelId = BenchState.modelId ?: "",
                        backend = "",
                        threads = 0,
                        nPrompt = 0,
                        nGenerate = 0,
                        nRepeat = 0,
                        error = e.message ?: "Unknown error"
                    )
                    BenchState.lastResult = errorResponse
                    call.respond(HttpStatusCode.InternalServerError, errorResponse)
                }
            }

            get("/bench/status") {
                call.respond(BenchStatusResponse(
                    status = BenchState.status,
                    modelId = BenchState.modelId,
                    lastResult = BenchState.lastResult
                ))
            }

            get("/logs") {
                val maxLines = call.request.queryParameters["lines"]?.toIntOrNull() ?: 200
                val logDir = File(appContext.filesDir, "logs")

                if (!logDir.exists() || !logDir.isDirectory) {
                    call.respond(LogsResponse(
                        lines = emptyList(),
                        fileLoggingEnabled = TimberConfig.isFileLoggingEnabled()
                    ))
                    return@get
                }

                val logFiles = logDir.listFiles()
                    ?.filter { it.name.startsWith("log.") }
                    ?.sortedBy { it.lastModified() }
                    ?: emptyList()

                val allLines = mutableListOf<String>()
                for (file in logFiles) {
                    allLines.addAll(file.readLines())
                }

                call.respond(LogsResponse(
                    lines = allLines.takeLast(maxLines),
                    fileLoggingEnabled = TimberConfig.isFileLoggingEnabled()
                ))
            }

            post("/logs/enable") {
                TimberConfig.setFileLoggingEnabled(true)
                call.respond(LogEnableResponse(
                    status = "ok",
                    fileLoggingEnabled = true
                ))
            }
        }
    }
}

// ============ Benchmark Execution ============

private suspend fun executeBenchmark(appContext: Context, request: BenchRequest): BenchResponse {
    val service = BenchmarkService.getInstance()

    val backendInt = when (request.backend.lowercase()) {
        "cpu" -> 0
        "opencl", "gpu" -> 3
        "metal" -> 1
        else -> 0
    }

    // Initialize model
    val initialized = service.initializeModel(
        modelId = request.modelId,
        backendType = request.backend
    )

    if (!initialized) {
        return BenchResponse(
            status = "error",
            modelId = request.modelId,
            backend = request.backend,
            threads = request.threads,
            nPrompt = request.nPrompt,
            nGenerate = request.nGenerate,
            nRepeat = request.nRepeat,
            error = "Failed to initialize model: ${request.modelId}"
        )
    }

    try {
        val runtimeParams = RuntimeParameters(
            backends = listOf(backendInt),
            threads = listOf(request.threads),
            power = listOf(0),
            precision = listOf(2),
            memory = listOf(2),
            dynamicOption = listOf(0)
        )

        // Only run combined prompt+generate test (skip separate prefill-only and decode-only)
        val testParams = TestParameters(
            nPrompt = emptyList(),
            nGenerate = emptyList(),
            nPrompGen = listOf(Pair(request.nPrompt, request.nGenerate)),
            nRepeat = listOf(request.nRepeat)
        )

        val resultDeferred = CompletableDeferred<BenchmarkResult>()

        service.runBenchmark(
            context = appContext,
            modelId = request.modelId,
            callback = object : BenchmarkCallback {
                override fun onProgress(progress: BenchmarkProgress) {
                    Timber.tag(TAG).d("Bench progress: ${progress.progress}%")
                }

                override fun onComplete(result: BenchmarkResult) {
                    resultDeferred.complete(result)
                }

                override fun onBenchmarkError(errorCode: Int, message: String) {
                    resultDeferred.completeExceptionally(
                        Exception("Benchmark error ($errorCode): $message")
                    )
                }
            },
            runtimeParams = runtimeParams,
            testParams = testParams
        )

        // Safety net: if the benchmark finishes without calling onComplete/onError,
        // detect via polling isBenchmarkRunning()
        val watchJob = CoroutineScope(Dispatchers.IO).launch {
            while (service.isBenchmarkRunning()) {
                delay(500)
            }
            if (!resultDeferred.isCompleted) {
                resultDeferred.completeExceptionally(
                    Exception("Benchmark finished without producing results")
                )
            }
        }

        try {
            val result = withTimeout(300_000L) { resultDeferred.await() }
            watchJob.cancel()

            val ti = result.testInstance
            val prefillSpeeds = ti.getTokensPerSecond(request.nPrompt, ti.prefillUs)
            val decodeSpeeds = ti.getTokensPerSecond(request.nGenerate, ti.decodeUs)

            return BenchResponse(
                status = "completed",
                modelId = request.modelId,
                backend = request.backend,
                threads = request.threads,
                nPrompt = request.nPrompt,
                nGenerate = request.nGenerate,
                nRepeat = request.nRepeat,
                prefillSpeed = if (prefillSpeeds.isNotEmpty()) SpeedStat(
                    avg = ti.getAvgUs(prefillSpeeds),
                    stdev = ti.getStdevUs(prefillSpeeds)
                ) else null,
                decodeSpeed = if (decodeSpeeds.isNotEmpty()) SpeedStat(
                    avg = ti.getAvgUs(decodeSpeeds),
                    stdev = ti.getStdevUs(decodeSpeeds)
                ) else null,
                rawPrefillUs = ti.prefillUs.toList(),
                rawDecodeUs = ti.decodeUs.toList()
            )
        } catch (e: TimeoutCancellationException) {
            watchJob.cancel()
            service.stopBenchmark()
            throw Exception("Benchmark timed out after 5 minutes")
        }
    } finally {
        service.release()
    }
}
