package com.alibaba.mnnllm.android.debug

import android.content.Context
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.Netty
import io.ktor.server.netty.NettyApplicationEngine
import io.ktor.server.plugins.contentnegotiation.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import timber.log.Timber

object DebugServer {
    private const val TAG = "DebugServer"
    private const val PORT = 18888

    private var server: EmbeddedServer<NettyApplicationEngine, NettyApplicationEngine.Configuration>? = null
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    fun start(context: Context) {
        if (server != null) return
        val appContext = context.applicationContext
        scope.launch {
            try {
                server = embeddedServer(Netty, port = PORT, host = "0.0.0.0") {
                    install(ContentNegotiation) {
                        json(Json {
                            prettyPrint = true
                            ignoreUnknownKeys = true
                            encodeDefaults = true
                        })
                    }
                    debugRoutes(appContext)
                }.start(wait = false)
                Timber.tag(TAG).i("Debug server started on port $PORT")
            } catch (e: Exception) {
                Timber.tag(TAG).e(e, "Failed to start debug server")
            }
        }
    }

    fun stop() {
        server?.stop(1000, 2000)
        server = null
        Timber.tag(TAG).i("Debug server stopped")
    }
}
