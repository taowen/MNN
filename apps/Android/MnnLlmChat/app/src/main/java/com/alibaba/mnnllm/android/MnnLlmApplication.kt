// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android

import android.app.Application
import android.system.Os
import android.util.Log
import com.facebook.stetho.Stetho
import com.facebook.stetho.dumpapp.DumperPlugin
import com.alibaba.mnnllm.android.debug.ModelListDumperPlugin
import com.alibaba.mnnllm.android.debug.LoggerDumperPlugin
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mnnllm.android.utils.CrashUtil
import com.alibaba.mnnllm.android.utils.CurrentActivityTracker
import com.alibaba.mnnllm.android.utils.TimberConfig
import timber.log.Timber
import android.content.Context
import com.jaredrummler.android.device.DeviceName
import com.alibaba.mnnllm.android.modelist.ModelListManager

class MnnLlmApplication : Application() {
    
    override fun onCreate() {
        super.onCreate()
        ApplicationProvider.set(this)
        CrashUtil.init(this)
        instance = this
        DeviceName.init(this)
        setupQnnEnvironment()

        // Initialize CurrentActivityTracker
        CurrentActivityTracker.initialize(this)

        // Initialize Timber logging based on configuration
        TimberConfig.initialize(this)
        
        // Set context for ModelListManager (enables auto-initialization)
        ModelListManager.setContext(getInstance())

        if (BuildConfig.DEBUG) {
            val initializer = Stetho.newInitializerBuilder(this)
                .enableDumpapp {
                    Stetho.DefaultDumperPluginsBuilder(this)
                        .provide(ModelListDumperPlugin())
                        .provide(LoggerDumperPlugin())
                        .finish()
                }
                .enableWebKitInspector(Stetho.defaultInspectorModulesProvider(this))
                .build()
            Stetho.initialize(initializer)
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

    companion object {
        private const val TAG = "MnnLlmApplication"
        private lateinit var instance: MnnLlmApplication

        fun getAppContext(): Context {
            return instance.applicationContext
        }
        
        /**
         * Get the application instance for accessing Timber configuration
         */
        fun getInstance(): MnnLlmApplication {
            return instance
        }
    }
}
