// Created by ruoyi.sjd on 2025/5/9.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.utils
import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Process
import android.util.Log
import androidx.core.content.FileProvider
import java.io.*
import java.text.SimpleDateFormat
import java.util.*
import kotlin.system.exitProcess

@SuppressLint("StaticFieldLeak")
object CrashUtil : Thread.UncaughtExceptionHandler {
    private const val TAG = "CrashUtil"
    private var defaultHandler: Thread.UncaughtExceptionHandler? = null
    private lateinit var context: Context
    private val dateFormat = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
    private lateinit var crashDir: File

    init {
        try {
            System.loadLibrary("mnnllmapp")
        } catch (e: Throwable) {
            Log.w(TAG, "Native crash library load failed", e)
        }
    }

    private external fun initNative(crashDir: String)

    fun init(ctx: Context) {
        context = ctx.applicationContext
        defaultHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler(this)
        crashDir = File(context.filesDir, "crash").apply { if (!exists()) mkdirs() }
        //seems not work fine for native crash
//        initNative(crashDir.absolutePath)
    }

    @JvmStatic
    fun onNativeCrash() {
        val ts = dateFormat.format(Date())
        val file = File(crashDir, "native_$ts.log")
        FileWriter(file).use { fw ->
            fw.appendLine("===== Native Crash =====")
            fw.appendLine()
            fw.appendLine("===== Logcat (main) =====")
            fw.append(getLogcat("-d", "-v", "time"))
            fw.appendLine()
            fw.appendLine("===== Logcat (crash buffer) =====")
            fw.append(getLogcat("-d", "-b", "crash", "-v", "time"))
        }
        Log.i(TAG, "Saved native crash log to ${file.absolutePath}")
    }

    override fun uncaughtException(thread: Thread, ex: Throwable) {
        Log.e(TAG, "UNCAUGHT EXCEPTION on thread ${thread.name}", ex)
        saveJavaCrash(thread, ex)
        defaultHandler?.uncaughtException(thread, ex) ?: run {
            Process.killProcess(Process.myPid())
            exitProcess(10)
        }
    }


    private fun saveJavaCrash(thread: Thread, ex: Throwable) {
        Log.e(TAG, "saveJavaCrash: saving crash from thread ${thread.name}: ${ex.message}")
        val ts = dateFormat.format(Date())
        val file = File(crashDir, "crash_$ts.log")
        FileWriter(file).use { fw ->
            fw.appendLine("===== Java Crash =====")
            fw.appendLine("Thread: ${thread.name}")
            fw.appendLine(Log.getStackTraceString(ex))
            fw.appendLine("\n===== Logcat (main) =====")
            fw.append(getLogcat("-d", "-v", "time"))
            fw.appendLine("\n===== Logcat (crash buffer) =====")
            fw.append(getLogcat("-d", "-b", "crash", "-v", "time"))
        }
        Log.i(TAG, "Saved crash log to ${file.absolutePath}")
    }

    private fun getLogcat(vararg args: String): String {
        val sb = StringBuilder()
        try {
            val process = Runtime.getRuntime().exec(arrayOf("logcat", *args))
            val reader = BufferedReader(InputStreamReader(process.inputStream))
            val startTime = System.currentTimeMillis()
            val timeoutMs = 3000L
            while (System.currentTimeMillis() - startTime < timeoutMs) {
                if (reader.ready()) {
                    val line = reader.readLine() ?: break
                    sb.appendLine(line)
                } else {
                    Thread.sleep(10)
                }
            }
            process.destroy()
        } catch (e: Exception) {
            sb.appendLine("Failed to collect logcat: ${e.message}")
        }
        return sb.toString()
    }

    fun shareLatestCrash(ctx: Context) {
        val files = crashDir.listFiles() ?: return
        val latest = files.maxByOrNull { it.lastModified() } ?: return
        val uri: Uri = FileProvider.getUriForFile(
            ctx,
            "${ctx.packageName}.fileprovider",
            latest
        )
        ClipboardUtils.copyToClipboard(ctx, latest.readText())
        UiUtils.showToast(ctx, "Crash report copied to clipboard")
        val shareIntent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        ctx.startActivity(Intent.createChooser(shareIntent, "Share Crash Report"))
    }

    fun hasCrash(): Boolean {
        return crashDir.listFiles()?.isNotEmpty() ?: false
    }
}