//
// Created by ruoyi.sjd on 2024/12/25.
//

#pragma once
#include <android/log.h>
#include <cstdio>
#include <cstdarg>
#include <mutex>

#define LOG_TAG_DEBUG "MNN_DEBUG"
#define LOG_TAG_ERROR "MNN_ERROR"

inline FILE* mls_get_log_file() {
    static FILE* f = nullptr;
    static std::once_flag flag;
    std::call_once(flag, []() {
        f = fopen("/data/data/com.alibaba.mnnllm.android/files/mnn_debug.log", "w");
    });
    return f;
}

inline void mls_log_to_file(const char* tag, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    __android_log_vprint(ANDROID_LOG_DEBUG, tag, fmt, args);
    va_end(args);

    FILE* f = mls_get_log_file();
    if (f) {
        va_list args2;
        va_start(args2, fmt);
        fprintf(f, "[%s] ", tag);
        vfprintf(f, fmt, args2);
        fprintf(f, "\n");
        fflush(f);
        va_end(args2);
    }
}

#define MNN_DEBUG(...) mls_log_to_file(LOG_TAG_DEBUG, __VA_ARGS__)
#ifndef MNN_ERROR
#define MNN_ERROR(...) mls_log_to_file(LOG_TAG_ERROR, __VA_ARGS__)
#endif
#define LOGD(...) mls_log_to_file(LOG_TAG_DEBUG, __VA_ARGS__)
#define LOGE(...) mls_log_to_file(LOG_TAG_ERROR, __VA_ARGS__)
