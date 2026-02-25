//
// Created by ruoyi.sjd on 2024/01/12.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "diffusion_session.h"
#include "mls_log.h"
#include <memory>
#include <utility>
mls::DiffusionSession::DiffusionSession(std::string resource_path, int memory_mode):
                                        resource_path_(std::move(resource_path)),
                                        memory_mode_(memory_mode){
    MNN_DEBUG("diffusion session init resource_path_: %s memory_mode: %d (diffusion not built)", resource_path_.c_str(), memory_mode);
}

void mls::DiffusionSession::Run(const std::string &prompt,
                                const std::string &image_path,
                                int iter_num,
                                int random_seed,
                                const std::function<void(int)>& progressCallback) {
    if (!diffusion_) {
        MNN_ERROR("diffusion not available");
        return;
    }
    if (!loaded_) {
        this->diffusion_->load();
        loaded_ = true;
    }
    this->diffusion_->run(prompt, image_path, iter_num, random_seed, progressCallback);
    if (memory_mode_ != 1) {
        loaded_ = false;
    }
}
