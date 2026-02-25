#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <cstdlib>
#include <ctime>

#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include "core/MNNFileUtils.h"

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

static void saveInputOutputs(const MNN::Express::Module::Info* info, std::vector<MNN::Express::VARP> inputs, std::vector<MNN::Express::VARP> outputs, const std::string & outputDir, int index) {
    MNN_ASSERT(info->inputNames.size() == inputs.size());
    MNN_ASSERT(info->outputNames.size() == outputs.size());
    for (int i=0; i<info->inputNames.size(); ++i) {
        inputs[i].fix(MNN::Express::VARP::CONSTANT);
        inputs[i]->setName(info->inputNames[i]);
    }
    for (int i=0; i<info->outputNames.size(); ++i) {
        outputs[i]->setName(info->outputNames[i]);
    }
    auto subDir = MNNFilePathConcat(outputDir, std::to_string(index));
    if (!(MNNCreateDir(subDir.c_str()))) {
        MNN_PRINT("Failed to create dir %s.\n", outputDir.c_str());
    }

    std::string inputPath = MNNFilePathConcat(subDir, "input.mnn");
    std::string outputPath = MNNFilePathConcat(subDir, "output.mnn");
    MNN::Express::Variable::save(inputs, inputPath.c_str());
    MNN::Express::Variable::save(outputs, outputPath.c_str());
    MNN_PRINT("Successfully generate %s and %s.\n", inputPath.c_str(), outputPath.c_str());
}

static void generateForAudio(const std::string& modelPath, const std::string& outputDir, const std::string& jsonPath) {
    int nWindow = 50;
    {
        std::ifstream ifs(jsonPath);
        if (!ifs.is_open()) {
            MNN_ERROR("Failed to open JSON config file: %s.\n", jsonPath.c_str());
            return;
        }
        rapidjson::IStreamWrapper isw(ifs);
        rapidjson::Document doc;
        doc.ParseStream(isw);

        if (doc.HasParseError() || !doc.IsObject()) {
            MNN_ERROR("Failed to parse JSON config file: %s.\n", jsonPath.c_str());
            return;
        }

        if (doc.HasMember("n_window") && doc["n_window"].IsInt()) {
            nWindow = doc["n_window"].GetInt();
        }
    }

    int chunkFrames = nWindow * 2;  // 100
    int tokensPerChunk = chunkFrames;
    for (int i = 0; i < 3; i++) {
        tokensPerChunk = (tokensPerChunk + 1) / 2;
    }
    // tokensPerChunk = 13 for nWindow=50

    MNN_PRINT("Audio encoder: n_window=%d, chunk_frames=%d, tokens_per_chunk=%d\n",
              nWindow, chunkFrames, tokensPerChunk);

    // Load Model
    MNN::ScheduleConfig config;
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr(MNN::Express::Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setExternalFile((modelPath + ".weight").c_str());

    std::shared_ptr<MNN::Express::Module> net;
    net.reset(MNN::Express::Module::load({}, {}, modelPath.c_str(), rtmgr), MNN::Express::Module::destroy);
    if (net == nullptr) {
        MNN_ERROR("Failed to load audio model: %s\n", modelPath.c_str());
        return;
    }

    auto info = net->getInfo();
    MNN_PRINT("Audio model inputs: %d, outputs: %d\n", (int)info->inputNames.size(), (int)info->outputNames.size());
    for (int i = 0; i < info->inputNames.size(); i++) {
        MNN_PRINT("  input[%d]: %s\n", i, info->inputNames[i].c_str());
    }
    for (int i = 0; i < info->outputNames.size(); i++) {
        MNN_PRINT("  output[%d]: %s\n", i, info->outputNames[i].c_str());
    }

    // Create test inputs for single chunk: input_features [1, 128, chunk_frames], attention_mask [1, tokens_per_chunk, tokens_per_chunk]
    std::vector<MNN::Express::VARP> inputs;

    MNN::Express::VARP inputFeatures = MNN::Express::_Input({1, 128, chunkFrames}, MNN::Express::NCHW, halide_type_of<float>());
    float* featData = inputFeatures->writeMap<float>();
    for (int i = 0; i < 128 * chunkFrames; i++) {
        featData[i] = (float)(rand()) / RAND_MAX;
    }
    inputs.push_back(inputFeatures);

    MNN::Express::VARP attentionMask = MNN::Express::_Input({1, tokensPerChunk, tokensPerChunk}, MNN::Express::NCHW, halide_type_of<float>());
    ::memset(attentionMask->writeMap<float>(), 0, tokensPerChunk * tokensPerChunk * sizeof(float));
    inputs.push_back(attentionMask);

    // Forward
    auto outputs = net->onForward(inputs);
    if (outputs.empty()) {
        MNN_ERROR("Audio model forward failed!\n");
        return;
    }

    MNN_PRINT("Output shape: ");
    auto outputInfo = outputs[0]->getInfo();
    for (int i = 0; i < outputInfo->dim.size(); i++) {
        MNN_PRINT("%d ", outputInfo->dim[i]);
    }
    MNN_PRINT("\n");

    saveInputOutputs(info, inputs, outputs, outputDir, 1);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./generateAudioIO model/config.json outputDir\n");
        MNN_PRINT("This program generates IO test data for the audio encoder model.\n");
        return 1;
    }

    srand(time(NULL));

    std::string modelDir = argv[1];
    std::string modelPath = modelDir + "/audio.mnn";
    std::string llmConfigPath = modelDir + "/llm_config.json";
    std::string outputDir = argv[2];

    FUNC_PRINT_ALL(modelPath.c_str(), s);
    FUNC_PRINT_ALL(llmConfigPath.c_str(), s);

    if (!(MNNCreateDir(outputDir.c_str()))) {
        MNN_PRINT("Failed to create dir %s.\n", outputDir.c_str());
    }

    generateForAudio(modelPath, outputDir, llmConfigPath);

    return 0;
}
