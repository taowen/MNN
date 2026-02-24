//
//  QNNBackend.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNBackend.hpp"
#include "core/MNNFileUtils.h"
#include "QnnTypeMacros.hpp"
// #define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "core/FileLoader.hpp"

// #define QNN_PROFILE_OP
// #define QNN_PROFILE_SUMMARIZE
// #define QNN_VERBOSE
#ifdef ENABLE_QNN_CONVERT_MODE
#define QNN_FORWARD_TYPE MNN_CONVERT_QNN
#else
#define QNN_FORWARD_TYPE MNN_FORWARD_NN
#endif

namespace MNN {
namespace QNN {

// MemObj subclass that frees host memory allocated for QNN output tensors.
// QNN output tensors need host memory so Tensor::clone() works across sub-modules.
class QnnHostMemObj : public Backend::MemObj {
public:
    QnnHostMemObj(void* ptr) : mPtr(ptr) {}
    ~QnnHostMemObj() override {
        if (mPtr) { free(mPtr); mPtr = nullptr; }
    }
private:
    void* mPtr;
};

struct QnnContext {
    QNN_INTERFACE_VER_TYPE interface{};
    QNN_SYSTEM_INTERFACE_VER_TYPE systemInterface{};
    Qnn_LogHandle_t logHandle = nullptr;
    Qnn_BackendHandle_t backendHandle = nullptr;
    Qnn_DeviceHandle_t deviceHandle = nullptr;
    int soc_id;
    int dsp_arch;
};

static QnnContext gContext;
static std::mutex gQnnContextMutex;

static void createQnnContext(){
    std::lock_guard<std::mutex> lck(gQnnContextMutex);
    QNN_INTERFACE_VER_TYPE qnnInterface{};
#ifndef ENABLE_QNN_CONVERT_MODE
    {
        QnnInterface_t** interfaceProviders = nullptr;
        uint32_t numProviders = 0;
        if (QNN::QnnInterface_getProviders((const QnnInterface_t***)&interfaceProviders, &numProviders) != QNN_SUCCESS) {
            MNN_PRINT("MNN_QNN: Failed to call 'QnnInterface_getProviders'.\n");
            return;
        }
        if (interfaceProviders == nullptr) {
            MNN_PRINT("MNN_QNN: Failed to get interface providers: null interface providers received.\n");
            return;
        }
        if (numProviders == 0) {
            MNN_PRINT("MNN_QNN: Failed to get interface providers: 0 interface providers.\n");
            return;
        }
        bool foundValidInterface = false;
        for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
            if (QNN_API_VERSION_MAJOR == interfaceProviders[pIdx]->apiVersion.coreApiVersion.major &&
                QNN_API_VERSION_MINOR <= interfaceProviders[pIdx]->apiVersion.coreApiVersion.minor) {
                foundValidInterface = true;
                qnnInterface = interfaceProviders[pIdx]->QNN_INTERFACE_VER_NAME;
                break;
            }
        }
        if (!foundValidInterface) {
            MNN_PRINT("MNN_QNN: Failed to find a valid interface.\n");
            return;
        }
    }
#else
    qnnInterface = QNN::gQnnConvertorInterface;
#endif

    // Create Log.
    Qnn_LogHandle_t logHandle = nullptr;
    {
        QnnLog_Callback_t logCallback = nullptr;
        if ((QNN_GET_ERROR_CODE(qnnInterface.logCreate(logCallback, QNN_LOG_LEVEL_ERROR, &logHandle)) != QNN_SUCCESS) ||
            (logHandle == nullptr)) {
            MNN_PRINT("MNN_QNN: Failed to initialize logging in the backend.\n");
            return;
        }
    }

    // Create Backend.
    Qnn_BackendHandle_t backendHandle = nullptr;
    {
        const QnnBackend_Config_t** backendConfig = nullptr;
        if ((QNN_GET_ERROR_CODE(qnnInterface.backendCreate(logHandle, backendConfig, &backendHandle)) != QNN_SUCCESS) ||
            (backendHandle == nullptr)) {
            MNN_PRINT("MNN_QNN: Failed to create the backend.\n");
            return;
        }
    }

    // Create Device.
    Qnn_DeviceHandle_t deviceHandle = nullptr;
    QnnHtpDevice_Arch_t dspArch = QNN_HTP_DEVICE_ARCH_NONE;
    uint32_t socId = 0;
    {
        // Check whether the device API is supported.
        bool supportDevice = QNN::checkCapability(qnnInterface, QNN_PROPERTY_GROUP_DEVICE);
        if (supportDevice) {
            const QnnDevice_Config_t ** deviceConfig = nullptr;
            auto qnnStatus = qnnInterface.deviceCreate(logHandle, deviceConfig, &deviceHandle);
            if (qnnStatus != QNN_SUCCESS || deviceHandle == nullptr) {
                // Newer SoCs (e.g. SM8750) require explicit SoC config
                MNN_PRINT("MNN_QNN: Default device creation failed (error:%lu), retrying with SoC config\n", (unsigned long)qnnStatus);
                deviceHandle = nullptr;

                QnnHtpDevice_CustomConfig_t htpSocConfig = {};
                htpSocConfig.option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
                htpSocConfig.socModel = 0;

                // Try known SoC models from newest to oldest
                static const uint32_t knownSocs[] = {
                    69,  // SM8750 (Snapdragon 8 Elite, HTP V79)
                    57,  // SM8650 (Snapdragon 8 Gen 3, HTP V75)
                    43,  // SM8550 (Snapdragon 8 Gen 2, HTP V73)
                    36,  // SM8450 (Snapdragon 8 Gen 1, HTP V69)
                };

                QnnDevice_Config_t socDeviceConfig = QNN_DEVICE_CONFIG_INIT;
                socDeviceConfig.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
                socDeviceConfig.customConfig = &htpSocConfig;
                const QnnDevice_Config_t* deviceConfigArray[] = {&socDeviceConfig, nullptr};

                for (uint32_t soc : knownSocs) {
                    htpSocConfig.socModel = soc;
                    qnnStatus = qnnInterface.deviceCreate(logHandle, deviceConfigArray, &deviceHandle);
                    if (qnnStatus == QNN_SUCCESS && deviceHandle != nullptr) {
                        MNN_PRINT("MNN_QNN: Device created with SoC model %u\n", soc);
                        break;
                    }
                    deviceHandle = nullptr;
                }

                if (deviceHandle == nullptr) {
                    MNN_PRINT("MNN_QNN: Failed to create device with any known SoC config, error:%lu\n", (unsigned long)qnnStatus);
                    return;
                }
            }

            if (qnnInterface.deviceGetPlatformInfo == nullptr) {
                MNN_PRINT("[Warning]: No QnnDevice_getPlatformInfo API");
            } else {
                // QnnDevice_PlatformInfo_t platformInfo = QNN_DEVICE_PLATFORM_INFO_INIT;
                const QnnDevice_PlatformInfo_t* backendPlatformInfoPtr = nullptr;
                qnnStatus = qnnInterface.deviceGetPlatformInfo(logHandle, &backendPlatformInfoPtr);
                if(qnnStatus != QNN_SUCCESS || backendPlatformInfoPtr == nullptr) {
                    MNN_PRINT("[Warning]: deviceGetPlatformInfo Failed to query platform info");
                } else {
                    QnnDevice_HardwareDeviceInfo_t* hwDeviceInfo = backendPlatformInfoPtr->v1.hwDevices;
                    dspArch = hwDeviceInfo->v1.deviceInfoExtension->onChipDevice.arch;
                    socId = hwDeviceInfo->v1.deviceInfoExtension->onChipDevice.socModel;
                }
            }
        } else {
            MNN_PRINT("MNN_QNN: Not supporting device API.\n");
            return;
        }
    }

    // Create System Interface
    QNN_SYSTEM_INTERFACE_VER_TYPE systemInterface{};
#ifndef ENABLE_QNN_CONVERT_MODE
    #ifdef MNN_WITH_PLUGIN
    {
        QnnSystemInterface_t** interfaceProviders = nullptr;
        uint32_t numProviders = 0;
        if (QNN::QnnSystemInterface_getProviders((const QnnSystemInterface_t***)&interfaceProviders, &numProviders) != QNN_SUCCESS) {
            MNN_PRINT("MNN_QNN: Failed to call 'QnnInterface_getProviders'.\n");
            return;
        }
        if (interfaceProviders == nullptr) {
            MNN_PRINT("MNN_QNN: Failed to get interface providers: null interface providers received.\n");
            return;
        }
        if (numProviders == 0) {
            MNN_PRINT("MNN_QNN: Failed to get interface providers: 0 interface providers.\n");
            return;
        }
        bool foundValidSystemInterface{false};
        for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
            if (QNN_SYSTEM_API_VERSION_MAJOR == interfaceProviders[pIdx]->systemApiVersion.major &&
                QNN_SYSTEM_API_VERSION_MINOR <= interfaceProviders[pIdx]->systemApiVersion.minor) {
                foundValidSystemInterface = true;
                systemInterface = interfaceProviders[pIdx]->QNN_SYSTEM_INTERFACE_VER_NAME;
                break;
            }
        }
        if (!foundValidSystemInterface) {
            MNN_PRINT("MNN_QNN: Failed to find a valid interface.\n");
            return;
        }
    }
    #endif
#else
    systemInterface = QNN::gQnnConvertorSystemInterface;
#endif


    QNN::gContext.interface = qnnInterface;
    QNN::gContext.systemInterface = systemInterface;
    QNN::gContext.backendHandle = backendHandle;
    QNN::gContext.deviceHandle = deviceHandle;
    QNN::gContext.logHandle = logHandle;
    QNN::gContext.soc_id = socId;
    QNN::gContext.dsp_arch = dspArch;
}

#ifdef QNN_PROFILE_SUMMARIZE
static std::string getOpTypeFromName(const std::string& nodeName) {
    // The pattern is usually "OpType_..."
    size_t pos = nodeName.find('_');
    if (pos != std::string::npos) {
        return nodeName.substr(0, pos);
    }
    // Fallback for names without '_', like "Input OpId_2 (cycles)"
    pos = nodeName.find(' ');
    if (pos != std::string::npos) {
        return nodeName.substr(0, pos);
    }
    // If no delimiter is found, return the whole name as the type
    return nodeName;
}
#endif

static void createProfileHandle(const QNN_INTERFACE_VER_TYPE& interface, const Qnn_BackendHandle_t& backend_handle, Qnn_ProfileHandle_t* profile_handle_ptr) {
    #if defined(QNN_PROFILE_SUMMARIZE) || defined(QNN_PROFILE_OP)
    if (*profile_handle_ptr == nullptr) {
        // set QNN_PROFILE_LEVEL_DETAILED
        QnnProfile_Level_t profileLevel = QNN_PROFILE_LEVEL_DETAILED;
        MNN_PRINT("[QNN Profile] Creating QNN Profile Handle with DETAILED level.\n");
        auto profile_err = interface.profileCreate(backend_handle, profileLevel, profile_handle_ptr);
        if (profile_err != QNN_SUCCESS || *profile_handle_ptr == nullptr) {
            MNN_ERROR("[QNN Profile] Failed to create QNN Profile Handle, error: %d\n", (int)profile_err);
            *profile_handle_ptr = nullptr;
        }
    }
    #endif
}

static void doProfile(const QNN_INTERFACE_VER_TYPE& interface, const Qnn_ProfileHandle_t& profile_handle) {
#ifdef QNN_PROFILE_OP
    if (profile_handle) {
        uint32_t numTopLevelEvents = 0;
        const QnnProfile_EventId_t* topLevelEvents = nullptr;

        auto get_err = interface.profileGetEvents(profile_handle, &topLevelEvents, &numTopLevelEvents);
        if (get_err != QNN_SUCCESS) {
            MNN_PRINT("[QNN Profile] Failed to get top-level events. Error: %d\n", (int)get_err);
            return;
        }

        MNN_PRINT("\n--- QNN Node-level Performance Report ---\n");
        bool foundNodeData = false;

        for (uint32_t i = 0; i < numTopLevelEvents; ++i) {
            QnnProfile_EventData_t eventData = QNN_PROFILE_EVENT_DATA_INIT;
            interface.profileGetEventData(topLevelEvents[i], &eventData);

            if (eventData.type) {
                MNN_PRINT("Found EXECUTE event. Total time: %llu us. Querying sub-events...\n", (unsigned long long)eventData.value);

                uint32_t numSubEvents = 0;
                const QnnProfile_EventId_t* subEvents = nullptr;

                // 3. GetSubEvents
                auto get_sub_err = interface.profileGetSubEvents(topLevelEvents[i], &subEvents, &numSubEvents);
                if (get_sub_err != QNN_SUCCESS) {
                    MNN_PRINT("[QNN Profile] Failed to get sub-events for EXECUTE event. Error: %d\n", (int)get_sub_err);
                    continue;
                }

                for (uint32_t j = 0; j < numSubEvents; ++j) {
                    QnnProfile_EventData_t subEventData = QNN_PROFILE_EVENT_DATA_INIT;
                    interface.profileGetEventData(subEvents[j], &subEventData);

                    if (subEventData.type == QNN_PROFILE_EVENTTYPE_NODE) {
                        foundNodeData = true;
                        const char* nodeName = subEventData.identifier;
                        uint64_t value = subEventData.value;

                        switch (subEventData.unit) {
                            case QNN_PROFILE_EVENTUNIT_MICROSEC:
                                MNN_PRINT("Node: %-45s | Time: %10llu us (%.3f ms)\n",
                                        nodeName, (unsigned long long)value, (double)value / 1000.0);
                                break;
                            case QNN_PROFILE_EVENTUNIT_CYCLES:
                                MNN_PRINT("Node: %-45s | Cycles: %.2f*10^6\n", nodeName, (double)value / 1000000.0);
                                break;
                            // ... other dealing ...
                            default:
                                MNN_PRINT("Node: %-45s | Value: %10llu (Unit: %u - Unknown)\n",
                                        nodeName, (unsigned long long)value, subEventData.unit);
                                break;
                        }
                    }
                }
            }
        }

        if (!foundNodeData) {
            MNN_PRINT("No node-specific performance data found. Please ensure you have set:\n");
            MNN_PRINT("1. Profile level to QNN_PROFILE_LEVEL_DETAILED.\n");
            MNN_PRINT("2. HTP graph config with QNN_HTP_GRAPH_CONFIG_OPTION_PERF_PROFILE (if available).\n");
        }
        MNN_PRINT("-----------------------------------------\n");
    }
#endif

#ifdef QNN_PROFILE_SUMMARIZE
    if (profile_handle) {
        std::map<std::string, uint64_t> opCycleStats;
        uint64_t totalNodeCycles = 0;

        uint32_t numTopLevelEvents = 0;
        const QnnProfile_EventId_t* topLevelEvents = nullptr;

        auto get_err = interface.profileGetEvents(profile_handle, &topLevelEvents, &numTopLevelEvents);
        if (get_err != QNN_SUCCESS) {
            MNN_PRINT("[QNN Profile] Failed to get top-level events. Error: %d\n", (int)get_err);
            return;
        }

        for (uint32_t i = 0; i < numTopLevelEvents; ++i) {
            QnnProfile_EventData_t eventData = QNN_PROFILE_EVENT_DATA_INIT;
            interface.profileGetEventData(topLevelEvents[i], &eventData);

            if (eventData.type) { // == QNN_PROFILE_EVENTTYPE_EXECUTE) {
                uint32_t numSubEvents = 0;
                const QnnProfile_EventId_t* subEvents = nullptr;
                auto get_sub_err = interface.profileGetSubEvents(topLevelEvents[i], &subEvents, &numSubEvents);
                if (get_sub_err != QNN_SUCCESS) continue;

                for (uint32_t j = 0; j < numSubEvents; ++j) {
                    QnnProfile_EventData_t subEventData = QNN_PROFILE_EVENT_DATA_INIT;
                    interface.profileGetEventData(subEvents[j], &subEventData);

                    if (subEventData.type == QNN_PROFILE_EVENTTYPE_NODE) {
                        if (subEventData.identifier) {
                            std::string opType = getOpTypeFromName(subEventData.identifier);
                            opCycleStats[opType] += subEventData.value;
                            totalNodeCycles += subEventData.value;
                        }
                    }
                }
            }
        }

        if (!opCycleStats.empty()) {
            MNN_PRINT("\n--- QNN Operator-wise Performance Summary ---\n");
            MNN_PRINT("%-20s | %15s | %s\n", "Operator Type", "Total Cycles", "Percentage");
            MNN_PRINT("--------------------------------------------------\n");

            std::vector<std::pair<std::string, uint64_t>> sortedStats(opCycleStats.begin(), opCycleStats.end());
            std::sort(sortedStats.begin(), sortedStats.end(), [](const std::pair<std::string, uint64_t>& a, const std::pair<std::string, uint64_t>& b) {
                return a.second > b.second; // sort by large -> small
            });

            for (const auto& pair : sortedStats) {
                double percentage = (totalNodeCycles > 0) ? ((double)pair.second * 100.0 / totalNodeCycles) : 0.0;
                MNN_PRINT("%-20s | %15llu | %.2f%%\n", pair.first.c_str(), pair.second, percentage);
            }
            MNN_PRINT("--------------------------------------------------\n");
            MNN_PRINT("%-20s | %15llu | 100.00%%\n", "Total", totalNodeCycles);
        }
    }
    // =========================================================
#endif
}
}
}

#ifdef MNN_WITH_PLUGIN

#include "MNN/plugin/PluginShapeInference.hpp"
#include "MNN/plugin/PluginContext.hpp"
#include "MNN/plugin/PluginKernel.hpp"
#include "shape/SizeComputer.hpp"

namespace MNN {
namespace plugin {

namespace shape_inference {
class PluginShapeRaw : public InferShapeKernel {
public:
    bool compute(InferShapeContext* ctx) override;
};
static bool computeIndex(PluginContext* ctx, int & index) {
    const std::vector<Tensor *> & inputs = ctx->inputs();
    auto attrAllShape = ctx->getAttr("allInputShape");
    if (nullptr == attrAllShape || nullptr == attrAllShape->list() || nullptr == attrAllShape->list()->i()) {
        MNN_ERROR("MNN_QNN: Incorrect Plugin Op, can't find 'allInputShape' attr.\n");
        return false;
    }
    int dimSum = 0;
    for (int i = 0; i < inputs.size(); i++) {
        auto inputDim = inputs[i]->dimensions();
        dimSum += inputDim;
    }
    if (0 == dimSum) {
        // Scalar
        index = 0;
        return true;
    }
    auto indexNumber = attrAllShape->list()->i()->size() / dimSum;
    for (int si=0; si<indexNumber; ++si) {
        auto dstSi = attrAllShape->list()->i()->data() + si * dimSum;
        bool valid = true;
        for (int i=0; i<inputs.size(); ++i) {
            auto inputDim = inputs[i]->dimensions();
            for (int j = 0; j < inputDim; j++) {
                if (inputs[i]->length(j) != dstSi[j]) {
                    valid = false;
                    break;
                }
            }
            dstSi += inputDim;
            if (!valid) {
                break;
            }
        }
        if (valid) {
            index = si;
            return true;
        }
    }
    return false;
}

bool PluginShapeRaw::compute(InferShapeContext* ctx) {
    if (ctx->hasAttr("op")) {
        auto attr = ctx->getAttr("op");
        if (nullptr != attr->tensor() && nullptr != attr->tensor()->int8s()) {
            auto realop = flatbuffers::GetRoot<Op>(attr->tensor()->int8s()->data());
            return SizeComputer::computeOutputSize(realop, ctx->inputs(), ctx->outputs());
        }
    } else {
        int shapeIndex = 0;
        if (!(computeIndex(ctx, shapeIndex))) {
            MNN_ERROR("MNN_QNN: Failed to compute shape for Plugin Op.\n");
            return false;
        }

        std::string prefix = "o_" + std::to_string(shapeIndex) + "_";
        for (int i=0; i<ctx->outputs().size(); ++i) {
            auto dst = ctx->output(i);
            std::string key = prefix + std::to_string(i);
            auto attr = ctx->getAttr(key.c_str());

            if (nullptr == attr || nullptr == attr->tensor()) {
                MNN_ERROR("MNN_QNN: Failed to find raw shape %s.\n", key.c_str());
                return false;
            }
            auto blob = attr->tensor();
            dst->setType(blob->dataType());
            if (nullptr != blob->dims()) {
                dst->buffer().dimensions = blob->dims()->size();
                for (int j=0; j<blob->dims()->size(); ++j) {
                    dst->setLength(j, blob->dims()->data()[j]);
                }
            } else {
                dst->buffer().dimensions = 0;
            }
            TensorUtils::getDescribe(dst)->dimensionFormat = blob->dataFormat();
        }
        return true;
    }
    return false;
}
}

namespace backend {
static bool freeQnnTensor(Qnn_Tensor_t &tensor) {
  // free all pointer allocations in struct
  free((void *)QNN_TENSOR_GET_NAME(tensor));
  free(QNN_TENSOR_GET_DIMENSIONS(tensor));
  free(QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(tensor));

  auto quant    = QNN_TENSOR_GET_QUANT_PARAMS(tensor);
  auto encoding = quant.quantizationEncoding;
  if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    free(quant.axisScaleOffsetEncoding.scaleOffset);
  } else if (encoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
    free(quant.bwAxisScaleOffsetEncoding.scales);
    if (quant.bwAxisScaleOffsetEncoding.offsets != nullptr) {
      free(quant.bwAxisScaleOffsetEncoding.offsets);
    }
  }
  return true;
}

static bool freeQnnTensors(Qnn_Tensor_t *&tensors, uint32_t numTensors) {
  // free all pointer allocations in struct
  for (size_t i = 0; i < numTensors; i++) {
    freeQnnTensor(tensors[i]);
  }
  free(tensors);

  return true;
}

struct GraphInfo {
  Qnn_GraphHandle_t graph;
  char *graphName;
  Qnn_Tensor_t *inputTensors;
  uint32_t numInputTensors;
  Qnn_Tensor_t *outputTensors;
  uint32_t numOutputTensors;
};

static bool deepCopyQnnTensorInfo(Qnn_Tensor_t *dst, const Qnn_Tensor_t *src) {
  if (nullptr == dst || nullptr == src) {
    return false;
  }
  // set tensor.version before using QNN_TENSOR_SET macros, as they require the version to be set
  // to correctly assign values
  dst->version           = src->version;
  const char *tensorName = QNN_TENSOR_GET_NAME(src);
  if (!tensorName) {
    QNN_TENSOR_SET_NAME(dst, nullptr);
  } else {
    QNN_TENSOR_SET_NAME(dst, ::strdup(tensorName));
  }
  QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
  QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
  QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
  QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));
  dst->v1.memType = QNN_TENSORMEMTYPE_RAW;
  Qnn_QuantizeParams_t qParams = QNN_QUANTIZE_PARAMS_INIT;
  qParams.encodingDefinition   = QNN_TENSOR_GET_QUANT_PARAMS(src).encodingDefinition;
  qParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding ==
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    qParams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
    qParams.scaleOffsetEncoding  = QNN_TENSOR_GET_QUANT_PARAMS(src).scaleOffsetEncoding;
  } else if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding ==
             QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    qParams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
    qParams.axisScaleOffsetEncoding.axis =
        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.axis;
    qParams.axisScaleOffsetEncoding.numScaleOffsets =
        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
    if (QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets > 0) {
      qParams.axisScaleOffsetEncoding.scaleOffset = (Qnn_ScaleOffset_t *)malloc(
          QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets *
          sizeof(Qnn_ScaleOffset_t));
      if (qParams.axisScaleOffsetEncoding.scaleOffset) {
        for (size_t idx = 0;
             idx < QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
             idx++) {
          qParams.axisScaleOffsetEncoding.scaleOffset[idx].scale =
              QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].scale;
          qParams.axisScaleOffsetEncoding.scaleOffset[idx].offset =
              QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].offset;
        }
      }
    }
  }
  QNN_TENSOR_SET_QUANT_PARAMS(dst, qParams);
  QNN_TENSOR_SET_RANK(dst, QNN_TENSOR_GET_RANK(src));
  QNN_TENSOR_SET_DIMENSIONS(dst, nullptr);
  if (QNN_TENSOR_GET_RANK(src) > 0) {
    QNN_TENSOR_SET_DIMENSIONS(dst, (uint32_t *)malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t)));
    if (QNN_TENSOR_GET_DIMENSIONS(dst)) {
      ::memcpy(QNN_TENSOR_GET_DIMENSIONS(dst),
                             QNN_TENSOR_GET_DIMENSIONS(src),
                             QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t));
    }
    if (QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(src)) {
      QNN_TENSOR_SET_IS_DYNAMIC_DIMENSIONS(
          dst, (uint8_t *)malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint8_t)));
      ::memcpy(QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(dst),
                             QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(src),
                             QNN_TENSOR_GET_RANK(src) * sizeof(uint8_t));
    }
  }
  QNN_TENSOR_SET_SPARSE_PARAMS(dst, QNN_TENSOR_GET_SPARSE_PARAMS(src));
  return true;
}

static bool copyTensorsInfo(const Qnn_Tensor_t *tensorsInfoSrc,
                                 Qnn_Tensor_t *&tensorWrappers,
                                 uint32_t tensorsCount) {
  auto returnStatus = true;
  tensorWrappers    = (Qnn_Tensor_t *)calloc(tensorsCount, sizeof(Qnn_Tensor_t));
  if (nullptr == tensorWrappers) {
    MNN_ERROR("Failed to allocate memory for tensorWrappers.");
    return false;
  }
  if (returnStatus) {
    for (size_t tIdx = 0; tIdx < tensorsCount; tIdx++) {
      #ifdef QNN_VERBOSE
      MNN_PRINT("Extracting tensorInfo for tensor Idx: %d.\n", (int) tIdx);
      #endif
      tensorWrappers[tIdx] = QNN_TENSOR_INIT;
      deepCopyQnnTensorInfo(&tensorWrappers[tIdx], &tensorsInfoSrc[tIdx]);
    }
  }
  return returnStatus;
}

template <typename T>
static bool copyGraphsInfoFromSrc(const T *graphInfoSrc, GraphInfo *graphInfoDst) {
  graphInfoDst->graphName = nullptr;
  if (graphInfoSrc->graphName) {
    graphInfoDst->graphName = ::strdup(graphInfoSrc->graphName);
  }
  graphInfoDst->inputTensors    = nullptr;
  graphInfoDst->numInputTensors = 0;
  if (graphInfoSrc->graphInputs) {
    if (!copyTensorsInfo(
            graphInfoSrc->graphInputs, graphInfoDst->inputTensors, graphInfoSrc->numGraphInputs)) {
      return false;
    }
    graphInfoDst->numInputTensors = graphInfoSrc->numGraphInputs;
  }
  graphInfoDst->outputTensors    = nullptr;
  graphInfoDst->numOutputTensors = 0;
  if (graphInfoSrc->graphOutputs) {
    if (!copyTensorsInfo(graphInfoSrc->graphOutputs,
                         graphInfoDst->outputTensors,
                         graphInfoSrc->numGraphOutputs)) {
      return false;
    }
    graphInfoDst->numOutputTensors = graphInfoSrc->numGraphOutputs;
  }
  return true;
}

static bool copyGraphsInfo(const QnnSystemContext_GraphInfo_t *graphsInput,
                                const uint32_t numGraphs,
                                GraphInfo **&graphsInfo) {
  if (!graphsInput) {
    MNN_ERROR("Received nullptr for graphsInput.");
    return false;
  }
  auto returnStatus = true;
  graphsInfo =
      (GraphInfo **)calloc(numGraphs, sizeof(GraphInfo *));
  GraphInfo *graphInfoArr =
      (GraphInfo *)calloc(numGraphs, sizeof(GraphInfo));
  if (nullptr == graphsInfo || nullptr == graphInfoArr) {
    MNN_ERROR("Failure to allocate memory for *graphInfo");
    returnStatus = false;
  }
  if (true == returnStatus) {
    for (size_t gIdx = 0; gIdx < numGraphs; gIdx++) {
      #ifdef QNN_VERBOSE
      MNN_PRINT("Extracting graphsInfo for graph Idx: %d", (int) gIdx);
      #endif
      if (graphsInput[gIdx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        copyGraphsInfoFromSrc(&graphsInput[gIdx].graphInfoV1, &graphInfoArr[gIdx]);
      } else if (graphsInput[gIdx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        copyGraphsInfoFromSrc(&graphsInput[gIdx].graphInfoV3, &graphInfoArr[gIdx]);
      }
      graphsInfo[gIdx] = graphInfoArr + gIdx;
    }
  }
  if (true != returnStatus) {
    MNN_ERROR("Received an ERROR during extractGraphsInfo. Freeing resources.");
    if (graphsInfo) {
      for (uint32_t gIdx = 0; gIdx < numGraphs; gIdx++) {
        if (graphsInfo[gIdx]) {
          if (nullptr != graphsInfo[gIdx]->graphName) {
            free(graphsInfo[gIdx]->graphName);
            graphsInfo[gIdx]->graphName = nullptr;
          }
          freeQnnTensors(graphsInfo[gIdx]->inputTensors,
                                          graphsInfo[gIdx]->numInputTensors);
          freeQnnTensors(graphsInfo[gIdx]->outputTensors,
                                          graphsInfo[gIdx]->numOutputTensors);
        }
      }
      free(*graphsInfo);
    }
    free(graphsInfo);
    graphsInfo = nullptr;
  }
  return true;
}

template<typename T>
static bool copyGraphsInfoFromBinaryInfo(const T & binaryInfo, GraphInfo **& graphsInfo, uint32_t & graphsCount) {
    if (binaryInfo.graphs) {
        if (!copyGraphsInfo(binaryInfo.graphs, binaryInfo.numGraphs, graphsInfo)) {
            MNN_ERROR("MNN_QNN: Failed while copying graphs Info.\n");
            return false;
        }
        graphsCount = binaryInfo.numGraphs;
        return true;
    }
    return false;
}

static bool copyMetadataToGraphsInfo(const QnnSystemContext_BinaryInfo_t *binaryInfo,
                                          GraphInfo **&graphsInfo,
                                          uint32_t &graphsCount) {
    if (nullptr == binaryInfo) {
        MNN_ERROR("MNN_QNN: binaryInfo is nullptr.\n");
        return false;
    }
    graphsCount = 0;
    switch (binaryInfo->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
            return copyGraphsInfoFromBinaryInfo(binaryInfo->contextBinaryInfoV1, graphsInfo, graphsCount);
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
            return copyGraphsInfoFromBinaryInfo(binaryInfo->contextBinaryInfoV2, graphsInfo, graphsCount);
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
            return copyGraphsInfoFromBinaryInfo(binaryInfo->contextBinaryInfoV3, graphsInfo, graphsCount);
        default:
            MNN_ERROR("MNN_QNN: Unrecognized system context binary info version.\n");
            return false;
    }
}

static bool freeGraphsInfo(GraphInfo ***graphsInfo, uint32_t numGraphs) {
  if (graphsInfo == nullptr || *graphsInfo == nullptr) {
    return false;
  }
  for (uint32_t i = 0; i < numGraphs; i++) {
    free((*graphsInfo)[i]->graphName);
    freeQnnTensors((*graphsInfo)[i]->inputTensors, (*graphsInfo)[i]->numInputTensors);
    freeQnnTensors((*graphsInfo)[i]->outputTensors, (*graphsInfo)[i]->numOutputTensors);
  }
  free(**graphsInfo);
  free(*graphsInfo);
  *graphsInfo = nullptr;
  return true;
}

class MMapReader {
private:
    void* mAddr = nullptr;
    file_t mFile = INVALID_FILE;
    size_t mSize = 0;
    void _clean() {
        if (nullptr != mAddr) {
            MNNUnmapFile(mAddr, mSize);
            mAddr = nullptr;
        }
        if (mFile != INVALID_FILE) {
            MNNCloseFile(mFile);
            mFile = INVALID_FILE;
        }
        mSize = 0;
    }
public:
    void* addr() const {
        return mAddr;
    }
    size_t size() const {
        return mSize;
    }
    MMapReader() {
        // Do nothing
    }
    ~MMapReader() {
        _clean();
    }
    bool open(const char* filename) {
        _clean();
        mFile = MNNOpenFile(filename, MNN_FILE_READ);
        mSize = MNNGetFileSize(mFile);
        mAddr = MNNMmapFile(mFile, mSize, true);
        return true;
    }
};

class RawExecutorWrapper {
private:
    Qnn_ContextHandle_t mQnnContextHandle = nullptr;
    const QnnContext_Config_t** mQnnContextConfig = nullptr;
    std::vector<Qnn_GraphHandle_t> mQnnGraphHandleVec = {};
    QnnHtpGraph_CustomConfig_t mQnnHtpGraphCustomConfig{};
    QnnGraph_Config_t mQnnGraphConfig{};
    Qnn_ProfileHandle_t mQnnProfileHandle = nullptr;
    GraphInfo **mGraphsInfo = nullptr;
    uint32_t mGraphCount = 0;
    std::string mPath;
    std::unique_ptr<QNN::QNNPerf> mPerf;

public:
    RawExecutorWrapper() {
        mPerf = QNN::QNNPerf::create(&QNN::gContext.interface);
        mPerf->setPowerConfigBurst();
        mPerf->setRpcLatencyAndPolling();
    }
    ~ RawExecutorWrapper() {
        if (mQnnProfileHandle) {
            QNN::gContext.interface.profileFree(mQnnProfileHandle);
            mQnnProfileHandle = nullptr;
        }
        if (nullptr != mQnnContextHandle) {
            CALL_QNN(QNN::gContext.interface.contextFree(mQnnContextHandle, nullptr));
        }
        freeGraphsInfo(&mGraphsInfo, mGraphCount);
    }

    bool compileModel(const std::string& path, size_t offset, size_t size, const std::vector<std::string>& allGraphName) {
        mPath = path;
        void* buffer = nullptr;
        std::vector<char> bufferVec(size, 0);
        MMapReader reader;
        if (size > 0) {
            buffer = bufferVec.data();
            std::unique_ptr<FileLoader> binaryFile(new FileLoader(path.c_str()));
            binaryFile->offset((int64_t)offset);
            binaryFile->read((char *)buffer, (int64_t)size);
        } else {
            reader.open(path.c_str());
            buffer = reader.addr();
            size = reader.size();
        }

        // 1. Set mGraphsInfo and mGraphCount from the buffer.
        {
            QnnSystemContext_Handle_t systemContextHandle = nullptr;
            if (QNN_SUCCESS != QNN::gContext.systemInterface.systemContextCreate(&systemContextHandle)) {
                MNN_ERROR("Could not create system context handle.");
                return false;
            }
            Qnn_ContextBinarySize_t binarySize = 0;
            const QnnSystemContext_BinaryInfo_t* binaryInfo = nullptr;
            CALL_QNN(QNN::gContext.systemInterface.systemContextGetBinaryInfo(systemContextHandle, buffer, size, &binaryInfo,&binarySize));
            copyMetadataToGraphsInfo(binaryInfo, mGraphsInfo, mGraphCount);
            if (QNN_SUCCESS != QNN::gContext.systemInterface.systemContextFree(systemContextHandle)) {
                MNN_ERROR("Could not free system context handle.");
                return false;
            }
        }

        // 2. Retrieve graphs.
        {
            auto error = QNN::gContext.interface.contextValidateBinary(QNN::gContext.backendHandle, QNN::gContext.deviceHandle, mQnnContextConfig, buffer, size);
            if (QNN_SUCCESS != error) {
                MNN_ERROR("QNN: Failed to validate binary: %d\n", (int) error);
                return false;
            }

            // Create Graph profile
            MNN::QNN::createProfileHandle(QNN::gContext.interface, QNN::gContext.backendHandle, &mQnnProfileHandle);

            CALL_QNN(QNN::gContext.interface.contextCreateFromBinary(QNN::gContext.backendHandle, QNN::gContext.deviceHandle, mQnnContextConfig, buffer, size, &mQnnContextHandle, mQnnProfileHandle));

            mQnnGraphHandleVec.resize(mGraphCount, nullptr);

            std::vector<GraphInfo*> sortedGraphsInfo(mGraphCount, nullptr);
            std::map<std::string, GraphInfo*> graphInfoMap;
            for (int i = 0; i < mGraphCount; ++i) {
                graphInfoMap[mGraphsInfo[i]->graphName] = mGraphsInfo[i];
            }

            for (int i = 0; i < mGraphCount; ++i) {
                auto it = graphInfoMap.find(allGraphName[i]);
                MNN_ASSERT(it != graphInfoMap.end());
                sortedGraphsInfo[i] = it->second;
            }
            for (int i = 0; i < mGraphCount; ++i) {
                mGraphsInfo[i] = sortedGraphsInfo[i];
            }

            for (int i = 0; i < mGraphCount; i++) {
                CALL_QNN(QNN::gContext.interface.graphRetrieve(mQnnContextHandle, mGraphsInfo[i]->graphName, &(mQnnGraphHandleVec[i])));
            }
        }


        return true;
    }

    void invokModel(const std::vector<std::pair<const MNN::Tensor *, std::string>>& inputs, std::vector<std::pair<const MNN::Tensor *, std::string>>& outputs, int shapeIndex) {
        GraphInfo* graph = mGraphsInfo[shapeIndex];
        Qnn_GraphHandle_t qnnGraphHandle = mQnnGraphHandleVec[shapeIndex];

        // MNN_PRINT("%s, Input:%d, output:%d\n", mPath.c_str(), inputs.size(), outputs.size());
        for (int i=0; i<inputs.size(); ++i) {
            auto t = inputs[i].first;
            bool find = false;
            for (int j=0; j<graph->numInputTensors; ++j) {
                auto& dstT = graph->inputTensors[j];
                #ifdef QNN_VERBOSE
                MNN_PRINT("input name: %s %s\n", inputs[i].second.c_str(), dstT.v1.name);
                #endif
                if (inputs[i].second == dstT.v1.name) {
                    dstT.v1.clientBuf.data = t->host<void>();
                    dstT.v1.clientBuf.dataSize = t->usize();
                    find = true;
                    break;
                }
            }
            if (!find) {
                FUNC_PRINT(i);
            }
        }
        for (int i=0; i<outputs.size(); ++i) {
            auto t = outputs[i].first;
            bool find = false;
            for (int j=0; j<graph->numOutputTensors; ++j) {
                auto& dstT = graph->outputTensors[j];
                #ifdef QNN_VERBOSE
                MNN_PRINT("output name: %s %s\n", outputs[i].second.c_str(), dstT.v1.name);
                #endif
                if (outputs[i].second == dstT.v1.name) {
                    dstT.v1.clientBuf.data = t->host<void>();
                    dstT.v1.clientBuf.dataSize = t->usize();
                    find = true;
                    break;
                }
            }
            if (!find) {
                FUNC_PRINT(i);
            }
        }
        CALL_QNN(QNN::gContext.interface.graphExecute(qnnGraphHandle, graph->inputTensors, graph->numInputTensors, \
            graph->outputTensors, graph->numOutputTensors, mQnnProfileHandle, nullptr));
        MNN::QNN::doProfile(QNN::gContext.interface, mQnnProfileHandle);
    }
};

class PluginExecuteRaw : public CPUComputeKernel {
private:
    std::unique_ptr<RawExecutorWrapper> mRawExecutor;
    std::vector<std::pair<const MNN::Tensor *, std::string>> mInputs;
    std::vector<std::pair<const MNN::Tensor *, std::string>> mOutputs;
    std::vector<std::shared_ptr<MNN::Tensor>> mRealInputs;
    std::vector<std::shared_ptr<MNN::Tensor>> mRealOutputs;
    int mShapeIndex;
public:
    ~ PluginExecuteRaw() {
        mRealInputs.clear();
        mRealOutputs.clear();
        mRawExecutor.reset();
    }
    bool init(CPUKernelContext* ctx) override {
        if (QNN::gContext.deviceHandle == nullptr){
            QNN::createQnnContext();
        }
        auto path = MNNFilePathConcat(ctx->dir_path(), ctx->getAttr("path")->s()->str());

        std::vector<std::string> allGraphName;
        auto allGraphNameAttr = ctx->getAttr("allGraphName");
        if (allGraphNameAttr && allGraphNameAttr->list() && allGraphNameAttr->list()->s()) {
            auto graphNames = allGraphNameAttr->list()->s();
            for (int i = 0; i < graphNames->size(); ++i) {
                allGraphName.push_back(graphNames->GetAsString(i)->str());
            }
        } else {
            MNN_ERROR("MNN_QNN: Incorrect Plugin Op, can't find 'allGraphName' attr.\n");
            return false;
        }

        size_t binaryOffset = 0;
        auto offsetAttr = ctx->getAttr("offset");
        if (offsetAttr && offsetAttr->list() && offsetAttr->list()->i()->size() == 2) {
            const int * dataPtr = offsetAttr->list()->i()->data();
            int lowSrc = dataPtr[0];
            int highSrc = dataPtr[1];

            uint32_t lowDst, highDst;
            ::memcpy(&lowDst, &lowSrc, sizeof(uint32_t));
            ::memcpy(&highDst, &highSrc, sizeof(uint32_t));

            binaryOffset = (static_cast<size_t>(highDst) << 32) | static_cast<size_t>(lowDst);
        }

        size_t binarySize = 0;
        auto sizeAttr = ctx->getAttr("size");
        if (sizeAttr && sizeAttr->list() && sizeAttr->list()->i()->size() == 2) {
            const int * dataPtr = sizeAttr->list()->i()->data();
            int lowSrc = dataPtr[0];
            int highSrc = dataPtr[1];

            uint32_t lowDst, highDst;
            ::memcpy(&lowDst, &lowSrc, sizeof(uint32_t));
            ::memcpy(&highDst, &highSrc, sizeof(uint32_t));

            binarySize = (static_cast<size_t>(highDst) << 32) | static_cast<size_t>(lowDst);
        }
        mRawExecutor.reset(new RawExecutorWrapper());
        return mRawExecutor->compileModel(path, binaryOffset, binarySize, allGraphName);
    }

    bool resize(CPUKernelContext* ctx) override {
        int shapeIndex = 0;
        if (!(shape_inference::computeIndex(ctx, shapeIndex))) {
            MNN_ERROR("MNN_QNN: Failed to execute Plugin Op.\n");
            return false;
        }
        mShapeIndex = shapeIndex;

        auto inputs = ctx->getAttr("inputs")->list();
        auto inputTensor = ctx->inputs();
        MNN_ASSERT(inputs->s()->size() == inputTensor.size());
        mInputs.resize(inputs->s()->size());
        mRealInputs.resize(inputTensor.size());
        for (int i=0; i<inputs->s()->size(); ++i) {
            mRealInputs[i].reset(new Tensor(inputTensor[i], Tensor::CAFFE));
            mInputs[i].second = inputs->s()->GetAsString(i)->str();
            mInputs[i].first = mRealInputs[i].get();
        }
        auto outputs = ctx->getAttr("outputs")->list();
        auto outputTensor = ctx->outputs();
        mOutputs.resize(outputs->s()->size());
        MNN_ASSERT(outputs->s()->size() == outputTensor.size());
        mRealOutputs.resize(outputTensor.size());
        for (int i=0; i<outputs->s()->size(); ++i) {
            mRealOutputs[i].reset(new Tensor(outputTensor[i], Tensor::CAFFE));
            mOutputs[i].second = outputs->s()->GetAsString(i)->str();
            mOutputs[i].first = mRealOutputs[i].get();
        }

        return true;
    }

    bool compute(CPUKernelContext* ctx) override {
        AUTOTIME;
        int shapeIndex = 0;
        if (!(shape_inference::computeIndex(ctx, shapeIndex))) {
            MNN_ERROR("MNN_QNN: Failed to execute Plugin Op.\n");
            return false;
        }
        std::string graphName = ctx->getAttr("allGraphName")->list()->s()->GetAsString(shapeIndex)->str();

        // compute and alloc real in-time inputs and outputs tensor
        {
            auto inputs = ctx->getAttr("inputs")->list();
            auto inputTensor = ctx->inputs();
            MNN_ASSERT(inputs->s()->size() == inputTensor.size());
            mInputs.resize(inputs->s()->size());
            mRealInputs.resize(inputTensor.size());
            for (int i=0; i<inputs->s()->size(); ++i) {
                mRealInputs[i].reset(new Tensor(inputTensor[i], Tensor::CAFFE));
                mInputs[i].second = inputs->s()->GetAsString(i)->str();
                mInputs[i].first = mRealInputs[i].get();
            }
            auto outputs = ctx->getAttr("outputs")->list();
            auto outputTensor = ctx->outputs();
            mOutputs.resize(outputs->s()->size());
            MNN_ASSERT(outputs->s()->size() == outputTensor.size());
            mRealOutputs.resize(outputTensor.size());
            for (int i=0; i<outputs->s()->size(); ++i) {
                mRealOutputs[i].reset(new Tensor(outputTensor[i], Tensor::CAFFE));
                mOutputs[i].second = outputs->s()->GetAsString(i)->str();
                mOutputs[i].first = mRealOutputs[i].get();
            }
        }

        #ifdef QNN_VERBOSE
        MNN_PRINT("Graph name:%s, %d\n", graphName.c_str(), shapeIndex);
        #endif
        auto inputTensor = ctx->inputs();
        auto outputTensor = ctx->outputs();

        for (int i=0; i<mInputs.size(); ++i) {
            ctx->backend()->onCopyBuffer(inputTensor[i], mRealInputs[i].get());
        }
        mRawExecutor->invokModel(mInputs, mOutputs, mShapeIndex);
        for (int i=0; i<mOutputs.size(); ++i) {
            ctx->backend()->onCopyBuffer(mRealOutputs[i].get(), outputTensor[i]);
        }

        return true;
    }
};

} // namespace backend
}
}

#endif

namespace MNN {
namespace QNN {

#ifdef ENABLE_QNN_ONLINE_FINALIZE
QnnBackend::QnnBackend(const QnnRuntime* runtime) : Backend(QNN_FORWARD_TYPE), mPower(runtime->mPower) {
    mRuntime = runtime;
    mUseFP16 = (runtime->mPrecision != BackendConfig::Precision_High) ? true : false;
    mPerf = QNNPerf::create(&mRuntime->mQnnInterface);
    if (mPower == BackendConfig::Power_High) {
        mPerf->setPowerConfigBurst();
        mPerf->setRpcLatencyAndPolling();
    }

    // Set mQnnGraphConfig.
    mQnnHtpGraphCustomConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
    mQnnHtpGraphCustomConfig.precision = QNN_PRECISION_FLOAT16;
    mQnnGraphConfig.option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    mQnnGraphConfig.customConfig = &mQnnHtpGraphCustomConfig;
}

QnnBackend::~QnnBackend() {
    clean();
    if (mPower == BackendConfig::Power_High) {
        mPerf->setPowerConfigBalanced();
    }
}

static inline std::map<OpType, QnnBackend::Creator*>* getCreatorMap() {
    static std::once_flag of;
    static std::map<OpType, QnnBackend::Creator*>* ret = nullptr;
    std::call_once(of, [&]() { ret = new std::map<OpType, QnnBackend::Creator*>; });
    return ret;
}

Execution* QnnBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {
    MNN_ASSERT(op != nullptr);
    auto map = getCreatorMap();
    auto iter = map->find(op->type());

    // MNN_PRINT("MNN_QNN::onCreate Type %d, Name %s.\n", op->type(), op->name()->c_str());

    if (iter == map->end()) {
        if(op->name() != nullptr){
            MNN_PRINT("MNN_QNN: Not registered type %d, %s.\n", op->type(), op->name()->c_str());
        } else {
            MNN_PRINT("MNN_QNN: Not registered type %d.\n", op->type());
        }
        return nullptr;
    }

    auto exe = iter->second->onCreate(inputs, outputs, op, this);

    if (nullptr == exe) {
        if(op->name() != nullptr){
            MNN_PRINT("MNN_QNN: Don't support type %d, %s.\n", op->type(), op->name()->c_str());
        } else {
            MNN_PRINT("MNN_QNN: Don't support type %d.\n", op->type());
        }
        return nullptr;
    }

    return exe;
}

bool QnnBackend::addCreator(OpType t, Creator* c) {
    auto map = getCreatorMap();
    if (map->find(t) != map->end()) {
        MNN_PRINT("MNN_QNN: %d type has be added.\n", t);
        return false;
    }
    map->insert(std::make_pair(t, c));
    return true;
}


void QnnBackend::onExecuteBegin() const {
    mGraphExecuted = false;
    if (mPower == BackendConfig::Power_Normal) {
        mPerf->setPowerConfigBurst();
        mPerf->setRpcLatencyAndPolling();
    }
    return;
}

void QnnBackend::startProfile() const{
    MNN::QNN::doProfile(mRuntime->mQnnInterface, mQnnProfileHandle);
}

void QnnBackend::onExecuteEnd() const {
    executeGraph();
    if (mPower == BackendConfig::Power_Normal) {
        mPerf->setPowerConfigBalanced();
    }
    startProfile();
    return;
}

void QnnBackend::onResizeBegin() {
    MNN_PRINT("QNN onResizeBegin[%p]: cleaning and recreating graph\n", this);
    clean();
    createContextAndGraph();
    return;
}

ErrorCode QnnBackend::onResizeEnd() {
    MNN_PRINT("QNN onResizeEnd[%p]: tensorWrappers=%d outputCastMap=%d dequantMap=%d "
              "inputIndexes=%d outputIndexes=%d extraInputs=%d extraOutputs=%d\n",
              this,
              (int)mQNNTensorWrappers.size(), (int)mOutputCastTensorMap.size(),
              (int)mDeQuantOutputTensorMap.size(),
              (int)mInputTensorIndexes.size(), (int)mOutputTensorIndexes.size(),
              (int)mExtraInputWrappers.size(), (int)mExtraOutputWrappers.size());
    // Check for entries with missing tensors BEFORE building cast nodes
    for (auto iter : mOutputCastTensorMap) {
        int idx1 = getTensorIdx(iter.second.first);
        int idx2 = getTensorIdx(iter.second.second.get());
        MNN_PRINT("QNN onResizeEnd: outputCast entry: origIdx=%d stageIdx=%d wrapperTotal=%d\n",
                  idx1, idx2, (int)mQNNTensorWrappers.size());
    }
    buildOutputCast();
    buildOutputDequant();
    finalizeGraph();
    MNN_PRINT("QNN onResizeEnd[%p]: after finalizeGraph, running %d release funcs\n",
              this, (int)mReleaseFunc.size());
    for(auto func : mReleaseFunc){
        func();
    }
    mReleaseFunc.clear();
    MNN_PRINT("QNN onResizeEnd[%p]: done\n", this);
    return NO_ERROR;
}

Backend::MemObj* QnnBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
    std::string tName = "QnnTensor_" + std::to_string(mTensorCounter);
    if (TensorUtils::getDescribe(tensor)->index >= 0) {
        tName = std::string("t") + std::to_string(TensorUtils::getDescribe(tensor)->index);
    }

    bool isInput = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
    bool isOutput = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
    bool isConst = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    if (isConst) {
        MNN_PRINT("QNN onAcquire[%p]: CONSTANT tensor '%s' shape=[", this, tName.c_str());
        for (int i = 0; i < (int)tensor->shape().size(); i++) {
            MNN_PRINT("%s%d", i ? "," : "", tensor->shape()[i]);
        }
        MNN_PRINT("] elemSize=%d\n", tensor->elementSize());
    }

    Qnn_TensorType_t tType = QNN_TENSOR_TYPE_NATIVE;
    if (isInput || isConst) {
        tType = QNN_TENSOR_TYPE_APP_WRITE;
    }
    if (isOutput) {
        tType = QNN_TENSOR_TYPE_APP_READ;
    }

    Qnn_DataType_t tDataType;
    Qnn_QuantizeParams_t tQuantizeParams{};
    tQuantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    tQuantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    Qnn_ScaleOffset_t tScaleOffsetEncoding;
    tScaleOffsetEncoding.scale = 0.0f;
    tScaleOffsetEncoding.offset = 0;
    auto quant = TensorUtils::getDescribe(tensor)->quantAttr.get();
    bool isQuant = quant != nullptr && TensorUtils::getDescribe(tensor)->applyQuant;
    //MNN_ASSERT((tensor->getType().code == halide_type_float) || (tensor->getType().code == halide_type_int && tensor->getType().bits == 32));
    if (mUseFP16 && tensor->getType().code == halide_type_float) {
        tType = QNN_TENSOR_TYPE_NATIVE;
        tDataType = QNN_DATATYPE_FLOAT_16;
    } else if (tensor->getType().code == halide_type_float) {
        tDataType = QNN_DATATYPE_FLOAT_32;
    } else if (tensor->getType().code == halide_type_int && tensor->getType().bits == 32) {
        tDataType = QNN_DATATYPE_INT_32;
    } else {
        MNN_PRINT("MNN_QNN: Not supported data type in <QnnBackend::onAcquire>.\n");
        return nullptr;
    }
    if(isQuant) {
        tType = QNN_TENSOR_TYPE_NATIVE;
        auto quantType = TensorUtils::getDescribe(tensor)->quantAttr->type;
        if(quantType == DataType_DT_INT8){
            tQuantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
            tQuantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            if(quant->zero != 0){
                MNN_PRINT("MNN_QNN: Not supported asymmetric quant in <QnnBackend::onAcquire>.\n");
                return nullptr;
            }
            tScaleOffsetEncoding.scale = quant->scale;
            tScaleOffsetEncoding.offset = 0;
            tDataType = QNN_DATATYPE_SFIXED_POINT_8;
            if (isOutput) {
                tType = QNN_TENSOR_TYPE_NATIVE;
            }
        }else if(quantType == DataType_DT_INT16){
            // uint16
            tQuantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
            tQuantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            tScaleOffsetEncoding.scale = quant->scale;
            tScaleOffsetEncoding.offset = quant->zero;
            tDataType = QNN_DATATYPE_UFIXED_POINT_16;
            if (isOutput) {
                tType = QNN_TENSOR_TYPE_NATIVE;
            }
        }
    }
    tQuantizeParams.scaleOffsetEncoding = tScaleOffsetEncoding;
    Tensor::DimensionType tensorDimType = tensor->getDimensionType();

    std::vector<int> tDims = tensor->shape();
    if(TensorUtils::getDescribe(tensor)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4){
        tensorDimType = gQnnTensorDimType;
        std::unique_ptr<Tensor> tempTensor(new Tensor(tensor, tensorDimType, false));
        if (!(tempTensor->shape().empty())) {
            tDims = tempTensor->shape();
        } else {
            tDims = {1};
        }
    }

    std::string suffix = "";
    if((isInput || isConst) && mUseFP16 && tensor->getType().code == halide_type_float){
        suffix = "_cast";
    }
    if(isOutput && isQuant){
        suffix = "_dequant";
    }
    if(isOutput && mUseFP16 && tensor->getType().code == halide_type_float){
        suffix = "_cast";
    }
    std::shared_ptr<QNNTensorWrapper> qnnTensorWrapper = QNNTensorWrapper::create(tName + suffix, tType, tDataType, tDims, tQuantizeParams);

    Qnn_Tensor_t * qnnTensor = qnnTensorWrapper->getNativeTensor();
    CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, qnnTensor));
    mQNNTensorWrappers.push_back(qnnTensorWrapper);
    mTensorMap.insert({TensorUtils::getDescribe(tensor), mTensorCounter});

    if (isInput || isConst) {
        // create stage tensor to cast
        if (mUseFP16 && tensor->getType().code == halide_type_float) {
            mTensorCounter += 1;
            std::shared_ptr<Tensor> stageTensor;
            stageTensor.reset(Tensor::create<float>(tensor->shape(), nullptr, tensorDimType));
            Qnn_QuantizeParams_t tQuantizeParamstmp = QNN_QUANTIZE_PARAMS_INIT;
            std::shared_ptr<QNNTensorWrapper> qnnCastTensorWrapper = QNNTensorWrapper::create(tName, QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, tDims, tQuantizeParamstmp);
            CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, qnnCastTensorWrapper->getNativeTensor()));
            mInputCastTensorMap.insert({TensorUtils::getDescribe(tensor), {tensor, stageTensor}});
            mQNNTensorWrappers.push_back(qnnCastTensorWrapper);
            mTensorMap.insert({TensorUtils::getDescribe(const_cast<const Tensor*>(stageTensor.get())), mTensorCounter});
            mInputTensorIndexes.push_back(mTensorCounter);
            qnnCastTensorWrapper->alloc(tensorDimType);
            buildInputCast(tensor);
        }else{
            mInputTensorIndexes.push_back(mTensorCounter);
            qnnTensorWrapper->alloc(tensorDimType);
        }
    }
    if (isOutput) {
        if(isQuant){
            mTensorCounter += 1;
            std::shared_ptr<Tensor> stageTensor;
            stageTensor.reset(Tensor::create<float>(tensor->shape(), nullptr, tensorDimType));
            if (tensor->getType().code == halide_type_float) {
                tDataType = QNN_DATATYPE_FLOAT_32;
            } else {
                MNN_PRINT("MNN_QNN: Not supported data type in <QnnBackend::onAcquire>.\n");
                return nullptr;
            }
            Qnn_QuantizeParams_t tQuantizeParamstmp = QNN_QUANTIZE_PARAMS_INIT;
            std::shared_ptr<QNNTensorWrapper> qnnOutputTensorWrapper = QNNTensorWrapper::create(tName, QNN_TENSOR_TYPE_APP_READ, tDataType, tDims, tQuantizeParamstmp);
            CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, qnnOutputTensorWrapper->getNativeTensor()));
            mDeQuantOutputTensorMap.insert({TensorUtils::getDescribe(tensor), {tensor, stageTensor}});
            mQNNTensorWrappers.push_back(qnnOutputTensorWrapper);
            mTensorMap.insert({TensorUtils::getDescribe(const_cast<const Tensor*>(stageTensor.get())), mTensorCounter});
            mOutputTensorIndexes.push_back(mTensorCounter);
            qnnOutputTensorWrapper->alloc(tensorDimType);
        } else{
            if (mUseFP16 && tensor->getType().code == halide_type_float) {
                mTensorCounter += 1;
                std::shared_ptr<Tensor> stageTensor;
                stageTensor.reset(Tensor::create<float>(tensor->shape(), nullptr, tensorDimType));
                Qnn_QuantizeParams_t tQuantizeParamstmp = QNN_QUANTIZE_PARAMS_INIT;
                std::shared_ptr<QNNTensorWrapper> qnnCastTensorWrapper = QNNTensorWrapper::create(tName, QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32, tDims, tQuantizeParamstmp);
                CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, qnnCastTensorWrapper->getNativeTensor()));
                mOutputCastTensorMap.insert({TensorUtils::getDescribe(tensor), {tensor, stageTensor}});
                mQNNTensorWrappers.push_back(qnnCastTensorWrapper);
                mTensorMap.insert({TensorUtils::getDescribe(const_cast<const Tensor*>(stageTensor.get())), mTensorCounter});
                mOutputTensorIndexes.push_back(mTensorCounter);
                qnnCastTensorWrapper->alloc(tensorDimType);
                MNN_PRINT("QNN onAcquire[%p]: OUTPUT+FP16 cast entry added, origDesc=%p stageDesc=%p "
                          "origIdx=%d stageIdx=%d shape=[%d",
                          this, TensorUtils::getDescribe(tensor),
                          TensorUtils::getDescribe(const_cast<const Tensor*>(stageTensor.get())),
                          mTensorCounter - 1, mTensorCounter, tensor->shape().empty() ? 0 : tensor->shape()[0]);
                for (int si = 1; si < (int)tensor->shape().size(); si++) {
                    MNN_PRINT(",%d", tensor->shape()[si]);
                }
                MNN_PRINT("]\n");
            }else{
                mOutputTensorIndexes.push_back(mTensorCounter);
                qnnTensorWrapper->alloc(tensorDimType);
            }
        }
    }

    // For OUTPUT tensors, allocate host memory so that Tensor::clone() works
    // across sub-module boundaries (PipelineModule passes VARPs between
    // sub-modules, and the receiving StaticModule calls copyFromHostTensor
    // which reads from the tensor's host pointer).
    void* hostMem = nullptr;
    if (isOutput) {
        auto mutableTensor = const_cast<Tensor*>(tensor);
        size_t hostSize = (size_t)tensor->elementSize() * tensor->getType().bytes();
        if (hostSize > 0) {
            hostMem = calloc(1, hostSize);
            mutableTensor->buffer().host = (uint8_t*)hostMem;
            mOutputTensors.push_back(mutableTensor);
            MNN_PRINT("QNN onAcquire[%p]: allocated %zu bytes host memory for OUTPUT tensor\n",
                      this, hostSize);
        }
    }

    mTensorCounter += 1;
    #ifdef QNN_VERBOSE
    MNN_PRINT("Total qnn tensor count:%d\n", mTensorCounter);
    #endif
    if (hostMem) {
        return new QnnHostMemObj(hostMem);
    }
    return new Backend::MemObj();
}


bool QnnBackend::onClearBuffer() {
    return true;
}


void QnnBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
    bool isInput = TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
    bool isOutput = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
    bool isConst = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT || TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    bool isDstConst = TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    if (!isInput && !isOutput && !isConst) {
        // Intermediate tensor copy (e.g., WrapCopyExecution for mixed pipelines).
        // Try to determine direction from the tensor map.
        int srcIdx = getTensorIdx(srcTensor);
        if (srcIdx >= 0) {
            // Source is a QNN tensor — treat as output copy (data leaving QNN)
            outputIO(srcTensor, dstTensor);
        } else {
            // Source is external — treat as input copy (data entering QNN)
            inputIO(srcTensor, dstTensor);
        }
        return;
    }

    if (isInput || isDstConst) {
        // CONSTANT destination tensors are registered as APP_WRITE in onAcquire,
        // so route them through inputIO just like INPUT tensors.
        inputIO(srcTensor, dstTensor);
    } else if (isOutput) {
        outputIO(srcTensor, dstTensor);
    } else {
        // Not support.
    }
}

void QnnBackend::inputIO(const Tensor* srcTensor, const Tensor* dstTensor) const {
    auto iter = mInputCastTensorMap.find(TensorUtils::getDescribe(dstTensor));
    int dstIndex = -1;
    if(iter != mInputCastTensorMap.end()){
        dstIndex = getTensorIdx(iter->second.second.get());
    } else{
        dstIndex = getTensorIdx(dstTensor);
    }
    if (dstIndex < 0 || dstIndex >= (int)mQNNTensorWrappers.size()) {
        MNN_ERROR("QNN inputIO[%p]: invalid dstIndex=%d (total=%d), skipping\n",
                  this, dstIndex, (int)mQNNTensorWrappers.size());
        return;
    }
    std::shared_ptr<QNNTensorWrapper> dstQnnTensorWrapper = mQNNTensorWrappers[dstIndex];
    std::shared_ptr<Tensor> dstDataContainer = dstQnnTensorWrapper->getDataContainer();

    bool valid0 = srcTensor->getType().code == halide_type_float;
    bool valid1 = srcTensor->getType().code == halide_type_int && srcTensor->getType().bits == 32;

    // Currently, support float and int input only.
    MNN_ASSERT(valid0 || valid1);

    // Guard against NULL source host pointer.  This can happen when the source
    // tensor comes from another QnnBackend whose graph was not executed (e.g.,
    // backup backend with no ops).  The QNN tensor wrapper has its own memory
    // but the MNN Tensor's host pointer is never set.
    if (srcTensor->host<void>() == nullptr) {
        MNN_ERROR("QNN inputIO[%p]: src host is NULL (elemSize=%d), zero-filling dst\n",
                  this, srcTensor->elementSize());
        ::memset(dstDataContainer.get()->host<void>(), 0,
                 srcTensor->elementSize() * srcTensor->getType().bytes());
        return;
    }
    if(TensorUtils::getDescribe(srcTensor)->dimensionFormat == TensorUtils::getDescribe(dstDataContainer.get())->dimensionFormat){
        ::memcpy(dstDataContainer.get()->host<float>(), srcTensor->host<float>(), srcTensor->elementSize() * sizeof(float));
    // Debug: dump first few input values for pipeline-level inputs
    if (dstIndex < 10) {
        float* dbg = srcTensor->host<float>();
        int n = srcTensor->elementSize() < 8 ? srcTensor->elementSize() : 8;
        char dbgBuf[256];
        int pos = 0;
        for (int d = 0; d < n; d++) {
            pos += snprintf(dbgBuf + pos, sizeof(dbgBuf) - pos, " %.4f", dbg[d]);
        }
        MNN_PRINT("QNN inputIO: input[%d] first %d values:%s (total=%d)\n", dstIndex, n, dbgBuf, srcTensor->elementSize());
    }
    }else{
        auto code = CPUTensorConverter::convert(srcTensor, dstDataContainer.get());
        if (NO_ERROR != code) {
            MNN_ERROR("MNN_QNN: Error in QNNBackend::onCopyBuffer.\n");
        }
    }
}

void QnnBackend::outputIO(const Tensor* srcTensor, const Tensor* dstTensor) const {
    // Lazy graph execution: if the graph hasn't been executed yet (e.g., called
    // from a WrapCopyExecution mid-pipeline before onExecuteEnd), run it now
    // so that the QNN data containers have valid output data.
    if (!mGraphExecuted && mGraphFinalized) {
        MNN_PRINT("QNN outputIO[%p]: triggering lazy graph execution\n", this);
        executeGraph();
    }

    auto iter = mDeQuantOutputTensorMap.find(TensorUtils::getDescribe(srcTensor));
    int srcIndex = -1;
    if(iter != mDeQuantOutputTensorMap.end()){
        srcIndex = getTensorIdx(iter->second.second.get());
    } else{
        if(mUseFP16){
            auto castIter = mOutputCastTensorMap.find(TensorUtils::getDescribe(srcTensor));
            if(castIter != mOutputCastTensorMap.end()){
                srcIndex = getTensorIdx(castIter->second.second.get());
            }else{
                MNN_ERROR("MNN_QNN: Error in QNNBackend::onCopyBuffer for cast float to half.\n");
                return;
            }
        }else{
            srcIndex = getTensorIdx(srcTensor);
        }
    }
    if (srcIndex < 0 || srcIndex >= (int)mQNNTensorWrappers.size()) {
        // Source tensor is not in QNN's tensor map — it's from a different
        // backend (e.g., CPU backup via WrapCopyExecution).  Fall back to
        // direct host-to-host copy.
        if (srcTensor->host<void>() != nullptr && dstTensor->host<void>() != nullptr) {
            size_t bytes = (size_t)srcTensor->elementSize() * srcTensor->getType().bytes();
            MNN_PRINT("QNN outputIO[%p]: foreign src, host-to-host copy %zu bytes\n", this, bytes);
            ::memcpy(const_cast<Tensor*>(dstTensor)->host<void>(),
                     srcTensor->host<void>(), bytes);
        } else {
            MNN_ERROR("QNN outputIO[%p]: foreign src with NULL host (src=%p dst=%p), skipping\n",
                      this, srcTensor->host<void>(), dstTensor->host<void>());
        }
        return;
    }
    std::shared_ptr<QNNTensorWrapper> srcQnnTensorWrapper = mQNNTensorWrappers[srcIndex];
    std::shared_ptr<Tensor> srcDataContainer = srcQnnTensorWrapper->getDataContainer();

    // Currently, support float output only.
    bool valid0 = dstTensor->getType().code == halide_type_float;
    bool valid1 = dstTensor->getType().code == halide_type_int && dstTensor->getType().bits == 32;

    // Currently, support float and int input only.
    MNN_ASSERT(valid0 || valid1);

    // Guard against NULL destination host pointer (same issue as inputIO above)
    if (dstTensor->host<void>() == nullptr) {
        MNN_ERROR("QNN outputIO[%p]: dst host is NULL, skipping copy\n", this);
        return;
    }
    if(TensorUtils::getDescribe(dstTensor)->dimensionFormat == TensorUtils::getDescribe(srcDataContainer.get())->dimensionFormat){
        ::memcpy(dstTensor->host<float>(), srcDataContainer.get()->host<float>(), srcTensor->elementSize() * sizeof(float));
    }else{
        auto code = CPUTensorConverter::convert(srcDataContainer.get(), dstTensor);
        if (NO_ERROR != code) {
            MNN_ERROR("MNN_QNN: Error in QNNBackend::onCopyBuffer.\n");
        }
    }
}
bool QnnBackend::useCache() const {
    return mRuntime->mUseCache;
}

void QnnBackend::createContextAndGraph() {
    // Each QnnBackend creates its own QNN context so that multiple backends
    // (main + backup + lm_head, etc.) don't clobber each other's contexts.
    // Previously all backends shared mRuntime->mQnnContextHandle, which caused
    // one backend's clean() to invalidate another backend's graph handles.
    CALL_QNN(mRuntime->mQnnInterface.contextCreate(
        mRuntime->mQnnBackendHandle, mRuntime->mQnnDeviceHandle,
        mQnnContextConfig, &mQnnContextHandle));
    if (mQnnContextHandle == nullptr) {
        MNN_ERROR("QNN createContextAndGraph[%p]: contextCreate returned null!\n", this);
        return;
    }
    const QnnGraph_Config_t * pGraphConfig[] = {&mQnnGraphConfig, nullptr};
    if (mRuntime->mUseCache) {
        CALL_QNN(mRuntime->mQnnInterface.graphRetrieve(mQnnContextHandle, mQnnGraphName.c_str(), &mQnnGraphHandle));
    } else {
        CALL_QNN(mRuntime->mQnnInterface.graphCreate(mQnnContextHandle, mQnnGraphName.c_str(), pGraphConfig, &mQnnGraphHandle));
    }
    MNN_ASSERT(mQnnGraphHandle != nullptr);
}

void QnnBackend::finalizeGraph() {
    // [TODO] Fix this. Add the following branch for empty resize.
    if (mTensorCounter == 0) {
        return;
    }
    // Skip finalization for graphs with no inputs - these are typically backup backend
    // instances that can't form a valid QNN graph.
    if (mInputTensorIndexes.empty() && mExtraInputWrappers.empty()) {
        MNN_PRINT("QNN finalizeGraph[%p]: skipping (no inputs), tensorCount=%d\n",
                  this, mTensorCounter);
        return;
    }
    MNN_PRINT("QNN finalizeGraph[%p]: tensorCount=%d inputs=%d outputs=%d extraInputs=%d extraOutputs=%d\n",
              this, mTensorCounter, (int)mInputTensorIndexes.size(), (int)mOutputTensorIndexes.size(),
              (int)mExtraInputWrappers.size(), (int)mExtraOutputWrappers.size());

    // Create Prefile Handle
    MNN::QNN::createProfileHandle(mRuntime->mQnnInterface, mRuntime->mQnnBackendHandle, &mQnnProfileHandle);

    auto ret = mRuntime->mQnnInterface.graphFinalize(mQnnGraphHandle, mQnnProfileHandle, mQnnSignalHandle);
    int errorCode = ret & 0xFFFF;
    if (errorCode != QNN_SUCCESS) {
        MNN_ERROR("QNN graphFinalize FAILED: error code %d\n", errorCode);
    } else {
        MNN_PRINT("QNN graphFinalize SUCCESS\n");
        mGraphFinalized = true;
    }
}

void QnnBackend::executeGraph() const {
    if (!mGraphFinalized) {
        MNN_PRINT("QNN executeGraph[%p]: skipping (graph not finalized)\n", this);
        return;
    }
    if (mGraphExecuted) {
        MNN_PRINT("QNN executeGraph[%p]: skipping (already executed this cycle)\n", this);
        return;
    }
    // Fill deferred inputs: copy data from original tensor host to APP_WRITE wrapper.
    // These are INPUT tensors discovered in getTensorIdx (not via onAcquire) whose
    // data was never transferred via onCopyBuffer.
    for (auto& entry : mDeferredInputs) {
        int wIdx = entry.first;
        const Tensor* srcTensor = entry.second;
        if (wIdx < 0 || wIdx >= (int)mQNNTensorWrappers.size()) continue;
        auto container = mQNNTensorWrappers[wIdx]->getDataContainer();
        if (srcTensor->host<void>() && container && container->host<void>()) {
            size_t bytes = (size_t)srcTensor->elementSize() * srcTensor->getType().bytes();
            ::memcpy(container->host<void>(), srcTensor->host<void>(), bytes);
            MNN_PRINT("QNN executeGraph: deferred input[%d] copied %zu bytes from host %p\n",
                      wIdx, bytes, srcTensor->host<void>());
        } else {
            MNN_ERROR("QNN executeGraph: deferred input[%d] src host=%p container=%p\n",
                      wIdx, srcTensor->host<void>(),
                      container ? container->host<void>() : nullptr);
        }
    }

    std::vector<Qnn_Tensor_t> inputs;
    std::vector<Qnn_Tensor_t> outputs;
    for (int i = 0; i <  mInputTensorIndexes.size(); i++) {
        inputs.push_back(*(mQNNTensorWrappers[mInputTensorIndexes[i]]->getNativeTensor()));
    }
    for (auto& wrapper : mExtraInputWrappers) {
        inputs.push_back(*(wrapper->getNativeTensor()));
    }
    for (int j = 0 ; j < mOutputTensorIndexes.size(); j++) {
        outputs.push_back(*(mQNNTensorWrappers[mOutputTensorIndexes[j]]->getNativeTensor()));
    }
    for (auto& wrapper : mExtraOutputWrappers) {
        outputs.push_back(*(wrapper->getNativeTensor()));
    }

    MNN_PRINT("QNN executeGraph: %d inputs, %d outputs\n", (int)inputs.size(), (int)outputs.size());
    for (int i = 0; i < (int)inputs.size(); i++) {
        MNN_PRINT("  input[%d] id=%u type=%d buf=%p size=%u\n",
                  i, inputs[i].v1.id, inputs[i].v1.type,
                  inputs[i].v1.clientBuf.data, inputs[i].v1.clientBuf.dataSize);
    }
    for (int i = 0; i < (int)outputs.size(); i++) {
        MNN_PRINT("  output[%d] id=%u type=%d buf=%p size=%u\n",
                  i, outputs[i].v1.id, outputs[i].v1.type,
                  outputs[i].v1.clientBuf.data, outputs[i].v1.clientBuf.dataSize);
    }

    auto ret = mRuntime->mQnnInterface.graphExecute(mQnnGraphHandle, inputs.data(), inputs.size(), outputs.data(), outputs.size(), mQnnProfileHandle, mQnnSignalHandle);
    int errorCode = ret & 0xFFFF;
    if (errorCode != QNN_SUCCESS) {
        MNN_ERROR("QNN graphExecute FAILED: error code %d\n", errorCode);
    } else {
        MNN_PRINT("QNN graphExecute SUCCESS\n");
        mGraphExecuted = true;
    }

    // Sync output data containers to host memory so that Tensor::clone()
    // across sub-module boundaries gets valid data.
    for (int i = 0; i < (int)mOutputTensors.size() && i < (int)mOutputTensorIndexes.size(); i++) {
        int wrapperIdx = mOutputTensorIndexes[i];
        if (wrapperIdx < 0 || wrapperIdx >= (int)mQNNTensorWrappers.size()) continue;
        auto container = mQNNTensorWrappers[wrapperIdx]->getDataContainer();
        auto tensor = mOutputTensors[i];
        if (tensor && tensor->host<void>() && container && container->host<void>()) {
            size_t bytes = (size_t)tensor->elementSize() * tensor->getType().bytes();
            ::memcpy(tensor->host<void>(), container->host<void>(), bytes);
            MNN_PRINT("QNN executeGraph: synced %zu bytes to OUTPUT tensor host\n", bytes);
            // Debug: dump first few float values of the output
            float* debugPtr = container->host<float>();
            if (debugPtr) {
                int numFloats = (int)(bytes / sizeof(float));
                int printCount = numFloats < 8 ? numFloats : 8;
                char dbgBuf[256];
                int pos = 0;
                for (int d = 0; d < printCount; d++) {
                    pos += snprintf(dbgBuf + pos, sizeof(dbgBuf) - pos, " %.4f", debugPtr[d]);
                }
                MNN_PRINT("QNN output[%d] first %d values:%s\n", i, printCount, dbgBuf);
            }
        }
    }
}

void QnnBackend::freeContextAndGraph() {
    if (mTensorCounter != 0) {
        mQnnGraphHandle = nullptr;
    }
    // Free this backend's own context (not the runtime's shared one).
    if (nullptr != mQnnContextHandle) {
        CALL_QNN(mRuntime->mQnnInterface.contextFree(mQnnContextHandle, nullptr));
        mQnnContextHandle = nullptr;
    }
}

void QnnBackend::addNodeToGraph(Qnn_OpConfigVersion_t version, const char* nodeName, const char* packageName, const char* nodeType, std::vector<Qnn_Param_t> & params, std::vector<Qnn_Tensor_t> & inputs, std::vector<Qnn_Tensor_t> & outputs) {
    MNN_ASSERT(nodeName != nullptr && packageName != nullptr && nodeType != nullptr && !(inputs.empty()) && !(outputs.empty()));

    Qnn_OpConfig_t opConfig = QNN_OPCONFIG_INIT;
    opConfig.version = version;
    opConfig.v1.name = nodeName;
    opConfig.v1.packageName = packageName;
    opConfig.v1.typeName = nodeType;
    opConfig.v1.numOfParams = params.size();
    opConfig.v1.params = params.data();
    opConfig.v1.numOfInputs = inputs.size();
    opConfig.v1.inputTensors = inputs.data();
    opConfig.v1.numOfOutputs = outputs.size();
    opConfig.v1.outputTensors = outputs.data();

    CALL_QNN(mRuntime->mQnnInterface.backendValidateOpConfig(mRuntime->mQnnBackendHandle, opConfig));

    CALL_QNN(mRuntime->mQnnInterface.graphAddNode(mQnnGraphHandle, opConfig));
}

int QnnBackend::getTensorIdx(const Tensor * tensor) const {
    const Tensor::InsideDescribe::NativeInsideDescribe * tensorKey = TensorUtils::getDescribe(tensor);
    auto iter = mTensorMap.find(tensorKey);
    int idx = -1;
    if (iter == mTensorMap.end()) {
        std::string tName = "QnnTensor_" + std::to_string(mTensorCounter);
        std::vector<uint32_t> tDims = getNHWCShape(tensor);
        Qnn_DataType_t tDataType;
        std::shared_ptr<QNNTensorWrapper> qnnTensorWrapper;

        if (TensorUtils::getDescribe(tensor)->usage == Tensor::InsideDescribe::Usage::CONSTANT) {
            // CONSTANT tensors: create static tensor with data
            #ifdef QNN_VERBOSE
            MNN_PRINT("qnn tenor usage:%d, dimension:%d\n", TensorUtils::getDescribe(tensor)->usage, tensor->dimensions());
            #endif
            if (tensor->getType().code == halide_type_int && tensor->getType().bits == 32) {
                tDataType = QNN_DATATYPE_INT_32;
                qnnTensorWrapper = QNNTensorWrapper::createStaticTensor(tName, tDataType, tDims, tensor->host<int>());
            } else if (tensor->getType().code == halide_type_float) {
                tDataType = mUseFP16 ? QNN_DATATYPE_FLOAT_16 : QNN_DATATYPE_FLOAT_32;
                qnnTensorWrapper = QNNTensorWrapper::createStaticFloatTensor(tName, tDataType, tDims, tensor->host<float>());
            } else {
                MNN_ASSERT(false);
            }
        } else if (TensorUtils::getDescribe(tensor)->usage == Tensor::InsideDescribe::Usage::INPUT) {
            // INPUT tensor not in map: this sub-module's input was allocated by a
            // different backend (the previous sub-module's output).  Register as
            // APP_WRITE so we can feed data before each graphExecute.
            MNN_PRINT("QNN getTensorIdx[%p]: creating APP_WRITE for unmapped INPUT tensor "
                      "dims=%d index=%d elemSize=%d\n",
                      this, tensor->dimensions(), TensorUtils::getDescribe(tensor)->index,
                      tensor->elementSize());
            Tensor::DimensionType tensorDimType = tensor->getDimensionType();
            if (mUseFP16 && tensor->getType().code == halide_type_float) {
                // FP16 path: NATIVE FP16 tensor (referenced by ops) + APP_WRITE FP32 staging + Cast
                tDataType = QNN_DATATYPE_FLOAT_16;
                Qnn_QuantizeParams_t qp = QNN_QUANTIZE_PARAMS_INIT;
                qnnTensorWrapper = QNNTensorWrapper::create(tName + "_cast", QNN_TENSOR_TYPE_NATIVE, tDataType, tDims, qp);
                Qnn_Tensor_t* qt = qnnTensorWrapper->getNativeTensor();
                CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, qt));
                mQNNTensorWrappers.push_back(qnnTensorWrapper);
                mTensorMap.insert({tensorKey, mTensorCounter});
                idx = mTensorCounter;
                mTensorCounter += 1;

                // FP32 APP_WRITE staging tensor
                std::shared_ptr<Tensor> stageTensor;
                stageTensor.reset(Tensor::create<float>(tensor->shape(), nullptr, tensorDimType));
                auto stageWrapper = QNNTensorWrapper::create(tName, QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, tDims, qp);
                CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, stageWrapper->getNativeTensor()));
                stageWrapper->alloc(tensorDimType);
                mInputCastTensorMap.insert({tensorKey, {tensor, stageTensor}});
                mQNNTensorWrappers.push_back(stageWrapper);
                int stageIdx = mTensorCounter;
                mTensorMap.insert({TensorUtils::getDescribe(const_cast<const Tensor*>(stageTensor.get())), stageIdx});
                mInputTensorIndexes.push_back(stageIdx);
                mDeferredInputs.push_back({stageIdx, tensor});
                mTensorCounter += 1;

                // Inline Cast node: FP32 APP_WRITE → FP16 NATIVE
                {
                    Qnn_Tensor_t castInput = *(mQNNTensorWrappers[stageIdx]->getNativeTensor());
                    Qnn_Tensor_t castOutput = *(qnnTensorWrapper->getNativeTensor());
                    std::vector<Qnn_Param_t> castParams;
                    std::vector<Qnn_Tensor_t> castInputs = {castInput};
                    std::vector<Qnn_Tensor_t> castOutputs = {castOutput};
                    std::string castName = "Cast_I_" + std::to_string(stageIdx) + "_O_" + std::to_string(idx);
                    const_cast<QnnBackend*>(this)->addNodeToGraph(
                        QNN_OPCONFIG_VERSION_1, castName.c_str(), "qti.aisw", "Cast",
                        castParams, castInputs, castOutputs);
                }
                return idx;
            } else {
                // Non-FP16: direct APP_WRITE
                if (tensor->getType().code == halide_type_float) {
                    tDataType = QNN_DATATYPE_FLOAT_32;
                } else if (tensor->getType().code == halide_type_int && tensor->getType().bits == 32) {
                    tDataType = QNN_DATATYPE_INT_32;
                } else {
                    tDataType = QNN_DATATYPE_FLOAT_32;
                }
                Qnn_QuantizeParams_t qp = QNN_QUANTIZE_PARAMS_INIT;
                qnnTensorWrapper = QNNTensorWrapper::create(tName, QNN_TENSOR_TYPE_APP_WRITE, tDataType, tDims, qp);
                Qnn_Tensor_t* qt = qnnTensorWrapper->getNativeTensor();
                CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, qt));
                qnnTensorWrapper->alloc(tensorDimType);
                mQNNTensorWrappers.push_back(qnnTensorWrapper);
                mTensorMap.insert({tensorKey, mTensorCounter});
                idx = mTensorCounter;
                mInputTensorIndexes.push_back(mTensorCounter);
                mDeferredInputs.push_back({mTensorCounter, tensor});
                mTensorCounter += 1;
                return idx;
            }
        } else {
            // Non-CONSTANT, non-INPUT tensors not in map: create as NATIVE tensor.
            MNN_PRINT("QNN getTensorIdx[%p]: creating NATIVE tensor for unmapped tensor "
                      "usage=%d dims=%d index=%d\n",
                      this, (int)TensorUtils::getDescribe(tensor)->usage, tensor->dimensions(),
                      TensorUtils::getDescribe(tensor)->index);
            if (tensor->getType().code == halide_type_int && tensor->getType().bits == 32) {
                tDataType = QNN_DATATYPE_INT_32;
            } else {
                tDataType = mUseFP16 ? QNN_DATATYPE_FLOAT_16 : QNN_DATATYPE_FLOAT_32;
            }
            Qnn_QuantizeParams_t tQuantizeParams = QNN_QUANTIZE_PARAMS_INIT;
            qnnTensorWrapper = QNNTensorWrapper::create(tName, QNN_TENSOR_TYPE_NATIVE, tDataType, tDims, tQuantizeParams);
        }

        Qnn_Tensor_t * qnnTensor = qnnTensorWrapper->getNativeTensor();
        CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, qnnTensor));
        mQNNTensorWrappers.push_back(qnnTensorWrapper);
        mTensorMap.insert({tensorKey, mTensorCounter});
        idx = mTensorCounter;
        mTensorCounter += 1;
    } else {
        idx = iter->second;
    }
    return idx;
}

void QnnBackend::addStaticTensorToGraph(Qnn_Tensor_t * staticTensor) {
    MNN_ASSERT(staticTensor->v1.type == QNN_TENSOR_TYPE_STATIC);
    CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, staticTensor));
}

void QnnBackend::addStageTensorToGraph(Qnn_Tensor_t * stageTensor) {
    MNN_ASSERT(stageTensor->v1.type == QNN_TENSOR_TYPE_NATIVE);
    CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, stageTensor));
}

Qnn_Tensor_t * QnnBackend::getNativeTensor(const Tensor * tensor) {
    int idx = getTensorIdx(tensor);
    if (idx < 0 || idx >= (int)mQNNTensorWrappers.size()) {
        MNN_ERROR("QNN getNativeTensor[%p]: invalid idx=%d (total=%d)\n",
                  this, idx, (int)mQNNTensorWrappers.size());
        return nullptr;
    }
    return mQNNTensorWrappers[idx]->getNativeTensor();
}

std::shared_ptr<QNNTensorWrapper> QnnBackend::getTensorWrapper(const Tensor * tensor) {
    const Tensor::InsideDescribe::NativeInsideDescribe * tensorKey = TensorUtils::getDescribe(tensor);
    auto iter = mTensorMap.find(tensorKey);
    MNN_ASSERT(iter != mTensorMap.end());
    return mQNNTensorWrappers[iter->second];
}

bool QnnBackend::getUseFP16() const {
    return mUseFP16;
}

void QnnBackend::clean() {
    MNN_PRINT("QNN clean[%p]: tensorWrappers=%d tensorMap=%d outputCastMap=%d "
              "extraInputs=%d extraOutputs=%d\n",
              this,
              (int)mQNNTensorWrappers.size(), (int)mTensorMap.size(),
              (int)mOutputCastTensorMap.size(),
              (int)mExtraInputWrappers.size(), (int)mExtraOutputWrappers.size());
    if (mQnnProfileHandle) {
        mRuntime->mQnnInterface.profileFree(mQnnProfileHandle);
        mQnnProfileHandle = nullptr;
    }
    freeContextAndGraph(); // This function must be called first.
    mTensorCounter = 0;
    mQNNTensorWrappers.clear();
    mTensorMap.clear();
    mInputTensorIndexes.clear();
    mOutputTensorIndexes.clear();
    mOutputTensors.clear();
    mDeQuantOutputTensorMap.clear();
    mInputCastTensorMap.clear();
    mOutputCastTensorMap.clear();
    mExtraInputWrappers.clear();
    mExtraOutputWrappers.clear();
    mDeferredInputs.clear();
    mGraphFinalized = false;
}

void QnnBackend::registerExtraTensor(Qnn_Tensor_t* tensor) {
    MNN_PRINT("QNN registerExtraTensor: name=%s type=%d dataType=%d rank=%d id_before=%u\n",
              tensor->v1.name, tensor->v1.type, tensor->v1.dataType, tensor->v1.rank, tensor->v1.id);
    auto ret = mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, tensor);
    int errorCode = ret & 0xFFFF;
    MNN_PRINT("QNN registerExtraTensor: id_after=%u errorCode=%d\n", tensor->v1.id, errorCode);
    if (errorCode != QNN_SUCCESS) {
        MNN_ERROR("QNN registerExtraTensor FAILED: error code %d\n", errorCode);
    }
}

void QnnBackend::registerExtraInput(std::shared_ptr<QNNTensorWrapper> wrapper) {
    mExtraInputWrappers.push_back(wrapper);
}

void QnnBackend::registerExtraOutput(std::shared_ptr<QNNTensorWrapper> wrapper) {
    mExtraOutputWrappers.push_back(wrapper);
}

std::shared_ptr<Tensor> QnnBackend::getInputDataContainer(const Tensor* tensor) const {
    // For FP16 mode, input tensors have a staging FP32 APP_WRITE tensor
    // in mInputCastTensorMap. Return that staging tensor's data container.
    auto castIter = mInputCastTensorMap.find(TensorUtils::getDescribe(tensor));
    if (castIter != mInputCastTensorMap.end()) {
        int idx = getTensorIdx(castIter->second.second.get());
        if (idx < 0 || idx >= (int)mQNNTensorWrappers.size()) {
            MNN_ERROR("QNN getInputDataContainer[%p]: invalid idx=%d (total=%d)\n",
                      this, idx, (int)mQNNTensorWrappers.size());
            return nullptr;
        }
        return mQNNTensorWrappers[idx]->getDataContainer();
    }
    // Non-FP16: the tensor itself is APP_WRITE with a data container
    int idx = getTensorIdx(tensor);
    if (idx < 0 || idx >= (int)mQNNTensorWrappers.size()) {
        MNN_ERROR("QNN getInputDataContainer[%p]: invalid idx=%d (total=%d)\n",
                  this, idx, (int)mQNNTensorWrappers.size());
        return nullptr;
    }
    return mQNNTensorWrappers[idx]->getDataContainer();
}
void QnnBackend::buildOutputDequant(){
    Qnn_OpConfigVersion_t mOpConfigVersion = QNN_OPCONFIG_VERSION_1;
    std::string mNodeName;
    std::string mPackageName = "qti.aisw";
    std::string mNodeType;
    std::vector<Qnn_Param_t> mParams;
    std::vector<Qnn_Tensor_t> mInputs;
    std::vector<Qnn_Tensor_t> mOutputs;
    for(auto iter : mDeQuantOutputTensorMap){
        auto* inputTensor = getNativeTensor(iter.second.first);
        auto* outputTensor = getNativeTensor(iter.second.second.get());
        if (!inputTensor || !outputTensor) {
            MNN_PRINT("QNN buildOutputDequant: skipping entry with missing tensor\n");
            continue;
        }
        mNodeType.clear();
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType = "Dequantize";
        std::string name = "Dequantize_I_" + std::to_string(getTensorIdx(iter.second.first)) + "_O_" + std::to_string(getTensorIdx(iter.second.second.get()));
        mInputs.push_back(*inputTensor);
        mOutputs.push_back(*outputTensor);
        addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
}

void QnnBackend::buildOutputCast(){
    MNN_PRINT("QNN buildOutputCast[%p]: entries=%d wrappers=%d tensorMap=%d\n",
              this, (int)mOutputCastTensorMap.size(), (int)mQNNTensorWrappers.size(),
              (int)mTensorMap.size());
    Qnn_OpConfigVersion_t mOpConfigVersion = QNN_OPCONFIG_VERSION_1;
    std::string mNodeName;
    std::string mPackageName = "qti.aisw";
    std::string mNodeType;
    std::vector<Qnn_Param_t> mParams;
    std::vector<Qnn_Tensor_t> mInputs;
    std::vector<Qnn_Tensor_t> mOutputs;
    for(auto iter : mOutputCastTensorMap){
        // Validate both tensors exist in mTensorMap before building Cast node
        int inputIdx = getTensorIdx(iter.second.first);
        int outputIdx = getTensorIdx(iter.second.second.get());
        if (inputIdx < 0 || inputIdx >= (int)mQNNTensorWrappers.size() ||
            outputIdx < 0 || outputIdx >= (int)mQNNTensorWrappers.size()) {
            MNN_PRINT("QNN buildOutputCast: skipping entry with invalid idx "
                      "(inputIdx=%d outputIdx=%d total=%d)\n",
                      inputIdx, outputIdx, (int)mQNNTensorWrappers.size());
            continue;
        }
        auto* inputTensor = mQNNTensorWrappers[inputIdx]->getNativeTensor();
        auto* outputTensor = mQNNTensorWrappers[outputIdx]->getNativeTensor();
        mNodeType.clear();
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType = "Cast";
        std::string name = "Cast_I_" + std::to_string(getTensorIdx(iter.second.first)) + "_O_" + std::to_string(getTensorIdx(iter.second.second.get()));
        mInputs.push_back(*inputTensor);
        mOutputs.push_back(*outputTensor);
        addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
}

void QnnBackend::buildInputCast(const Tensor *tensor){
    Qnn_OpConfigVersion_t mOpConfigVersion = QNN_OPCONFIG_VERSION_1;
    std::string mNodeName;
    std::string mPackageName = "qti.aisw";
    std::string mNodeType;
    std::vector<Qnn_Param_t> mParams;
    std::vector<Qnn_Tensor_t> mInputs;
    std::vector<Qnn_Tensor_t> mOutputs;
    mNodeType.clear();
    mParams.clear();
    mInputs.clear();
    mOutputs.clear();
    mNodeType = "Cast";
    auto iter = mInputCastTensorMap.find(TensorUtils::getDescribe(tensor));
    if(iter != mInputCastTensorMap.end()){
        auto* inputNative = getNativeTensor(iter->second.second.get());
        auto* outputNative = getNativeTensor(iter->second.first);
        if (!inputNative || !outputNative) {
            MNN_ERROR("QNN buildInputCast[%p]: skipping, null tensor (input=%p output=%p)\n",
                      this, inputNative, outputNative);
            return;
        }
        std::string name = "Cast_I_" + std::to_string(getTensorIdx(iter->second.second.get())) + "_O_" + std::to_string(getTensorIdx(iter->second.first));
        mInputs.push_back(*inputNative);
        mOutputs.push_back(*outputNative);
        addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
}

QnnRuntime::QnnRuntime(const Backend::Info& info, QNN_INTERFACE_VER_TYPE qnnInterface, Qnn_LogHandle_t qnnLogHandle, Qnn_BackendHandle_t qnnBackendHandle, Qnn_DeviceHandle_t qnnDeviceHandle) {
    // MNN_PRINT("QnnRuntime is constructing.\n");
    mInfo = info;
    // Default setting
    mPower = BackendConfig::Power_Normal;
    mMemory = BackendConfig::Memory_Normal;
    mPrecision = BackendConfig::Precision_Normal;
    // User setting
    if (info.user != nullptr) {
        mPrecision = info.user->precision;
        mPower = info.user->power;
        mMemory = info.user->memory;
    }
    mQnnInterface = qnnInterface;
    mQnnLogHandle = qnnLogHandle;
    mQnnBackendHandle = qnnBackendHandle;
    mQnnDeviceHandle = qnnDeviceHandle;
}

QnnRuntime::~QnnRuntime() {
    if (nullptr != mQnnContextHandle) {
        CALL_QNN(mQnnInterface.contextFree(mQnnContextHandle, nullptr));
    }
}
bool QnnRuntime::onSetCache(const void* buffer, size_t size) {
    // TODO: Fix bug and complete
    return false;
    if (nullptr == buffer) {
        return false;
    }
    auto error = mQnnInterface.contextValidateBinary(mQnnBackendHandle, mQnnDeviceHandle, mQnnContextConfig, buffer, size);
    if (QNN_SUCCESS != error) {
        MNN_ERROR("QNN: Failed to validate binary: %d\n", (int) error);
        return false;
    }
    freeContext();
    CALL_QNN(mQnnInterface.contextCreateFromBinary(mQnnBackendHandle, mQnnDeviceHandle, mQnnContextConfig, buffer, size, &mQnnContextHandle, nullptr));
    mUseCache = true;
    return true;
}
void QnnRuntime::allocContext() const {
    CALL_QNN(mQnnInterface.contextCreate(mQnnBackendHandle, mQnnDeviceHandle, mQnnContextConfig, &mQnnContextHandle));
    MNN_ASSERT(mQnnContextHandle != nullptr);
}
void QnnRuntime::freeContext() const {
    if (nullptr != mQnnContextHandle) {
        CALL_QNN(mQnnInterface.contextFree(mQnnContextHandle, nullptr));
        mQnnContextHandle = nullptr;
        mBinaryBuffer.clear();
    }
}

std::pair<const void*, size_t> QnnRuntime::onGetCache() {
    return std::make_pair(nullptr, 0);
    if (!mBinaryBuffer.empty()) {
        return std::make_pair(mBinaryBuffer.data(), mBinaryBuffer.size());
    }
    if (nullptr == mQnnContextHandle) {
        return std::make_pair(nullptr, 0);
    }
    Qnn_ContextBinarySize_t size = 0;
    CALL_QNN(mQnnInterface.contextGetBinarySize(mQnnContextHandle, &size));
    FUNC_PRINT(size);
    if (0 == size) {
        return std::make_pair(nullptr, 0);
    }
    mBinaryBuffer.resize(size);
    Qnn_ContextBinarySize_t writesize = 0;
    CALL_QNN(mQnnInterface.contextGetBinary(mQnnContextHandle, mBinaryBuffer.data(), size, &writesize));
    return std::make_pair(mBinaryBuffer.data(), mBinaryBuffer.size());
}

Backend* QnnRuntime::onCreate(const BackendConfig* config, Backend* origin) const {
    return new QnnBackend(this);
}

QnnRuntime* QnnRuntime::create(const Backend::Info& info) {
    if (QNN::gContext.deviceHandle == nullptr){
        QNN::createQnnContext();
    }
    // Create Interface.
    return new QnnRuntime(info, gContext.interface, gContext.logHandle, gContext.backendHandle, gContext.deviceHandle);
}

// Do nothing
void QnnRuntime::onGabageCollect(int level) {}

Runtime::CompilerType QnnRuntime::onGetCompilerType() const {
    return Compiler_Origin;
}

bool QnnRuntime::onSetCachePath(const char* path, int mode) {
#ifdef ENABLE_QNN_CONVERT_MODE
    MNN_ASSERT(path != nullptr);
    QNNConvertor::OutputDir = std::string(path);
    MNNCreateDir(path);
#endif
    return true;
}

bool QnnRuntime::registerCustomOpPackage(QNN_INTERFACE_VER_TYPE qnnInterface, Qnn_BackendHandle_t backendHandle, const std::string & path, const std::string & interfaceProvider, const std::string & target) {
    if (QNN_GET_ERROR_CODE(qnnInterface.backendRegisterOpPackage(backendHandle, path.c_str(), interfaceProvider.c_str(), target.c_str())) != QNN_SUCCESS) {
        MNN_PRINT("MNN_QNN: Failed to register the Op Package: %s.\n", path.c_str());
        return false;
    }
    return true;
}

class QnnRuntimeCreator : public RuntimeCreator {
public:
    virtual Runtime* onCreate(const Backend::Info& info) const override {
        return QnnRuntime::create(info);
    }
    static bool _supportQuant(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
        auto otype = op->type();
        std::set<OpType> judgOneInputOpTypes = { OpType_Slice, OpType_StridedSlice, OpType_GatherV2, OpType_Reshape, OpType_Unsqueeze, OpType_Flatten, OpType_Squeeze};
        if(judgOneInputOpTypes.find(otype) != judgOneInputOpTypes.end()){
            if (TensorUtils::getDescribe(inputs[0])->quantAttr == nullptr) {
                return false;
            }
        }else{
            for (auto t : inputs) {
                auto des = TensorUtils::getDescribe(t);
                if (des->quantAttr == nullptr) {
                    return false;
                }
            }
        }
        auto quantType = TensorUtils::getDescribe(inputs[0])->quantAttr->type;
        switch (otype) {
            case OpType_Convolution:
            case OpType_ConvolutionDepthwise:
                if (inputs.size() > 1) {
                    return false;
                }
                if (op->main_as_Convolution2D() && op->main_as_Convolution2D()->weight() != nullptr) {
                    return false;
                } else {
                    return true;
                }
            case OpType_ReLU:
                if ((op->main_as_Relu() == nullptr) || op->main_as_Relu()->slope() == 0.f) {
                    return true;
                } else {
                    return false;
                }
            case OpType_LayerNorm:
                if(quantType == DataType_DT_INT16){
                    return true;
                }else{
                    //ToDo :support featuremap int8 quant
                    return false;
                }
            case OpType_Scale:
            case OpType_Attention:
                return false;
            default:
                break;
        }
        return true;
    }
    virtual bool onSetQuantInfo(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) const override {
        if (nullptr == op) {
            return true;
        }
        auto res = _supportQuant(op, inputs, outputs);
        for (auto t : outputs) {
            TensorUtils::getDescribe(t)->applyQuant = res;
        }
        return res;
    }
    virtual bool onValid(Backend::Info& info) const override {
        return true;
    }
    virtual bool onGetDeviceInfo(const std::string& deviceKey, std::string& deviceValue) const override {
        if(deviceKey == "soc_id" && gContext.soc_id != 0) {
            deviceValue = std::to_string(gContext.soc_id);
            return true;
        }
        if(deviceKey == "dsp_arch" && gContext.dsp_arch != 0) {
            deviceValue = "v" + std::to_string(gContext.dsp_arch);
            return true;
        }
        return false;
    }
};
#endif
} // end namespace QNN

void registerQNNRuntimeCreator() {
#ifndef ENABLE_QNN_CONVERT_MODE
    // check whether the qnn lib is available
    if (!QNN::loadQNNSymbol()) {
        return;
    }
#endif

#ifdef ENABLE_QNN_ONLINE_FINALIZE
    QNN::registerQNNOps();
    MNNInsertExtraRuntimeCreator(QNN_FORWARD_TYPE, new QNN::QnnRuntimeCreator, false);
#endif

#ifdef MNN_WITH_PLUGIN
    plugin::InferShapeKernelRegister::add("QNN", []() { // NOLINT
        return new plugin::shape_inference::PluginShapeRaw;               // NOLINT
    });
    plugin::ComputeKernelRegistry<plugin::backend::PluginExecuteRaw::KernelT>::add("QNN", []() {
        return new plugin::backend::PluginExecuteRaw;
    });
#endif
}

} // end namespace MNN
