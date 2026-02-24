#include "QNNAttention.hpp"
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include "core/OpCommonUtils.hpp"
#endif
namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

// #define GQA_USE_GATHER
/*
seqLenQ == seqLenKV
query : [Batch, seqLenQ,  headNum, headDim] -> (real layout) [Batch, headNum, headDim, seqLenQ]
key   : [Batch, seqLenKV, headNum, headDim] -> (real layout) [Batch, headNum, headDim, seqLenKV]
value : [Batch, seqLenKV, headNum, headDim] -> (real layout) [Batch, headNum, headDim, seqLenKV]
ouput : [Batch, seqLenQ, headNum * headDim] -> (real layout) [Batch, headNum * headDim, seqLenQ]
*/
ErrorCode QNNAttention::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // Validate inputs (may be empty at onCreate time in fused transformer mode)
    if (inputs.size() < 3 || inputs.size() > 4 || outputs.size() != 1) {
        MNN_ERROR("QNN Attention: invalid inputs(%d)/outputs(%d)\n",
                  (int)inputs.size(), (int)outputs.size());
        return NOT_SUPPORT;
    }
    if (inputs[0]->dimensions() != 4 || inputs[1]->dimensions() != 4 ||
        inputs[2]->dimensions() != 4 || outputs[0]->dimensions() != 3) {
        MNN_ERROR("QNN Attention: invalid tensor dimensions\n");
        return NOT_SUPPORT;
    }
#ifdef QNN_VERBOSE
    MNN_PRINT("QNN Attention inputs shape:\n");
    for(int i = 0; i < inputs.size(); i++) {
        auto shape = inputs[i]->shape();
        for(int j = 0; j < shape.size(); j++) {
            MNN_PRINT("%d ", shape[j]);
        }
        MNN_PRINT("\n");
    }
    MNN_PRINT("QNN Attention outputs shape:\n");
    for(int i = 0; i < outputs.size(); i++) {
        auto shape = outputs[i]->shape();
        for(int j = 0; j < shape.size(); j++) {
            MNN_PRINT("%d ", shape[j]);
        }
        MNN_PRINT("\n");
    }
#endif
    auto shape = inputs[0]->shape();
    int batch = shape[0];
    int seqLen = shape[1];
    int headNum = shape[2];
    int headDim = shape[3];
    int seqLenQ = seqLen;
    int kvHeadNum = inputs[1]->length(2);
    int seqLenKV = inputs[1]->length(1);
    float scale = 1.0 / sqrt(headDim);
    Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;
    this->createStageTensor("Query_perm", dataType, std::vector<int>({batch, headNum, headDim, seqLenQ})); // mTempTensorWrappers[0], stage query
    this->createStageTensor("Key_perm", dataType, std::vector<int>({batch, kvHeadNum, headDim, seqLenKV})); // mTempTensorWrappers[1], stage key
    this->createStageTensor("Value_perm", dataType, std::vector<int>({batch, kvHeadNum, headDim, seqLenKV})); // mTempTensorWrappers[2], stage value
    this->createStageTensor("ScaleQ", dataType, std::vector<int>({batch, headNum, headDim, seqLenQ})); // mTempTensorWrappers[3], stage Scale
    this->createStageTensor("QK", dataType, std::vector<int>({batch, headNum, seqLenQ, seqLenKV})); // mTempTensorWrappers[4], stage QK
    this->createStageTensor("Softmax", dataType, std::vector<int>({batch, headNum, seqLenQ, seqLenKV})); // mTempTensorWrappers[5], stage Softmax
    this->createStageTensor("QKV", dataType, std::vector<int>({batch, headNum, seqLenQ, headDim})); // mTempTensorWrappers[6], stage QKV
    this->createStageTensor("Transpose", dataType, std::vector<int>({batch, seqLenQ, headNum, headDim})); // mTempTensorWrappers[7], stage Transpose

    size_t totalSize = batch * headNum * seqLenQ * headDim;
    std::vector<float> scaleVec(totalSize, scale);
    // mTempTensorWrappers[5], static coef
    this->createStaticFloatTensor("coef", dataType, std::vector<uint32_t>({(uint32_t)batch, (uint32_t)headNum, (uint32_t)headDim, (uint32_t)seqLenQ}), scaleVec.data());

    std::vector<uint32_t> mapReal{0, 2, 3, 1};
    std::vector<uint32_t> mapOutputReal{0, 2, 1, 3};
    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) 4}, mapReal.data(), "input_query"); // mParamTensorWrappers[0]
    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) 4}, mapReal.data(), "input_key"); // mParamTensorWrappers[1]
    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) 4}, mapReal.data(), "input_value"); // mParamTensorWrappers[2]
    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) 4}, mapOutputReal.data(), "output_trans"); // mParamTensorWrappers[3]

    // transpose input
    {
        // transpose query
        {
            std::string name = mNodeName + "_Transpose_query";
            mNodeType = "Transpose";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); // input0
            mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // perm_query
            mOutputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // stage query

            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        // transpose key
        {
            std::string name = mNodeName + "_Transpose_key";
            mNodeType = "Transpose";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[1]))); // input1
            mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // perm_key
            mOutputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // stage key

            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        // transpose value
        {
            std::string name = mNodeName + "_Transpose_value";
            mNodeType = "Transpose";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[2]))); // input2
            mParams.push_back(*(mParamTensorWrappers[2]->getNativeParam())); // perm_value
            mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // stage value

            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
    }

    // GQA
    bool isGQA = (headNum != kvHeadNum);
    int tensorNumGQA = 0;
    int group = headNum / kvHeadNum;
    if(isGQA) {
        this->createStageTensor("RepeatedKey", dataType, std::vector<int>({batch, headNum, headDim, seqLenKV})); // mTempTensorWrappers[9], stage RepeatedKey
        this->createStageTensor("RepeatedValue", dataType, std::vector<int>({batch, headNum, headDim, seqLenKV})); // mTempTensorWrappers[10], stage RepeatedValue

        #ifdef GQA_USE_GATHER
        // index: fill in Key and Value to shape of Query
        // [a0, a1, ..., a(kvHeadNum-1)] -> [a0 ... a0, a1 ... a1, a(kvHeadNum-1) ... a(kvHeadNum-1)]
        std::vector<int32_t> index(totalSize);
        for(int b = 0; b < batch; b++) {
            int base_index = 0;
            for(int h = 0; h < kvHeadNum; h++) {
                for(int a = 0; a < group * headDim * seqLenKV; a++) {
                    index[(b * kvHeadNum + h) * group * headDim * seqLenKV + a] = base_index;
                }
                base_index++;
            }
        }
        this->createStaticTensor("gather_index", QNN_DATATYPE_INT_32, {(uint32_t)batch, (uint32_t)headNum, (uint32_t)headDim, (uint32_t)seqLenKV}, index.data());
        tensorNumGQA = 3;
        #else

        std::vector<uint32_t> splitIndex(kvHeadNum-1);
        for(int i = 0; i < splitIndex.size(); i++) {
            splitIndex[i] = i + 1;
        }
        // mParamTensorWrappers[4]
        this->createParamTensor("split_index", QNN_DATATYPE_UINT_32, {(uint32_t)kvHeadNum-1}, (void *)splitIndex.data(), "K_Split");
        // mTempTensorWrappers[11] .. [10+kvHeadNum] stage SplitKV_Temp
        for(int i = 0; i < kvHeadNum; i++) {
            this->createStageTensor("SplitK_Temp" + std::to_string(i), dataType, std::vector<int>({batch, 1, headDim, seqLenKV}));
        }
        // mParamTensorWrappers[5]
        this->createParamTensor("split_index", QNN_DATATYPE_UINT_32, {(uint32_t)kvHeadNum-1}, (void *)splitIndex.data(), "V_Split");
        // mTempTensorWrappers[11+kvHeadNum] .. [10+2*kvHeadNum] stage SplitKV_Temp
        for(int i = 0; i < kvHeadNum; i++) {
            this->createStageTensor("SplitV_Temp" + std::to_string(i), dataType, std::vector<int>({batch, 1, headDim, seqLenKV}));
        }
        tensorNumGQA = 2 + 2*kvHeadNum;
        #endif

        this->createParamScalar("axis", (uint32_t)1);
    }
    bool hasMask = (inputs.size() > 3);
    int maskPosIndex = 9 + tensorNumGQA;
    int scalarBaseIndex = isGQA ? 1 : 0;
    if(hasMask) {
        this->createStageTensor("tempMask", dataType, std::vector<int>({batch, 1, seqLenQ, seqLenKV})); // mTempTensorWrappers[maskPosIndex], stage Mask
        this->createStageTensor("maskResult", dataType, std::vector<int>({batch, headNum, seqLenQ, seqLenKV})); // mTempTensorWrappers[maskPosIndex+1], stage Mask
    }

    // scale
    {
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_Scale";
        mNodeType = "ElementWiseMultiply";
        mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); //stage query
        mInputs.push_back(*(mTempTensorWrappers[8]->getNativeTensor())); // coef
        mOutputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); // ScaleQ

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // Q * K
    {
        auto tempK = *(mTempTensorWrappers[1]->getNativeTensor());
        if(isGQA) {
            #ifdef GQA_USE_GATHER
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            std::string name = mNodeName + "_K_Repeat";
            mNodeType = "GatherElements";

            mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // stage key
            mInputs.push_back(*(mTempTensorWrappers[11]->getNativeTensor())); // gather_index
            mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // axis
            mOutputs.push_back(*(mTempTensorWrappers[9]->getNativeTensor())); // stage RepeatedKey

            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            #else
            {
                mParams.clear();
                mInputs.clear();
                mOutputs.clear();
                std::string name = mNodeName + "_K_Split";
                mNodeType = "Split";

                mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // stage key
                mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // axis
                mParams.push_back(*(mParamTensorWrappers[4]->getNativeParam())); // split_index
                for(int i = 0; i < kvHeadNum; i++) {
                    mOutputs.push_back(*(mTempTensorWrappers[11+i]->getNativeTensor())); // stage TempKey
                }
                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }
            {
                mParams.clear();
                mInputs.clear();
                mOutputs.clear();
                std::string name = mNodeName + "_K_Concat";
                mNodeType = "Concat";

                for(int i = 0; i < kvHeadNum; i++) {
                    for(int j = 0; j < group; j++) {
                        mInputs.push_back(*(mTempTensorWrappers[11+i]->getNativeTensor())); // stage TempKey
                    }
                }
                mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // axis
                mOutputs.push_back(*(mTempTensorWrappers[9]->getNativeTensor())); // stage TempKey
                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }
            #endif
            tempK = *(mTempTensorWrappers[9]->getNativeTensor());
        }
        bool transpose0 = true;
        bool transpose1 = false;
        this->createParamScalar("transpose_in0", transpose0); // mParamScalarWrappers[scalarBaseIndex + 0], transpose_in0
        this->createParamScalar("transpose_in1", transpose1); // mParamScalarWrappers[scalarBaseIndex + 1], transpose_in1

        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_MatMul_QK";
        mNodeType = "MatMul";
        mInputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); //ScaleQ
        mInputs.push_back(tempK); // input1
        mParams.push_back(*(mParamScalarWrappers[scalarBaseIndex + 0]->getNativeParam()));  // transpose0
        mParams.push_back(*(mParamScalarWrappers[scalarBaseIndex + 1]->getNativeParam()));  // transpose1
        mOutputs.push_back(*(mTempTensorWrappers[4]->getNativeTensor())); // QK

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    auto softmax_in = *(mTempTensorWrappers[4]->getNativeTensor());

    // mask
    if(hasMask)
    {
        if(inputs[3]->getType() != halide_type_of<float>()) {
            MNN_ERROR("Qnn attention only support float mask currently\n");
        }
        // mask reshape
        {
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            std::string name = mNodeName + "_Mask_Reshape";
            mNodeType = "Reshape";
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[3]))); // stage mask
            mOutputs.push_back(*(mTempTensorWrappers[maskPosIndex]->getNativeTensor())); // tempMask

            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }

        // mask compute
        {
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            std::string name = mNodeName + "_Mask_Add";
            mNodeType = "ElementWiseAdd";
            mInputs.push_back(*(mTempTensorWrappers[4]->getNativeTensor())); // QK stage
            mInputs.push_back(*(mTempTensorWrappers[maskPosIndex]->getNativeTensor())); // stage tempMask
            mOutputs.push_back(*(mTempTensorWrappers[maskPosIndex + 1]->getNativeTensor())); //

            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        softmax_in = *(mTempTensorWrappers[maskPosIndex + 1]->getNativeTensor());
    }

    // softmax
    {
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_Softmax";
        mNodeType = "Softmax";
        mInputs.push_back(softmax_in);
        mOutputs.push_back(*(mTempTensorWrappers[5]->getNativeTensor()));// Stage Softmax

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    // QK * V
    {
        auto tempV = *(mTempTensorWrappers[2]->getNativeTensor());
        if(isGQA) {
            #ifdef GQA_USE_GATHER
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            std::string name = mNodeName + "_V_Repeat";
            mNodeType = "GatherElements";

            mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // stage value
            mInputs.push_back(*(mTempTensorWrappers[11]->getNativeTensor())); // gather_index
            mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // axis
            mOutputs.push_back(*(mTempTensorWrappers[10]->getNativeTensor())); // stage RepeatedValue

            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            #else
            {
                mParams.clear();
                mInputs.clear();
                mOutputs.clear();
                std::string name = mNodeName + "_V_Split";
                mNodeType = "Split";

                mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // stage value
                mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // axis
                mParams.push_back(*(mParamTensorWrappers[5]->getNativeParam())); // split_index
                for(int i = 0; i < kvHeadNum; i++) {
                    mOutputs.push_back(*(mTempTensorWrappers[11+kvHeadNum+i]->getNativeTensor())); // stage TempValue
                }
                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }
            {
                mParams.clear();
                mInputs.clear();
                mOutputs.clear();
                std::string name = mNodeName + "_V_Concat";
                mNodeType = "Concat";

                for(int i = 0; i < kvHeadNum; i++) {
                    for(int j = 0; j < group; j++) {
                        mInputs.push_back(*(mTempTensorWrappers[11+kvHeadNum+i]->getNativeTensor())); // stage TempKey
                    }
                }
                mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // axis
                mOutputs.push_back(*(mTempTensorWrappers[10]->getNativeTensor())); // stage TempKey
                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }
            #endif
            tempV = *(mTempTensorWrappers[10]->getNativeTensor());
        }
        bool transpose0 = false;
        bool transpose1 = true;
        this->createParamScalar("transpose_in0", transpose0); // mParamScalarWrappers[scalarBaseIndex + 2], transpose_in0
        this->createParamScalar("transpose_in1", transpose1); // mParamScalarWrappers[scalarBaseIndex + 3], transpose_in1

        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_MatMul_QKV";
        mNodeType = "MatMul";
        mInputs.push_back(*(mTempTensorWrappers[5]->getNativeTensor())); //Softmax
        mInputs.push_back(tempV); // input2
        mParams.push_back(*(mParamScalarWrappers[scalarBaseIndex + 2]->getNativeParam()));  // transpose0
        mParams.push_back(*(mParamScalarWrappers[scalarBaseIndex + 3]->getNativeParam()));  // transpose1
        mOutputs.push_back(*(mTempTensorWrappers[6]->getNativeTensor())); // QKV

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // Transpose
    {
        std::string name = mNodeName + "_Transpose";
        mNodeType = "Transpose";
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mInputs.push_back(*(mTempTensorWrappers[6]->getNativeTensor())); // QKV
        mParams.push_back(*(mParamTensorWrappers[3]->getNativeParam())); // perm
        mOutputs.push_back(*(mTempTensorWrappers[7]->getNativeTensor())); // Transpose

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    // Reshape
    {
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_Reshape";
        mNodeType = "Reshape";

        mInputs.push_back(*(mTempTensorWrappers[7]->getNativeTensor())); // Transpose
        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    return NO_ERROR;
}

// ============================================================================
// QNNKVCacheAttention: fixed-size KV buffer approach for KV cache on QNN/NPU
// ============================================================================
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

QNNKVCacheAttention::QNNKVCacheAttention(Backend *backend, const Op *op)
    : QNNCommonExecution(backend, op) {
    mState = std::make_shared<KVState>();
}

ErrorCode QNNKVCacheAttention::onResize(const std::vector<Tensor *> &inputs,
                                        const std::vector<Tensor *> &outputs) {
    if (inputs.size() < 3 || outputs.size() < 1) {
        MNN_ERROR("QNNKVCacheAttention: invalid inputs(%d)/outputs(%d)\n",
                  (int)inputs.size(), (int)outputs.size());
        return NOT_SUPPORT;
    }

    // Q: [batch, seqQ, numHead, headDim],  K: [batch, seqQ, kvNumHead, headDim]
    mNumHead = inputs[0]->length(2);
    mKvNumHead = inputs[1]->length(2);
    mHeadDim = inputs[0]->length(3);

    int elemSize = mBackend->getUseFP16() ? 2 : 4;
    int cacheSizeBytes = mMaxKVLen * mKvNumHead * mHeadDim * elemSize;
    if ((int)mState->keyCache.size() != cacheSizeBytes) {
        mState->keyCache.resize(cacheSizeBytes, 0);
        mState->valueCache.resize(cacheSizeBytes, 0);
        mState->elemSize = elemSize;
    }

    return QNNCommonExecution::onResize(inputs, outputs);
}

ErrorCode QNNKVCacheAttention::onEncode(const std::vector<Tensor *> &inputs,
                                        const std::vector<Tensor *> &outputs) {
    auto shape = inputs[0]->shape();
    int batch = shape[0];
    int seqQ = shape[1];
    int headNum = shape[2];
    int headDim = shape[3];
    int kvHeadNum = inputs[1]->length(2);
    int seqNewKV = inputs[1]->length(1);
    float scale = 1.0 / sqrt(headDim);
    mGraphDataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;

    MNN_PRINT("QNNKVCacheAttention::onEncode batch=%d seqQ=%d headNum=%d kvHeadNum=%d "
              "headDim=%d maxKVLen=%d seqNew=%d dtype=%d\n",
              batch, seqQ, headNum, kvHeadNum, headDim, mMaxKVLen, seqNewKV,
              (int)mGraphDataType);

    // ================================================================
    // Stage tensors — all KV dims are fixed at maxKVLen (no +seqNew)
    // ================================================================
    // [0] Q_perm: [B, H, D, seqQ]
    this->createStageTensor("Q_perm", mGraphDataType,
        std::vector<int>({batch, headNum, headDim, seqQ}));
    // [1] K_full: ScatterElements output [B, maxKVLen, kvH, D]
    this->createStageTensor("K_full", mGraphDataType,
        std::vector<int>({batch, mMaxKVLen, kvHeadNum, headDim}));
    // [2] V_full: ScatterElements output [B, maxKVLen, kvH, D]
    this->createStageTensor("V_full", mGraphDataType,
        std::vector<int>({batch, mMaxKVLen, kvHeadNum, headDim}));
    // [3] K_perm: [B, kvH, D, maxKVLen]
    this->createStageTensor("K_perm", mGraphDataType,
        std::vector<int>({batch, kvHeadNum, headDim, mMaxKVLen}));
    // [4] V_perm: [B, kvH, D, maxKVLen]
    this->createStageTensor("V_perm", mGraphDataType,
        std::vector<int>({batch, kvHeadNum, headDim, mMaxKVLen}));
    // [5] ScaleQ: [B, H, D, seqQ]
    this->createStageTensor("ScaleQ", mGraphDataType,
        std::vector<int>({batch, headNum, headDim, seqQ}));
    // [6] QK: [B, H, seqQ, maxKVLen]
    this->createStageTensor("QK", mGraphDataType,
        std::vector<int>({batch, headNum, seqQ, mMaxKVLen}));
    // [7] Softmax: [B, H, seqQ, maxKVLen]
    this->createStageTensor("Softmax", mGraphDataType,
        std::vector<int>({batch, headNum, seqQ, mMaxKVLen}));
    // [8] QKV: [B, H, seqQ, D]
    this->createStageTensor("QKV", mGraphDataType,
        std::vector<int>({batch, headNum, seqQ, headDim}));
    // [9] Transpose_out: [B, seqQ, H, D]
    this->createStageTensor("Transpose_out", mGraphDataType,
        std::vector<int>({batch, seqQ, headNum, headDim}));
    // [10] scale coef (static)
    size_t scaleSize = batch * headNum * seqQ * headDim;
    std::vector<float> scaleVec(scaleSize, scale);
    this->createStaticFloatTensor("coef", mGraphDataType,
        std::vector<uint32_t>({(uint32_t)batch, (uint32_t)headNum,
                               (uint32_t)headDim, (uint32_t)seqQ}),
        scaleVec.data());

    // ================================================================
    // Param tensors
    // ================================================================
    std::vector<uint32_t> mapReal{0, 2, 3, 1};       // [B,S,H,D] → [B,H,D,S]
    std::vector<uint32_t> mapOutputReal{0, 2, 1, 3};  // [B,H,S,D] → [B,S,H,D]
    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t)4},
                            mapReal.data(), "input_query");       // [0]
    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t)4},
                            mapReal.data(), "input_fullK");       // [1]
    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t)4},
                            mapReal.data(), "input_fullV");       // [2]
    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t)4},
                            mapOutputReal.data(), "output_trans"); // [3]

    // ================================================================
    // Param scalars: axis=1 for ScatterElements and GQA Concat
    // ================================================================
    this->createParamScalar("axis", (uint32_t)1);  // [0]

    // ================================================================
    // APP_WRITE: KV cache (clientBuf redirected to KVState in onExecute)
    // ================================================================
    mKCacheWrapper = QNNTensorWrapper::create(
        mNodeName + "_Kcache", QNN_TENSOR_TYPE_APP_WRITE, mGraphDataType,
        std::vector<int>({batch, mMaxKVLen, kvHeadNum, headDim}));
    mBackend->registerExtraTensor(mKCacheWrapper->getNativeTensor());
    mKCacheWrapper->alloc();
    mBackend->registerExtraInput(mKCacheWrapper);

    mVCacheWrapper = QNNTensorWrapper::create(
        mNodeName + "_Vcache", QNN_TENSOR_TYPE_APP_WRITE, mGraphDataType,
        std::vector<int>({batch, mMaxKVLen, kvHeadNum, headDim}));
    mBackend->registerExtraTensor(mVCacheWrapper->getNativeTensor());
    mVCacheWrapper->alloc();
    mBackend->registerExtraInput(mVCacheWrapper);

    // APP_WRITE: scatter indices [B, seqNew, kvH, D] — int32
    mIndicesWrapper = QNNTensorWrapper::create(
        mNodeName + "_indices", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_INT_32,
        std::vector<int>({batch, seqNewKV, kvHeadNum, headDim}));
    mBackend->registerExtraTensor(mIndicesWrapper->getNativeTensor());
    mIndicesWrapper->alloc();
    mBackend->registerExtraInput(mIndicesWrapper);

    // APP_WRITE: mask [B, 1, seqQ, maxKVLen]
    mMaskWrapper = QNNTensorWrapper::create(
        mNodeName + "_mask", QNN_TENSOR_TYPE_APP_WRITE, mGraphDataType,
        std::vector<int>({batch, 1, seqQ, mMaxKVLen}));
    mBackend->registerExtraTensor(mMaskWrapper->getNativeTensor());
    mMaskWrapper->alloc();
    mBackend->registerExtraInput(mMaskWrapper);

    // ================================================================
    // APP_READ sinks: capture K_new / V_new during graphExecute
    // ================================================================
    mKNewSinkWrapper = QNNTensorWrapper::create(
        mNodeName + "_KnewSink", QNN_TENSOR_TYPE_APP_READ, mGraphDataType,
        std::vector<int>({batch, seqNewKV, kvHeadNum, headDim}));
    mBackend->registerExtraTensor(mKNewSinkWrapper->getNativeTensor());
    mKNewSinkWrapper->alloc();
    mBackend->registerExtraOutput(mKNewSinkWrapper);

    mVNewSinkWrapper = QNNTensorWrapper::create(
        mNodeName + "_VnewSink", QNN_TENSOR_TYPE_APP_READ, mGraphDataType,
        std::vector<int>({batch, seqNewKV, kvHeadNum, headDim}));
    mBackend->registerExtraTensor(mVNewSinkWrapper->getNativeTensor());
    mVNewSinkWrapper->alloc();
    mBackend->registerExtraOutput(mVNewSinkWrapper);

    // ================================================================
    // Graph ops
    // ================================================================

    // 1. Identity copy: inputs[1]→Knew sink, inputs[2]→Vnew sink
    {
        CLEAR_BEFORE_ADDING_NODE;
        std::string name = mNodeName + "_Copy_Knew";
        mNodeType = "Reshape";
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[1])));
        mOutputs.push_back(*(mKNewSinkWrapper->getNativeTensor()));
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(),
            mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    {
        CLEAR_BEFORE_ADDING_NODE;
        std::string name = mNodeName + "_Copy_Vnew";
        mNodeType = "Reshape";
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[2])));
        mOutputs.push_back(*(mVNewSinkWrapper->getNativeTensor()));
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(),
            mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // 2. ScatterElements: merge K_cache + K_new at positions [pastLen..pastLen+seqNew)
    //    output is fixed [B, maxKVLen, kvH, D] — no growing tensor
    {
        CLEAR_BEFORE_ADDING_NODE;
        std::string name = mNodeName + "_Scatter_K";
        mNodeType = "ScatterElements";
        mInputs.push_back(*(mKCacheWrapper->getNativeTensor()));           // data [B,maxKVLen,kvH,D]
        mInputs.push_back(*(mIndicesWrapper->getNativeTensor()));          // indices [B,seqNew,kvH,D]
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[1])));        // updates [B,seqNew,kvH,D]
        mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam()));   // axis=1
        mOutputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor()));  // K_full
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(),
            mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    {
        CLEAR_BEFORE_ADDING_NODE;
        std::string name = mNodeName + "_Scatter_V";
        mNodeType = "ScatterElements";
        mInputs.push_back(*(mVCacheWrapper->getNativeTensor()));
        mInputs.push_back(*(mIndicesWrapper->getNativeTensor()));
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[2])));
        mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam()));
        mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor()));  // V_full
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(),
            mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // 3. Transpose Q, K_full, V_full
    {
        CLEAR_BEFORE_ADDING_NODE;
        std::string name = mNodeName + "_Transpose_Q";
        mNodeType = "Transpose";
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
        mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam()));
        mOutputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(),
            mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    {
        CLEAR_BEFORE_ADDING_NODE;
        std::string name = mNodeName + "_Transpose_K";
        mNodeType = "Transpose";
        mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor()));
        mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam()));
        mOutputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor()));
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(),
            mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    {
        CLEAR_BEFORE_ADDING_NODE;
        std::string name = mNodeName + "_Transpose_V";
        mNodeType = "Transpose";
        mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor()));
        mParams.push_back(*(mParamTensorWrappers[2]->getNativeParam()));
        mOutputs.push_back(*(mTempTensorWrappers[4]->getNativeTensor()));
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(),
            mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // ================================================================
    // GQA: repeat K/V heads if needed
    // ================================================================
    bool isGQA = (headNum != kvHeadNum);
    int group = headNum / kvHeadNum;
    int gqaTempBase = 11;  // next available stage index
    int scalarBaseIndex = 1;  // [0]=scatter axis

    if (isGQA) {
        this->createStageTensor("RepeatedKey", mGraphDataType,
            std::vector<int>({batch, headNum, headDim, mMaxKVLen}));  // [11]
        this->createStageTensor("RepeatedValue", mGraphDataType,
            std::vector<int>({batch, headNum, headDim, mMaxKVLen}));  // [12]

        std::vector<uint32_t> splitIndex(kvHeadNum - 1);
        for (int i = 0; i < (int)splitIndex.size(); i++) splitIndex[i] = i + 1;

        this->createParamTensor("split_index", QNN_DATATYPE_UINT_32,
            {(uint32_t)kvHeadNum - 1}, (void*)splitIndex.data(), "K_Split"); // [4]
        int splitTempBase = (int)mTempTensorWrappers.size();
        for (int i = 0; i < kvHeadNum; i++)
            this->createStageTensor("SplitK_Temp" + std::to_string(i), mGraphDataType,
                std::vector<int>({batch, 1, headDim, mMaxKVLen}));
        this->createParamTensor("split_index", QNN_DATATYPE_UINT_32,
            {(uint32_t)kvHeadNum - 1}, (void*)splitIndex.data(), "V_Split"); // [5]
        for (int i = 0; i < kvHeadNum; i++)
            this->createStageTensor("SplitV_Temp" + std::to_string(i), mGraphDataType,
                std::vector<int>({batch, 1, headDim, mMaxKVLen}));

        this->createParamScalar("axis", (uint32_t)1);  // [1] GQA axis
        scalarBaseIndex = 2;

        // Split K_perm along head dim (axis=1)
        {
            CLEAR_BEFORE_ADDING_NODE;
            std::string name = mNodeName + "_K_Split";
            mNodeType = "Split";
            mInputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor()));
            mParams.push_back(*(mParamScalarWrappers[1]->getNativeParam()));
            mParams.push_back(*(mParamTensorWrappers[4]->getNativeParam()));
            for (int i = 0; i < kvHeadNum; i++)
                mOutputs.push_back(*(mTempTensorWrappers[splitTempBase + i]->getNativeTensor()));
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(),
                mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        {
            CLEAR_BEFORE_ADDING_NODE;
            std::string name = mNodeName + "_K_GQA_Concat";
            mNodeType = "Concat";
            for (int i = 0; i < kvHeadNum; i++)
                for (int j = 0; j < group; j++)
                    mInputs.push_back(*(mTempTensorWrappers[splitTempBase + i]->getNativeTensor()));
            mParams.push_back(*(mParamScalarWrappers[1]->getNativeParam()));
            mOutputs.push_back(*(mTempTensorWrappers[gqaTempBase]->getNativeTensor()));
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(),
                mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        {
            CLEAR_BEFORE_ADDING_NODE;
            std::string name = mNodeName + "_V_Split";
            mNodeType = "Split";
            mInputs.push_back(*(mTempTensorWrappers[4]->getNativeTensor()));
            mParams.push_back(*(mParamScalarWrappers[1]->getNativeParam()));
            mParams.push_back(*(mParamTensorWrappers[5]->getNativeParam()));
            for (int i = 0; i < kvHeadNum; i++)
                mOutputs.push_back(*(mTempTensorWrappers[splitTempBase + kvHeadNum + i]->getNativeTensor()));
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(),
                mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        {
            CLEAR_BEFORE_ADDING_NODE;
            std::string name = mNodeName + "_V_GQA_Concat";
            mNodeType = "Concat";
            for (int i = 0; i < kvHeadNum; i++)
                for (int j = 0; j < group; j++)
                    mInputs.push_back(*(mTempTensorWrappers[splitTempBase + kvHeadNum + i]->getNativeTensor()));
            mParams.push_back(*(mParamScalarWrappers[1]->getNativeParam()));
            mOutputs.push_back(*(mTempTensorWrappers[gqaTempBase + 1]->getNativeTensor()));
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(),
                mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
    }

    auto attnK = isGQA ? *(mTempTensorWrappers[gqaTempBase]->getNativeTensor())
                       : *(mTempTensorWrappers[3]->getNativeTensor());
    auto attnV = isGQA ? *(mTempTensorWrappers[gqaTempBase + 1]->getNativeTensor())
                       : *(mTempTensorWrappers[4]->getNativeTensor());

    // ================================================================
    // Mask stage tensors
    // ================================================================
    int maskIdx = (int)mTempTensorWrappers.size();
    this->createStageTensor("tempMask", mGraphDataType,
        std::vector<int>({batch, 1, seqQ, mMaxKVLen}));
    this->createStageTensor("maskResult", mGraphDataType,
        std::vector<int>({batch, headNum, seqQ, mMaxKVLen}));

    // ================================================================
    // Scale Q
    // ================================================================
    {
        CLEAR_BEFORE_ADDING_NODE;
        std::string name = mNodeName + "_Scale";
        mNodeType = "ElementWiseMultiply";
        mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));   // Q_perm
        mInputs.push_back(*(mTempTensorWrappers[10]->getNativeTensor()));  // coef
        mOutputs.push_back(*(mTempTensorWrappers[5]->getNativeTensor()));  // ScaleQ
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(),
            mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // ================================================================
    // Q * K
    // ================================================================
    {
        bool t0 = true, t1 = false;
        this->createParamScalar("transpose_in0", t0);
        this->createParamScalar("transpose_in1", t1);

        CLEAR_BEFORE_ADDING_NODE;
        std::string name = mNodeName + "_MatMul_QK";
        mNodeType = "MatMul";
        mInputs.push_back(*(mTempTensorWrappers[5]->getNativeTensor()));  // ScaleQ
        mInputs.push_back(attnK);
        mParams.push_back(*(mParamScalarWrappers[scalarBaseIndex]->getNativeParam()));
        mParams.push_back(*(mParamScalarWrappers[scalarBaseIndex + 1]->getNativeParam()));
        mOutputs.push_back(*(mTempTensorWrappers[6]->getNativeTensor())); // QK
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(),
            mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // ================================================================
    // Mask
    // ================================================================
    {
        CLEAR_BEFORE_ADDING_NODE;
        std::string name = mNodeName + "_Mask_Reshape";
        mNodeType = "Reshape";
        mInputs.push_back(*(mMaskWrapper->getNativeTensor()));
        mOutputs.push_back(*(mTempTensorWrappers[maskIdx]->getNativeTensor()));
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(),
            mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    {
        CLEAR_BEFORE_ADDING_NODE;
        std::string name = mNodeName + "_Mask_Add";
        mNodeType = "ElementWiseAdd";
        mInputs.push_back(*(mTempTensorWrappers[6]->getNativeTensor()));
        mInputs.push_back(*(mTempTensorWrappers[maskIdx]->getNativeTensor()));
        mOutputs.push_back(*(mTempTensorWrappers[maskIdx + 1]->getNativeTensor()));
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(),
            mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // ================================================================
    // Softmax
    // ================================================================
    {
        CLEAR_BEFORE_ADDING_NODE;
        std::string name = mNodeName + "_Softmax";
        mNodeType = "Softmax";
        mInputs.push_back(*(mTempTensorWrappers[maskIdx + 1]->getNativeTensor()));
        mOutputs.push_back(*(mTempTensorWrappers[7]->getNativeTensor()));
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(),
            mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // ================================================================
    // Softmax * V
    // ================================================================
    {
        bool t0 = false, t1 = true;
        this->createParamScalar("transpose_in0", t0);
        this->createParamScalar("transpose_in1", t1);

        CLEAR_BEFORE_ADDING_NODE;
        std::string name = mNodeName + "_MatMul_QKV";
        mNodeType = "MatMul";
        mInputs.push_back(*(mTempTensorWrappers[7]->getNativeTensor()));
        mInputs.push_back(attnV);
        mParams.push_back(*(mParamScalarWrappers[scalarBaseIndex + 2]->getNativeParam()));
        mParams.push_back(*(mParamScalarWrappers[scalarBaseIndex + 3]->getNativeParam()));
        mOutputs.push_back(*(mTempTensorWrappers[8]->getNativeTensor()));
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(),
            mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // ================================================================
    // Transpose output + Reshape
    // ================================================================
    {
        CLEAR_BEFORE_ADDING_NODE;
        std::string name = mNodeName + "_Transpose_out";
        mNodeType = "Transpose";
        mInputs.push_back(*(mTempTensorWrappers[8]->getNativeTensor()));
        mParams.push_back(*(mParamTensorWrappers[3]->getNativeParam()));
        mOutputs.push_back(*(mTempTensorWrappers[9]->getNativeTensor()));
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(),
            mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    {
        CLEAR_BEFORE_ADDING_NODE;
        std::string name = mNodeName + "_Reshape_out";
        mNodeType = "Reshape";
        mInputs.push_back(*(mTempTensorWrappers[9]->getNativeTensor()));
        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(),
            mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    return NO_ERROR;
}

ErrorCode QNNKVCacheAttention::onExecute(const std::vector<Tensor *> &inputs,
                                         const std::vector<Tensor *> &outputs) {
    int seqQ = inputs[0]->length(1);
    int seqNewKV = inputs[1]->length(1);
    int tokenStride = mKvNumHead * mHeadDim;
    int elemSize = mState->elemSize;

    // ================================================================
    // Step 1: Consume pending sink data from the PREVIOUS graphExecute.
    // Sink data is raw bytes (FP16 or FP32) — direct memcpy, no conversion.
    // ================================================================
    if (mState->hasPending && mState->pendingKeySinkPtr != nullptr) {
        int pendingBytes = mState->pendingLen * tokenStride * elemSize;
        int dstOffsetBytes = mState->pastLen * tokenStride * elemSize;

        if (mState->pastLen + mState->pendingLen > mMaxKVLen) {
            MNN_ERROR("QNNKVCacheAttention: cache overflow (%d + %d > %d)\n",
                      mState->pastLen, mState->pendingLen, mMaxKVLen);
            return INPUT_DATA_ERROR;
        }

        ::memcpy(mState->keyCache.data() + dstOffsetBytes,
                 mState->pendingKeySinkPtr, pendingBytes);
        ::memcpy(mState->valueCache.data() + dstOffsetBytes,
                 mState->pendingValueSinkPtr, pendingBytes);
        mState->pastLen += mState->pendingLen;
        mState->hasPending = false;

        MNN_PRINT("QNNKVCacheAttention: consumed pending %d tokens, pastLen now %d\n",
                  mState->pendingLen, mState->pastLen);
    }

    MNN_PRINT("QNNKVCacheAttention::onExecute seqQ=%d seqNew=%d pastLen=%d\n",
              seqQ, seqNewKV, mState->pastLen);

    // ================================================================
    // Step 2: Redirect K_cache/V_cache clientBuf to persistent KVState.
    // QNN reads directly from the persistent buffer — no full cache copy.
    // ================================================================
    mKCacheWrapper->getNativeTensor()->v1.clientBuf.data = mState->keyCache.data();
    mVCacheWrapper->getNativeTensor()->v1.clientBuf.data = mState->valueCache.data();

    // ================================================================
    // Step 3: Fill scatter indices [B, seqNew, kvH, D] — int32.
    // For each new token position s, index = pastLen + s (broadcast over h,d).
    // ================================================================
    auto indicesBuf = mIndicesWrapper->getDataContainer();
    int32_t* indicesData = indicesBuf->host<int32_t>();
    int pastLen = mState->pastLen;

    for (int s = 0; s < seqNewKV; s++) {
        int32_t idx = pastLen + s;
        int offset = s * tokenStride;
        for (int i = 0; i < tokenStride; i++) {
            indicesData[offset + i] = idx;
        }
    }

    // ================================================================
    // Step 4: Fill mask [B, 1, seqQ, maxKVLen].
    // [0, pastLen): cached → attend (0)
    // [pastLen, pastLen+seqNew): new tokens → causal mask
    // [pastLen+seqNew, maxKVLen): padding → masked (-10000)
    // ================================================================
    int totalValid = pastLen + seqNewKV;
    int maskSize = seqQ * mMaxKVLen;

    auto maskBuf = mMaskWrapper->getDataContainer();
    if (mGraphDataType == QNN_DATATYPE_FLOAT_32) {
        float* maskData = maskBuf->host<float>();
        for (int q = 0; q < seqQ; q++) {
            float* row = maskData + q * mMaxKVLen;
            for (int k = 0; k < mMaxKVLen; k++) {
                if (k < totalValid && (k < pastLen || k - pastLen <= q)) {
                    row[k] = 0.0f;
                } else {
                    row[k] = -10000.0f;
                }
            }
        }
    } else {
        // FP16: build in float then convert
        std::vector<float> maskFloat(maskSize);
        for (int q = 0; q < seqQ; q++) {
            float* row = maskFloat.data() + q * mMaxKVLen;
            for (int k = 0; k < mMaxKVLen; k++) {
                if (k < totalValid && (k < pastLen || k - pastLen <= q)) {
                    row[k] = 0.0f;
                } else {
                    row[k] = -10000.0f;
                }
            }
        }
        FLOAT_TO_HALF(maskFloat.data(), (int16_t*)maskBuf->host<void>(), maskSize);
    }

    // ================================================================
    // Step 5: Set up pending pointers for sinks.
    // After graphExecute, APP_READ sinks will contain K_new/V_new data.
    // The NEXT onExecute reads them and appends to the persistent cache.
    // ================================================================
    auto kSinkBuf = mKNewSinkWrapper->getDataContainer();
    auto vSinkBuf = mVNewSinkWrapper->getDataContainer();
    mState->pendingKeySinkPtr = kSinkBuf->host<void>();
    mState->pendingValueSinkPtr = vSinkBuf->host<void>();
    mState->pendingLen = seqNewKV;
    mState->hasPending = true;

    return NO_ERROR;
}

bool QNNKVCacheAttention::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto tmp = new QNNKVCacheAttention(bn, op);
    tmp->mState = mState;  // Share persistent KV cache between prefill/decode
    tmp->mMaxKVLen = mMaxKVLen;
    tmp->mNumHead = mNumHead;
    tmp->mKvNumHead = mKvNumHead;
    tmp->mHeadDim = mHeadDim;
    tmp->mGraphDataType = mGraphDataType;
    *dst = tmp;
    return true;
}

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

class QNNAttentionCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        auto param = op->main_as_AttentionParam();
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
        if (param->kv_cache()) {
            // KV cache mode: use fixed-size buffer approach for QNN
            return new QNNKVCacheAttention(backend, op);
        }
#endif
        // Non KV cache: input validation deferred to onEncode
        // (inputs may be empty at onCreate time in fused transformer mode)
        return new QNNAttention(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNAttentionCreator, OpType_Attention)
#endif
} // end namespace QNN
} // end namespace MNN
