//
//  QNNStridedSlice.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNStridedSlice.hpp"

#define CLIP(input, min, max) ((input) < (min) ? (min) : ((input) > (max) ? (max) : (input)))

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

QNNStridedSlice::QNNStridedSlice(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {
    if(op->type() == OpType_Slice) {
        mIsSlice = true;
    }
}

ErrorCode QNNStridedSlice::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto inputTensor = inputs[0];
    mInputDim = inputTensor->dimensions();
    mDimType = inputTensor->getDimensionType();

    if(mIsSlice) {
        auto param = mOp->main_as_Slice();
        auto axis = param->axis();
        if (axis < 0) {
            axis = inputTensor->dimensions() + axis;
        }
        int64_t slice_num = 0;
        if (param->slicePoints() != nullptr) {
            if (param->slicePoints()->size() < outputs.size()) {
                slice_num = static_cast<int64_t>(outputs.size());
            } else if (param->slicePoints()->size() == 1) {
                slice_num = static_cast<int64_t>(param->slicePoints()->Get(0));
            } else {
                slice_num = static_cast<int64_t>(param->slicePoints()->size());
            }
        } else {
            slice_num = static_cast<int64_t>(outputs.size());
        }
        auto shape = inputs[0]->shape();
        #ifdef QNN_VERBOSE
        MNN_PRINT("slice:%d %d %d %d, axis:%d, slice_num:%d output_num:%d, dim:%d\n", shape[0], shape[1], shape[2], shape[3], axis, slice_num, outputs.size(), mInputDim);
        #endif
        int realAxis = axis;
        int slice_size = inputs[0]->length(axis) / slice_num;
        for(int index = 0; index < slice_num; index++) {
            std::vector<int> rangeData(mInputDim * 3, 0);
            for (int i = 0; i < mInputDim; i++) {
                rangeData[3 * i + 0] = 0;
                rangeData[3 * i + 1] = inputs[0]->length(i);
                rangeData[3 * i + 2] = 1;
            }
            rangeData[3 * realAxis + 0] = index * slice_size;
            rangeData[3 * realAxis + 1] = index * slice_size + slice_size;
            this->createParamTensor("ranges", QNN_DATATYPE_INT_32, {(uint32_t) mInputDim, 3}, (void *) rangeData.data(), std::to_string(index));

            // Add Node.
            mNodeType = "StridedSlice";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            std::string name =  mNodeName + "_part" + std::to_string(index);
            mParams.push_back(*(mParamTensorWrappers[index]->getNativeParam()));
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
            mOutputs.push_back(*(mBackend->getNativeTensor(outputs[index])));

            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        return NO_ERROR;
    }

    auto param = mOp->main_as_StridedSliceParam();
    mNodeType = "StridedSlice";

    // Deal with ranges.
    std::vector<int> beginRaw(mInputDim, 0);
    std::vector<int> endRaw = inputTensor->shape();
    std::vector<int> strideRaw(mInputDim, 1);
    if (param->fromType() == 0) {
        this->computeRangesType0(inputs, beginRaw, endRaw, strideRaw);
    } else {
        this->computeRangesType1(inputs, beginRaw, endRaw, strideRaw);
    }

    std::vector<int> rangeData(mInputDim * 3, 0);
    for (int axis = 0; axis < mInputDim; axis++) {
        rangeData[3 * axis + 0] = beginRaw[axis];
        rangeData[3 * axis + 1] = endRaw[axis];
        rangeData[3 * axis + 2] = strideRaw[axis];
    }
    this->createParamTensor("ranges", QNN_DATATYPE_INT_32, {(uint32_t) mInputDim, 3}, (void *) rangeData.data());

    // Deal with masks.
    uint32_t beginMaskData = computeMask(param->beginMask(), mInputDim, mDimType);
    uint32_t endMaskData =  computeMask(param->endMask(), mInputDim, mDimType);
    uint32_t shrinkAxesData =  computeMask(param->shrinkAxisMask(), mInputDim, mDimType);
    uint32_t newAxesMaskData = computeMask(param->newAxisMask(), mInputDim, mDimType);

    // Debug: log all StridedSlice parameters for diagnosis
    {
        char shapeBuf[128], rangeBuf[256];
        int sp = 0, rp = 0;
        for (int i = 0; i < mInputDim && i < 8; i++) {
            sp += snprintf(shapeBuf + sp, sizeof(shapeBuf) - sp, "%d ", inputTensor->length(i));
        }
        for (int i = 0; i < mInputDim && i < 8; i++) {
            rp += snprintf(rangeBuf + rp, sizeof(rangeBuf) - rp, "[%d,%d,%d] ",
                           rangeData[3*i], rangeData[3*i+1], rangeData[3*i+2]);
        }
        char outShapeBuf[128];
        int op2 = 0;
        for (int i = 0; i < outputs[0]->dimensions() && i < 8; i++) {
            op2 += snprintf(outShapeBuf + op2, sizeof(outShapeBuf) - op2, "%d ", outputs[0]->length(i));
        }
        MNN_PRINT("QNN StridedSlice[%s]: fromType=%d ndim=%d inShape=[%s] outShape=[%s] "
                  "ranges=%s masks: begin=0x%x end=0x%x shrink=0x%x newaxis=0x%x\n",
                  mNodeName.c_str(), param->fromType(), mInputDim,
                  shapeBuf, outShapeBuf, rangeBuf,
                  beginMaskData, endMaskData, shrinkAxesData, newAxesMaskData);
    }

    this->createParamScalar("begin_mask", beginMaskData);
    this->createParamScalar("end_mask", endMaskData);
    this->createParamScalar("shrink_axes", shrinkAxesData);
    this->createParamScalar("new_axes_mask", newAxesMaskData);

    // Add Node.
    mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam()));
    for (int i = 0; i < mParamScalarWrappers.size(); i++) {
        mParams.push_back(*(mParamScalarWrappers[i]->getNativeParam()));
    }
    mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
    mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));

    mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);

    return NO_ERROR;
}

void QNNStridedSlice::computeRangesType0(const std::vector<Tensor *> &inputs, std::vector<int> & beginRaw, std::vector<int> & endRaw, std::vector<int> & strideRaw) {
    auto inputTensor = inputs[0];
    auto beginTensor = inputs[1];
    auto endTensor = inputs[2];
    auto strideTensor = inputs[3];
    auto beginRawSource = beginTensor->host<int>();
    auto endRawSource = endTensor->host<int>();
    auto strideRawSource = strideTensor->host<int>();

    int sliceDim = beginTensor->length(0);
    MNN_ASSERT(sliceDim == endTensor->length(0) && sliceDim == strideTensor->length(0));

    for (int i = 0; i < sliceDim; i++) {
        int dimSize = inputs[0]->length(i);
        int b = beginRawSource[i];
        int e = endRawSource[i];
        // Convert negative indices to positive (e.g., -1 → dimSize-1)
        if (b < 0) b += dimSize;
        if (e < 0) e += dimSize;
        beginRaw[i] = CLIP(b, 0, dimSize - 1);
        endRaw[i] = CLIP(e, 1, dimSize);
        strideRaw[i] = strideRawSource[i];
    }
    return;
}

void QNNStridedSlice::computeRangesType1(const std::vector<Tensor *> &inputs, std::vector<int> & beginRaw, std::vector<int> & endRaw, std::vector<int> & strideRaw) {
    auto inputTensor = inputs[0];
    auto beginTensor = inputs[1];
    auto endTensor = inputs[2];
    auto strideTensor = inputs[4];
    auto beginRawSource = beginTensor->host<int>();
    auto endRawSource = endTensor->host<int>();
    auto strideRawSource = strideTensor->host<int>();

    auto axisTensor = inputs[3];
    int sliceDim = beginTensor->length(0);
    MNN_ASSERT(sliceDim == endTensor->length(0) && sliceDim == axisTensor->length(0) && sliceDim == strideTensor->length(0));

    for (int i = 0; i < sliceDim; i++) {
        int tempAxis = axisTensor->host<int>()[i];
        tempAxis = tempAxis >= 0 ? tempAxis : (tempAxis + mInputDim);
        int dimSize = inputs[0]->length(tempAxis);
        int b = beginRawSource[i];
        int e = endRawSource[i];
        // Convert negative indices to positive (e.g., -1 → dimSize-1)
        if (b < 0) b += dimSize;
        if (e < 0) e += dimSize;
        beginRaw[tempAxis] = CLIP(b, 0, dimSize - 1);
        endRaw[tempAxis] = CLIP(e, 1, dimSize);
        strideRaw[tempAxis] = strideRawSource[i];
    }
    return;
}


uint32_t QNNStridedSlice::computeMask(uint32_t rawMask, int dim, Tensor::DimensionType mDimType) {
    if (rawMask == 0) return 0;

    uint32_t result = 0;
    for (int axis = 0; axis < dim; axis++) {
        int realAxis = axis;
        result |= ((rawMask >> axis) & 1) << realAxis; // If the axis-th bit of rawMask is 1, set the realAxis-th bit of result to 1.
    }

    return result;
}

class QNNStridedSliceCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        if(op->type() == OpType_Slice) {
            return new QNNStridedSlice(backend, op);
        }
        auto param = op->main_as_StridedSliceParam();

        // <begin>, <end> and <stride> should be static.
        for (int i = 1; i < inputs.size(); i++) {
            MNN_ASSERT(TensorUtils::getDescribe(inputs[i])->usage == Tensor::InsideDescribe::Usage::CONSTANT);
        }

        if (param->fromType() == 1) {
            MNN_ASSERT(param->shrinkAxisMask() == 0 && param->newAxisMask() == 0 && param->ellipsisMask() == 0);
            if (inputs.size() != 5) {
                return nullptr;
            }
            return new QNNStridedSlice(backend, op);
        }

        // [TODO] 把newAxisMask和ellipsisMask考虑在内
        if (param->fromType() == 0) {
            if (inputs.size() == 4 && param->newAxisMask() == 0 && param->ellipsisMask() == 0) {
                return new QNNStridedSlice(backend, op);
            } else {
                return nullptr;
            }
        }

        // Shouldn't reach here.
        return nullptr;
    }
};

REGISTER_QNN_OP_CREATOR(QNNStridedSliceCreator, OpType_StridedSlice)
REGISTER_QNN_OP_CREATOR(QNNStridedSliceCreator, OpType_Slice)
#endif
} // end namespace QNN
} // end namespace MNN

