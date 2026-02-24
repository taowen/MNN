#ifndef MNN_QNNAttention_HPP
#define MNN_QNNAttention_HPP

#include "QNNCommonExecution.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNAttention : public QNNCommonExecution {
public:
    QNNAttention(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

};

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
// QNN Attention with KV cache support using in-graph ScatterElements.
//
// Architecture: K_cache (APP_WRITE, clientBuf redirected to persistent byte
// buffer) and K_new (NATIVE from Linear) are merged via ScatterElements
// inside the QNN graph.  The output is always [B, maxKVLen, kvH, D] — fixed
// size regardless of context growth.  APP_READ sinks capture K_new/V_new
// during graphExecute; the *next* onExecute appends them to the persistent
// cache.  A causal mask [B, 1, seqQ, maxKVLen] gates invalid positions.
class QNNKVCacheAttention : public QNNCommonExecution {
public:
    QNNKVCacheAttention(Backend *backend, const Op *op);
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                                const std::vector<Tensor *> &outputs) override;
    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    int mMaxKVLen = 2048;
    int mNumHead = 0;
    int mKvNumHead = 0;
    int mHeadDim = 0;

    // Persistent KV cache shared between cloned instances (prefill/decode).
    // Stores raw bytes (FP16 or FP32).  Sink data from the previous
    // graphExecute is appended in the next onExecute via memcpy.
    struct KVState {
        std::vector<uint8_t> keyCache;   // [maxKVLen * kvH * D * elemSize]
        std::vector<uint8_t> valueCache;
        int pastLen = 0;
        int pendingLen = 0;
        int elemSize = 4;  // 2 for FP16, 4 for FP32

        // Pending sink data from previous graphExecute
        void* pendingKeySinkPtr = nullptr;
        void* pendingValueSinkPtr = nullptr;
        bool  hasPending = false;
    };
    std::shared_ptr<KVState> mState;

    // APP_WRITE: KV cache [batch, maxKVLen, kvHead, headDim] — clientBuf
    // redirected to KVState persistent buffer in onExecute
    std::shared_ptr<QNNTensorWrapper> mKCacheWrapper;
    std::shared_ptr<QNNTensorWrapper> mVCacheWrapper;
    // APP_WRITE: scatter indices [batch, seqNew, kvHead, headDim] — int32
    std::shared_ptr<QNNTensorWrapper> mIndicesWrapper;
    // APP_WRITE: attention mask [batch, 1, seqQ, maxKVLen]
    std::shared_ptr<QNNTensorWrapper> mMaskWrapper;
    // APP_READ: sinks that capture K_new/V_new during graphExecute
    std::shared_ptr<QNNTensorWrapper> mKNewSinkWrapper;
    std::shared_ptr<QNNTensorWrapper> mVNewSinkWrapper;

    Qnn_DataType_t mGraphDataType = QNN_DATATYPE_FLOAT_32;
};
#endif // MNN_SUPPORT_TRANSFORMER_FUSE

#endif // ENABLE_QNN_ONLINE_FINALIZE
} // end namespace QNN
} // end namespace MNN

#endif // end MNN_QNNAttention_HPP
