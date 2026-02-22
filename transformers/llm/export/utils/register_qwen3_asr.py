def register_qwen3_asr():
    """Register Qwen3-ASR classes with AutoConfig and AutoModel."""
    from transformers import AutoConfig, AutoModel
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
        Qwen3ASRConfig, Qwen3ASRThinkerConfig, Qwen3ASRAudioEncoderConfig
    )
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration
    )

    try:
        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    except ValueError:
        pass
    try:
        AutoConfig.register("qwen3_asr_thinker", Qwen3ASRThinkerConfig)
    except ValueError:
        pass
    try:
        AutoConfig.register("qwen3_asr_audio_encoder", Qwen3ASRAudioEncoderConfig)
    except ValueError:
        pass
    try:
        AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
    except ValueError:
        pass
