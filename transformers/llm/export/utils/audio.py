import torch
from .transformers import Decoder
from .spinner import spinner_run
from .torch_utils import onnx_export

class Audio(torch.nn.Module):
    def __init__(self, audio, base):
        super().__init__()
        self.model_type = base.config.model_type
        self.audio = audio
        self.embed_ = base.embed
        self.tokenizer = base.tokenizer
        self.config = base.config.origin_config
        self.hidden_size = base.config.hidden_size
        self.llm_config = { 'is_audio': True }
        self.rope_ratio = 1.0
        self.quant_bit = 16
        self.init_config()
        self.load()

    def get_config(self):
        return self.llm_config

    @staticmethod
    def get_audio(model_type):
        audio_models = {
            'qwen2_audio_encoder': Qwen2Audio,
            'qwen2_5_omni_audio_encoder': Qwen2_5OmniAudio,
            'funaudiochat_audio_encoder': FunAudioChatAudio,
            'qwen3_asr_audio_encoder': Qwen3AsrAudio,
        }
        if model_type in audio_models:
            return audio_models[model_type]
        return None

    def init_config(self):
        pass

    def load(self):
        raise NotImplementedError

    def str_to_ids(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids']
        return input_ids

    def forward(self, images):
        raise NotImplementedError

    def embed(self, input_ids, images = None, videos = None):
        raise NotImplementedError

    def export(self, onnx_path):
        raise NotImplementedError

class Qwen2Audio(Audio):
    def __init__(self, audio, base):
        super().__init__(audio, base)
        self.audio_embeds = None
        self.audio_pad_id = 151646
        self.n_fft = 400
        self.sampling_rate = 16000
        self.hop_length = 160
        self.chunk_length = 30
        self.feature_size = 128
        self.n_samples = self.chunk_length * self.sampling_rate
        self.max_length = self.n_samples // self.hop_length
        from transformers.audio_utils import mel_filter_bank
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + self.n_fft // 2,
            num_mel_filters=self.feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=self.sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def load(self):
        # model
        self.audio_tower = self.audio
        self.multi_modal_projector = self.audio.multi_modal_projector
        # config
        self.llm_config['is_audio'] = True

    def str_to_ids(self, prompt):
        if '<audio>' in prompt and '</audio>' in prompt:
            import re
            from io import BytesIO
            from urllib.request import urlopen
            import librosa
            pattern = r'(<audio>.*?</audio>)'
            parts = re.split(pattern, prompt)
            txt_prompt = ''
            for part in parts:
                if re.match(pattern, part):
                    audio_content = re.search(r'<audio>(.*?)</audio>', part).group(1)
                    if audio_content.startswith('http://') or audio_content.startswith('https://'):
                        audio_obj = librosa.load(BytesIO(urlopen(audio_content).read()), sr=self.sampling_rate)[0]
                    else:
                        # local file
                        audio_obj = librosa.load(audio_content, sr=self.sampling_rate)[0]
                    audio_embed_len = self.audio_process(audio_obj)
                    audio_pad_str = '<|AUDIO|>' * audio_embed_len
                    txt_prompt += audio_pad_str
                else:
                    txt_prompt += part
        else:
            txt_prompt = prompt
        input_ids = self.tokenizer(txt_prompt, return_tensors="pt")['input_ids']
        return input_ids

    def forward(self, input_features):
        input_features = input_features.to(dtype=self.audio_tower.conv1.weight.dtype, device=self.audio_tower.conv1.weight.device)
        inputs_embeds = torch.nn.functional.gelu(self.audio_tower.conv1(input_features))
        inputs_embeds = torch.nn.functional.gelu(self.audio_tower.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        _, seq_len, _ = inputs_embeds.shape
        embed_pos = self.audio_tower.embed_positions.weight[:seq_len, :]
        hidden_states = inputs_embeds + embed_pos
        for encoder_layer in self.audio_tower.layers:
            hidden_states = encoder_layer(hidden_states, None, None)[0]
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.audio_tower.avg_pooler(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.audio_tower.layer_norm(hidden_states)
        audio_features = self.multi_modal_projector(hidden_states)
        return audio_features

    def _torch_extract_fbank_features(self, waveform):
        window = torch.hann_window(self.n_fft)
        stft = torch.stft(waveform, self.n_fft, self.hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        mel_filters = torch.from_numpy(self.mel_filters).type(torch.float32)
        mel_spec = mel_filters.T @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if waveform.dim() == 2:
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def audio_process(self, audio_obj):
        # audio_obj = np.pad(audio_obj, (0, self.n_samples - audio_obj.shape[0]))
        waveform = torch.from_numpy(audio_obj).type(torch.float32)
        input_features = self._torch_extract_fbank_features(waveform).unsqueeze(0)
        audio_embeds = self.forward(input_features)
        self.audio_embeds = audio_embeds.permute([1, 0, 2])
        return self.audio_embeds.shape[0]

    def embed(self, input_ids, images = None, videos = None):
        input_embeds = self.embed_(input_ids)
        if self.audio_embeds is not None:
            audio_mask = (input_ids == self.audio_pad_id).squeeze()
            input_embeds[audio_mask] = self.audio_embeds.type(input_embeds.dtype)
        return input_embeds

    @spinner_run(f'export audio to ')
    def export(self, onnx_path):
        input_features = torch.randn((1, self.feature_size, self.max_length))

        model = self.float()
        onnx_model = f'{onnx_path}/audio.onnx'
        onnx_export(model, (input_features),
                    onnx_model,
                    input_names=['input_features'],
                    output_names=['audio_embeds'],
                    dynamic_axes={"input_features": {
                        2: "size"
                    }})
        return onnx_model

class AudioMlp(torch.nn.Module):
    def __init__(self, fc1, fc2, act):
        super().__init__()
        self.fc1 = fc1
        self.fc2 = fc2
        self.act = act

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class Qwen2_5OmniAudio(Qwen2Audio):
    def __init__(self, audio, base):
        super().__init__(audio, base)
        self.quant_bit = 4

    def load(self):
        # config
        config = self.audio.config
        self.n_window = config.n_window
        self.llm_config['is_audio'] = True
        self.llm_config['n_window'] = self.n_window
        self.hidden_size = config.d_model
        self.num_attention_heads = config.encoder_attention_heads
        self.num_key_value_heads = self.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.rotary = None
        self.model_map = {
            'decoder': {
                'self_attn': 'self_attn',
                'input_layernorm': 'self_attn_layer_norm',
                'post_attention_layernorm': 'final_layer_norm'
            },
            'attention': {
                'q_proj': 'q_proj',
                'k_proj': 'k_proj',
                'v_proj': 'v_proj',
                'o_proj': 'out_proj'
            }
        }
        self.blocks = []
        for layer in self.audio.layers:
            layer_id = len(self.blocks)
            block = Decoder(layer, layer_id, self)
            block.mlp = AudioMlp(layer.fc1, layer.fc2, layer.activation_fn)
            self.blocks.append(block)

    def forward(self, input_features, attention_mask = None):
        input_features = input_features.to(dtype=self.audio.conv1.weight.dtype, device=self.audio.conv1.weight.device)
        inputs_embeds = torch.nn.functional.gelu(self.audio.conv1(input_features))
        inputs_embeds = torch.nn.functional.gelu(self.audio.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        _, seq_len, _ = inputs_embeds.shape
        embed_pos = self.audio.positional_embedding.positional_embedding[:seq_len, :]
        hidden_states = inputs_embeds + embed_pos
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.audio.avg_pooler(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.audio.ln_post(hidden_states)
        audio_features = self.audio.proj(hidden_states)
        return audio_features

    def audio_process(self, audio_obj):
        # audio_obj = np.pad(audio_obj, (0, self.n_samples - audio_obj.shape[0]))
        waveform = torch.from_numpy(audio_obj).type(torch.float32)
        input_features = self._torch_extract_fbank_features(waveform).unsqueeze(0)
        _, _, seq_len = input_features.shape
        seq_len = int(seq_len // 2)
        cu_seqlens = [i for i in range(0, seq_len, self.n_window)]
        if seq_len % self.n_window != 0:
            cu_seqlens.append(seq_len)
        cu_seqlens = torch.tensor(cu_seqlens)
        attention_mask = torch.full(
            [1, seq_len, seq_len], torch.finfo(torch.float32).min
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        audio_embeds = self.forward(input_features, attention_mask)
        self.audio_embeds = audio_embeds.permute([1, 0, 2])
        return self.audio_embeds.shape[0]

    @spinner_run(f'export audio to ')
    def export(self, onnx_path):
        input_features = torch.randn((1, self.feature_size, self.max_length))
        seq_len = self.max_length // 2
        attention_mask = torch.randn([1, seq_len, seq_len])
        model = self.float()
        onnx_model = f'{onnx_path}/audio.onnx'
        onnx_export(model, (input_features, attention_mask),
                    onnx_model,
                    input_names=['input_features', 'attention_mask'],
                    output_names=['audio_embeds'],
                    dynamic_axes={"input_features": {
                        0: "size"
                    }, "attention_mask": {
                        1: "size", 2: "size"
                    }})
        return onnx_model

class FunAudioChatAudio(Qwen2_5OmniAudio):
    def __init__(self, audio, base):
        super().__init__(audio, base)
        self.audio_pad_id = 151669

    def load(self):
        # model
        self.audio = self.audio.float()
        self.audio_tower = self.audio.audio_tower.float()
        # config
        self.group_size = self.audio.config.group_size
        # call parent load
        super().load()

    def forward(self, input_features, attention_mask = None):
        # call parent forward to get audio_features before group pooling
        audio_features = super().forward(input_features, attention_mask)
        # group pooling and continual_output_matching
        batch, seqlen, hidden_size = audio_features.shape
        padding_feature = torch.zeros(
            (batch, (self.group_size - seqlen % self.group_size) % self.group_size, hidden_size),
            dtype=torch.long,
            device=audio_features.device,
        )
        audio_features = torch.cat([audio_features, padding_feature], dim=1)
        audio_features = audio_features.reshape(batch, -1, self.group_size, hidden_size)
        audio_features = audio_features.mean(dim=2)
        audio_features = self.audio_tower.continual_output_matching(audio_features)
        return audio_features


class Qwen3AsrAudio(Qwen2_5OmniAudio):
    """Audio encoder for Qwen3-ASR models.

    Key differences from Qwen2.5-Omni:
    - Conv2D stem (3 layers, 8x downsample) instead of Conv1D (2 layers, 2x downsample)
    - Per-chunk convolution: mel is split into chunks of n_window*2 frames
    - Sinusoidal positional embeddings (per-chunk, computed not stored)
    - Projection: proj1 (Linear+GELU) + proj2 (Linear) instead of avg_pooler + proj
    """

    def load(self):
        self.audio_pad_id = self.config.thinker_config.audio_token_id
        config = self.audio.config
        self.n_window = config.n_window
        self.n_window_infer = getattr(config, 'n_window_infer', 800)
        self.chunk_frames = self.n_window * 2  # 100 mel frames per conv chunk
        self.llm_config['is_audio'] = True
        self.llm_config['audio_pad'] = self.audio_pad_id
        self.llm_config['n_window'] = self.n_window
        self.llm_config['n_window_infer'] = self.n_window_infer
        self.hidden_size = config.d_model
        self.num_attention_heads = config.encoder_attention_heads
        self.num_key_value_heads = self.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.output_dim = config.output_dim
        self.rotary = None
        self.model_map = {
            'decoder': {
                'self_attn': 'self_attn',
                'input_layernorm': 'self_attn_layer_norm',
                'post_attention_layernorm': 'final_layer_norm'
            },
            'attention': {
                'q_proj': 'q_proj',
                'k_proj': 'k_proj',
                'v_proj': 'v_proj',
                'o_proj': 'out_proj'
            }
        }
        self.blocks = []
        for layer in self.audio.layers:
            layer_id = len(self.blocks)
            block = Decoder(layer, layer_id, self)
            block.mlp = AudioMlp(layer.fc1, layer.fc2, layer.activation_fn)
            self.blocks.append(block)

    def _sinusoidal_embedding(self, length, channels):
        import math
        log_timescale_increment = math.log(10000) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2, dtype=torch.float32))
        scaled_time = torch.arange(length, dtype=torch.float32).unsqueeze(1) * inv_timescales.unsqueeze(0)
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

    def forward(self, input_features, attention_mask=None):
        # input_features: [1, 128, T] where T = n_chunks * chunk_frames
        dtype = self.audio.conv2d1.weight.dtype
        device = self.audio.conv2d1.weight.device
        input_features = input_features.to(dtype=dtype, device=device)
        _, n_mels, total_frames = input_features.shape
        n_chunks = total_frames // self.chunk_frames

        # Per-chunk Conv2D: split time dim into chunks, then permute
        # [1, 128, T] -> [1, 128, n_chunks, chunk_frames] -> [n_chunks, 1, 128, chunk_frames]
        x = input_features.reshape(1, n_mels, -1, self.chunk_frames).permute(2, 0, 1, 3)
        x = torch.nn.functional.gelu(self.audio.conv2d1(x))
        x = torch.nn.functional.gelu(self.audio.conv2d2(x))
        x = torch.nn.functional.gelu(self.audio.conv2d3(x))

        # x: [n_chunks, 480, freq_bins, tokens_per_chunk]
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)  # [n_chunks, t, 7680]
        x = self.audio.conv_out(x)  # [n_chunks, t, d_model]

        # Per-chunk sinusoidal positional embedding
        pos_embed = self.audio.positional_embedding.positional_embedding[:t, :]
        x = x + pos_embed.unsqueeze(0)

        # Flatten to single sequence: [1, n_chunks*t, d_model]
        x = x.reshape(1, -1, self.hidden_size)

        # Transformer layers with windowed attention mask
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        # Output projection: LayerNorm -> proj1 + GELU -> proj2
        x = self.audio.ln_post(x)
        x = torch.nn.functional.gelu(self.audio.proj1(x))
        x = self.audio.proj2(x)

        return x

    def audio_process(self, audio_obj):
        waveform = torch.from_numpy(audio_obj).type(torch.float32)
        input_features = self._torch_extract_fbank_features(waveform).unsqueeze(0)
        _, _, total_frames = input_features.shape

        # Pad to multiple of chunk_frames
        if total_frames % self.chunk_frames != 0:
            pad_frames = self.chunk_frames - (total_frames % self.chunk_frames)
            input_features = torch.nn.functional.pad(input_features, (0, pad_frames))
            total_frames = input_features.shape[2]

        n_chunks = total_frames // self.chunk_frames
        # tokens_per_chunk after 3x stride-2 conv on time dim
        tokens_per_chunk = self.chunk_frames
        for _ in range(3):
            tokens_per_chunk = (tokens_per_chunk + 1) // 2
        seq_len = n_chunks * tokens_per_chunk

        # Windowed attention mask
        tokens_per_window = tokens_per_chunk * (self.n_window_infer // self.chunk_frames)
        cu_seqlens = list(range(0, seq_len, tokens_per_window))
        if seq_len % tokens_per_window != 0:
            cu_seqlens.append(seq_len)
        attention_mask = torch.full(
            [1, seq_len, seq_len], torch.finfo(torch.float32).min
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i], cu_seqlens[i - 1]:cu_seqlens[i]] = 0

        audio_embeds = self.forward(input_features, attention_mask)
        self.audio_embeds = audio_embeds.permute([1, 0, 2])
        return self.audio_embeds.shape[0]

    @spinner_run(f'export audio to ')
    def export(self, onnx_path):
        # Use 30 chunks (3000 mel frames = 30s audio) as trace example
        n_chunks = 30
        total_frames = n_chunks * self.chunk_frames
        tokens_per_chunk = self.chunk_frames
        for _ in range(3):
            tokens_per_chunk = (tokens_per_chunk + 1) // 2
        seq_len = n_chunks * tokens_per_chunk

        input_features = torch.randn((1, self.feature_size, total_frames))
        attention_mask = torch.randn([1, seq_len, seq_len])
        model = self.float()
        onnx_model = f'{onnx_path}/audio.onnx'
        onnx_export(model, (input_features, attention_mask),
                    onnx_model,
                    input_names=['input_features', 'attention_mask'],
                    output_names=['audio_embeds'],
                    dynamic_axes={"input_features": {
                        2: "size"
                    }, "attention_mask": {
                        1: "size", 2: "size"
                    }})
        return onnx_model