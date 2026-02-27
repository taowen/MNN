import os
import math
import numpy as np
import torch
import torch.nn.functional as F
torch.set_printoptions(precision=4, sci_mode=False)
from .model_mapper import ModelMapper
from .transformers import Rotary, Embedding, Decoder, Attention
from .spinner import spinner_run
from .torch_utils import onnx_export

class Token2Wav(torch.nn.Module):
    def __init__(self,token2wav, base):
        super().__init__()
        self.args = base.args
        self.token2wav = token2wav.float()
        self.config = base.config
        self.rope_ratio = 1.0
        self.quant_bit = 8
        self.load()

    def load(self):
        raise NotImplementedError

    def add_token_embeds(self, thinker_embeds):
        raise NotImplementedError

    def add_hidden_states(self, thinker_hidden_states):
        raise NotImplementedError

    def add_generate_ids(self, token_id):
        raise NotImplementedError

    def forward(self, inputs_embeds, attention_mask, position_ids):
        raise NotImplementedError

    def export(self, onnx_path):
        raise NotImplementedError

class UpSample1d(torch.nn.Module):
    def __init__(self, upsample, channel):
        super().__init__()
        self.ratio = upsample.ratio
        self.stride = upsample.stride
        self.pad = upsample.pad
        self.pad_left = upsample.pad_left
        self.pad_right = upsample.pad_right
        self.filter = upsample.filter.expand(channel, -1, -1).clone()
        self.channel = channel

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter, stride=self.stride, groups=self.channel)
        x = x[..., self.pad_left : -self.pad_right]
        return x

class DownSample1d(torch.nn.Module):
    def __init__(self, downsample, channel):
        super().__init__()
        self.pad_left = downsample.pad_left
        self.pad_right = downsample.pad_right
        self.stride = downsample.stride
        self.filter = downsample.filter.expand(channel, -1, -1).clone()
        self.channel = channel

    def forward(self, x):
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        out = F.conv1d(x, self.filter, stride=self.stride, groups=self.channel)
        return out

class TorchActivation1d(torch.nn.Module):
    def __init__(
        self,
        activation
    ):
        super().__init__()
        self.act = activation.act
        channel = self.act.in_features
        self.upsample = UpSample1d(activation.upsample, channel)
        self.downsample = DownSample1d(activation.downsample, channel)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x

# DiT model code
class ECAPA_TDNN(torch.nn.Module):
    def __init__(self, spk_encoder):
        super().__init__()
        self.blocks = spk_encoder.blocks
        self.mfa = spk_encoder.mfa
        self.asp = spk_encoder.asp
        self.fc = spk_encoder.fc

    def forward(self, x):
        # Minimize transpose for efficiency
        x = x.transpose(1, 2)
        xl = []
        for layer in self.blocks:
            x = layer(x)
            xl.append(x)
        # Multi-layer feature aggregation
        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)
        # Attentive Statistical Pooling
        x = self.asp(x)
        # Final linear transformation
        x = self.fc(x)
        # x = x.squeeze(-1) # avoid If when export to onnx
        x = x.permute(0, 2, 1)
        return x

class DitRotary(Rotary):
    def __init__(self):
        super().__init__(None)
        self.model_type = 'dit'
        self.rope_theta = 10000
        self.rotary_dim = 64
        self.theta = 1.0 / (self.rope_theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))

    def forward(self, position_ids):
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * self.theta
        rotary_pos_emb = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)])
        rotary_pos_emb = torch.stack((rotary_pos_emb, rotary_pos_emb), dim=-1)
        rotary_pos_emb = rotary_pos_emb.reshape(*rotary_pos_emb.shape[:-2], -1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(2).unsqueeze(1)
        return rotary_pos_emb

    @staticmethod
    def apply_rotary_pos(x, cos, sin):
        def rotate_half(x):
            x = x.reshape(*x.shape[:-1], -1, 2)
            x1, x2 = x.unbind(dim=-1)
            x = torch.stack((-x2, x1), dim=-1)
            return x.reshape(*x.shape[:-2], -1)

        x = (x * cos) + (rotate_half(x) * sin)
        return x

import math
class DiTAttention(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.dim = attn.dim
        self.heads = attn.heads
        self.inner_dim = attn.inner_dim
        self.to_q = attn.to_q
        self.to_k = attn.to_k
        self.to_v = attn.to_v
        self.to_out = attn.to_out

    def forward(
        self,
        x,
        rope=None,
        mask=None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        # `sample` projections.
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim)
        key = key.view(batch_size, -1, self.heads, head_dim)
        value = value.view(batch_size, -1, self.heads, head_dim)
        # apply rotary position embedding
        # Due to training process, only first head is applied with RoPE, will be fixed at next release
        cos, sin = rope[0], rope[1]
        first_query = query[:, :, :1, :]
        first_key   = key[:, :, :1, :]
        other_query = query[:, :, 1:, :]
        other_key   = key[:, :, 1:, :]
        first_query = DitRotary.apply_rotary_pos(first_query, cos, sin)
        first_key   = DitRotary.apply_rotary_pos(first_key, cos, sin)
        query = torch.concat([first_query, other_query], dim=2)
        key = torch.concat([first_key, other_key], dim=2)

        attention_mask = (~mask) * torch.finfo(torch.float32).min

        query = query.transpose(1, 2)
        key   = key.permute([0, 2, 3, 1])
        value = value.transpose(1, 2)
        attn_weights = torch.matmul(query, key) / math.sqrt(head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)
        x = attn_output.transpose(1, 2)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        x = x.reshape(batch_size, -1, self.heads * head_dim)
        x = x.to(query.dtype)

        # linear proj
        x = self.to_out[0](x)
        # dropout
        x = self.to_out[1](x)

        return x

class DiTBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.attn_norm = block.attn_norm
        self.attn = DiTAttention(block.attn)
        self.attn_ = block.attn
        self.look_ahead_block = block.look_ahead_block
        self.look_backward_block = block.look_backward_block
        self.ff_norm = block.ff_norm
        self.ff = block.ff

    def forward(self, x, t, rope=None, block_diff=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        attn_output = self.attn(
            x=norm,
            rope=rope,
            mask=(block_diff >= -float(self.look_backward_block)) & (block_diff <= float(self.look_ahead_block)),
        )

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x

class DitPreprocess(torch.nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.code_embed = dit.code_embed
        self.input_proj = dit.proj_in_other
        self.rotary_embed = DitRotary()
        self.block_size = 24

    def forward(self, cond, spk, code):
        max_duration = code.shape[1] * 2
        spk = spk.repeat(1, max_duration, 1)
        cond = cond.repeat(1, max_duration, 1)
        code_embed = self.code_embed(code)
        input_embeds = torch.cat((cond, code_embed, spk), dim=-1)
        code_embeds = self.input_proj(input_embeds)
        position_ids = torch.arange(max_duration)
        rope = self.rotary_embed(position_ids)

        block_indices = position_ids // self.block_size
        block_i = block_indices.unsqueeze(1)
        block_j = block_indices.unsqueeze(0)
        block_diff = block_j - block_i
        mask = block_diff.reshape(1, 1, max_duration, max_duration)
        return code_embeds, rope, mask

class DitWrapper(torch.nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.dit = dit
        self.cfg = False
        self.time_embed = dit.time_embed
        self.code_embed = dit.text_embed
        self.rotary_embed = DitRotary()
        self.transformer_blocks = torch.nn.ModuleList()
        for i in range(len(dit.transformer_blocks)):
            self.transformer_blocks.append(DiTBlock(dit.transformer_blocks[i]))
        self._create_block_diff = dit._create_block_diff
        self.norm_out = dit.norm_out
        self.proj_out = dit.proj_out
        proj_in = dit.input_embed.proj
        oc, ic = proj_in.weight.shape
        x_ic = 80
        other_ic = ic - x_ic
        self.proj_in_x = torch.nn.Linear(x_ic, oc)
        self.proj_in_x.weight.data = proj_in.weight[:, :x_ic]
        self.proj_in_x.bias = None
        self.proj_in_other = torch.nn.Linear(other_ic, oc)
        self.proj_in_other.weight.data = proj_in.weight[:, x_ic:]
        self.proj_in_other.bias = proj_in.bias
        self.spk_encoder = ECAPA_TDNN(dit.input_embed.spk_encoder)
        self.preprocess = DitPreprocess(self)

    def spk_encode(self, spk):
        return self.spk_encoder(spk)

    def forward(self, x, code_embeds, rope, mask, time):
        t = self.time_embed(time)
        hidden = self.proj_in_x(x) + code_embeds
        for block in self.transformer_blocks:
            hidden = block(hidden, t, rope=rope, block_diff=mask)
        hidden = self.norm_out(hidden, t)
        output = self.proj_out(hidden)
        return output

# end

class Qwen2_5OmniToken2Wav(Token2Wav):
    def __init__(self, token2wav, base):
        super().__init__(token2wav, base)

    def load(self):
        self.dit = self.token2wav.code2wav_dit_model
        self.bigvgan = self.token2wav.code2wav_bigvgan_model
        # some code change for export
        self.dit = DitWrapper(self.dit)
        # bigvgan.resblocks.activations.up/downsample contain conv weight channel by input
        for i in range(len(self.bigvgan.resblocks)):
            for j in range(len(self.bigvgan.resblocks[i].activations)):
                old_act = self.bigvgan.resblocks[i].activations[j]
                self.bigvgan.resblocks[i].activations[j] = TorchActivation1d(old_act)
        self.bigvgan.activation_post = TorchActivation1d(self.bigvgan.activation_post)
        # spk
        path = os.path.join(self.args.path, 'spk_dict.pt')
        self.speaker_map = {}
        for key, value in torch.load(path).items():
            spk = value["cond"].float()
            cond = value['ref_mel'].float()
            value.pop("ref_mel", None)
            value['spk'] = spk.unsqueeze(1)
            value['cond'] =self.dit.spk_encode(cond)
            self.speaker_map[key] = value
        spk = "Chelsie"
        self.speaker_params = self.speaker_map[spk]

    def dit_forward(self, code, initial_noise = None):
        spk = self.speaker_params["spk"].float()
        cond = self.speaker_params["cond"].float()
        max_duration = code.shape[1] * 2
        code_embeds, rope, mask = self.dit.preprocess(cond, spk, code)
        def func(t, x):
            pred = self.dit(x=x, code_embeds=code_embeds, rope=rope, mask=mask, time=torch.tensor([t]))
            return pred

        steps = 5
        t = torch.linspace(0, 1, steps, dtype=cond.dtype)
        t = 1 - torch.cos(torch.pi / 2 * t)

        if initial_noise is None:
            torch.manual_seed(42)
            y0 = torch.randn([1, max_duration, 80], dtype=cond.dtype)
        else:
            y0 = initial_noise.clone()

        for t0, t1 in zip(t[:-1], t[1:]):
            dt = t1 - t0
            k1 = func(t0, y0)
            k2 = func(t0 + dt * 1/3, y0 + dt * k1 * 1/3)
            k3 = func(t0 + dt * 2/3, y0 + dt * (k2 - k1 * 2/3))
            k4 = func(t1, y0 + dt * (k1 - k2 + k3))
            dy = (k1 + 3 * (k2 + k3) + k4) * dt * 0.125
            y0 += dy

        generated_mel = y0.permute(0, 2, 1)
        # print('generated_mel = ', generated_mel, generated_mel.shape)
        # print('generated_mel.shape = ', generated_mel.shape)
        return generated_mel

    @torch.no_grad()
    def generate(self, code):
        generated_mel = self.dit_forward(code)
        waveform = self.bigvgan(generated_mel)
        return waveform

    @torch.no_grad()
    def generate_stream(self, code):
        # Defeine dit streaming parameters
        dit_chunk_size = 48
        dit_left_context = 24
        dit_right_context = 12
        dit_left_padding = 0
        dit_right_padding = dit_right_context
        dit_start_index = 0
        dit_mel_len = 0

        # Define vocoder streaming parameters
        vocoder_left_context = 10
        vocoder_right_context = 10
        vocoder_left_pad = 0
        vocoder_right_pad = vocoder_right_context
        vocoder_upsample_rate = 240

        torch.manual_seed(42)
        initial_noise = torch.randn([1, 30000, 80], dtype=torch.float32)
        code_buffer = torch.full((1, 0), 0, dtype=torch.long, device=code.device)
        mel_buffer = torch.full((1, 80, 0), 0, dtype=torch.float32, device=code.device)
        waveform_buffer = torch.full((0,), 0, dtype=torch.float32)
        for next_code in code[0]:
            code_buffer = torch.cat([code_buffer, next_code.reshape(1, 1)], dim=1)
            if code_buffer.size(1) == dit_left_padding + dit_chunk_size + dit_right_padding:
                # dit
                generated_mel = self.dit_forward(code_buffer, initial_noise[:, dit_start_index: dit_start_index + code_buffer.size(1) * 2])
                generated_mel = generated_mel[:, :, dit_left_padding * 2: -dit_right_padding * 2]
                dit_left_padding = dit_left_context
                code_buffer = code_buffer[:, -(dit_left_padding + dit_right_padding):]
                dit_mel_len += generated_mel.size(-1)
                dit_start_index = dit_mel_len - dit_left_context * 2
                # bigvgan
                mel_buffer = torch.cat([mel_buffer, generated_mel], dim=-1)
                waveform = self.bigvgan(mel_buffer)
                waveform = waveform[vocoder_left_pad * vocoder_upsample_rate: -vocoder_right_pad * vocoder_upsample_rate]
                waveform_buffer = torch.cat([waveform_buffer, waveform], dim=-1)
                vocoder_left_pad = vocoder_left_context
                mel_buffer = mel_buffer[:, :, -(vocoder_left_pad + vocoder_right_pad):]

        if code_buffer.size(1) > 0:
            generated_mel = self.dit_forward(code_buffer, initial_noise[:, dit_start_index: dit_start_index + code_buffer.size(1) * 2])
            generated_mel = generated_mel[:, :, dit_left_padding * 2:]
            mel_buffer = torch.cat([mel_buffer, generated_mel], dim=-1)
            waveform = self.bigvgan(mel_buffer)
            waveform = waveform[vocoder_left_pad * vocoder_upsample_rate:]
            waveform_buffer = torch.cat([waveform_buffer, waveform], dim=-1)

        return waveform_buffer

    def export_spk(self):
        import MNN.expr as expr
        def torch_to_mnn(x):
            return expr.const(x.data_ptr(), x.shape)
        var_list = []
        for key, value in self.speaker_map.items():
            for k, v in value.items():
                if type(v) is not torch.Tensor:
                    v = torch.tensor(v)
                mnn_var = torch_to_mnn(v.contiguous().float())
                mnn_var.name = f'{key}_{k}'
                var_list.append(mnn_var)
        expr.save(var_list, f'{self.args.dst_path}/spk_dict.mnn')

    @spinner_run(f'export token2wav.predit to ')
    def export_predit(self, onnx_path):
        cond = torch.randn([1, 1, 128], dtype=torch.float32)
        spk = torch.randn([1, 192], dtype=torch.float32)
        code = torch.ones([1, 256], dtype=torch.int32)
        onnx_model = f'{onnx_path}/predit.onnx'
        onnx_export(self.dit.preprocess, (cond, spk, code),
                    onnx_model,
                    input_names=['cond', 'spk', 'code'],
                    output_names=['code_embeds', 'rope', 'mask'],
                    dynamic_axes={
                        "code": { 1: "size" },
                    })
        return onnx_model

    @spinner_run(f'export token2wav.dit to ')
    def export_dit(self, onnx_path):
        x = torch.randn([1, 512, 80], dtype=torch.float32)
        code_embeds = torch.randn([1, 512, 1024], dtype=torch.float32)
        rope = torch.randn([2, 1, 512, 1, 64], dtype=torch.float32)
        mask = torch.ones([1, 1, 512, 512], dtype=torch.int32)
        time = torch.tensor([0.0])
        onnx_model = f'{onnx_path}/dit.onnx'
        onnx_export(self.dit, (x, code_embeds, rope, mask, time),
                    onnx_model,
                    input_names=['x', 'code_embeds', 'rope', 'mask', 'time'],
                    output_names=['mel'],
                    dynamic_axes={
                        "x": { 1: "size" },
                        "code_embeds": { 1: "size" },
                        "rope": { 2: "size" },
                        "mask": { 2: "size", 3: "size" },
                    })
        return onnx_model

    @spinner_run(f'export token2wav.bigvgan to ')
    def export_bigvgan(self, onnx_path):
        generated_mel = torch.randn([1, 80, 512], dtype=torch.float32)
        onnx_model = f'{onnx_path}/bigvgan.onnx'
        onnx_export(self.bigvgan, (generated_mel),
                    onnx_model,
                    input_names=['generated_mel'],
                    output_names=['waveform'],
                    dynamic_axes={
                        "generated_mel": { 2: "size" },
                    })
        return onnx_model

    def export(self, onnx_path):
        self.export_spk()
        predit = self.export_predit(onnx_path)
        dit = self.export_dit(onnx_path)
        bigvgan = self.export_bigvgan(onnx_path)
        return predit, dit, bigvgan

# CosyVoice3 classes

class CosyVoice3DiTAttention(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.heads = attn.heads
        self.to_q = attn.to_q
        self.to_k = attn.to_k
        self.to_v = attn.to_v
        self.to_out = attn.to_out

    def forward(self, x, rope=None, mask=None):
        batch_size = x.shape[0]
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)
        head_dim = query.shape[-1] // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim)
        key = key.view(batch_size, -1, self.heads, head_dim)
        value = value.view(batch_size, -1, self.heads, head_dim)
        # Apply RoPE to ALL heads (unlike Qwen2.5-Omni which only applies to first head)
        cos, sin = rope[0], rope[1]
        query = DitRotary.apply_rotary_pos(query, cos, sin)
        key = DitRotary.apply_rotary_pos(key, cos, sin)
        # Manual attention
        query = query.transpose(1, 2)  # [B, H, T, D]
        key = key.permute([0, 2, 3, 1])  # [B, H, D, T]
        value = value.transpose(1, 2)  # [B, H, T, D]
        attn_weights = torch.matmul(query, key) / math.sqrt(head_dim)
        attn_weights = attn_weights + mask  # mask is float: 0 or -inf
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)
        x = attn_output.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        x = self.to_out[0](x)
        x = self.to_out[1](x)
        return x

class CosyVoice3DiTBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.attn_norm = block.attn_norm
        self.attn = CosyVoice3DiTAttention(block.attn)
        self.ff_norm = block.ff_norm
        self.ff = block.ff

    def forward(self, x, t, rope=None, mask=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(x=norm, rope=rope, mask=mask)
        x = x + gate_msa.unsqueeze(1) * attn_output
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output
        return x

class CosyVoice3DitPreprocess(torch.nn.Module):
    def __init__(self, dit_wrapper):
        super().__init__()
        self.input_embedding = dit_wrapper.input_embedding
        self.pre_lookahead_conv1 = dit_wrapper.pre_lookahead_layer.conv1
        self.pre_lookahead_conv2 = dit_wrapper.pre_lookahead_layer.conv2
        self.pre_lookahead_len = dit_wrapper.pre_lookahead_layer.pre_lookahead_len
        self.spk_embed_affine_layer = dit_wrapper.spk_embed_affine_layer
        self.input_proj = dit_wrapper.proj_in_other
        self.rotary_embed = DitRotary()
        self.token_mel_ratio = dit_wrapper.token_mel_ratio
        self.static_chunk_size = dit_wrapper.static_chunk_size

    def forward(self, cond, spk, code):
        # cond: [1, Tc, 80] — reference mel (speech_feat)
        # spk: [1, 1, 192] — speaker embedding
        # code: [1, N] — prompt_tokens + generated codec tokens (prepended in C++)

        # Token embedding
        token_embed = self.input_embedding(code)  # [1, N, 80]

        # Pre-lookahead (inline, always non-streaming with empty context)
        outputs = token_embed.transpose(1, 2)  # [1, 80, N]
        outputs = F.pad(outputs, (0, self.pre_lookahead_len))  # pad right
        outputs = F.leaky_relu(self.pre_lookahead_conv1(outputs))
        outputs = F.pad(outputs, (self.pre_lookahead_conv2.kernel_size[0] - 1, 0))  # pad left
        outputs = self.pre_lookahead_conv2(outputs)
        mu = outputs.transpose(1, 2) + token_embed  # residual [1, N, 80]

        # Upsample
        mu = mu.repeat_interleave(self.token_mel_ratio, dim=1)  # [1, 2N, 80]
        mel_len = mu.shape[1]

        # Speaker
        spk_norm = F.normalize(spk, dim=-1)  # [1, 192]
        spk_proj = self.spk_embed_affine_layer(spk_norm)  # [1, 80]
        spk_expanded = spk_proj.unsqueeze(1).expand(-1, mel_len, -1)  # [1, mel_len, 80]

        # Pad cond
        cond_padded = F.pad(cond, (0, 0, 0, mel_len - cond.shape[1]))  # [1, mel_len, 80]

        # Project other
        other = torch.cat([cond_padded, mu, spk_expanded], dim=-1)  # [1, mel_len, 240]
        code_embeds = self.input_proj(other)  # [1, mel_len, 1024]

        # Rope
        position_ids = torch.arange(mel_len)
        rope = self.rotary_embed(position_ids)  # [2, 1, mel_len, 1, 64]

        # Chunk attention mask (0=attend, -inf=don't attend)
        chunk_ids = torch.arange(mel_len) // self.static_chunk_size
        attend = chunk_ids.unsqueeze(0) <= chunk_ids.unsqueeze(1)  # [mel_len, mel_len]
        mask = torch.where(attend, torch.tensor(0.0), torch.tensor(torch.finfo(torch.float32).min))
        mask = mask.reshape(1, 1, mel_len, mel_len)

        return code_embeds, rope, mask

class CosyVoice3DitWrapper(torch.nn.Module):
    def __init__(self, flow_model):
        super().__init__()
        dit = flow_model.decoder.estimator
        self.time_embed = dit.time_embed

        # Split proj
        proj_in = dit.input_embed.proj
        oc, ic = proj_in.weight.shape  # [1024, 320]
        x_ic = 80
        other_ic = ic - x_ic
        self.proj_in_x = torch.nn.Linear(x_ic, oc)
        self.proj_in_x.weight.data = proj_in.weight[:, :x_ic].clone()
        self.proj_in_x.bias = None
        self.proj_in_other = torch.nn.Linear(other_ic, oc)
        self.proj_in_other.weight.data = proj_in.weight[:, x_ic:].clone()
        self.proj_in_other.bias = proj_in.bias

        self.conv_pos_embed = dit.input_embed.conv_pos_embed

        self.transformer_blocks = torch.nn.ModuleList()
        for block in dit.transformer_blocks:
            self.transformer_blocks.append(CosyVoice3DiTBlock(block))

        self.norm_out = dit.norm_out
        self.proj_out = dit.proj_out

        # Pre-processing components stored for preprocess access
        self.input_embedding = flow_model.input_embedding
        self.pre_lookahead_layer = flow_model.pre_lookahead_layer
        self.spk_embed_affine_layer = flow_model.spk_embed_affine_layer
        self.token_mel_ratio = flow_model.token_mel_ratio
        self.static_chunk_size = dit.static_chunk_size

        self.preprocess = CosyVoice3DitPreprocess(self)

    def forward(self, x, code_embeds, rope, mask, time):
        # x: [1, mel_len, 80] — noised mel
        # code_embeds: [1, mel_len, 1024]
        # rope: [2, 1, mel_len, 1, 64]
        # mask: [1, 1, mel_len, mel_len] — float mask
        # time: [1]
        t = self.time_embed(time)
        hidden = self.proj_in_x(x) + code_embeds
        hidden = self.conv_pos_embed(hidden) + hidden
        for block in self.transformer_blocks:
            hidden = block(hidden, t, rope=rope, mask=mask)
        hidden = self.norm_out(hidden, t)
        output = self.proj_out(hidden)
        return output

class CosyVoice3HiftWrapper(torch.nn.Module):
    def __init__(self, hift):
        super().__init__()
        self.hift = hift
        # Remove weight_norm parametrizations
        import torch.nn.utils.parametrize as parametrize
        for name, module in self.hift.named_modules():
            if hasattr(module, 'parametrizations') and hasattr(module.parametrizations, 'weight'):
                try:
                    parametrize.remove_parametrizations(module, 'weight')
                except:
                    pass
        try:
            self.hift.remove_weight_norm()
        except:
            pass

        # Build Conv1d kernels for STFT/ISTFT replacement
        n_fft = hift.istft_params['n_fft']   # 16
        hop_len = hift.istft_params['hop_len']  # 4
        n_freq = n_fft // 2 + 1  # 9
        self.n_fft = n_fft
        self.hop_len = hop_len

        # Periodic Hann window (matches scipy fftbins=True used by original code)
        window = torch.hann_window(n_fft, periodic=True)

        # DFT basis
        freqs = torch.arange(n_freq, dtype=torch.float32)
        times = torch.arange(n_fft, dtype=torch.float32)
        angles = 2.0 * math.pi * freqs.unsqueeze(1) * times.unsqueeze(0) / n_fft  # [9, 16]

        # STFT analysis kernel: [18, 1, 16]
        cos_basis = torch.cos(angles) * window  # [9, 16]
        sin_basis = -torch.sin(angles) * window  # [9, 16]
        stft_kernel = torch.cat([cos_basis, sin_basis], dim=0).unsqueeze(1)  # [18, 1, 16]
        self.register_buffer('stft_kernel', stft_kernel)

        # ISTFT synthesis kernel: [18, 1, 16]
        # Scaling: c[0]=c[N/2]=1/N, c[k]=2/N for 0<k<N/2
        c = torch.ones(n_freq) * 2.0 / n_fft
        c[0] = 1.0 / n_fft
        c[-1] = 1.0 / n_fft
        istft_cos = (c.unsqueeze(1) * torch.cos(angles) * window)  # [9, 16]
        istft_sin = (-c.unsqueeze(1) * torch.sin(angles) * window)  # [9, 16]
        istft_kernel = torch.cat([istft_cos, istft_sin], dim=0).unsqueeze(1)  # [18, 1, 16]
        self.register_buffer('istft_kernel', istft_kernel)

        # Window^2 kernel for overlap normalization: [1, 1, 16]
        win_sq_kernel = (window ** 2).unsqueeze(0).unsqueeze(0)  # [1, 1, 16]
        self.register_buffer('win_sq_kernel', win_sq_kernel)

    def _conv_stft(self, x):
        # x: [B, T_wave] — replaces torch.stft with center=True
        pad = self.n_fft // 2
        x = F.pad(x, (pad, pad), mode='reflect')
        x = x.unsqueeze(1)  # [B, 1, T_wave+n_fft]
        spec = F.conv1d(x, self.stft_kernel, stride=self.hop_len)  # [B, 18, T_frames]
        n_freq = self.n_fft // 2 + 1
        return spec[:, :n_freq], spec[:, n_freq:]  # real, imag

    def _conv_istft(self, magnitude, phase):
        # magnitude, phase: [B, 9, T_frames] — replaces torch.istft with center=True
        magnitude = torch.clamp(magnitude, max=100.0)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        spec = torch.cat([real, imag], dim=1)  # [B, 18, T_frames]
        # Overlap-add via ConvTranspose1d
        wave = F.conv_transpose1d(spec, self.istft_kernel, stride=self.hop_len)  # [B, 1, out_len]
        # Window normalization
        T_frames = spec.shape[2]
        ones = torch.ones(1, 1, T_frames, device=spec.device, dtype=spec.dtype)
        norm = F.conv_transpose1d(ones, self.win_sq_kernel, stride=self.hop_len)  # [1, 1, out_len]
        wave = wave / (norm + 1e-8)
        # Trim center padding
        pad = self.n_fft // 2
        wave = wave[:, :, pad:-pad]
        return wave.squeeze(1)  # [B, T_wave]

    def forward(self, generated_mel):
        # generated_mel: [1, 80, T]
        # f0 prediction
        f0 = self.hift.f0_predictor(generated_mel)  # [1, T]
        # f0 → source
        s = self.hift.f0_upsamp(f0[:, None]).transpose(1, 2)  # [1, T*480, 1]
        s, _, _ = self.hift.m_source(s)
        s = s.transpose(1, 2)  # [1, 1, T*480]

        # === Inlined decode (replaces self.hift.decode) ===
        # STFT on source signal
        s_stft_real, s_stft_imag = self._conv_stft(s.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)  # [B, 18, T_frames]

        x = self.hift.conv_pre(generated_mel)
        for i in range(self.hift.num_upsamples):
            x = F.leaky_relu(x, self.hift.lrelu_slope)
            x = self.hift.ups[i](x)
            if i == self.hift.num_upsamples - 1:
                x = self.hift.reflection_pad(x)
            # Source fusion
            si = self.hift.source_downs[i](s_stft)
            si = self.hift.source_resblocks[i](si)
            x = x + si
            # ResBlocks
            xs = None
            for j in range(self.hift.num_kernels):
                if xs is None:
                    xs = self.hift.resblocks[i * self.hift.num_kernels + j](x)
                else:
                    xs = xs + self.hift.resblocks[i * self.hift.num_kernels + j](x)
            x = xs / self.hift.num_kernels

        x = F.leaky_relu(x)
        x = self.hift.conv_post(x)
        n_freq = self.hift.istft_params['n_fft'] // 2 + 1
        magnitude = torch.exp(x[:, :n_freq, :])
        phase = torch.sin(x[:, n_freq:, :])

        # ISTFT
        waveform = self._conv_istft(magnitude, phase)
        waveform = torch.clamp(waveform, -self.hift.audio_limit, self.hift.audio_limit)
        return waveform

class CosyVoice3Token2Wav:
    def __init__(self, args, cosyvoice_model_path):
        self.args = args
        self.cosyvoice_model_path = cosyvoice_model_path
        self.quant_bit = 8
        self.load()

    def load(self):
        import sys
        sys.path.insert(0, '/home/taowen/Fun-Audio-Chat/third_party/CosyVoice')
        sys.path.insert(0, '/home/taowen/Fun-Audio-Chat/third_party/CosyVoice/third_party/Matcha-TTS')

        import torch
        from omegaconf import DictConfig
        from cosyvoice.transformer.upsample_encoder import PreLookaheadLayer
        from cosyvoice.flow.flow_matching import CausalConditionalCFM
        from cosyvoice.flow.DiT.dit import DiT
        from cosyvoice.flow.flow import CausalMaskedDiffWithDiT
        from cosyvoice.hifigan.generator import CausalHiFTGenerator
        from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor

        model_path = self.cosyvoice_model_path

        # Instantiate flow model
        pre_lookahead_layer = PreLookaheadLayer(in_channels=80, channels=1024, pre_lookahead_len=3)
        estimator = DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2,
                       mel_dim=80, mu_dim=80, spk_dim=80, out_channels=80,
                       static_chunk_size=50, num_decoding_left_chunks=-1)
        cfm_params = DictConfig({
            'sigma_min': 1e-6, 'solver': 'euler', 't_scheduler': 'cosine',
            'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'
        })
        decoder = CausalConditionalCFM(in_channels=240, n_spks=1, spk_emb_dim=80,
                                       cfm_params=cfm_params, estimator=estimator)
        flow_model = CausalMaskedDiffWithDiT(
            input_size=80, output_size=80, spk_embed_dim=192, output_type='mel',
            vocab_size=6561, input_frame_rate=25, only_mask_loss=True,
            token_mel_ratio=2, pre_lookahead_len=3,
            pre_lookahead_layer=pre_lookahead_layer, decoder=decoder
        )
        state_dict = torch.load(f'{model_path}/flow.pt', map_location='cpu')
        flow_model.load_state_dict(state_dict)
        flow_model.eval().float()

        # Wrap DiT
        self.dit = CosyVoice3DitWrapper(flow_model)

        # Instantiate HIFT
        f0_pred = CausalConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512)
        hift = CausalHiFTGenerator(
            in_channels=80, base_channels=512, nb_harmonics=8,
            sampling_rate=24000, nsf_alpha=0.1, nsf_sigma=0.003,
            nsf_voiced_threshold=10, upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            istft_params={'n_fft': 16, 'hop_len': 4},
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            lrelu_slope=0.1, audio_limit=0.99, conv_pre_look_right=4,
            f0_predictor=f0_pred
        )
        hift_sd = torch.load(f'{model_path}/hift.pt', map_location='cpu')
        hift.load_state_dict(hift_sd)
        hift.eval().float()
        self.bigvgan = CosyVoice3HiftWrapper(hift)

        # Load speaker map
        spk_path = '/home/taowen/Fun-Audio-Chat/utils/new_spk2info.pt'
        self.speaker_map = torch.load(spk_path, map_location='cpu')

    def export_spk(self):
        import MNN.expr as expr
        def torch_to_mnn(x):
            return expr.const(x.data_ptr(), x.shape)
        var_list = []
        for key, value in self.speaker_map.items():
            # spk: speaker embedding [1, 192]
            spk = value['embedding'].contiguous().float()
            mnn_var = torch_to_mnn(spk)
            mnn_var.name = f'{key}_spk'
            var_list.append(mnn_var)
            # cond: speech_feat [1, T, 80]
            cond = value['speech_feat'].contiguous().float()
            mnn_var = torch_to_mnn(cond)
            mnn_var.name = f'{key}_cond'
            var_list.append(mnn_var)
            # prompt_token: speech_token [1, N]
            prompt_token = value['speech_token'].contiguous().int()
            mnn_var = torch_to_mnn(prompt_token)
            mnn_var.name = f'{key}_prompt_token'
            var_list.append(mnn_var)
            # bos_token: dummy (not used for CosyVoice3 but C++ expects it)
            bos = torch.tensor([0], dtype=torch.int32).contiguous()
            mnn_var = torch_to_mnn(bos)
            mnn_var.name = f'{key}_bos_token'
            var_list.append(mnn_var)
        expr.save(var_list, f'{self.args.dst_path}/spk_dict.mnn')

    @spinner_run(f'export token2wav.predit to ')
    def export_predit(self, onnx_path):
        cond = torch.randn([1, 100, 80], dtype=torch.float32)
        spk = torch.randn([1, 192], dtype=torch.float32)
        code = torch.ones([1, 256], dtype=torch.int32)
        onnx_model = f'{onnx_path}/predit.onnx'
        onnx_export(self.dit.preprocess, (cond, spk, code),
                    onnx_model,
                    input_names=['cond', 'spk', 'code'],
                    output_names=['code_embeds', 'rope', 'mask'],
                    dynamic_axes={
                        "cond": { 1: "cond_size" },
                        "code": { 1: "size" },
                    })
        return onnx_model

    @spinner_run(f'export token2wav.dit to ')
    def export_dit(self, onnx_path):
        x = torch.randn([1, 512, 80], dtype=torch.float32)
        code_embeds = torch.randn([1, 512, 1024], dtype=torch.float32)
        rope = torch.randn([2, 1, 512, 1, 64], dtype=torch.float32)
        mask = torch.zeros([1, 1, 512, 512], dtype=torch.float32)
        time = torch.tensor([0.0])
        onnx_model = f'{onnx_path}/dit.onnx'
        onnx_export(self.dit, (x, code_embeds, rope, mask, time),
                    onnx_model,
                    input_names=['x', 'code_embeds', 'rope', 'mask', 'time'],
                    output_names=['mel'],
                    dynamic_axes={
                        "x": { 1: "size" },
                        "code_embeds": { 1: "size" },
                        "rope": { 2: "size" },
                        "mask": { 2: "size", 3: "size" },
                    })
        return onnx_model

    @spinner_run(f'export token2wav.bigvgan to ')
    def export_bigvgan(self, onnx_path):
        generated_mel = torch.randn([1, 80, 60], dtype=torch.float32)
        onnx_model = f'{onnx_path}/bigvgan.onnx'
        onnx_export(self.bigvgan, (generated_mel,),
                    onnx_model,
                    input_names=['generated_mel'],
                    output_names=['waveform'],
                    dynamic_axes={
                        "generated_mel": { 2: "size" },
                    })
        return onnx_model

    def export(self, onnx_path):
        self.export_spk()
        predit = self.export_predit(onnx_path)
        dit = self.export_dit(onnx_path)
        bigvgan = self.export_bigvgan(onnx_path)
        return predit, dit, bigvgan