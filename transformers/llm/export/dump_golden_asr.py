#!/usr/bin/env python3
"""Dump golden tensors from Qwen3-ASR-0.6B for verification against MNN C++ inference.

Outputs:
  golden_mel.npy        — mel features [1, 128, T]
  golden_audio_embed.npy — audio encoder output [1, seq_len, hidden]
  golden_output_ids.txt  — generated token IDs, one per line
"""

import os
import sys
import argparse
import numpy as np
import torch
import librosa

# Add export utils to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.model import LlmModel
from utils.tokenizer import LlmTokenizer


def main():
    parser = argparse.ArgumentParser(description='Dump golden tensors for Qwen3-ASR verification')
    parser.add_argument('--model_path', required=True, help='Path to Qwen3-ASR-0.6B model')
    parser.add_argument('--audio_path', required=True, help='Path to test WAV file')
    parser.add_argument('--output_dir', required=True, help='Directory to save golden tensors')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='Max tokens to generate')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model_path} ...")

    class FakeArgs:
        path = args.model_path
        tokenizer_path = args.model_path
        quant_bit = 4
        quant_block = 0
        lm_quant_bit = None
        lm_quant_block = None
        awq = False
        smooth = False
        export = None
        dst_path = '/tmp/golden_export'
        seperate_embed = False
        embed_bit = 8
        eagle_path = None
        mtp = False
        sym = False
        lora_path = None
        lora_split = False
        test = None
        skip_weight = False

    fake_args = FakeArgs()

    model = LlmModel.from_pretrained(args.model_path, args=fake_args)
    model.float()
    tokenizer = LlmTokenizer.from_pretrained(args.model_path, model_type=model.config.model_type)
    model.tokenizer = tokenizer
    audio = model.audio

    # Load audio
    print(f"Loading audio from {args.audio_path} ...")
    waveform_np, sr = librosa.load(args.audio_path, sr=16000)
    waveform = torch.from_numpy(waveform_np).float()
    print(f"  waveform shape: {waveform.shape}, duration: {len(waveform_np)/16000:.2f}s")

    # 1. Dump mel features
    print("Extracting mel features ...")
    with torch.no_grad():
        mel = audio._torch_extract_fbank_features(waveform).unsqueeze(0)
    print(f"  mel shape: {mel.shape}")
    assert mel.shape[1] == 128, f"Expected 128 mel bins, got {mel.shape[1]}"
    assert mel.abs().sum() > 0, "Mel features are all zeros!"
    np.save(os.path.join(args.output_dir, 'golden_mel.npy'), mel.numpy())
    print(f"  saved golden_mel.npy")

    # 2. Dump audio encoder output
    # Build prompt to run audio_process via str_to_ids (this also sets audio.audio_embeds)
    prompt = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|audio_start|><audio>{args.audio_path}</audio><|audio_end|><|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    print("Running audio encoder via str_to_ids ...")
    with torch.no_grad():
        input_ids = audio.str_to_ids(prompt)
    audio_embed = audio.audio_embeds  # [seq_len, 1, hidden]
    audio_embed_np = audio_embed.permute(1, 0, 2).numpy()  # [1, seq_len, hidden]
    embed_len = audio_embed_np.shape[1]
    print(f"  audio_embed shape: {audio_embed_np.shape}, embed_len: {embed_len}")
    assert audio_embed_np.shape[1] > 0, "Audio embedding has zero length!"
    assert np.abs(audio_embed_np).sum() > 0, "Audio embedding is all zeros!"
    np.save(os.path.join(args.output_dir, 'golden_audio_embed.npy'), audio_embed_np)
    print(f"  saved golden_audio_embed.npy")

    # 3. Generate output tokens
    print("Generating output tokens ...")
    seq_len = input_ids.numel()
    output_ids = []

    print("  Prompt tokens:", seq_len)
    print("  Generating: ", end="", flush=True)

    with torch.no_grad():
        new_tokens = 0
        while new_tokens < args.max_new_tokens:
            attention_mask = model.get_attention_mask(seq_len, new_tokens)
            position_ids = model.get_position_ids(seq_len, new_tokens, input_ids)
            input_embeds = model.embedding(input_ids)
            logits, _, _ = model.forward(
                input_ids=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                logits_index=torch.tensor([-1], dtype=torch.int32),
                deepstack_embeds=None
            )
            token_id = torch.argmax(logits[:, -1, :]).item()
            output_ids.append(token_id)
            seq_len += 1
            new_tokens += 1

            if token_id in tokenizer.stop_ids:
                print(" [EOS]")
                break

            word = tokenizer.id_to_str(token_id)
            print(word, end="", flush=True)
            input_ids = torch.tensor([[token_id]])

    print(f"\n  Generated {len(output_ids)} tokens")

    # Save output IDs
    ids_path = os.path.join(args.output_dir, 'golden_output_ids.txt')
    with open(ids_path, 'w') as f:
        for tid in output_ids:
            f.write(f"{tid}\n")
    print(f"  saved golden_output_ids.txt")

    # Save decoded text
    text = tokenizer.decode(output_ids)
    text_path = os.path.join(args.output_dir, 'golden_output_text.txt')
    with open(text_path, 'w') as f:
        f.write(text)
    print(f"  saved golden_output_text.txt")
    print(f"\nAll golden tensors saved to {args.output_dir}")


if __name__ == '__main__':
    main()
