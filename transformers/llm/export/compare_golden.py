#!/usr/bin/env python3
"""Compare MNN C++ dumped tensors against Python golden outputs.

Usage:
    python3 compare_golden.py --golden_dir /path/to/golden/ --mnn_mel mel_features.mnn --mnn_embed audio_embed.mnn [--mnn_tokens token_log.txt]

The MNN .mnn files are saved by C++ DEBUG_AUDIO code via Variable::save().
Golden files are .npy files from dump_golden_asr.py.
"""

import os
import sys
import argparse
import numpy as np

def load_mnn_tensor(mnn_path):
    """Load a tensor from an MNN .mnn file using MNN Python bindings."""
    try:
        import MNN.expr as F
        var_dict = F.load_as_dict(mnn_path)
        results = {}
        for name, var in var_dict.items():
            arr = np.array(var.read())
            results[name] = arr
        return results
    except ImportError:
        print("WARNING: MNN Python module not available. Trying numpy fallback.")
        print("Install MNN Python: pip install MNN")
        return None

def compare_tensors(name, golden, mnn, atol=1e-4):
    """Compare two numpy arrays and report differences."""
    print(f"\n--- {name} ---")
    print(f"  Golden shape: {golden.shape}, dtype: {golden.dtype}")
    print(f"  MNN    shape: {mnn.shape}, dtype: {mnn.dtype}")

    if golden.shape != mnn.shape:
        print(f"  SHAPE MISMATCH!")
        # Try to compare overlapping region
        min_shape = tuple(min(g, m) for g, m in zip(golden.shape, mnn.shape))
        print(f"  Comparing overlapping region: {min_shape}")
        slices = tuple(slice(0, s) for s in min_shape)
        golden = golden[slices]
        mnn = mnn[slices]

    diff = np.abs(golden.astype(np.float32) - mnn.astype(np.float32))
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"  Max diff:  {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")
    print(f"  Golden range: [{golden.min():.6f}, {golden.max():.6f}]")
    print(f"  MNN range:    [{mnn.min():.6f}, {mnn.max():.6f}]")

    if max_diff < atol:
        print(f"  PASS (max diff < {atol})")
        return True
    else:
        print(f"  FAIL (max diff {max_diff:.6e} >= {atol})")
        # Show where the largest differences are
        flat_idx = np.argmax(diff.flatten())
        idx = np.unravel_index(flat_idx, diff.shape)
        print(f"  Worst at index {idx}: golden={golden[idx]:.6f}, mnn={mnn[idx]:.6f}")
        return False

def compare_token_ids(golden_path, mnn_token_log):
    """Compare generated token IDs."""
    print(f"\n--- Token IDs ---")
    with open(golden_path) as f:
        golden_ids = [int(line.strip()) for line in f if line.strip()]

    mnn_ids = []
    with open(mnn_token_log) as f:
        for line in f:
            line = line.strip()
            # Parse "DEBUG_AUDIO: token[N] = ID" format
            if 'token[' in line and '=' in line:
                token_id = int(line.split('=')[-1].strip())
                mnn_ids.append(token_id)
            elif line.isdigit():
                mnn_ids.append(int(line))

    print(f"  Golden tokens: {len(golden_ids)}")
    print(f"  MNN tokens:    {len(mnn_ids)}")

    min_len = min(len(golden_ids), len(mnn_ids))
    matches = sum(1 for i in range(min_len) if golden_ids[i] == mnn_ids[i])
    print(f"  Matching: {matches}/{min_len}")

    if matches == min_len and len(golden_ids) == len(mnn_ids):
        print(f"  PASS (exact match)")
        return True
    else:
        # Show first divergence
        for i in range(min_len):
            if golden_ids[i] != mnn_ids[i]:
                print(f"  First divergence at index {i}: golden={golden_ids[i]}, mnn={mnn_ids[i]}")
                break
        return False


def main():
    parser = argparse.ArgumentParser(description='Compare MNN C++ outputs against golden tensors')
    parser.add_argument('--golden_dir', required=True, help='Directory with golden_*.npy files')
    parser.add_argument('--mnn_mel', default=None, help='Path to mel_features.mnn from C++')
    parser.add_argument('--mnn_embed', default=None, help='Path to audio_embed.mnn from C++')
    parser.add_argument('--mnn_tokens', default=None, help='Path to token log (from logcat/stdout)')
    parser.add_argument('--mel_atol', type=float, default=1e-4, help='Tolerance for mel comparison')
    parser.add_argument('--embed_atol', type=float, default=1e-3, help='Tolerance for embedding comparison')
    args = parser.parse_args()

    all_pass = True

    if args.mnn_mel:
        golden_mel = np.load(os.path.join(args.golden_dir, 'golden_mel.npy'))
        mnn_data = load_mnn_tensor(args.mnn_mel)
        if mnn_data:
            # The saved tensor has name "mel_features"
            mnn_mel = list(mnn_data.values())[0]
            if not compare_tensors("Mel Features", golden_mel, mnn_mel, atol=args.mel_atol):
                all_pass = False
        else:
            print("Could not load MNN mel tensor")
            all_pass = False

    if args.mnn_embed:
        golden_embed = np.load(os.path.join(args.golden_dir, 'golden_audio_embed.npy'))
        mnn_data = load_mnn_tensor(args.mnn_embed)
        if mnn_data:
            mnn_embed = list(mnn_data.values())[0]
            if not compare_tensors("Audio Embedding", golden_embed, mnn_embed, atol=args.embed_atol):
                all_pass = False
        else:
            print("Could not load MNN embed tensor")
            all_pass = False

    if args.mnn_tokens:
        golden_ids_path = os.path.join(args.golden_dir, 'golden_output_ids.txt')
        if os.path.exists(golden_ids_path) and os.path.exists(args.mnn_tokens):
            if not compare_token_ids(golden_ids_path, args.mnn_tokens):
                all_pass = False
        else:
            print(f"Token files not found: golden={golden_ids_path}, mnn={args.mnn_tokens}")

    print(f"\n{'='*50}")
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
