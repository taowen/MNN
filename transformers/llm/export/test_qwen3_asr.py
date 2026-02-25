#!/usr/bin/env python3
"""Test Qwen3-ASR on Android device via adb broadcast + file-based status polling.

Supports two modes:
  batch     - sends full audio in one generate command (existing behavior)
  streaming - splits audio into chunks, uses streaming_generate command
"""

import argparse
import json
import subprocess
import sys
import time

PACKAGE = "com.alibaba.mnnllm.android"
COMPONENT = f"{PACKAGE}/{PACKAGE}.test.TestBackdoorReceiver"


def adb(*args, device=None):
    cmd = ["adb"]
    if device:
        cmd += ["-s", device]
    cmd += list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"adb command failed: {' '.join(cmd)}", file=sys.stderr)
        print(f"stderr: {result.stderr}", file=sys.stderr)
    return result.stdout.strip()


def adb_broadcast(command, device=None, **extras):
    # Build as single shell command string to preserve quoting on device shell.
    # Must use explicit broadcast (-n component) — Android 8+ blocks implicit broadcasts
    # to manifest-registered receivers.
    parts = [f"am broadcast -n {COMPONENT}"]
    parts.append(f"--es command '{command}'")
    for key, value in extras.items():
        escaped = str(value).replace("'", "'\\''")
        parts.append(f"--es {key} '{escaped}'")
    shell_cmd = " ".join(parts)
    output = adb("shell", shell_cmd, device=device)
    print(f"broadcast {command}: {output}")
    return output


def read_device_file(name, device=None):
    return adb("shell", "run-as", PACKAGE, "cat", f"files/{name}", device=device)


def wait_for_status(target, device=None, timeout=300):
    """Poll test_status.json until status matches target or error. Returns parsed JSON."""
    start = time.time()
    last_status = None
    while time.time() - start < timeout:
        raw = read_device_file("test_status.json", device=device)
        if raw:
            try:
                data = json.loads(raw)
                status = data.get("status")
                if status != last_status:
                    print(f"  status: {status}")
                    last_status = status
                if status == target:
                    return data
                if status == "error":
                    print(f"ERROR: {data.get('error', 'unknown')}", file=sys.stderr)
                    sys.exit(1)
            except json.JSONDecodeError:
                pass
        time.sleep(0.5)
    print(f"TIMEOUT waiting for status '{target}' after {timeout}s", file=sys.stderr)
    sys.exit(1)


def print_debug_log(device=None, lines=50):
    print(f"\n=== Native debug log (last {lines} lines) ===")
    log = read_device_file("mnn_debug.log", device=device)
    if log:
        for line in log.split("\n")[-lines:]:
            print(f"  {line}")
    else:
        print("  (no debug log found)")


def run_batch_test(args):
    """Run batch (non-streaming) ASR test."""
    prompt = args.prompt
    if not prompt:
        prompt = (
            "<|audio_start|>"
            f"<audio>{args.audio}</audio>"
            "<|audio_end|>"
        )
    print(f"=== Generating (batch) with prompt ({len(prompt)} chars) ===")
    adb_broadcast("generate", device=args.device, prompt=prompt)
    status_data = wait_for_status("done", device=args.device, timeout=args.timeout)
    return status_data


def run_streaming_test(args):
    """Run streaming ASR test — audio split into chunks on device."""
    PREFIX = "<|im_start|>user\n<|audio_start|>"
    SUFFIX = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"
    chunk_seconds = str(args.chunk_seconds)

    print(f"=== Streaming generate: audio={args.audio}, chunk={chunk_seconds}s ===")
    adb_broadcast("streaming_generate", device=args.device,
                  audio_file=args.audio,
                  prompt_prefix=PREFIX,
                  prompt_suffix=SUFFIX,
                  chunk_seconds=chunk_seconds)
    status_data = wait_for_status("done", device=args.device, timeout=args.timeout)
    return status_data


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-ASR on Android")
    parser.add_argument("--device", "-s", default=None, help="adb device serial")
    parser.add_argument("--model_dir", default="/data/local/tmp/Qwen3-ASR/Qwen3-ASR-MNN-q4",
                        help="Model directory on device")
    parser.add_argument("--mode", choices=["batch", "streaming"], default="batch",
                        help="Test mode: batch (full audio) or streaming (chunked)")
    parser.add_argument("--audio", default="/data/local/tmp/test_speech.wav",
                        help="Audio file path on device")
    parser.add_argument("--prompt", default=None,
                        help="Prompt for batch mode (overrides --audio)")
    parser.add_argument("--chunk_seconds", type=float, default=1.0,
                        help="Chunk duration in seconds for streaming mode")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("--config", default="config.json",
                        help="Config filename (e.g. config_qnn.json for QNN)")
    parser.add_argument("--skip_load", action="store_true",
                        help="Skip model loading (assume already loaded)")
    args = parser.parse_args()

    if not args.skip_load:
        print(f"=== Loading model: {args.model_dir} (config: {args.config}) ===")
        adb_broadcast("load", device=args.device, model_dir=args.model_dir, config_file=args.config)
        wait_for_status("loaded", device=args.device, timeout=args.timeout)
        print("Model loaded.\n")

    if args.mode == "batch":
        status_data = run_batch_test(args)
    else:
        status_data = run_streaming_test(args)

    print()
    output = read_device_file("test_output.txt", device=args.device)
    print(f"=== Generated output ===\n{output}\n")

    print("=== Metrics ===")
    for key, value in status_data.items():
        if key != "status":
            print(f"  {key}: {value}")

    if not args.skip_load:
        print("\n=== Releasing session ===")
        adb_broadcast("release", device=args.device)
        time.sleep(1)

    print_debug_log(device=args.device)
    print("\nDone.")


if __name__ == "__main__":
    main()
