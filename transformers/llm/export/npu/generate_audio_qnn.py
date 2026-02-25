#!/usr/bin/python

import sys
import os
import argparse
import subprocess
import json
import shutil
import time

def makeIO(args):
    exe = os.path.join(os.getcwd(), args.mnn_path, "generateAudioIO")
    output = os.path.join(args.cache_path, 'testdir')
    print(os.popen(exe + " " + args.model + " " + output).read())

def seperate(args):
    exe = os.path.join(os.getcwd(), args.mnn_path, "compilefornpu")
    model = os.path.join(os.getcwd(), args.model, 'audio.mnn')
    print("model:", model)
    config = {
        "type": "QNN",
        "skips": [],
        "testdir": [
            os.path.join("testdir", "1")
        ],
        "cache": "qnn_audio"
    }
    cache = os.path.join(os.getcwd(), args.cache_path)
    with open(os.path.join(cache, 'qnn.json'), 'w') as f:
        f.write(json.dumps(config, indent=4))

    process = subprocess.Popen(exe + ' ' + model + ' qnn_audio/audio.mnn qnn.json', bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cache, text=True, shell=True)
    for line in process.stdout:
        print(line, end='')

    process.wait()

def compile_qnn(args):
    exe = os.path.join(os.getcwd(), args.mnn_path, "..", "source", "backend", "qnn", "npu_convert.py")
    cache = os.path.join(os.getcwd(), args.cache_path)
    process = subprocess.Popen("python3 " + exe + ' npu_postreat.json %d ' % args.soc_id + ' ' + args.dsp_arch, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cache, text=True, shell=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()

def output_qnn(args):
    if os.path.exists(os.path.join(args.model, 'qnn_audio')):
        shutil.rmtree(os.path.join(args.model, 'qnn_audio'))
    shutil.move(os.path.join(args.cache_path, 'qnn_audio'), os.path.join(args.model, 'qnn_audio'))

    # Update config_qnn.json with audio_model path
    config_path = os.path.join(args.model, "config_qnn.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_npu = json.load(f)
    else:
        config_npu = {
            "llm_model": "qnn/llm.mnn",
            "backend_type": "cpu",
            "thread_num": 1,
            "precision": "low",
            "chunk_limits": [128, 1],
            "memory": "low",
            "sampler_type": "penalty",
            "penalty": 1.1
        }
    config_npu["audio_model"] = "qnn_audio/audio.mnn"
    with open(config_path, 'w') as f:
        f.write(json.dumps(config_npu, indent=4))
    shutil.rmtree(args.cache_path)

def convert(args):
    cache = os.path.join(os.getcwd(), args.cache_path)
    os.makedirs(cache, exist_ok=True)
    sta = time.time()
    print("Step1: Make IO")
    makeIO(args)
    end = time.time()
    print("Cost: ", end - sta, ' s')
    sta = end
    print("Step2: Seperate Model")
    seperate(args)
    end = time.time()
    print("Cost: ", end - sta, ' s')
    sta = end
    print("Step3: Compile to QNN")
    compile_qnn(args)
    end = time.time()
    print("Cost: ", end - sta, ' s')
    print("Step4: Move result file to ", args.model)
    output_qnn(args)

    print("End")

def main():
    parser = argparse.ArgumentParser(description='generate_audio_qnn', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', type=str, required=True,
                        help='model directory path')
    parser.add_argument('--soc_id', type=int, required=True,
                        help='The soc_id, for 8gen4 is 69, for 8gen3 is 57')
    parser.add_argument('--dsp_arch', type=str, required=True,
                        help='The dsp_arch, for 8gen4 is v79, for 8gen3 is v75')
    parser.add_argument('--mnn_path', type=str, default="../../../build/",
                        help='mnn build path')
    parser.add_argument('--cache_path', type=str, default="tmp_audio",
                        help='cache path for work')
    args = parser.parse_args()
    convert(args)


if __name__ == '__main__':
    main()
