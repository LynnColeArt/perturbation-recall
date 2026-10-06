#!/usr/bin/env python3
"""Build the native worker against the exact audited llama.cpp revision."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--llama-source', type=Path, required=True)
    p.add_argument('--cuda-compiler', type=Path, default=Path('/usr/local/cuda/bin/nvcc'))
    p.add_argument('--jobs', type=int, default=8)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    runtime = json.loads((root/'models/llama-runtime.json').read_text())
    revision = subprocess.check_output(['git','-C',str(args.llama_source),'rev-parse','HEAD'],text=True).strip()
    if revision != runtime['revision']:
        p.error('llama.cpp checkout does not match the pinned runtime revision')
    if subprocess.check_output(['git','-C',str(args.llama_source),'diff','HEAD','--','include','src','ggml'],text=True):
        p.error('Pinned llama.cpp runtime sources have local changes')
    build = root/'.cache/build'
    subprocess.run(['cmake','-S',str(root/'native'),'-B',str(build),
        '-DLLAMA_SOURCE_DIR='+str(args.llama_source.resolve()), '-DGGML_CUDA=ON',
        '-DCMAKE_CUDA_COMPILER='+str(args.cuda_compiler), '-DCMAKE_CUDA_ARCHITECTURES=121',
        '-DCMAKE_BUILD_TYPE=Release', '-DGGML_CUDA_GRAPHS=OFF'],check=True)
    subprocess.run(['cmake','--build',str(build),'--target','perturbation-worker','-j',str(args.jobs)],check=True)
    print(build/'perturbation-worker')


if __name__ == '__main__':
    main()
