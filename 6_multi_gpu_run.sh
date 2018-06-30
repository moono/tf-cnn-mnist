#!/usr/bin/env bash

python multi_gpu.py --num_gpus=1 --batch_size=128
python multi_gpu.py --num_gpus=1 --batch_size=1024
python multi_gpu.py --num_gpus=1 --batch_size=2048
python multi_gpu.py --num_gpus=1 --batch_size=4096

python multi_gpu.py --num_gpus=2 --batch_size=128
python multi_gpu.py --num_gpus=2 --batch_size=1024
python multi_gpu.py --num_gpus=2 --batch_size=2048
python multi_gpu.py --num_gpus=2 --batch_size=4096