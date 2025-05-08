#!/bin/bash

cur_dir=$(cd $(dirname $0); pwd)
ws_dir=${cur_dir}/../
source ${cur_dir}/datasets_dir.config

python ${ws_dir}/launch.py --model resnet50 --precision AMP --mode convergence --platform DGXA100 ${datasets_dir} --batch-size 256 --memory-format nhwc --epochs 1 --prof 20 --no-checkpoints --mixup 0.0 --seed 0 --enable_profiler
rm -rf ${cur_dir}/experiment_raport.json