#!/bin/bash

cur_dir=$(cd $(dirname $0); pwd)
ws_dir=${cur_dir}/../
source ${cur_dir}/spd_datasets_dir.config

python ${ws_dir}/launch.py --model resnet50 --precision AMP --mode convergence --platform DGXA100 ${datasets_dir} --batch-size 512 --memory-format nhwc --epochs 1 --prof 100 --no-checkpoints --mixup 0.0 --seed 0 --enable_perf
rm -rf ${cur_dir}/experiment_raport.json