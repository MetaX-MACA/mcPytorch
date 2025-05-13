#!/bin/bash

cur_dir=$(cd $(dirname $0); pwd)
ws_dir=${cur_dir}/../
result_dir=${ws_dir}/result_resnet50/result_C500_resnet50_AMP_8C_nchw_512/
rm -rf ${result_dir}
mkdir -p ${result_dir}
source ${cur_dir}/datasets_dir.config

python ${ws_dir}/multiproc.py --nproc_per_node 8 ${ws_dir}/launch.py --model resnet50 --precision AMP --mode convergence --platform DGXA100 ${datasets_dir} --batch-size 512 --memory-format nchw --epochs 50 --mixup 0.0 --seed 0 --workspace ${result_dir} --raport-file raport.json
