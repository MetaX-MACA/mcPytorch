#!/usr/bin/env bash

# Copyright (c) 2019-2020 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# echo "Container nvidia build = " $NVIDIA_BUILD_ID

cur_path=`pwd`
init_checkpoint=${1:-"${cur_path}/../bert_large_pretrained_amp.pt"}
epochs=${2:-"1.0"}
batch_size=${3:-"24"}
learning_rate=${4:-"3e-5"}
warmup_proportion=${5:-"0.1"}
precision=${6:-"fp16"}
num_gpu=${7:-"1"}
seed=${8:-"1"}
squad_dir=${9:-"${cur_path}/../squad_data/"}
vocab_file=${10:-"${cur_path}/vocab/vocab"}
OUT_DIR=${11:-"${cur_path}/workspace/bert/results/SQuAD"}
mode=${12:-"train"}
CONFIG_FILE=${13:-"${cur_path}/bert_configs/large.json"}
max_steps=${14:-"-1"}

echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16=" --fp16 "
fi

if [ "$num_gpu" = "1" ] ; then
  export CUDA_VISIBLE_DEVICES=0
  mpi_command=""
else
  unset CUDA_VISIBLE_DEVICES
  mpi_command=" -m torch.distributed.launch --nproc_per_node=$num_gpu"
fi

CMD="python  $mpi_command run_squad_profile2.py "
CMD+="--init_checkpoint=$init_checkpoint "
if [ "$mode" = "train" ] ; then
  CMD+="--do_train "
  CMD+="--train_file=$squad_dir/train-v1.1.json "
  CMD+="--train_batch_size=$batch_size "
elif [ "$mode" = "eval" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
  CMD+="--eval_script=$squad_dir/evaluate-v1.1.py "
  CMD+="--do_eval "
elif [ "$mode" = "prediction" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
else
  CMD+=" --do_train "
  CMD+=" --train_file=$squad_dir/train-v1.1.json "
  CMD+=" --train_batch_size=$batch_size "
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
  CMD+="--eval_script=$squad_dir/evaluate-v1.1.py "
  CMD+="--do_eval "
fi

CMD+=" --do_lower_case "
CMD+=" --bert_model=bert-large-uncased "
CMD+=" --learning_rate=$learning_rate "
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --seed=$seed "
CMD+=" --num_train_epochs=$epochs "
CMD+=" --max_seq_length=384 "
CMD+=" --doc_stride=128 "
CMD+=" --output_dir=$OUT_DIR "
CMD+=" --vocab_file=$vocab_file "
CMD+=" --config_file=$CONFIG_FILE "
CMD+=" --max_steps=$max_steps "
CMD+=" $use_fp16"

LOGFILE=$OUT_DIR/logfile.txt
echo "$CMD |& tee $LOGFILE"
time $CMD |& tee $LOGFILE
