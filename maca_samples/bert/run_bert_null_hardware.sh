#!/bin/bash
# set -x

cd $(dirname $(realpath $0))
echo "****************************set environment variables***************************"
export USE_NULL_HARDWARE=ON

if [[ $1 == "" ]]; then
    echo "Error: Please input loop number."
    exit -1
fi

echo "****************************install required packages***************************"
pip install -r requirements.txt

echo "*************************save mode related modification************************"

config_file_path="configs/cfg_batch_1_seqlength_128_for_sample.py"
config_file_path_copy="configs/cfg_batch_1_seqlength_128_for_sample_auto_generate.py"
echo "******************copy $config_file_path to $config_file_path_copy**************"
cp $config_file_path $config_file_path_copy

src_device="device = torch.device(\"cuda:0\")"
dst_device="device = torch.device(\"cpu:0\")"
echo "===== replace $src_device -----> with $dst_device"
sed -i "s/$src_device/$dst_device/" $config_file_path_copy

src_hidden_dropout_prob="hidden_dropout_prob = 0.1"
dst_hidden_dropout_prob="hidden_dropout_prob = 0.0"
echo "===== replace $src_hidden_dropout_prob -----> with $dst_hidden_dropout_prob"
sed -i "s/$src_hidden_dropout_prob/$dst_hidden_dropout_prob/" $config_file_path_copy

src_attention_probs_dropout_prob="attention_probs_dropout_prob = 0.1"
dst_attention_probs_dropout_prob="attention_probs_dropout_prob = 0.0"
echo "===== replace $src_attention_probs_dropout_prob -----> with $dst_attention_probs_dropout_prob"
sed -i "s/$src_attention_probs_dropout_prob/$dst_attention_probs_dropout_prob/" $config_file_path_copy

golden_path=${PYTORCH_TEST_GOLDEN_PATH}
if [ -z “$golden_path” ];then
    echo “The environment variable MY_PATH is not set.”
fi
src_root_path="golden_root_path = \"/netapp/pytorch/golden/bert/\""
if [ -d “$golden_path” ];then
    src_root_path="golden_root_path = \"$golden_path\""
fi
golden_dir=$(cd "$(dirname "$0")";pwd)/golden_data_tmp/`date +%Y%m%d-%H:%M:%S`
dst_root_path="golden_root_path = \"$golden_dir\""
echo "===== replace $src_root_path -----> with $dst_root_path"
sed -i "s?$src_root_path?$dst_root_path?" $config_file_path_copy

echo "===== mkdir $golden_dir"
mkdir -p $golden_dir/

src_weight_path="pretrained_weight_path = os.path.join(golden_root_path, \"weights/pytorch_model.bin\")"
dst_weight_path="pretrained_weight_path = None"
echo "===== replace $src_weight_path -----> with $dst_weight_path"
sed -i "s@$src_weight_path@$dst_weight_path@" $config_file_path_copy

python run_bert.py --config_file_path $config_file_path_copy --mode save --null_hardware

echo "*************************verify mode related modification***********************"
src_device="device = torch.device(\"cpu:0\")"
dst_device="device = torch.device(\"cuda:0\")"
echo "===== replace $src_device -----> with $dst_device"
sed -i "s/$src_device/$dst_device/" $config_file_path_copy
src_backward_flag="run_backward_flag = False"
dst_backward_flag="run_backward_flag = True"
echo "===== replace $src_backward_flag -----> with $dst_backward_flag"
sed -i "s/$src_backward_flag/$dst_backward_flag/" $config_file_path_copy

python run_bert.py --config_file_path $config_file_path_copy --mode verify --null_hardware --loop_num $1

if [[ $? != 0 ]];then
    echo "Error: run fail"
    exit 1
else
    echo "run success"
fi
