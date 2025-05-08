#!/bin/bash
cd $(dirname $0)
startTime=`date +%Y%m%d-%H:%M:%S`
startTime_s=`date +%s`

if [[ $1 == "daily" || $1 == "device" ]]; then
  kernel_test_py+=("../maca_samples/bert/run_bert.py --config_file_path ../maca_samples/bert/configs_daily/cfg_batch_4_seqlength_128_daily.py --mode fast_verify")

  kernel_test_py+=("../maca_samples/resnet50/run_resnet50.py --batch_size 1")

  kernel_test_py+=("../maca_samples/dlrm/run_dlrm.py --batch_size 1")
  kernel_test_py+=("../maca_samples/dlrm/run_dlrm.py --batch_size 2")
  kernel_test_py+=("../maca_samples/dlrm/run_dlrm.py --batch_size 4")

  if [[ $1 == "daily" ]]; then
    kernel_test_py+=("./test_ddp_with_gloo_mlp_mp_spawn.py")
    kernel_test_py+=("./test_ddp_with_mccl_mlp_mp_spawn.py")
  fi
fi

if [[ $1 == "weekly" || $1 == "device" ]]; then
    kernel_test_py+=("../maca_samples/bert/run_bert.py --config_file_path ../maca_samples/bert/configs_daily/cfg_batch_1_seqlength_128_daily.py --mode verify")
    kernel_test_py+=("../maca_samples/bert/run_bert.py --config_file_path ../maca_samples/bert/configs_daily/cfg_batch_2_seqlength_128_daily.py --mode verify")
    kernel_test_py+=("../maca_samples/bert/run_bert.py --config_file_path ../maca_samples/bert/configs_daily/cfg_batch_1_seqlength_256_daily.py --mode fast_verify")
    kernel_test_py+=("../maca_samples/bert/run_bert.py --config_file_path ../maca_samples/bert/configs_daily/cfg_batch_2_seqlength_256_daily.py --mode fast_verify")
    kernel_test_py+=("../maca_samples/bert/run_bert.py --config_file_path ../maca_samples/bert/configs_daily/cfg_batch_4_seqlength_256_daily.py --mode fast_verify")
    kernel_test_py+=("../maca_samples/bert/run_bert.py --config_file_path ../maca_samples/bert/configs_daily/cfg_batch_1_seqlength_512_daily.py --mode fast_verify")
    kernel_test_py+=("../maca_samples/bert/run_bert.py --config_file_path ../maca_samples/bert/configs_daily/cfg_batch_2_seqlength_512_daily.py --mode fast_verify")
    kernel_test_py+=("../maca_samples/bert/run_bert.py --config_file_path ../maca_samples/bert/configs_daily/cfg_batch_4_seqlength_512_daily.py --mode fast_verify")

    kernel_test_py+=("../maca_samples/resnet50/run_resnet50.py --batch_size 2")
    kernel_test_py+=("../maca_samples/resnet50/run_resnet50.py --batch_size 4")

    pip install -r ../maca_samples/yolov5/requirements.txt
    kernel_test_py+=("../maca_samples/yolov5/train.py --batch_size 1 --mode ci")
    kernel_test_py+=("../maca_samples/yolov5/train.py --batch_size 2 --mode ci")
    pip install -r ../maca_samples/amp/requirements.txt
    kernel_test_py+=("../maca_samples/amp/bert_amp.py --epochs 1 --offline --data_path ../maca_samples/amp/data/")
    kernel_test_py+=("./resnet50/test_resnet50_ops.py --op_type conv2d --half")
    kernel_test_py+=("./test_mcdnn_conv.py")
    kernel_test_py+=("./test_attention_total.py")
    if [[ $1 == "weekly" ]]; then
      kernel_test_py+=("./compile/bert/test_bert_bertintermediate.py")
      kernel_test_py+=("./compile/bert/test_bert_bertoutput.py")
      kernel_test_py+=("./compile/bert/test_bert_bertpooler.py")
      kernel_test_py+=("./compile/bert/test_bert_bertselfattention.py")
      kernel_test_py+=("./compile/bert/test_bert_bertselfoutput.py")
      kernel_test_py+=("./compile/bert/test_bert_embeddings.py")
    fi
fi

source ../maca_tools/env/env_run_fast.sh
PYTORCH_ROOT="$(cd $(dirname $0);pwd)/.."

ERR=0
PASS=0
err_files=()
res=""
for file in "${kernel_test_py[@]}";do
    echo "Start test $file"
    testStartTime=$(date +%s)
    python $file
    if [[ $? != 0 ]];then
        res+="${file},fail,$(($(date +%s) - ${testStartTime}))#"
        ERR=$(expr ${ERR} + 1)
        err_files[${#err_files[*]}]=${file}
        echo "Error in tests of ${file}."
    else
        PASS=$(expr ${PASS} + 1)
        res+="${file},pass,$(($(date +%s) - ${testStartTime}))#"
        echo "Success in tests of ${file}."
    fi
    echo "End test $file"
done
endTime=`date +%Y%m%d-%H:%M:%S`
endTime_s=`date +%s`
sumTime=$[ $endTime_s-$startTime_s ]
timeMinu=$[ $sumTime / 60 ]
echo "===== $startTime -----> $endTime Total run $timeMinu minutes"

if [[ $2 != "" ]]; then
    PYTORCH_ROOT="$(cd $(dirname $0);pwd)/.."
    source ${PYTORCH_ROOT}/maca_tools/utils.sh
    xml_date=`date +%Y%m%d-%H-%M-%S`
    xml_path=$2/${xml_date}-pytest.xml
    generate_xml ${xml_path} ${res// /_} $PASS $ERR
fi

if [[ ${ERR} != 0 ]];then
    echo "*****Below test failed."
    for((i=0;i<${#err_files[@]};i++)); do
        echo ${err_files[$i]}
    done
    exit 1
else
    exit 0
fi

