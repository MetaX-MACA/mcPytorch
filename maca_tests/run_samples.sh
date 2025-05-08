startTime=`date +%Y%m%d-%H:%M:%S`
startTime_s=`date +%s`

PYTORCH_ROOT="$(cd $(dirname $0);pwd)/.."
ERR=0
PASS=0
res=""
err_files=()

sample_tests=(
    # "cpp_extension_cuda_demo"
    "libtorch_demo"
)

# run cpp extension and libtorch test
for file in "${sample_tests[@]}";do
    shell_name="run_${file}.sh"
    pushd ${PYTORCH_ROOT}/maca_samples/${file}
    testStartTime=$(date +%s)
    bash ${shell_name}
    ret=$?
    popd

    if [[ ${ret} != 0 ]]; then
        res+="${file},fail,$(($(date +%s) - ${testStartTime}))#"
        ERR=$(expr ${ERR} + 1)
        err_files[${#err_files[*]}]=${file}
    else
        PASS=$(expr ${PASS} + 1)
        res+="${file},pass,$(($(date +%s) - ${testStartTime}))#"
    fi
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
    xml_path=$2/${xml_date}-samples.xml
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