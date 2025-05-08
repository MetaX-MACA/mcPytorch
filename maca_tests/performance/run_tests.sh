#!/bin/bash

export PYTORCH_DISABLE_ACTIVITY_PROFILER_INFO=1
cd $(dirname $0)

# $1: file name
# $2: platform
# $3: save log directory

PLATFORM="c500_chip"
if [ -n "$2" ]; then
    if [[ $2 == "C500" || $2 == "c500_chip" ]]; then
        PLATFORM="c500_chip"
    elif [[ $2 == "A100" || $2 == "a100" ]]; then
        PLATFORM="a100"
    else
        echo "invalid hardware platform!" >&2
        exit 1
    fi
fi

LOG_DIR="./"
if [ -n "$3" ]; then
    LOG_DIR="$3"
fi

if [ -d "./perf_json" ]; then
    rm -rf ./perf_json
fi

pushd ./tests
if [ -n "$1" ] && [ "$1" = "all" ]; then
    for file in *.py
    do
        filename=$(basename "$file")
        if [[ "$filename" == "resnet50_fw.py" ]]; then
            continue
        fi
        echo "[PyTorch Perf Start] ${file}..."
        python "$file" -platform "$PLATFORM" -branch "dev" -path "../perf_json"
        echo "[PyTorch Perf End] ${file}..."
        if [[ $? != 0 ]];then
            echo "Error in ${file}."
            exit 1
        fi
    done
else
    python "$1" -platform "$PLATFORM" -branch "dev" -path "../perf_json"
    if [[ $? != 0 ]];then
        echo "Error in ${file}."
        exit 1
    fi
fi
popd

echo "[PyTorch Perf Collect Data Start] ..."
pushd ./perf_json
rm -rf ./total_benckmark.json
python ../merge_json.py
# read database
python ../tests/utils/query_database.py -name "total_benckmark.json"
if [[ $? != 0 ]]; then
    echo "Error when reading database and processing."
    exit 1
fi
# update database and generate log
python ../performance_analyzer/analyzer.py "total_benckmark.json" --save_log "$LOG_DIR"
echo "[PyTorch Perf Collect Data End]"
if [[ $? != 0 ]]; then
    echo "Error when analyzing perf data."
    exit 1
fi
popd

exit 0
