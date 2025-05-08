#!/bin/bash
set -e

function launch_and_check_slurm_jobs() {
    local wrapper_script=$1
    local cmd_prefix=$2
    IFS=\;; local cmd_args_lists=($3)
    local slurm_tests_name=$4
    local job_query_duration=60
    all_succeed=0
    declare -A job_id_name_dict=()
    failed_job=()
    local pytest=$5
    local xml_path=$6
    ERR=0
    PASS=0
    res=""
    declare -A job_start_time_dict=()

    for file in ${cmd_args_lists[@]}; do
        if [[ (${pytest} == 0) && (${xml_path} != '') ]]; then
            job_info=$(sbatch --job-name=${file}-[$(date +%s%6N)] \
                ${wrapper_script} \
                ${cmd_prefix}${file} --gtest_output=xml:"${xml_path}/${file}.xml")
        else
            job_info=$(sbatch --job-name=${file}-[$(date +%s%6N)] \
                ${wrapper_script} \
                ${cmd_prefix}${file})
        fi
        job_id=$(echo ${job_info} | awk '{print $4}')
        job_id_name_dict[${job_id}]=${file}
        job_start_time_dict[${job_id}]=$(date +%s)
    done

    while [[ ${#job_id_name_dict[@]} != 0 ]]; do 
        for key in ${!job_id_name_dict[*]}; do
            job_name=${job_id_name_dict[${key}]}
            sacct_res=$(sacct -j ${key} -P -b | tail -n 1)
            status=$(echo ${sacct_res} | awk -F '|' '{ print $2 }')
            testEndTime=$(date +%s)
            testDurTime=$((${testEndTime} - ${job_start_time_dict[${key}]}))
            if [[ ${status} == "COMPLETED" ]]; then
                PASS=$(expr ${PASS} + 1)
                res+="${job_name},pass,${testDurTime}#"
                unset job_id_name_dict[${key}]
                echo "job_id: ${key}, job_name: ${job_name} succeed."
            elif [[ ${status} == "FAILED" || ${status} == "CANCELLED" || \
                    ${status} == "TIMEOUT" || ${status} == "RESIZING" || \
                    ${status} == "DEADLINE" || ${status} == "NODE_FAIL" ]]; then
                ERR=$(expr ${ERR} + 1)
                res+="${job_name},fail,${testDurTime}#"
                all_succeed=1
                unset job_id_name_dict[${key}]
                echo "job_id: ${key}, job_name: ${job_name} failed or cancelled."
                failed_job[${#failed_job[@]}]="job_id: ${job_id}, job_name: ${job_name}"
            else
                echo -e "job_id: ${key}, job_name: ${job_name} not yet finished: \n${sacct_res}"
                break
            fi
        done
        if [[ ${#job_id_name_dict[@]} == 0 ]]; then
            break
        fi
        sleep ${job_query_duration}
    done 

    echo "****** Failed(or Cancelled) slurm tests [${slurm_tests_name}] summary: ******"
    for((i=0;i<${#failed_job[@]};i++)); do
        echo ${failed_job[$i]}
    done

    if [[ (${pytest} == 1) && (${xml_path} != '') ]]; then
        PYTORCH_ROOT="$(cd $(dirname $0);pwd)/.."
        xml_date=`date +%Y%m%d-%H-%M-%S`
        pytest_xml=${xml_path}/${xml_date}-pytest.xml
        source ${PYTORCH_ROOT}/maca_tools/utils.sh
        IFS=$'\n'
        generate_xml ${pytest_xml} ${res// /_} $PASS $ERR
    fi

    return ${all_succeed}
}
