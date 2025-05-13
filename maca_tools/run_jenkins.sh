#!/bin/bash
set -e
set -x
set -o pipefail

function help() {
    echo "Usage: bash run_jenkins.sh [checkin|daily|weekly]"
    echo -e "\tBuild project and run tests."
}

function print_env_info() {
    env
    ifconfig
}

print_env_info
pytorch_root="$(cd $(dirname $0);pwd)/.."
wheel_path="${pytorch_root}/../wheel/"
cd ${pytorch_root}
rm -rf ${wheel_path}
# in jenkins env, add conda dir to path
export PATH="${PATH}:${HOME}/anaconda3/condabin/"
pytorch_daily_build_package=/netapp/pytorch/dailybuild/

function run_checkin_test() {
    export WCUDA_HOME=${HOME}/wcuda_dir/checkin/
    source ./maca_tools/maca_version_chip.txt
    bash ./maca_tools/build_and_run_impl.sh                      \
        --maca_path ${MACA_PATH}                            \
        --conda_env_dst_python_version "3.8.16"             \
        --remove_cache                                      \
        --clean_conda_env_dst                               \
        --py_setup_cmd "bdist_wheel"                        \
        --run_test "checkin"                                \
        --enable_coding_style_check                         \
        --dst_wheel_dir_path ${wheel_path}
    unset WCUDA_HOME
    return $?
}

function run_daily_test() {
    export WCUDA_HOME=${HOME}/wcuda_dir/nightly/
    source ./maca_tools/maca_version_chip.txt
    bash ./maca_tools/build_and_run_impl.sh                 \
        --maca_path ${MACA_PATH}                            \
        --conda_env_dst_python_version "3.8.16"             \
        --remove_cache                                      \
        --clean_conda_env_dst                               \
        --py_setup_cmd "bdist_wheel"                        \
        --run_test "daily"                                  \
        --dst_wheel_dir_path ${wheel_path}                  \
        --send_mail
    ret=$?
#    bash ./maca_tools/build_and_run_impl.sh                 \
#        --maca_path ${MACA_PATH}                            \
#        --conda_env_dst_python_version "3.8.16"             \
#        --remove_cache                                      \
#        --clean_conda_env_dst                               \
#        --py_setup_cmd "bdist_wheel"                        \
#        --build_type debug                                  \
#        --dst_wheel_dir_path ${pytorch_daily_build_package}/$(date +%m%d)/debug/ &> daily_debug_build.log
#    ret_debug=$?
#    export PYTORCH_KEEP_IR_BYTECODE=1
#    bash ./maca_tools/build_and_run_impl.sh                 \
#        --maca_path ${MACA_PATH}                            \
#        --conda_env_dst_python_version "3.8.16"             \
#        --remove_cache                                      \
#        --clean_conda_env_dst                               \
#        --py_setup_cmd "bdist_wheel"                        \
#        --dst_wheel_dir_path ${pytorch_daily_build_package}/$(date +%m%d)/release/ &> daily_release_build.log
#    ret_release=$?
#    if [[ ${ret_debug} == 0 ]]; then
#        echo "[RESULT] Debug build success"
#    else
#        echo "[RESULT] Debug build fail"
#    fi
#    if [[ ${ret_release} == 0 ]]; then
#        echo "[RESULT] Release build success"
#    else
#        echo "[RESULT] Release build fail"
#    fi
    unset WCUDA_HOME
    return $ret
}

function run_benchmark_test() {
    export WCUDA_HOME=${HOME}/wcuda_dir/nightly/
    source ./maca_tools/maca_version_chip.txt
    bash ./maca_tools/build_and_run_impl.sh                 \
        --maca_path ${MACA_PATH}                            \
        --conda_env_dst_python_version "3.8.16"             \
        --remove_cache                                      \
        --clean_conda_env_dst                               \
        --py_setup_cmd "bdist_wheel"                        \
        --run_test "benchmark"                              \
        --dst_wheel_dir_path ${wheel_path}                  \
        --send_mail
    ret=$?
    unset WCUDA_HOME
    return $ret
}

function run_benchmark_master_test() {
    export WCUDA_HOME=${HOME}/wcuda_dir/nightly/
    source ./maca_tools/maca_version_chip.txt
    bash ./maca_tools/build_and_run_impl.sh                 \
        --maca_path ${MACA_PATH}                            \
        --conda_env_dst_python_version "3.8.16"             \
        --skip_build                                        \
        --remove_cache                                      \
        --clean_conda_env_dst                               \
        --py_setup_cmd "bdist_wheel"                        \
        --run_test "benchmark_master"                       \
        --dst_wheel_dir_path ${wheel_path}                  \
        --send_mail
    ret=$?
    unset WCUDA_HOME
    return $ret
}

function run_benchmark_fw_test() {
    export WCUDA_HOME=${HOME}/wcuda_dir/nightly/
    source ./maca_tools/maca_version_chip.txt
    bash ./maca_tools/build_and_run_impl.sh                 \
        --maca_path ${MACA_PATH}                            \
        --conda_env_src_name "pytorch"             \
        --skip_build                                        \
        --remove_cache                                      \
        --clean_conda_env_dst                               \
        --py_setup_cmd "bdist_wheel"                        \
        --run_test "benchmark_fw"                       \
        --dst_wheel_dir_path ${wheel_path}                  \
        --send_mail
    ret=$?
    unset WCUDA_HOME
    return $ret
}

function run_weekly_test() {
    export WCUDA_HOME=${HOME}/wcuda_dir/weekly/
    source ./maca_tools/maca_version_chip.txt
    ./maca_tools/build_and_run_impl.sh                        \
        --maca_path ${MACA_PATH}                              \
        --conda_env_dst_python_version "3.8.16"               \
        --remove_cache                                        \
        --clean_conda_env_dst                                 \
        --py_setup_cmd "bdist_wheel"                          \
        --run_test "weekly"                                   \
        --dst_wheel_dir_path ${wheel_path}                    \
        --send_mail
    unset WCUDA_HOME
    return $?
}

case $1 in
    checkin)
        run_checkin_test
        exit $?
        ;;
    daily)
        run_daily_test
        exit $?
        ;;
    benchmark)
        run_benchmark_test
        exit $?
        ;;
    benchmark_master)
        run_benchmark_master_test
        exit $?
        ;;
    benchmark_fw)
        run_benchmark_fw_test
        exit $?
        ;;
    weekly)
        run_weekly_test
        exit $?
        ;;
    -h | --help | *)
        help
        exit 1
        ;;
esac
