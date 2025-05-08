#!/bin/bash

function usage() {
    echo
    echo "Usage: $(basename $0) [options ...]"
    echo
    echo "Options:"
    echo " --maca_path <maca_path>"
    echo " --maca_version <maca_version>"
    echo " --maca_compiler_path <maca_path_for_mxcc>"
    echo " --force_pull. Default: 0"
    echo " --download_maca_dir <download_maca_dir>"
    echo " --download_maca_version <download_maca_version>"
    echo " --download_maca_compiler_version <download_maca_version_for_compiler>"
    echo " --platform <maca|cuda|rocm>. Default: maca"
    echo " --max_jobs <max_jobs>"
    echo " --enable_debug. Default: 0"
    echo " --enable_ninja. Default: 0"
    echo " --enable_clang. Default: 0"
    echo " --py_setup_cmd <install|bdist_wheel>. Default: install"
    echo " --conda_env_dst_name <conda_env_name_to_run>"
    echo " --conda_env_src_name <conda_env_name_to_clone>"
    echo " --conda_env_dst_python_version <conda_env_dst_python_version>"
    echo " --run_test= <''|checkin|daily|weekly>. Default: ''"
    echo " --skip_build. Default: 0"
    echo " --enable_coding_style_check. Default: 0"
    echo " --interactive. Default: 0"
    echo " -v|--verbose. Default: 0"
    echo " --remove_cache. Default: 0"
    echo " --use_slurm. Default: 0"
    echo " --send_mail. Default: 0"
    echo " --build_type <debug|release>. Higher priority than enable_debug. If not set, use enable_debug option"
    echo " --clean_conda_env_dst. Default: 0"
    echo " --clean_pytorch_build. Default: 0"
    echo " --dst_wheel_dir_path. Default: ''"
    echo " --src_wheel_dir_path. Default: ''"
    echo " --pytorch_enable_mkl_dynamic_lib. Default: 0"
    return 0
}

function err_msg() {
    echo "Error when parsing args. Abort"
}

function print_info() {
    echo -e "====== CONFIG ======"
    echo -e "\tmaca_path: ${maca_path}"
    echo -e "\tmaca_version: ${maca_version}"
    echo -e "\tmaca_compiler_path: ${maca_compiler_path}"
    echo -e "\tforce_pull: ${force_pull}"
    echo -e "\tdownload_maca_dir: ${download_maca_dir}"
    echo -e "\tdownload_maca_version: ${download_maca_version}"
    echo -e "\tdownload_maca_compiler_version: ${download_maca_compiler_version}"
    echo -e "\tplatform: ${platform}"
    echo -e "\tmax_jobs: ${max_jobs}"
    echo -e "\tenable_debug: ${enable_debug}"
    echo -e "\tenable_ninja: ${enable_ninja}"
    echo -e "\tenable_clang: ${enable_clang}"
    echo -e "\tpy_setup_cmd: ${py_setup_cmd}"
    echo -e "\tconda_env_dst_name: ${conda_env_dst_name}"
    echo -e "\tconda_env_src_name: ${conda_env_src_name}"
    echo -e "\tconda_env_dst_python_version: ${conda_env_dst_python_version}"
    echo -e "\trun_test: ${run_test}"
    echo -e "\tskip_build: ${skip_build}"
    echo -e "\tenable_coding_style_check: ${enable_coding_style_check}"
    echo -e "\tinteractive: ${interactive}"
    echo -e "\tverbose: ${enable_verbose}"
    echo -e "\tremove_cache: ${remove_cache}"
    echo -e "\tuse_slurm: ${use_slurm}"
    echo -e "\tsend_mail: ${send_mail}"
    echo -e "\tbuild_type: ${build_type}"
    echo -e "\tclean_conda_env_dst: ${clean_conda_env_dst}"
    echo -e "\tclean_conda_env_dst: ${clean_pytorch_build}"
    echo -e "\tdst_wheel_dir_path: ${dst_wheel_dir_path}"
    echo -e "\tsrc_wheel_dir_path: ${src_wheel_dir_path}"
    echo -e "\tpytorch_enable_mkl_dynamic_lib: ${pytorch_enable_mkl_dynamic_lib}"
    echo -e "====== CONFIG ======"
}

function download_maca_package() {
    rm -rf $1 && mkdir $1 && cd $1
    wget -O - http://172.161.13.22:9000/jenkins/MXC500/daily/c500.release.install.sh  | DOWNLOAD_VERSION=$1 bash -
    cd ..
}

function clone_and_enter_conda_env() {
    local conda_root="$(dirname $(which conda))/../"
    local conda=$(which conda)
    local dst_env_name=$1
    local src_env_name=$2
    ${conda} remove -n ${dst_env_name} --all -y
    ${conda} create -n ${dst_env_name} --clone ${src_env_name}
    source ${conda_root}/bin/activate ${dst_env_name}
}

function recreate_and_enter_conda_env() {
    local conda_root="$(dirname $(which conda))/../"
    local conda=$(which conda)
    local dst_env_name=$1
    local python_version=$2
    ${conda} remove -n ${dst_env_name} --all -y
    ${conda} create -n ${dst_env_name} python=${python_version} -y
    source ${conda_root}/bin/activate ${dst_env_name}
    which cmake
    cmake --version
    export PATH="${conda_root}/envs/${dst_env_name}/bin:${PATH}"
}

function enter_conda_env() {
    local conda_root="$(dirname $(which conda))/../"
    local dst_env_name=$1
    source ${conda_root}/bin/activate ${dst_env_name}
}

function leave_conda_env() {
    local conda_root="$(dirname $(which conda))/../"
    source ${conda_root}/bin/deactivate
}

function clean_conda_env() {
    local conda_root="$(dirname $(which conda))/../"
    local conda=$(which conda)
    local env_name=$1
    ${conda} remove -n ${env_name} --all -y
}

function main() {
    local maca_path=""
    local maca_version=""
    local maca_compiler_path=""
    local force_pull=0
    local download_maca_dir=""
    local download_maca_version=""
    local download_maca_compiler_version=""
    local platform="maca"
    local max_jobs=8
    local enable_debug=0
    local enable_ninja=0
    local enable_clang=0
    local py_setup_cmd="install"
    local conda_env_dst_name=""
    local conda_env_src_name=""
    local conda_env_dst_python_version=""
    local run_test=""
    local skip_build=0
    local enable_coding_style_check=0
    local interactive=0
    local enable_verbose=0
    local remove_cache=0
    local use_slurm=0
    local send_mail=0
    local build_type=""
    local clean_conda_env_dst=0
    local clean_pytorch_build=0
    local dst_wheel_dir_path=""
    local src_wheel_dir_path=""
    local pytorch_enable_mkl_dynamic_lib=0

    while [ -n "$1" ]; do
        case "$1" in
            -i)
                interactive=1
                shift 1
                ;;
            --maca_version)
                maca_version="$2"
                shift 2
                ;;
            --maca_path)
                maca_path="$2"
                shift 2
                ;;
            --maca_compiler_path)
                maca_compiler_path="$2"
                shift 2
                ;;
            --force_pull)
                force_pull=1
                shift
                ;;
            --download_maca_dir)
                download_maca_dir="$2"
                shift 2
                ;;
            --download_maca_version)
                download_maca_version="$2"
                shift 2
                ;;
            --download_maca_compiler_version)
                download_maca_compiler_version="$2"
                shift 2
                ;;
            --platform)
                platform="$2"
                shift 2
                ;;
            --max_jobs)
                max_jobs="$2"
                shift 2
                ;;
            --enable_debug)
                enable_debug=1
                shift
                ;;
            --enable_ninja)
                enable_ninja=1
                shift
                ;;
            --enable_clang)
                enable_clang=1
                shift
                ;;
            --py_setup_cmd)
                py_setup_cmd="$2"
                shift 2
                ;;
            --conda_env_dst_name)
                conda_env_dst_name="$2"
                shift 2
                ;;
            --conda_env_src_name)
                conda_env_src_name="$2"
                shift 2
                ;;
            --conda_env_dst_python_version)
                conda_env_dst_python_version="$2"
                shift 2
                ;;
            --run_test)
                run_test="$2"
                shift 2
                ;;
            --skip_build)
                skip_build=1
                shift 1
                ;;
            --enable_coding_style_check)
                enable_coding_style_check=1
                shift 1
                ;;
            -v|--verbose)
                enable_verbose=1
                shift 1
                ;;
            --remove_cache)
                remove_cache=1
                shift 1
                ;;
            --use_slurm)
                use_slurm=1
                shift 1
                ;;
            --send_mail)
                send_mail=1
                shift 1
                ;;
            --build_type)
                build_type="$2"
                shift 2
                ;;
            --clean_conda_env_dst)
                clean_conda_env_dst=1
                shift 1
                ;;
            --clean_pytorch_build)
                clean_pytorch_build=1
                shift 1
                ;;
            --dst_wheel_dir_path)
                dst_wheel_dir_path="$2"
                shift 2
                ;;
            --src_wheel_dir_path)
                src_wheel_dir_path="$2"
                shift 2
                ;;
            --pytorch_enable_mkl_dynamic_lib)
                pytorch_enable_mkl_dynamic_lib=1
                shift 1
                ;;
            -h|--help)
                usage
                return 0
                ;;
            *)
                err_msg
                usage
                return 1
                ;;
       esac
    done

    if [[ ${maca_version} != "" ]]; then
      export PYTORCH_MACA_VERSION=${maca_version}
    fi

    if [[ -e /.dockerenv ]]; then
        conda_env_dst_python_version=""
        clean_conda_env_dst=0
    fi

    # if [[ ${enable_verbose} != 0 ]]; then
    #     set -x
    # fi
    export PATH="${PATH}:${HOME}/anaconda3/condabin/"
    env

    pytorch_dir="$(cd $(dirname $0);pwd)/.."
    framework_dir="${pytorch_dir}/../"
    if [[ ${maca_path} == "" && ${force_pull} == 0 ]]; then
        echo "Neither maca_path nor force_pull is set. Use setting from maca_version_cmodel.txt instead"
        source ${pytorch_dir}/maca_tools/maca_version_cmodel.txt
        maca_path=${MACA_PATH}
    fi
    if [[ ${maca_path} != "" && ${MACA_PATH} == "" ]]; then
        echo "Set env MACA_PATH to ${maca_path}"
        export MACA_PATH=${maca_path}
    fi
    if [[ ${force_pull} == 1 && ( ${download_maca_dir} == "" || ${download_maca_version} == "" ) ]]; then
        echo "ERROR: If force_pull set, download_maca_dir and download_maca_version should be set."
        return 1
    fi

    if [[ ${download_maca_version} == "latest" || ${download_maca_compiler_version} == "latest" ]]; then
        latest_maca_version=$(wget -qO - http://172.161.13.22:9000/jenkins/MXC500/daily/ubuntu18.04/x86_64/mxc500_latest.txt)
        if [[ ${download_maca_version} == "latest" ]]; then
            download_maca_version=${latest_maca_version}
        fi
        if [[ ${download_maca_compiler_version} == "latest" ]]; then
            download_maca_compiler_version=${latest_maca_version}
        fi
    fi
    if [[ ${download_maca_compiler_version} == "" ]]; then
        download_maca_compiler_version=${download_maca_version}
    fi
    if [[ ${download_maca_dir} != "" && ${download_maca_dir} != /* ]]; then
        download_maca_dir=${pytorch_dir}/../${download_maca_dir}
    fi

    # modify enable_debug using build_type option
    if [[ ${build_type,,} == "debug" ]]; then
      enable_debug=1
    fi
    if [[ ${build_type,,} == "release" ]]; then
      enable_debug=0
    fi

    if [[ ${conda_env_dst_name} == "" && ${conda_env_src_name} != "" ]]; then
      conda_env_dst_name="${conda_env_src_name}-tmp$(date +%s%6N)"
    fi
    if [[ ${conda_env_dst_name} == "" && ${conda_env_src_name} == "" && ${conda_env_dst_python_version} != "" ]]; then
      conda_env_dst_name="mcpytorch_test-${conda_env_dst_python_version}-tmp$(date +%s%6N)"
      echo "Using conda dst env: ${conda_env_dst_name}"
    fi

    print_info
    if [[ ${interactive} == 1 ]]; then
        read -p $'Please verify above config and press [y|Y] to continue...\n' should_continue
        if [[ ${should_continue,,} != "y" ]]; then
            return 1
        fi
    fi


    if [[ ${use_slurm} == 1 ]]; then
        export PYTORCH_TEST_USE_SLRUM=1
    fi
    if [[ ${send_mail} == 1 ]]; then
        export PYTORCH_SEND_MAIL=1
    fi

    ### download maca package
    cd ${framework_dir}
    if [[ ${force_pull} == 1 ]]; then
        mkdir -p ${download_maca_dir} && cd ${download_maca_dir}
        download_maca_package ${download_maca_version}
        export PYTORCH_MACA_PATH="${download_maca_dir}/${download_maca_version}/maca-${download_maca_version}/"
        if [[ ${download_maca_version} != ${download_maca_compiler_version} ]]; then
            download_maca_package ${download_maca_compiler_version}
            export PYTORCH_MACA_COMPILER_PATH="${download_maca_dir}/${download_maca_compiler_version}/maca-${download_maca_compiler_version}/mxgpu_llvm/bin/"
        else
            export PYTORCH_MACA_COMPILER_PATH=${PYTORCH_MACA_PATH}/mxgpu_llvm/bin/
        fi
    else
        export PYTORCH_MACA_PATH=${maca_path}
        export PYTORCH_MACA_COMPILER_PATH=${maca_compiler_path}
    fi

    ### check environment
    if [[ ${conda_env_dst_name} != "" ]]; then
        export PYTHONNOUSERSITE=1
        if [[ $(which conda) == "" ]]; then
            echo "Conda environment must be installed."
            return 1
        fi
        if [[ ${conda_env_src_name} != "" && ${conda_env_dst_python_version} == "" ]]; then
            clone_and_enter_conda_env ${conda_env_dst_name} ${conda_env_src_name}
        elif [[ ${conda_env_src_name} == "" && ${conda_env_dst_python_version} == "" ]]; then
            enter_conda_env ${conda_env_dst_name}
        else  # // ${conda_env_dst_python_version} != ""
            recreate_and_enter_conda_env ${conda_env_dst_name} ${conda_env_dst_python_version}
        fi
    fi

    pip install -r ${pytorch_dir}/requirements.txt

    arch=$(uname -m)
    if [[ $arch != "aarch64" && $arch != "arm64" ]]; then
        pip install -r ${pytorch_dir}/requirements-noarm.txt
    fi

    ### check coding style
    # cd ${framework_dir}
    # if [[ ${enable_coding_style_check} != 0 ]]; then
    #     pip install -r ${pytorch_dir}/requirements-flake8.txt
    #     source ./pytorch/maca_tools/coding_style_check.sh
    #     coding_style_check
    #     if [[ $? != 0 ]]; then
    #         if [[ ${conda_env_dst_name} != "" ]]; then
    #             leave_conda_env
    #             if [[ ${clean_conda_env_dst} != 0 ]]; then
    #                 clean_conda_env ${conda_env_dst_name}
    #             fi
    #         fi
    #         return 1
    #     fi
    # fi

    ### remove cache
    cd ${pytorch_dir}
    if [[ ${remove_cache} != 0 ]]; then
        python setup.py clean
    fi

    ### build
    cd ${pytorch_dir}
    if [[ ${skip_build} == 0 ]]; then
        export TORCH_CUDA_ARCH_LIST="8.0"
        export BUILD_CAFFE2_OPS=0
        export BUILD_CAFFE2=0
        export USE_OPENMP=1
        export USE_NNPACK=0
        export USE_QNNPACK=0
        export USE_CCACHE=0
        export USE_SYSTEM_NCCL=1
        export CUDA_PATH=/home/bxiong/cu-bridge/CUDA_DIR
        if [[ ${enable_ninja} == 1 ]]; then
            export USE_NINJA=1
        else
            export USE_NINJA=0
        fi
        if [[ ${enable_clang} == 1 ]]; then
            export LD_LIBRARY_PATH="${CLANG_PATH}/lib:${LD_LIBRAY_PATH}"
            export CC="${CLANG_PATH}/bin/clang"
            export CXX="${CLANG_PATH}/bin/clang++"
        fi
        export USE_DISTRIBUTED=1
        export USE_MPI=0
        export USE_GLOO=1
        if [[ ${enable_debug} == 1 ]]; then
            export DEBUG=1
        fi
        if [[ ${max_jobs} != 0 ]]; then
            export MAX_JOBS=${max_jobs}
        fi
        if [[ ${pytorch_enable_mkl_dynamic_lib} == 1 ]]; then
            export PYTORCH_ENABLE_MKL_DYNAMIC_LIB=1
        fi
        if [[ ${platform} == "rocm" ]]; then
            python3 tools/amd_build/build_amd.py
        fi
        echo -e "PATH before: ${PATH}"
        source ./maca_tools/env/env_build.sh
        echo -e "PATH after: ${PATH}"

        python setup.py ${py_setup_cmd}
        ret=$?
        if [[ ${ret} != 0 ]]; then
            ## clean
            cd ${pytorch_dir}
            if [[ ${clean_pytorch_build} != 0 ]]; then
                python setup.py clean
            fi
            if [[ ${conda_env_dst_name} != "" ]]; then
                leave_conda_env
                if [[ ${clean_conda_env_dst} != 0 ]]; then
                    clean_conda_env ${conda_env_dst_name}
                fi
            fi
            return ${ret}
        fi
    fi

    ## run test
    cd ${pytorch_dir}
    test_ret=0
    if [[ ${run_test} != "" ]]; then
        if [[ ${PYTORCH_PRIVATE_WHEEL_PATH} != "" ]]; then
            pip install ${PYTORCH_PRIVATE_WHEEL_PATH}/flash_attn*.whl
            pip install ${PYTORCH_PRIVATE_WHEEL_PATH}/triton*.whl
        else
            pip install ${maca_path}/wheel/flash_attn*.whl
            pip install ${maca_path}/wheel/triton*.whl
        fi

        pip install -r ${pytorch_dir}/requirements-test.txt
        pip install -r ${pytorch_dir}/requirements-perf.txt
        if [[ ${py_setup_cmd} != "install" ]]; then
            pip install dist/torch-2.4.0*-cp38-cp38-*.whl
        fi
        source ./maca_tools/env/env_run_fast.sh
        ./maca_tools/run_tests.sh ${run_test}
        if [[ $? != 0 ]]; then
            test_ret=1
        fi
    fi

    ## copy result file
    cd ${pytorch_dir}/../
    if [[ ${py_setup_cmd} == "bdist_wheel" && ${dst_wheel_dir_path} != "" ]]; then
        mkdir -p ${dst_wheel_dir_path}
        cp -rf ${pytorch_dir}/dist/*  ${dst_wheel_dir_path}/
    fi

    ## clean
    cd ${pytorch_dir}
    if [[ ${clean_pytorch_build} != 0 ]]; then
        python setup.py clean
    fi
    if [[ ${conda_env_dst_name} != "" ]]; then
        leave_conda_env
        if [[ ${clean_conda_env_dst} != 0 ]]; then
            clean_conda_env ${conda_env_dst_name}
        fi
    fi

    return ${test_ret}
}

main "$@"
exit $?
