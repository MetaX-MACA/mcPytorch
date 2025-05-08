#!/bin/bash
echo "------ run pytorch tests ------"

export PYTORCH_TEST_INTERNAL=1
export MXC_PRIVATE_MEMORY=4096

export SEND_MAIL_SUBJECT="pytorch2.4 tests ${1}"
export PYTORCH_TEST_GOLDEN_PATH="/netapp/pytorch/golden/"
names=$(cat mail_names.txt)
if [[ $1 == "daily" ]]; then
    export SEND_MAIL_TO=$names
else
    export SEND_MAIL_TO=$names
fi

PYTORCH_ROOT="$(cd $(dirname $0);pwd)/.."

if [ $1 == "checkin" ]; then
    XML_PATH=${PYTORCH_ROOT}/test_report/xmls
    rm -rf ${XML_PATH}
    source ${PYTORCH_ROOT}/maca_tools/utils.sh
    make_xmls_dir ${XML_PATH}

    bash ${PYTORCH_ROOT}/maca_tests/run_c10_test.sh
    if [[ $? != 0 ]]; then
        echo "Error found in c10 tests."
        exit 1
    fi

    bash ${PYTORCH_ROOT}/maca_tests/run_aten_test.sh
    if [[ $? != 0 ]]; then
        echo "Error found in aten tests."
        exit 1
    fi

    bash ${PYTORCH_ROOT}/maca_tests/run_pytest.sh $1
    if [[ $? != 0 ]]; then
        echo "Error found in python tests."
        exit 1
    fi

    bash ${PYTORCH_ROOT}/maca_tests/run_pynet.sh $1
    if [[ $? != 0 ]]; then
        echo "Error found in python tests."
        exit 1
    fi


    bash ${PYTORCH_ROOT}/maca_tests/run_libtorch_cuda_size_test.sh
    if [[ $? != 0 ]]; then
        echo "Error found in libtorch_cuda size check."
        exit 1
    fi

    # use cmodel env to do multi-card tests
    # source ${PYTORCH_ROOT}/maca_tools/maca_version_cmodel.txt
    # source ${PYTORCH_ROOT}/maca_tools/env/env_run_fast.sh
    # bash ${PYTORCH_ROOT}/test/run_maca_pytest.sh mccl_and_gloo_checkin ${XML_PATH}
    # if [[ $? != 0 ]]; then
    #     echo "Error found in mccl_and_gloo_checkin tests."
    #     exit 1
    # fi
fi

if [[ $1 == "daily" ]]; then
    ERR=0
    XML_PATH=${PYTORCH_ROOT}/test_report/xmls
    rm -rf ${XML_PATH}
    source ${PYTORCH_ROOT}/maca_tools/utils.sh
    make_xmls_dir ${XML_PATH}

    bash ${PYTORCH_ROOT}/maca_tests/run_c10_test.sh ${XML_PATH}
    if [[ $? != 0 ]]; then
        echo "Error found in c10_test."
        ERR=$(expr ${ERR} + 1)
    fi
    bash ${PYTORCH_ROOT}/maca_tests/run_aten_test.sh ${XML_PATH}
    if [[ $? != 0 ]]; then
        echo "Error found in aten_test."
        ERR=$(expr ${ERR} + 1)
    fi
    bash ${PYTORCH_ROOT}/maca_tests/run_pytest.sh $1 ${XML_PATH}
    if [[ $? != 0 ]]; then
        ERR=$(expr ${ERR} + 1)
        echo "Error found in pytest."
    fi
    bash ${PYTORCH_ROOT}/maca_tests/run_pynet.sh $1 ${XML_PATH}
    if [[ $? != 0 ]]; then
        echo "Error found in python tests."
        ERR=$(expr ${ERR} + 1)
    fi
    bash ${PYTORCH_ROOT}/maca_tests/run_samples.sh $1 ${XML_PATH}
    if [[ $? != 0 ]]; then
        echo "Error found in maca samples tests."
        ERR=$(expr ${ERR} + 1)
    fi
    bash ${PYTORCH_ROOT}/test/run_runtest_nightly.sh $1
    if [[ $? != 0 ]]; then
        echo "Error found in runtest nightly"
        ERR=$(expr ${ERR} + 1)
    fi
    # export PYTORCH_SKIP_COMPILE_CHECK=1
    # bash ${PYTORCH_ROOT}/test/run_maca_pytest.sh dynamo ${XML_PATH}
    # if [[ $? != 0 ]]; then
    #     echo "Error found in dynamo tests."
    #     ERR=$(expr ${ERR} + 1)
    # fi
    # bash ${PYTORCH_ROOT}/test/run_maca_pytest.sh inductor ${XML_PATH}
    # if [[ $? != 0 ]]; then
    #     echo "Error found in inductor tests."
    #     ERR=$(expr ${ERR} + 1)
    # fi
    # unset PYTORCH_SKIP_COMPILE_CHECK

    if [[ ${PYTORCH_SEND_MAIL} == 1 ]]; then
        send_html_report ${XML_PATH} || true
    fi

    if [[ ${ERR} != 0 ]]; then
        echo "------------------Error found in pytorch tests ${1}------------------"
        exit 1
    fi
fi

if [[ $1 == "benchmark" ]]; then
    export RUN_BENCHMARK_CI=1
    pip install /home/jenkins_sw_bot/ws_pytorch/tools/DLLogger-1.0.0-py3-none-any.whl
    pip install /home/jenkins_sw_bot/ws_pytorch/tools/nvidia_dali_cuda110-1.27.0-8625303-py3-none-manylinux2014_x86_64.whl
    pip install pynvml
    pip install ${STORAGE_PATH}/${MACA_VERSION}/${MACA_VERSION}/wheel/torchvision-*.whl
    pip install tqdm
    pip install boto3
    pip install requests
    pip install six
    pip install pandas
    ERR=0
    bash ${PYTORCH_ROOT}/maca_tests/performance/run_tests.sh all
    if [[ $? != 0 ]]; then
        echo "Error found in perf test."
        ERR=$(expr ${ERR} + 1)
    fi

    if [[ ${ERR} != 0 ]];then
        echo "------------------Error found in pytorch tests ${1}------------------"
        exit 1
    fi

    bash ${PYTORCH_ROOT}/maca_tests/performance/spd_db/run_spd.sh
    if [[ $? != 0 ]];then
        echo " Failed run spd case_upload.py"
        exit 1
    fi
fi

if [[ $1 == "benchmark_master" ]]; then
    export RUN_BENCHMARK_CI=1
    pip install /home/jenkins_sw_bot/ws_pytorch/tools/DLLogger-1.0.0-py3-none-any.whl
    pip install /home/jenkins_sw_bot/ws_pytorch/tools/nvidia_dali_cuda110-1.27.0-8625303-py3-none-manylinux2014_x86_64.whl
    pip install pynvml
    pip install ${STORAGE_PATH}/${MACA_VERSION}/${MACA_VERSION}/wheel/torch-*.whl
    pip install ${STORAGE_PATH}/${MACA_VERSION}/${MACA_VERSION}/wheel/torchvision-*.whl
    pip install tqdm
    pip install boto3
    pip install requests
    pip install six
    pip install pandas
    ERR=0
    bash ${PYTORCH_ROOT}/maca_tests/performance/run_tests_master.sh all
    if [[ $? != 0 ]]; then
        echo "Error found in perf test."
        ERR=$(expr ${ERR} + 1)
    fi

    if [[ ${ERR} != 0 ]];then
        echo "------------------Error found in pytorch tests ${1}------------------"
        exit 1
    fi

    bash ${PYTORCH_ROOT}/maca_tests/performance/spd_db/run_spd_master.sh
    if [[ $? != 0 ]];then
        echo " Failed run spd master case_upload.py"
        exit 1
    fi
fi

if [[ $1 == "benchmark_fw" ]]; then
    export RUN_BENCHMARK_CI=1
    read -r first_line < "/opt/maca/Version.txt"
    export MACA_VERSION=${first_line:8}
    pip install /home/jenkins/workspace/tools/DLLogger-1.0.0-py3-none-any.whl
    pip install /home/jenkins/workspace/tools/nvidia_dali_cuda110-1.27.0-8625303-py3-none-manylinux2014_x86_64.whl
    pip install pynvml
    pip install /home/jenkins/testline/maca-mxc500-package/wheel/torch-*.whl
    pip install /home/jenkins/testline/maca-mxc500-package/wheel/torchvision-*.whl
    pip install tqdm
    pip install boto3
    pip install requests
    pip install six
    pip install pandas
    ERR=0
    bash ${PYTORCH_ROOT}/maca_tests/performance/run_tests_fw.sh all
    if [[ $? != 0 ]]; then
        echo "Error found in perf test."
        ERR=$(expr ${ERR} + 1)
    fi

    if [[ ${ERR} != 0 ]];then
        echo "------------------Error found in pytorch tests ${1}------------------"
        exit 1
    fi

    bash ${PYTORCH_ROOT}/maca_tests/performance/spd_db/run_spd_fw.sh
    if [[ $? != 0 ]];then
        echo " Failed run spd master case_upload.py"
        exit 1
    fi
fi

if [[ $1 == "weekly" ]]; then
    ERR=0
    XML_PATH=${PYTORCH_ROOT}/test_report/xmls
    rm -rf ${XML_PATH}
    source ${PYTORCH_ROOT}/maca_tools/utils.sh
    make_xmls_dir ${XML_PATH}

    pytest -n 1 -v ${PYTORCH_ROOT}/test/test_nn.py
    if [[ $? != 0 ]]; then
       echo "Error found in test_nn.py"
       ERR=$(expr ${ERR} + 1)
    fi
    pytest -n 1 -v ${PYTORCH_ROOT}/test/test_torch.py
    if [[ $? != 0 ]]; then
       echo "Error found in test_torch.py"
       ERR=$(expr ${ERR} + 1)
    fi
    pytest -n 1 -v ${PYTORCH_ROOT}/test/test_optim.py
    if [[ $? != 0 ]]; then
       echo "Error found in test_optim.py"
       ERR=$(expr ${ERR} + 1)
    fi
    pytest -n 1 -v ${PYTORCH_ROOT}/test/test_reductions.py
    if [[ $? != 0 ]]; then
       echo "Error found in test_reductions.py"
       ERR=$(expr ${ERR} + 1)
    fi
    pytest -n 1 -v ${PYTORCH_ROOT}/test/onnx/test_onnx_opset.py
    if [[ $? != 0 ]]; then
       echo "Error found in test_onnx_opset.py"
       ERR=$(expr ${ERR} + 1)
    fi
    pytest -n 1 -v ${PYTORCH_ROOT}/test/test_spectral_ops.py
    if [[ $? != 0 ]]; then
       echo "Error found in test_spectral_ops.py"
       ERR=$(expr ${ERR} + 1)
    fi
    pytest -n 1 -v ${PYTORCH_ROOT}/test/test_ops.py
    if [[ $? != 0 ]]; then
       echo "Error found in test_ops.py"
       ERR=$(expr ${ERR} + 1)
    fi
    python ${PYTORCH_ROOT}/test/distribued/test_c10d_nccl.py
    if [[ $? != 0 ]]; then
       echo "Error found in test_c10d_nccl.py"
       ERR=$(expr ${ERR} + 1)
    fi
    python ${PYTORCH_ROOT}/test/distribued/test_c10d_gloo.py
    if [[ $? != 0 ]]; then
       echo "Error found in test_c10d_gloo.py"
       ERR=$(expr ${ERR} + 1)
    fi

    bash ${PYTORCH_ROOT}/maca_tests/run_pytest.sh $1 ${XML_PATH}
    if [[ $? != 0 ]]; then
        ERR=$(expr ${ERR} + 1)
    fi
    bash ${PYTORCH_ROOT}/maca_tests/run_pynet.sh $1 ${XML_PATH}
    if [[ $? != 0 ]]; then
        ERR=$(expr ${ERR} + 1)
    fi
    bash ${PYTORCH_ROOT}/test/run_runtest_nightly.sh $1 ${XML_PATH}
    if [[ $? != 0 ]]; then
        echo "Error found in runtest nightly"
        ERR=$(expr ${ERR} + 1)
    fi

    bash ${PYTORCH_ROOT}/maca_tests/libtorch/build_libtorch.sh ${XML_PATH}
    if [[ $? != 0 ]]; then
        ERR=$(expr ${ERR} + 1)
    fi

    if [[ ${PYTORCH_SEND_MAIL} == 1 ]]; then
        send_html_report ${XML_PATH} || true
    fi

    if [[ ${ERR} != 0 ]]; then
        echo "------------------Error found in pytorch tests ${1}------------------"
        exit 1
    fi
fi

if [[ $1 == "device" ]]; then
    ERR=0
    XML_PATH=${PYTORCH_ROOT}/test_report/xmls
    rm -rf ${XML_PATH}
    source ${PYTORCH_ROOT}/maca_tools/utils.sh
    make_xmls_dir ${XML_PATH}

    bash ${PYTORCH_ROOT}/maca_tests/run_c10_test.sh
    if [[ $? != 0 ]]; then
        ERR=$(expr ${ERR} + 1)
    fi

    bash ${PYTORCH_ROOT}/maca_tests/run_aten_test.sh
    if [[ $? != 0 ]]; then
        ERR=$(expr ${ERR} + 1)
    fi

    bash ${PYTORCH_ROOT}/maca_tests/run_pytest.sh $1 ${XML_PATH}
    if [[ $? != 0 ]]; then
        ERR=$(expr ${ERR} + 1)
    fi

    bash ${PYTORCH_ROOT}/maca_tests/run_pynet.sh $1 ${XML_PATH}
    if [[ $? != 0 ]]; then
        ERR=$(expr ${ERR} + 1)
    fi

    if [[ ${PYTORCH_SEND_MAIL} == 1 ]]; then
        send_html_report ${XML_PATH} || true
    fi

    if [[ ${ERR} != 0 ]];then
        echo "------------------Error found in pytorch tests in ${1}------------------"
        exit 1
    fi
fi
