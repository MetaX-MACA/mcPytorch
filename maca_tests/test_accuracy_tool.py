#!/usr/bin/env python
import os
import copy
import shutil
import glob
import torch
import torch.nn as nn
import torch.profiler as profiler

def test_linear():
    dtype = torch.float32
    m = nn.Linear(20, 30)
    input = torch.randn(128, 20)
    output = m(input)
    m_cuda = copy.deepcopy(m).cuda()
    input_cuda = input.cuda()

    for file_name in os.listdir():
        if file_name.startswith("op_logs"):
            shutil.rmtree(file_name)

    os.environ['PYTORCH_ACC_CHECK_OP']="all"
    os.environ['PYTORCH_ACC_CHECK_LEVEL']="2"
    os.environ['RTOL']="1e-4"
    os.environ['ATOL']="1e-4"

    with profiler.profile(with_stack=True):
        output_cuda = m_cuda(input_cuda)
        grad = torch.randn_like(output_cuda)
        output_cuda.backward(grad)

    base_dir = glob.glob("op_logs_*")
    if not os.path.exists(base_dir[0]):
        raise Exception("Accuracy tool creates directory error!")

    if not os.path.exists(base_dir[0] + "/linear_0"):
        raise Exception("Accuracy tool creates directory error!")

    if not os.path.exists(base_dir[0] + "/linear_0/t_0"):
        raise Exception("Accuracy tool creates directory error!")

    if not os.path.exists(base_dir[0] + "/linear_0/addmm_0"):
        raise Exception("Accuracy tool creates directory error!")

    status = torch.allclose(output_cuda.cpu(), output, rtol = 1e-4, atol = 1e-4)
    diff = torch.abs(output_cuda.cpu() - output)
    status_err = not os.path.exists(base_dir[0] + "/linear_0/err_info.txt")

    if status != status_err:
        raise Exception("Accuracy tool makes fault comparsion!")

    os.environ['PYTORCH_ACC_CHECK_OP']="linear"

    with profiler.profile(with_stack=True):
        output = m(input)

    if not os.path.exists(base_dir[0] + "/linear_1"):
        raise Exception("Accuracy tool creates directory error!")

    with profiler.profile(with_stack=True):
        output_cuda = m_cuda(input_cuda)
        grad = torch.randn_like(output_cuda)
        output_cuda.backward(grad)

def test_nan():
    input = torch.randn(128, 20).cuda()
    input[0] = torch.nan

    # remove all fileis in op_logs
    cur_dir = os.getcwd()
    for file_name in os.listdir():
        if file_name.startswith("op_logs"):
            log_dir = os.path.join(cur_dir, file_name)
            for subfile_name in os.listdir(log_dir):
                sublog_dir = os.path.join(log_dir, subfile_name)
                shutil.rmtree(sublog_dir)

    os.environ['PYTORCH_ACC_CHECK_OP']="all"

    with profiler.profile(with_stack=True):
        output_cuda = input + 1

    base_dir = glob.glob("op_logs_*")
    status_err = os.path.exists(base_dir[0] + "/add_0/err_info.txt")

    if not status_err:
        raise Exception("Nan outputs did not detected!")


def test_conv_meta():
    case = {"input_shape": (1, 64, 56, 56), "in_channels": 64, "out_channels": 64, "kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1), "bias": False}

    inp = torch.rand(case["input_shape"])
    m = nn.Conv2d(case["in_channels"], case["out_channels"], case["kernel_size"], stride=case["stride"], padding=case["padding"], bias=case["bias"])
    output = m(inp)

    inp_cuda = inp.cuda()
    m_cuda = copy.deepcopy(m).cuda()

    # remove all fileis in op_logs
    cur_dir = os.getcwd()
    for file_name in os.listdir():
        if file_name.startswith("op_logs"):
            log_dir = os.path.join(cur_dir, file_name)
            for subfile_name in os.listdir(log_dir):
                sublog_dir = os.path.join(log_dir, subfile_name)
                shutil.rmtree(sublog_dir)

    os.environ['PYTORCH_ACC_CHECK_OP']="all"
    os.environ['PYTORCH_ACC_CHECK_LEVEL']="2"
    os.environ['CONV_META']="1"
    os.environ['RTOL']='1e-4'
    os.environ['ATOL']='1e-4'

    with profiler.profile(with_stack=True):
        output_cuda = m_cuda(inp_cuda)
    status = torch.allclose(output_cuda.cpu(), output, rtol = 1e-4, atol = 1e-4)
    diff = torch.abs(output_cuda.cpu() - output)
    base_dir = glob.glob("op_logs_*")
    status_err = not os.path.exists(base_dir[0] + "/conv2d_0/err_info.txt")

    if status != status_err:
        raise Exception("Accuracy tool makes fault comparsion!")


if __name__ == '__main__':
    test_linear()
    test_conv_meta()
    test_nan()
