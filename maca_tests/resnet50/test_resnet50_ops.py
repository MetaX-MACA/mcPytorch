#!/usr/bin/env python
import torch
import torch.nn as nn
import sys
import os
import time
import argparse
import copy
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import RunCpuAndGpuTest
from utils import RunSimpleGpuTest

dtype = torch.float32


def test_batchnorm(case_id, batch_size, only_run, only_fwd):
    test_cases = {
        0: {"input_shape": (batch_size, 1024, 14, 14), "eps": 1e-05, "momentum": 0.1, "affine": True, "track_running_stats": True, "time_hint_min": 11},  # noqa: B950
        1: {"input_shape": (batch_size, 128, 28, 28), "eps": 1e-05, "momentum": 0.1, "affine": True, "track_running_stats": True, "time_hint_min": 2},  # noqa: B950
        2: {"input_shape": (batch_size, 128, 56, 56), "eps": 1e-05, "momentum": 0.1, "affine": True, "track_running_stats": True, "time_hint_min": 3},  # noqa: B950
        3: {"input_shape": (batch_size, 2048, 7, 7), "eps": 1e-05, "momentum": 0.1, "affine": True, "track_running_stats": True, "time_hint_min": 18},  # noqa: B950
        4: {"input_shape": (batch_size, 256, 14, 14), "eps": 1e-05, "momentum": 0.1, "affine": True, "track_running_stats": True, "time_hint_min": 3},  # noqa: B950
        5: {"input_shape": (batch_size, 256, 28, 28), "eps": 1e-05, "momentum": 0.1, "affine": True, "track_running_stats": True, "time_hint_min": 4},  # noqa: B950
        6: {"input_shape": (batch_size, 256, 56, 56), "eps": 1e-05, "momentum": 0.1, "affine": True, "track_running_stats": True, "time_hint_min": 7},  # noqa: B950
        7: {"input_shape": (batch_size, 512, 14, 14), "eps": 1e-05, "momentum": 0.1, "affine": True, "track_running_stats": True, "time_hint_min": 6},  # noqa: B950
        8: {"input_shape": (batch_size, 512, 28, 28), "eps": 1e-05, "momentum": 0.1, "affine": True, "track_running_stats": True, "time_hint_min": 11},  # noqa: B950
        9: {"input_shape": (batch_size, 512, 7, 7), "eps": 1e-05, "momentum": 0.1, "affine": True, "track_running_stats": True, "time_hint_min": 5},  # noqa: B950
        10: {"input_shape": (batch_size, 64, 112, 112), "eps": 1e-05, "momentum": 0.1, "affine": True, "track_running_stats": True, "time_hint_min": 6},  # noqa: B950
        11: {"input_shape": (batch_size, 64, 56, 56), "eps": 1e-05, "momentum": 0.1, "affine": True, "track_running_stats": True, "time_hint_min": 2}  # noqa: B950
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        m = nn.BatchNorm2d(case["input_shape"][1], eps=case["eps"], momentum=case["momentum"],
                           affine=case["affine"], track_running_stats=case["track_running_stats"])
        m.state_dict()["weight"].copy_(torch.rand(case["input_shape"][1], dtype=dtype))
        m.state_dict()["bias"].copy_(torch.rand(case["input_shape"][1], dtype=dtype))
        m.state_dict()["running_mean"].copy_(torch.rand(case["input_shape"][1], dtype=dtype))
        m.state_dict()["running_var"].copy_(torch.rand(case["input_shape"][1], dtype=dtype))
        input = torch.rand(case["input_shape"], dtype=dtype)
        input.requires_grad_(True)
        if only_run:
            RunSimpleGpuTest(m, input, backward=(not only_fwd))
        else:
            if not RunCpuAndGpuTest(m, input, backward=(not only_fwd), loop=True):
                print("!!!!!! Error raise with batchnorm case:", case)
                ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret


def test_add(case_id, batch_size, only_run, only_fwd):
    test_cases = {
        0: {"input_shape": (batch_size, 256, 56, 56)},
        1: {"input_shape": (batch_size, 512, 28, 28)},
        2: {"input_shape": (batch_size, 1024, 14, 14)},
        3: {"input_shape": (batch_size, 2048, 7, 7)}
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        m = torch.add
        input_a = torch.rand(case["input_shape"], dtype=dtype)
        input_b = torch.rand(case["input_shape"], dtype=dtype)
        input_a.requires_grad_(True)
        input_b.requires_grad_(True)
        if only_run:
            RunSimpleGpuTest(m, input_a, input_b, backward=(not only_fwd))
        else:
            if not RunCpuAndGpuTest(m, input_a, input_b, backward=(not only_fwd), loop=True):
                print("!!!!!! Error raise with avgpool case:", case)
                ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret


def test_avgpool(case_id, batch_size, only_run, only_fwd):
    test_cases = {
        0: {"input_shape": (batch_size, 2048, 7, 7), "output_size": (1, 1)}
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        m = nn.AdaptiveAvgPool2d(output_size=case["output_size"])
        input = torch.rand(case["input_shape"], dtype=dtype)
        input.requires_grad_(True)

        if only_run:
            RunSimpleGpuTest(m, input, backward=(not only_fwd))
        else:
            if not RunCpuAndGpuTest(m, input, backward=(not only_fwd), loop=True):
                print("!!!!!! Error raise with avgpool case:", case)
                ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret


def test_maxpool(case_id, batch_size, only_run, only_fwd):
    test_cases = {
        0: {"input_shape": (batch_size, 64, 112, 112), "kernel_size": 3, "stride": 2, "padding": 1, "dilation": 1, "ceil_mode": False}  # noqa: B950
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        m = nn.MaxPool2d(kernel_size=case["kernel_size"], stride=case["stride"],
                         padding=case["padding"], dilation=case["dilation"], ceil_mode=case["ceil_mode"])
        input = torch.rand(case["input_shape"], dtype=dtype)
        input.requires_grad_(True)
        if only_run:
            RunSimpleGpuTest(m, input, backward=(not only_fwd))
        else:
            if not RunCpuAndGpuTest(m, input, backward=(not only_fwd), loop=True):
                print("!!!!!! Error raise with maxpool case:", case)
                ret = False

        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret


def test_linear(case_id, batch_size, only_run, only_fwd):
    test_cases = {
        0: {"input_shape": (batch_size, 2048), "in_features": 2048, "out_features": 1000, "bias": True}
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        m = nn.Linear(in_features=case["in_features"], out_features=case["out_features"])
        input = torch.rand(case["input_shape"], dtype=dtype)
        input.requires_grad_(True)

        if only_run:
            RunSimpleGpuTest(m, input, backward=(not only_fwd))
        else:
            if not RunCpuAndGpuTest(m, input, backward=(not only_fwd), loop=True, ftol=1e-3, btol=1e-3):
                print("!!!!!! Error raise with linear case:", case)
                ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret


def test_relu(case_id, batch_size, only_run, only_fwd):
    test_cases = {
        0: {"input_shape": (batch_size, 1024, 14, 14), "inplace": False},
        1: {"input_shape": (batch_size, 128, 28, 28), "inplace": False},
        2: {"input_shape": (batch_size, 128, 56, 56), "inplace": False},
        3: {"input_shape": (batch_size, 2048, 7, 7), "inplace": False},
        4: {"input_shape": (batch_size, 256, 14, 14), "inplace": False},
        5: {"input_shape": (batch_size, 256, 28, 28), "inplace": False},
        6: {"input_shape": (batch_size, 256, 56, 56), "inplace": False},
        7: {"input_shape": (batch_size, 512, 14, 14), "inplace": False},
        8: {"input_shape": (batch_size, 512, 28, 28), "inplace": False},
        9: {"input_shape": (batch_size, 512, 7, 7), "inplace": False},
        10: {"input_shape": (batch_size, 64, 112, 112), "inplace": False},
        11: {"input_shape": (batch_size, 64, 56, 56), "inplace": False}
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        m = nn.ReLU(inplace=case["inplace"])
        input = torch.rand(case["input_shape"], dtype=dtype)
        input.requires_grad_(True)

        if only_run:
            RunSimpleGpuTest(m, input, backward=(not only_fwd))
        else:
            if not RunCpuAndGpuTest(m, input, backward=(not only_fwd), loop=True):
                print("!!!!!! Error raise with relu case:", case)
                ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)

        time_s = time.time()
        m = nn.ReLU(inplace=True)
        input = torch.rand(case["input_shape"], dtype=dtype)
        if only_run:
            print("WARNING: skip inplace relu test")
        else:
            if not RunCpuAndGpuTest(m, input, backward=False, loop=True):
                print("!!!!!! Error raise with relu case:", case)
                ret = False
        duration = time.time() - time_s
        print(case, "(inplace), time: ", duration)
    return ret


def test_misc(case_id, batch_size, only_run, only_fwd):
    test_cases = {
        0: {"input_shape": (batch_size, 256, 56, 56)},
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        if only_run:
            print("Warning: skip the misc case for 'only run' option")
        else:
            input_a = torch.rand(case["input_shape"], dtype=dtype)
            input_b = torch.rand(case["input_shape"], dtype=dtype)
            input_a_d = input_a.cuda()
            input_b_d = input_b.cuda()
            input_a = input_a + input_b
            input_a_d = input_a_d + input_b_d
            input_a_d_h = input_a_d.cpu()
            ret = ret and torch.allclose(input_a, input_a_d_h)
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret



def test_conv2d(case_id, batch_size, only_run, only_fwd, mcdnn_tf32, half):
    test_cases = {
        0: {"input_shape": (batch_size, 1024, 14, 14), "in_channels": 1024, "out_channels": 2048, "kernel_size": (1, 1), "stride": (2, 2), "padding": (0, 0), "bias": False},  # noqa: B950
        1: {"input_shape": (batch_size, 1024, 14, 14), "in_channels": 1024, "out_channels": 256, "kernel_size": (1, 1), "stride": (1, 1), "padding": (0, 0), "bias": False},   # noqa: B950
        2: {"input_shape": (batch_size, 1024, 14, 14), "in_channels": 1024, "out_channels": 512, "kernel_size": (1, 1), "stride": (1, 1), "padding": (0, 0), "bias": False},   # noqa: B950
        3: {"input_shape": (batch_size, 128, 28, 28), "in_channels": 128, "out_channels": 128, "kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1), "bias": False},     # noqa: B950
        4: {"input_shape": (batch_size, 128, 56, 56), "in_channels": 128, "out_channels": 128, "kernel_size": (3, 3), "stride": (2, 2), "padding": (1, 1), "bias": False},     # noqa: B950
        5: {"input_shape": (batch_size, 128, 28, 28), "in_channels": 128, "out_channels": 512, "kernel_size": (1, 1), "stride": (1, 1), "padding": (0, 0), "bias": False},     # noqa: B950
        6: {"input_shape": (batch_size, 2048, 7, 7), "in_channels": 2048, "out_channels": 512, "kernel_size": (1, 1), "stride": (1, 1), "padding": (0, 0), "bias": False},     # noqa: B950
        7: {"input_shape": (batch_size, 256, 14, 14), "in_channels": 256, "out_channels": 1024, "kernel_size": (1, 1), "stride": (1, 1), "padding": (0, 0), "bias": False},    # noqa: B950
        8: {"input_shape": (batch_size, 256, 56, 56), "in_channels": 256, "out_channels": 128, "kernel_size": (1, 1), "stride": (1, 1), "padding": (0, 0), "bias": False},     # noqa: B950
        9: {"input_shape": (batch_size, 256, 14, 14), "in_channels": 256, "out_channels": 256, "kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1), "bias": False},     # noqa: B950
        10: {"input_shape": (batch_size, 256, 28, 28), "in_channels": 256, "out_channels": 256, "kernel_size": (3, 3), "stride": (2, 2), "padding": (1, 1), "bias": False},    # noqa: B950
        11: {"input_shape": (batch_size, 256, 56, 56), "in_channels": 256, "out_channels": 512, "kernel_size": (1, 1), "stride": (2, 2), "padding": (0, 0), "bias": False},    # noqa: B950
        12: {"input_shape": (batch_size, 256, 56, 56), "in_channels": 256, "out_channels": 64, "kernel_size": (1, 1), "stride": (1, 1), "padding": (0, 0), "bias": False},     # noqa: B950
        13: {"input_shape": (batch_size, 3, 224, 224), "in_channels": 3, "out_channels": 64, "kernel_size": (7, 7), "stride": (2, 2), "padding": (3, 3), "bias": False},       # noqa: B950
        14: {"input_shape": (batch_size, 512, 28, 28), "in_channels": 512, "out_channels": 1024, "kernel_size": (1, 1), "stride": (2, 2), "padding": (0, 0), "bias": False},   # noqa: B950
        15: {"input_shape": (batch_size, 512, 28, 28), "in_channels": 512, "out_channels": 128, "kernel_size": (1, 1), "stride": (1, 1), "padding": (0, 0), "bias": False},    # noqa: B950
        16: {"input_shape": (batch_size, 512, 7, 7), "in_channels": 512, "out_channels": 2048, "kernel_size": (1, 1), "stride": (1, 1), "padding": (0, 0), "bias": False},     # noqa: B950
        17: {"input_shape": (batch_size, 512, 28, 28), "in_channels": 512, "out_channels": 256, "kernel_size": (1, 1), "stride": (1, 1), "padding": (0, 0), "bias": False},    # noqa: B950
        18: {"input_shape": (batch_size, 512, 7, 7), "in_channels": 512, "out_channels": 512, "kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1), "bias": False},      # noqa: B950
        19: {"input_shape": (batch_size, 512, 14, 14), "in_channels": 512, "out_channels": 512, "kernel_size": (3, 3), "stride": (2, 2), "padding": (1, 1), "bias": False},    # noqa: B950
        20: {"input_shape": (batch_size, 64, 56, 56), "in_channels": 64, "out_channels": 256, "kernel_size": (1, 1), "stride": (1, 1), "padding": (0, 0), "bias": False},      # noqa: B950
        21: {"input_shape": (batch_size, 64, 56, 56), "in_channels": 64, "out_channels": 64, "kernel_size": (1, 1), "stride": (1, 1), "padding": (0, 0), "bias": False},       # noqa: B950
        22: {"input_shape": (batch_size, 64, 56, 56), "in_channels": 64, "out_channels": 64, "kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1), "bias": False}        # noqa: B950
    }

    case_id = int(case_id)
    ret = True

    tf32_fwd_badcase_id = {13}
    tf32_bd_badcase_id = {0, 1, 2, 6, 7, 9, 10, 13, 14, 16, 18, 19}

    torch.backends.cudnn.allow_tf32 = mcdnn_tf32
    for id, case in test_cases.items():
        if mcdnn_tf32:
            if only_fwd and id in tf32_fwd_badcase_id:
                continue
            if not only_fwd and id in tf32_bd_badcase_id:
                continue
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        m = nn.Conv2d(case["in_channels"], case["out_channels"], case["kernel_size"], stride=case["stride"], padding=case["padding"], bias=case["bias"])  # noqa: B950
        input = torch.rand(case["input_shape"])

        if only_run:
            RunSimpleGpuTest(m, input, backward=(not only_fwd), half=half)
        else:
            if not RunCpuAndGpuTest(m, input, backward=(not only_fwd), loop=True, ftol=1e-3, btol=1e-3, half=half):
                print("!!!!!! Error raise with conv2d case:", case, flush=True)
                ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration, flush=True)

    return ret

def test_tf32_set_get():
    ret = True
    torch.backends.cudnn.allow_tf32 = True
    if torch.backends.cudnn.allow_tf32 is not True:
        print("cudnn allow_tf32 is set to True, but not effective")
        ret = False

    torch.backends.cudnn.allow_tf32 = False
    if torch.backends.cudnn.allow_tf32 is not False:
        print("cudnn allow_tf32 is set to False, but not effective")
        ret = False
    return ret


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--op_type", default="all",
                        help="<all|batchnorm|avgpool|maxpool|relu|linear|add|misc|conv2d|tf32_set_get>")
    parser.add_argument("--id", default=-1, help="op_type's test case id")
    parser.add_argument("--batch_size", default=1, help="batch size")
    parser.add_argument("--only_run", action="store_true", help="only run and not checkout precision")
    parser.add_argument("--only_fwd", action="store_true", help="only run forward")
    parser.add_argument("--mcdnn_tf32", action="store_true", help="conv run with mcdnn tf32")
    parser.add_argument("--half", action="store_true", help="half type")

    args = parser.parse_args()
    op_type = args.op_type
    case_id = args.id
    batch_size = int(args.batch_size)
    only_run = args.only_run
    only_fwd = args.only_fwd
    mcdnn_tf32 = args.mcdnn_tf32
    half = args.half

    ret = True
    if op_type == "all":
        ret = ret and test_batchnorm(case_id, batch_size, only_run, only_fwd)
        ret = ret and test_avgpool(case_id, batch_size, only_run, only_fwd)
        ret = ret and test_maxpool(case_id, batch_size, only_run, only_fwd)
        ret = ret and test_linear(case_id, batch_size, only_run, only_fwd)
        ret = ret and test_relu(case_id, batch_size, only_run, only_fwd)
        ret = ret and test_add(case_id, batch_size, only_run, only_fwd)
        ret = ret and test_misc(case_id, batch_size, only_run, only_fwd)
        ret = ret and test_conv2d(case_id, batch_size, only_run, only_fwd, mcdnn_tf32, half)
        ret = ret and test_tf32_set_get()
    elif op_type == "batchnorm":
        ret = ret and test_batchnorm(case_id, batch_size, only_run, only_fwd)
    elif op_type == "avgpool":
        ret = ret and test_avgpool(case_id, batch_size, only_run, only_fwd)
    elif op_type == "maxpool":
        ret = ret and test_maxpool(case_id, batch_size, only_run, only_fwd)
    elif op_type == "relu":
        ret = ret and test_relu(case_id, batch_size, only_run, only_fwd)
    elif op_type == "linear":
        ret = ret and test_linear(case_id, batch_size, only_run, only_fwd)
    elif op_type == "add":
        ret = ret and test_add(case_id, batch_size, only_run, only_fwd)
    elif op_type == "misc":
        ret = ret and test_misc(case_id, batch_size, only_run, only_fwd)
    elif op_type == "conv2d":
        ret = ret and test_conv2d(case_id, batch_size, only_run, only_fwd, mcdnn_tf32, half)
    elif op_type == "tf32_set_get":
        ret = ret and test_tf32_set_get()
    else:
        print("Error: Not support op_type: ", op_type)

    exit(0 if ret is True else 1)
