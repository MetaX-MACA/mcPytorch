import functools
from datetime import datetime
import re
import torch
import torch.nn as nn
import copy
import os
from functools import reduce
import math 
import numpy as np
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(dir_path + "/../../spd_db/")
from DatabaseManager import DatabaseManager

# use to control accuracy test, run accuracy test when WARMUP_IDX == 1 only
WARMUP_IDX = 0  

# return time (seconds)
# currently only support ns, us, ms & s
def get_time_s(info):
    line = info.strip().split("\n")[-1]
    # if not grep data return -1
    if "Self CUDA time total:" not in line:
        return -1
    str_val = line[22:].strip()
    if "ns" in line :
        val = float(str_val[:-2])
        val = float(val / 1000000000)
    elif "us" in line:
        val = float(str_val[:-2])
        val = float(val / 1000000)
    elif "ms" in line:
        val = float(str_val[:-2])
        val = float(val / 1000)
    elif "s" in line:
        val = float(str_val[:-1])
    else:
        assert 0
        
    return val


def get_db176(table_name2='reports'):
    db_config2 = {
        'user': 'root',
        'password': 'metax1234',
        'host': '10.2.120.176',
        'database': 'acl_performance',
        'port': 30000
    }
    db_176 = DatabaseManager(table_name2, db_config2)
    return db_176


def launch_prof(json_path, warmup_num=30, active_num=8, max_repeat_num=10):
    def warper(func):
        @functools.wraps(func)
        def run(instance, *args, **kwargs):
            print('[launch_prof] func={}, args={}, kwargs={}'.format(func, args, kwargs), flush=True)
            date_start = datetime.now()
            if "func_name" in kwargs and kwargs["func_name"] != "":
                instance.function = kwargs["func_name"]
            if "dtype" in kwargs:
                instance.testgroup = str(kwargs["dtype"])[6:]
            if "feature" in kwargs:
                instance.feature = str(kwargs["feature"])
            case_info = f"function:{instance.function}, job_name:{instance.job_name}, branch:{instance.branch}, testcase:{instance.testcase}, testgroup:{instance.testgroup}"
            global WARMUP_IDX
            WARMUP_IDX = 0
            # warmup
            for _ in range(warmup_num):
                func(instance, *args, **kwargs)
                WARMUP_IDX += 1

            # test
            t = []
            for _ in range(active_num):
                tt = func(instance, *args, **kwargs)
                repeat_num = 0
                while tt == -1 and repeat_num < max_repeat_num:
                    tt = func(instance, *args, **kwargs)
                    repeat_num += 1
                    print(f"### try {repeat_num} times to re-run test to get kernel cuda time during run {case_info}")
                assert tt != -1, f"torch.profiler cannot grep kernel cuda time after try {max_repeat_num} times during run {case_info}"
                t.append(tt)
            mean_time = sum(t) / len(t)
            metrics = {"second": mean_time}
            # print("mean: ", metrics)
            instance.performance = {"metrics":metrics}

            date_end = datetime.now()
            instance.teststart = date_start.strftime("%Y-%m-%d %H:%M:%S")
            instance.duration = "%H:%M:%S"
            time_duration = date_end - date_start
            instance.duration = re.sub("%H", str(time_duration.seconds // 3600), instance.duration)
            instance.duration = re.sub("%M", str(time_duration.seconds // 60), instance.duration)
            instance.duration = re.sub("%S", str(time_duration.seconds % 60), instance.duration)
            if "is_optim" in kwargs:
                instance.is_optim = kwargs["is_optim"]
            if "shape" in kwargs:
                instance.testcase = str(kwargs["shape"])

            # db_176 = get_db176()
            # last_data = db_176.query_data_order([instance.function, instance.job_name, instance.branch, instance.testcase, instance.testgroup], \
            #     "function = %s and job_name = %s and branch = %s and testcase = %s and testgroup = %s")
            # if last_data is not None and len(last_data) is not 0:
            #     metrics2 = {"last_second": last_data[0][20]}
            #     instance.performance["metrics"].update(metrics2)
            #     ref_err = cal_rel_err_new(mean_time, last_data[0][20])
            #     metrics3 = {"ref_err": ref_err}
            #     instance.performance["metrics"].update(metrics3)

            # last_data_10 = db_176.query_data_order_10([instance.function, instance.job_name, instance.branch, instance.testcase, instance.testgroup], \
            #     "function = %s and job_name = %s and branch = %s and testcase = %s and testgroup = %s")
            # if last_data_10 is not None and len(last_data) is not 0:
            #     avg_10 = 0
            #     for i in range(len(last_data_10)):
            #         avg_10 += last_data_10[0][20]
            #     avg_10 = avg_10 / len(last_data_10)
            #     metrics4 = {"last_10day_avg_second": avg_10}
            #     instance.performance["metrics"].update(metrics4)
            #     ref_err_10 = cal_rel_err_new(mean_time, avg_10)
            #     metrics5 = {"ref_err_10day": ref_err_10}
            #     instance.performance["metrics"].update(metrics5)

            instance.dumpJson(json_path, instance.function + "_" + instance.testgroup + "_" +
                                instance.testcase + ".json")
            return mean_time
        return run
    return warper


def peek(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return first


def cal_rel_err(infer, golden):
    diff = infer - golden
    diff_square = diff * diff
    infer_result_square_double = 2 * infer * infer
    sum_diff_square = torch.sum(diff_square)
    sum_infer_result_square_double = torch.sum(infer_result_square_double)
    result = torch.sqrt(sum_diff_square / (sum_infer_result_square_double + 1e-12))
    return result


def cal_rel_err_new(infer, golden):
    ret = (golden - infer)/golden
    return ret


# the inputs is the list of all input of model
def accuracy_check(model_device, inputs_device:list, bwd_input_device:torch.Tensor = None, fwd_tol=1e-4, bwd_tol=1e-4) -> bool:
    global WARMUP_IDX
    if WARMUP_IDX != 0:
        return True
    
    if not os.getenv("PYTORCH_ENABLE_ACCURACY_TEST", 0):
        print(f"skip accuracy check as PYTORCH_ENABLE_ACCURACY_TEST not set")
        return True
    
    assert inputs_device != [], "inputs should not be empty"
    # check forward
    if hasattr(model_device, "parameters") and peek(model_device.parameters()) is not None:
        model_cpu = copy.deepcopy(model_device).to(torch.float32).cpu()
    else:
        model_cpu = model_device

    if hasattr(model_device, "train") and model_device.training:
        model_cpu.train()

    inputs_cpu = []
    for input in inputs_device:
        if torch.is_tensor(input):
            if input.dtype == torch.float16 or input.dtype == torch.bfloat16:
                input = input.clone().detach().float().cpu().requires_grad_(True)
            else:
                input = input.clone().detach().cpu().requires_grad_(True)
        inputs_cpu.append(input)

    output_device = model_device(*inputs_device)
    output_cpu = model_cpu(*inputs_cpu)
    if isinstance(output_device, tuple):
        for i in range(len(output_device)):
            output_device_i = output_device[i]
            output_cpu_i = output_cpu[i]
            rel_error = cal_rel_err(output_cpu_i, output_device_i.float().cpu())
            if rel_error > fwd_tol:
                print(f"forward No.{i} output check fail: rel_error = {rel_error} > {fwd_tol}")
                return False
    else:
        rel_error = cal_rel_err(output_cpu, output_device.float().cpu())
        if rel_error > fwd_tol:
            print(f"forward check fail: rel_error = {rel_error} > {fwd_tol}")
            return False

    # check backward
    bwd_input_cpu = None
    if bwd_input_device != None:
        bwd_input_cpu = bwd_input_device.cpu()

        output_cpu.backward(bwd_input_cpu)
        output_device.backward(bwd_input_device)

        # check input grad
        for i, input in enumerate(inputs_device):
            if torch.is_tensor(input) and input.requires_grad:
                rel_error = cal_rel_err(inputs_cpu[i].grad, input.grad.float().cpu())
                if rel_error > bwd_tol:
                    print(f"backward check fail: rel_error = {rel_error} > {bwd_tol}")
                    return False

        # check parameters grad
        if hasattr(model_device, "parameters"):
            param_device = [i for i in model_device.parameters()]
            param_cpu = [i for i in model_cpu.parameters()]

            for param_d, param_c in zip(param_device, param_cpu):
                if torch.is_tensor(param_d) and param_d.requires_grad:
                    rel_error = cal_rel_err(param_c.grad, param_d.grad.float().cpu())
                    if rel_error > bwd_tol:
                        print(f"backward check fail: rel_error = {rel_error} > {bwd_tol}")
                        return False
    print(f"accuracy check pass")
    return True

def make_pattern_tensor(shape, dtype, output_strides=[], input_strides=[], offset = 0):
    # assert num_input == len(input_strides), "set input_strides.length == num_input"
    shape_mul = reduce(lambda x, y: x * y, shape)
    outputs = []
    if len(output_strides) > 0:
        for i in range(len(output_strides)):
            output_stride_max = max(np.array(shape) * np.array(output_strides[i]))
            output_ceil = math.ceil(output_stride_max / shape_mul)
            output_shape = copy.deepcopy(shape)
            if output_ceil > 0:
              output_shape[-1] = shape[-1] * output_ceil + offset
            output = torch.randn(output_shape, dtype=dtype, device="cuda")
            output = output[..., offset:output_shape[-1] + offset]
            try:
                output = output.as_strided(shape, output_strides[i])
            except Exception:
                return [], []
            outputs.append(output)

    inputs = []
    for i in range(len(input_strides)):
        input_stride_max = max(np.array(shape) * np.array(input_strides[i]))
        input_ceil = math.ceil(input_stride_max / shape_mul)
        input_shape = copy.deepcopy(shape)
        if input_ceil > 0:
          input_shape[-1] = shape[-1] * input_ceil + offset
        input = torch.randn(input_shape, dtype=dtype, device="cuda")
        input = input[..., offset:input_shape[-1] + offset]
        try:
            input = input.as_strided(shape, input_strides[i])
        except Exception:
            return [], []
        inputs.append(input)
    return outputs, inputs