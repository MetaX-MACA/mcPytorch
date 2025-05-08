#!/usr/bin/env python
import socket
import torch
import copy
import os
import torch.nn as nn
import math
from functools import reduce
import numpy as np
import functools
from enum import Enum

w, h = 224, 224
TEST_MODE = os.getenv('TEST_MODE')
if TEST_MODE is None or TEST_MODE.lower() == "checkin":
    BERT_TEST_BATCHSIZE = [1]  # bert test batch size for checkin
    RESNET50_TEST_BATCHSIZE = [1]
    resnet_50_shapes = [(64, h // 4, w // 4)]
else:
    BERT_TEST_BATCHSIZE = [1, 2, 4]  # bert test batch size
    RESNET50_TEST_BATCHSIZE = [1, 4, 8]
    resnet_50_shapes = [(64, h // 2, w // 2), (64, h // 4, w // 4), (256, h // 4, w // 4),
                        (128, h // 8, w // 8), (512, h // 8, w // 8), (256, h // 16, w // 16),
                        (1024, h // 16, w // 16), (512, h // 32, w // 32), (2048, h // 32, w // 32)]

GOLDEN_DIR = r"/netapp/pytorch/golden/"   # dir of golden data


def check_close(infer_result_data, golden_data, eps=1e-4):
    diff = infer_result_data - golden_data
    diff_square = diff * diff
    infer_result_square_double = 2 * infer_result_data * infer_result_data
    sum_diff_square = torch.sum(diff_square)
    sum_infer_result_square_double = torch.sum(infer_result_square_double)
    result = torch.sqrt(sum_diff_square / (sum_infer_result_square_double + 1e-12))
    print(f"check close error: {result}")

    return result < eps

def RunSimpleGpuTest(*args, **kwargs):
    r"""Assume:
        args[0]: pytorch func or module on cpu
        args[1:]: inputs on cpu
        kwargs: 'backward'
    """
    enable_backward = False
    enable_half = False
    if "backward" in kwargs.keys():
        enable_backward = kwargs["backward"]
    if "half" in kwargs.keys():
        enable_half = kwargs["half"]

    param_gpu = []
    for param in args[1:]:
        param_g = param.cuda()
        if enable_half and hasattr(param_g, "half"):
            param_g = param_g.detach.half()
        param_g.requires_grad_(True)
        param_gpu.append(param_g)

    fun = args[0]
    if hasattr(fun, "cuda"):
        fun_cuda = copy.deepcopy(fun).cuda()
    else:
        fun_cuda = copy.deepcopy(fun)

    if enable_half and hasattr(fun, "half"):
        fun_cuda = fun_cuda.half()

    output = fun_cuda(*param_gpu)
    print("run fwd finish")

    if enable_backward:
        backward_input = torch.randn(output.shape).cuda().to(dtype=output.dtype)
        output.backward(backward_input)
        print("run bwd finish")
    print("### output: ", output)


def RunCpuAndGpuTest(*args, **kwargs):
    r"""for function test, e.g. we want to test torch.nn.AdaptiveAvgPool2d(),
    the args are torch.nn.AdaptiveAvgPool2d(4), input_tensor.
    for class method test, e.g. we want to test tensor.expand(), the args are
    "expand", input_tensor, expand params.
    kwargs use to set tol, e.g. ftol = 1e-4, btol = 1e-4, if without setting,
    use allclose default tol.

    Args:
    backward = True to run backward, default is False
    backward_input is the input for backward, default is None, and would gen by randn
    loop use to run loop test if set as True, default is False, when loop is true,
        it would return test result instead of exit
    ftol & btol is tol for forward and backward
    """

    ftol = 1e-4
    btol = 1e-4
    backward = False
    loop = False
    backward_input = None
    half = False

    if "ftol" in kwargs.keys():
        ftol = kwargs["ftol"]
    if "btol" in kwargs.keys():
        btol = kwargs["btol"]
    if "backward" in kwargs.keys():
        backward = kwargs["backward"]
    if "backward_input" in kwargs.keys():
        backward_input = kwargs["backward_input"]
    if "loop" in kwargs.keys():
        loop = kwargs["loop"]
    if "half" in kwargs.keys():
        half = kwargs["half"]

    if isinstance(args[0], str):
        status = RunMethod(*args, ftol=ftol)
    else:
        status = RunFunction(*args, backward=backward, backward_input=backward_input, ftol=ftol, btol=btol, half=half)

    if loop:
        return status
    else:
        if status:
            exit(0)
        else:
            exit(1)


def RunFunction(*args, backward, backward_input, ftol, btol, half):
    fun = args[0]

    param_cpu = []
    param_gpu = []
    for param in args[1:]:
        param_c = param
        if hasattr(param, "cuda"):
            param_g = param.detach().clone().cuda()
        else:
            param_g = param.detach().clone()

        if half and hasattr(param, "half"):
            param_g = param_g.detach().half()

        if backward and hasattr(param, "requires_grad_"):
            param_g = param_g.requires_grad_(True)
            param_c = param_c.requires_grad_(True)

        param_gpu.append(param_g)     # some type not support requires_grad_
        param_cpu.append(param_c)

    output_golden = fun(*param_cpu)

    if hasattr(fun, "cuda"):
        fun_cuda = copy.deepcopy(fun).cuda()
    else:
        fun_cuda = copy.deepcopy(fun)

    if half and hasattr(fun_cuda, "half"):
        fun_cuda = fun_cuda.half()

    output = fun_cuda(*param_gpu)
    print("### output: ", output)

    if backward:
        # gen backward_input by randn if it is None,
        if backward_input is None:
            backward_input = torch.randn(output_golden.shape, dtype=output_golden.dtype)

        output_golden.backward(backward_input)
        output.backward(backward_input.detach().clone().cuda().to(dtype=output.dtype))
    diff = output.float().cpu() - output_golden.float()
    print(f"###forward max diff {torch.max(diff)}")
    fw_status = check_close(output.float().cpu(), output_golden, eps=ftol)
    bw_status = True

    if backward:
        for i in range(len(param_cpu)):
            g_g = param_gpu[i].grad.float().cpu()
            g_c = param_cpu[i].grad.float()
            diff = torch.abs(g_g - g_c)
            print(f"###g_g\n: {g_g}")
            print(f"###g_c\n: {g_c}")
            print(f"###backward max diff {torch.max(diff)}")
            bw_status = (bw_status and check_close(g_g, g_c, eps=btol))
    print(f"$$$fw_status: {fw_status}, bw_status: {bw_status}")
    return (fw_status and bw_status)


def RunMethod(*args, ftol):
    r"""instance.method(params)
    """
    method = args[0]
    instance = args[1]
    params = args[2:]

    output_golden = getattr(instance, method)(*params)

    if hasattr(instance, "cuda"):
        instance_gpu = copy.deepcopy(instance).cuda()
    else:
        instance_gpu = copy.deepcopy(instance)

    param_gpu = []
    for param in params:
        if hasattr(param, "cuda"):
            param_g = param.detach().clone().cuda()
        else:
            param_g = param
        param_gpu.append(param_g)

    output = getattr(instance_gpu, method)(*param_gpu).cpu()

    return check_close(output, output_golden, ftol)


def RunSimpleMethodTest(*args):
    r"""instance.method(params)
    """
    method = args[0]
    instance = args[1]
    params = args[2:]

    if hasattr(instance, "cuda"):
        instance_gpu = copy.deepcopy(instance).cuda()
    else:
        instance_gpu = copy.deepcopy(instance)

    param_gpu = []
    for param in params:
        if hasattr(param, "cuda"):
            param_g = param.detach().clone().cuda()
        else:
            param_g = param
        param_gpu.append(param_g)

    output = getattr(instance_gpu, method)(*param_gpu).cpu()

    return

def generate_num(start, end=None, mul=2, ascend=True, times=5):
    """Generate num list.

    Args:
        start: the start num.
        end: the end num, if None, end = int(start * mul ** times) if ascend else int(start / (mul ** times)).
        mul: the mul num.
        ascend: Default: True, output data in ascending order.
        times: the num of the generated list.

    Examples:
        >>> generate_num(224, ascend=False);
        >>> output = [224, 112, 56, 28, 14, 7]
        >>> generate_num(64, ascend=True);
        >>> output = [64, 128, 256, 512, 1024, 2048]
        >>> generate_num(64, aend=65);
        >>> output = [64]
    """
    if not end:
        end = int(start * mul ** times) if ascend else int(start / (mul ** times))
    if start > end:
        start, end = end, start
    num = []
    while start <= end:
        num.append(start)
        start *= mul
    if not ascend:
        num.sort(reverse=True)
    return num


def MMTest(m, n, k, batch1=1, batch2=1, op=torch.matmul, input1=None, input2=None, bias=None, bw_input=None):
    """ checking forward & backwards of matmul/linear.
        if only mnkb specified, would gen random inputs/weight/bias/bw_input, if inputs/weight/bias/bw_input is specified,
    the specified data would be used.

    Args:
        mnkb: mat1's shape is [batch1, m, k], mat2's shape is [batch2, k, n], and result is [batch, m, n]
        input1: its shape should be batch1*m*k
        input2: if op is torch.nn.Linear, input2's shape should be batch2*n*k(batch2=1), or it should be batch2*k*n for matmul.
        op: torch.matmul or torch.nn.Linear
        bias: only for linear
        bw_input is the input of backward
    """
    assert m > 0 and n > 0 and k > 0 and batch1 > 0 and batch2 >0, "mnkb value check fail"
    assert op is torch.nn.Linear or torch.matmul, "op check fail: only linear and matmul supported"
    if batch1 != batch2:
        assert batch1 ==1 or batch2 == 1, "batch check fail"

    module = None
    if batch1 == 1:
        batch = batch2
    else:
        batch = batch1

    if bw_input is not None:
        if len(bw_input.shape) == 2:
            assert bw_input.shape[0] == m and bw_input.shape[1] == n and batch == 1, "bw_input shape check fail"
        elif len(bw_input.shape) == 3:
            assert bw_input.shape[0] == batch and bw_input.shape[1] == m and bw_input.shape[2] == n, f"bw_input shape check fail [{bw_input.shape[0]}, {bw_input.shape[1]}, {bw_input.shape[2]}], [{batch}, {m}, {n}]"
    else:
        bw_input = torch.rand(batch, m, n)

    if op is torch.nn.Linear:
        assert batch2 == 1, "batch2 should be 1 for linear"
        if input1 is not None:
            if len(input1.shape) == 2:
                assert input1.shape[0] == m and input1.shape[1] == k and batch1 == 1, "input1 shape check fail"
            elif len(input1.shape) == 3:
                assert input1.shape[0] == batch1 and input1.shape[1] == m and input1.shape[2] == k, "input1 shape check fail"
            else:
                assert 0, "input1 shape check fail"
        else:
            input1 = torch.rand(batch1, m, k)

        if input2 is not None:
            assert len(input2.shape) == 2, "input2 shape check fail"
            assert input2.shape[0] == n and input2.shape[1] == k, "input2 shape check fail"
        else:
            input2 = torch.rand(n, k)

        if bias is not None:
            assert len(bias.shape) == 1 and bias.shape[0] == n, "bias shape check fail"
        else:
            bias = torch.rand(n)

        module = torch.nn.Linear(n, k)

        input1 = input1.cpu()
        module.weight = torch.nn.Parameter(input2.cpu())
        module.bias = torch.nn.Parameter(bias.cpu())
        return checkFwAndBw(input1, input2=None, bw_input=bw_input, op=module)
    else:   # torch.matmul
        if input1 is not None:
            if len(input1.shape) == 2:
                assert input1.shape[0] == m and input1.shape[1] == k and batch1 == 1, "input1 shape check fail"
            elif len(input1.shape) == 3:
                assert input1.shape[0] == batch1 and input1.shape[1] == m and input1.shape[2] == k, "input1 shape check fail"
            else:
                assert 0, "input1 shape check fail"
        else:
            input1 = torch.rand(batch1, m, k)
        if input2 is not None:
            if len(input2.shape) == 2:
                assert input2.shape[0] == k and input2.shape[1] == n and batch2 == 1, "input2 shape check fail"
            elif len(input2.shape) == 3:
                assert input2.shape[0] == batch2 and input2.shape[1] == k and input2.shape[2] == n, "input2 shape check fail"
            else:
                assert 0, "input2 shape check fail"
        else:
            input2 = torch.rand(batch2, k, n)
        module = torch.matmul
        return checkFwAndBw(input1, input2,  bw_input, module)


def checkFwAndBw(input1, input2=None, bw_input=None, op=None):
    op_c = op
    input1_c = input1.requires_grad_(True)
    input1_g = input1.detach().clone().cuda().requires_grad_(True)

    bw_input_c = bw_input
    bw_input_g = bw_input.detach().clone().cuda()


    if input2 == None:  # linear
        op_g = copy.deepcopy(op).cuda()
        out_c = op(input1)
        out_g = op_g(input1_g.cuda())

    else:
        op_g = op
        input2_c = input2.requires_grad_(True)
        input2_g = input2.detach().clone().cuda().requires_grad_(True)
        out_c = op_c(input1_c, input2_c)
        out_g = op_g(input1_g, input2_g)

    fw_status = check_close(out_g.cpu(), out_c)

    out_g.backward(bw_input_g)
    out_c.backward(bw_input_c)

    input1_g_grad = input1_g.grad.cpu()
    input1_c_grad = input1_c.grad
    input1_bw_status = check_close(input1_g_grad, input1_c_grad)

    if input2 is None:  # linear
        input2_g_grad = op_g.weight.grad.cpu()
        input2_c_grad = op_c.weight.grad
    else:
        input2_g_grad = input2_g.grad.cpu()
        input2_c_grad = input2_c.grad

    input2_bw_status = check_close(input2_g_grad, input2_c_grad)

    print(f"$$$ forward check: {fw_status}")
    if(not fw_status):
        print(f"cpu forward result:\n {out_c}")
        print(f"gpu forward result:\n {out_g.cpu()}")


    print(f"$$$ input1 check: {input1_bw_status}")
    if(not input1_bw_status):
        print(f"cpu input1 backward result:\n {input1_c_grad}")
        print(f"gpu input1 backward result:\n {input1_g_grad}")

    print(f"$$$ input2 check: {input2_bw_status}")
    if(not input2_bw_status):
        print(f"cpu input2/weight backward result:\n {input2_c_grad}")
        print(f"gpu input2/weight backward result:\n {input2_g_grad}")

    print(f"### check result(fw_status, input1_bw_status, input2_bw_status/weight_bw_status): {fw_status, input1_bw_status, input2_bw_status}")
    return fw_status and input1_bw_status and input2_bw_status

class perfModeEnvGuard():
    '''
    class for auto set and restore environment variable `MACA_TORCH_PERF_MODE`
    Args:
        ops_env:new value to set for environment variable `MACA_TORCH_PERF_MODE`
    '''
    def __init__(self, ops_env):
        assert isinstance(ops_env, str)
        self.ops_env = ops_env
        if not self.ops_env:
            return
        self.exist_perf_env_flag = "MACA_TORCH_PERF_MODE" in os.environ.keys()
        if self.exist_perf_env_flag:
            self.src_perf_env = os.environ["MACA_TORCH_PERF_MODE"]
            if os.environ["MACA_TORCH_PERF_MODE"].endswith(","):
                os.environ["MACA_TORCH_PERF_MODE"] += self.ops_env
            else:
                os.environ["MACA_TORCH_PERF_MODE"] += ("," + self.ops_env)
        else:
            os.environ["MACA_TORCH_PERF_MODE"] = self.ops_env

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.ops_env:
            return
        if self.exist_perf_env_flag:
            os.environ["MACA_TORCH_PERF_MODE"] = self.src_perf_env
        else:
            del os.environ["MACA_TORCH_PERF_MODE"]

class EnvGuard:
    '''
    General class for auto setting and restoring environment variable`
    Args:
        env: Environment variable to be set and restored.
        flag: Flags to be setting to specific environment variable.
    '''
    def __init__(self, env, flag):
        assert isinstance(env, str)
        assert isinstance(flag, str)
        self.env = env
        self.flag =flag
        if not env or not flag:
            return
        self.stored_flag = os.environ[env] if env in os.environ.keys() else None
        os.environ[env] = flag

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.env or not self.flag:
            return
        if self.stored_flag:
            os.environ[self.env] = self.stored_flag
        else:
            del os.environ[self.env]

def selectPort(ip_addr, port):
    for _ in range(10):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((ip_addr, port))
            s.shutdown(2)
            print("port %d has been used." % port)
            port += 100
        except:
            print("port %d is unused." % port)
            return str(port)
    raise(Exception("Couldn't find an available port."))

class Result(Enum):
    Success = 0
    Error = 1
    OutMemory = 2


def make_pattern_tensor(shape, dtypes=[], output_strides=[], input_strides=[], offset = 0):
    # assert num_input == len(input_strides), "set input_strides.length == num_input"
    shape_mul = reduce(lambda x, y: x * y, shape)
    outputs = []
    if len(output_strides) > 0:
        for i in range(len(output_strides)):
            output_stride_max = max(np.array(shape) * np.array(output_strides[i]))
            output_ceil = math.ceil(output_stride_max / shape_mul)
            output_shape = copy.deepcopy(shape)
            output_shape[-1] = shape[-1] * output_ceil + offset
            output = torch.randn(output_shape, dtype=dtypes[0], device="cuda")
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
        input_shape[-1] = shape[-1] * input_ceil + offset
        if len(dtypes) > 1:
          input = torch.randn(input_shape, dtype=dtypes[i], device="cuda")
        else:
          input = torch.randn(input_shape, dtype=dtypes[0], device="cuda")
        input = input[..., offset:input_shape[-1] + offset]
        try:
            input = input.as_strided(shape, input_strides[i])
        except Exception:
            return [], []
        inputs.append(input)
    return outputs, inputs

def launch_coverage_test():
    def warper(func):
        @functools.wraps(func)
        def run(*args, **kwargs):
            shape, dtypes, test_not_align = kwargs["shape"], kwargs["dtypes"], kwargs["test_not_align"]
            output_strides, input_strides = kwargs["output_strides"], kwargs["input_strides"]
            if test_not_align:
                outputs, inputs = make_pattern_tensor(shape, dtypes, output_strides, input_strides, 3)
                if len(outputs) == 0 and len(inputs) == 0:
                    print(f"---------Skip test shape:{shape}, dtype:{dtypes[0]} due to as strided fail!")
                    return Result.OutMemory
                return func(shape=shape, dtypes=dtypes, outputs=outputs, inputs=inputs)
            else:
                outputs, inputs = make_pattern_tensor(shape, dtypes, output_strides, input_strides)
                if len(outputs) == 0 and len(inputs) == 0:
                    print(f"---------Skip test shape:{shape}, dtype:{dtypes[0]} due to as strided fail!")
                    return Result.OutMemory
                return func(shape=shape, dtypes=dtypes, outputs=outputs, inputs=inputs)
        return run
    return warper

@launch_coverage_test()
def test_copy(*args, **kwargs):
    shape, dtypes, outputs, inputs = kwargs["shape"], kwargs["dtypes"], kwargs["outputs"], kwargs["inputs"]
    output, output_c = outputs[0], outputs[0].cpu()
    input, input_c = inputs[0], inputs[0].cpu()

    output.copy_(input)
    output_c.copy_(input_c)

    if not torch.allclose(output.cpu(), output_c):
        print(f"--------Error with shape:{shape}, dtype:{dtypes[0]}")
        return Result.Error
    
    return Result.Success

@launch_coverage_test()
def test_add(*args, **kwargs):
    shape, dtypes, inputs = kwargs["shape"], kwargs["dtypes"], kwargs["inputs"]
    a, a_c = inputs[0], inputs[0].cpu()
    b, b_c = inputs[1], inputs[1].cpu()

    output = a + b
    output_c = a_c + b_c

    if not torch.allclose(output.cpu(), output_c):
        print(f"--------Error with shape:{shape}, dtype:{dtypes[0]}")
        return Result.Error
    
    return Result.Success


@launch_coverage_test()
def test_mul(*args, **kwargs):
    shape, dtypes, inputs = kwargs["shape"], kwargs["dtypes"], kwargs["inputs"]
    a, a_c = inputs[0], inputs[0].cpu()
    b, b_c = inputs[1], inputs[1].cpu()

    output = a * b
    output_c = a_c * b_c

    if not torch.allclose(output.cpu(), output_c):
        print(f"--------Error with shape:{shape}, dtype:{dtypes[0]}")
        return Result.Error
    
    return Result.Success

@launch_coverage_test()
def test_sub(*args, **kwargs):
    shape, dtypes, inputs = kwargs["shape"], kwargs["dtypes"], kwargs["inputs"]
    a, a_c = inputs[0], inputs[0].cpu()
    b, b_c = inputs[1], inputs[1].cpu()

    output = a - b
    output_c = a_c - b_c

    if not torch.allclose(output.cpu(), output_c):
        print(f"--------Error with shape:{shape}, dtype:{dtypes[0]}")
        return Result.Error
    
    return Result.Success

@launch_coverage_test()
def test_sub_inplace(*args, **kwargs):
    shape, dtypes, inputs = kwargs["shape"], kwargs["dtypes"], kwargs["inputs"]
    a, a_c = inputs[0], inputs[0].cpu()
    b, b_c = inputs[1], inputs[1].cpu()

    a -= b
    a_c -= b_c

    if not torch.allclose(a.cpu(), a_c):
        print(f"--------Error with shape:{shape}, dtype:{dtypes[0]}")
        return Result.Error
    
    return Result.Success