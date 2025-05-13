import torch
import torch.nn as nn
import os
import copy
import argparse
import math
from functools import reduce
import numpy as np
import functools
from enum import Enum

torch.manual_seed(0)
torch.set_printoptions(precision=5, threshold=10000)

class Result(Enum):
    Success = 0
    Error = 1
    OutMemory = 2


def make_pattern_tensor(shape, dtype, output_strides=[], input_strides=[], offset = 0):
    # assert num_input == len(input_strides), "set input_strides.length == num_input"
    shape_mul = reduce(lambda x, y: x * y, shape)
    outputs = []
    if len(output_strides) > 0:
        for i in range(len(output_strides)):
            output_stride_max = max(np.array(shape) * np.array(output_strides[i]))
            output_ceil = math.ceil(output_stride_max / shape_mul)
            output_shape = copy.deepcopy(shape)
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
        input_shape[-1] = shape[-1] * input_ceil + offset
        input = torch.randn(input_shape, dtype=dtype, device="cuda")
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
            shape, dtype, test_not_align = kwargs["shape"], kwargs["dtype"], kwargs["test_not_align"]
            output_strides, input_strides = kwargs["output_strides"], kwargs["input_strides"]
            if test_not_align:
                outputs, inputs = make_pattern_tensor(shape, dtype, output_strides, input_strides, 3)
                if len(outputs) == 0 and len(inputs) == 0:
                    print(f"---------Skip test shape:{shape}, dtype:{dtype} due to as strided fail!")
                    return Result.OutMemory
                return func(shape=shape, dtype=dtype, outputs=outputs, inputs=inputs)
            else:
                outputs, inputs = make_pattern_tensor(shape, dtype, output_strides, input_strides)
                if len(outputs) == 0 and len(inputs) == 0:
                    print(f"---------Skip test shape:{shape}, dtype:{dtype} due to as strided fail!")
                    return Result.OutMemory
                return func(shape=shape, dtype=dtype, outputs=outputs, inputs=inputs)
        return run
    return warper

@launch_coverage_test()
def test_copy(*args, **kwargs):
    shape, dtype, outputs, inputs = kwargs["shape"], kwargs["dtype"], kwargs["outputs"], kwargs["inputs"]
    output, output_c = outputs[0], outputs[0].cpu()
    input, input_c = inputs[0], inputs[0].cpu()

    output.copy_(input)
    output_c.copy_(input_c)

    if not torch.allclose(output.cpu(), output_c):
        print(f"--------Error with shape:{shape}, dtype:{dtype}")
        return Result.Error
    
    return Result.Success

@launch_coverage_test()
def test_add(*args, **kwargs):
    shape, dtype, inputs = kwargs["shape"], kwargs["dtype"], kwargs["inputs"]
    a, a_c = inputs[0], inputs[0].cpu()
    b, b_c = inputs[1], inputs[1].cpu()

    output = a + b
    output_c = a_c + b_c

    if not torch.allclose(output.cpu(), output_c):
        print(f"--------Error with shape:{shape}, dtype:{dtype}")
        return Result.Error
    
    return Result.Success

def test_elementwise_3_1():
    start, end, step = 2, 500, 33
    align_error_test, not_align_error_test, tests, out_memory_test, success_test = 0, 0, 0, 0, 0
    dtypes = [torch.float16, torch.float32]
    for dtype in dtypes:
        # for s0 in range(end-1, start-1, -step):
        for s0 in [256, 32, 64, 128]:
            for s1 in range(end-1, start-1, -step):
                for s2 in range(end-1, start-1, -step):
                    shape = [s0, s1, s2]
                    ###################################### modify the stride by your self ######################################

                    # swin-transformer 3-1 pattern
                    # shape = [s0, s1, s2+s0*4*s1]
                    # output_strides = [(1, shape[0] * 4, shape[2])]
                    # input_strides = [(1, shape[0] * 2, shape[2])]

                    output_strides = [(1, shape[0] * 4, shape[2] * 2)]
                    input_strides = [(1, shape[0] * 2, shape[2] * 2)]

                    # output_strides = [(1, shape[0] * 4, shape[2] * 4)]
                    # input_strides = [(1, shape[0] * 2, shape[2] * 4)]
                    # end swin-transformer

                    # longformer 3-1 pattern
                    output_strides = [(1, 513, shape[0]*2*513)]
                    input_strides = [(1, 513, shape[0]*2*513)]

                    output_strides = [(1, shape[0], shape[0] * shape[1])]
                    input_strides = [(1, shape[0] // 3, shape[0] // 3 * 4)]

                    output_strides = [(1, shape[0], shape[0] * shape[1])]
                    input_strides = [(1, shape[0] * shape[2], shape[0])]

                    output_strides = [(1, shape[0], shape[0] * shape[1] * 2)]
                    input_strides = [(1, shape[0] * shape[2], shape[0])]
                    # end longformer

                    ############################################################################################################
                    tests += 1
                    align_result = test_copy(shape=shape, dtype=dtype, test_not_align=False, output_strides=output_strides, input_strides=input_strides)
                    if align_result == Result.Error:
                        align_error_test += 1
                    elif align_result == Result.OutMemory:
                        out_memory_test += 1
                    else:
                        success_test += 1
                    tests += 1
                    not_align_result = test_copy(shape=shape, dtype=dtype, test_not_align=True, output_strides=output_strides, input_strides=input_strides)
                    if not_align_result == Result.Error:
                        not_align_error_test += 1
                    elif not_align_result == Result.OutMemory:
                        out_memory_test += 1
                    else:
                        success_test += 1
    print(f"-----------All tests:{tests}")
    print(f"-----------Success tests:{success_test}")
    print(f"-----------OutMemory tests:{out_memory_test}")
    print(f"-----------Align error test:{align_error_test}")
    print(f"-----------Not align error test:{not_align_error_test}")

def test_elementwise_4_1():
    start, end, step = 2, 200, 3
    align_error_test, not_align_error_test, tests, out_memory_test, success_test = 0, 0, 0, 0, 0
    dtypes = [torch.float16, torch.float32]
    for dtype in dtypes:
        # for s0 in range(end-1, start-1, -step):
        for s0 in [256, 32, 64, 128]:
            for s1 in range(end-1, start-1, -step):
                for s2 in range(end-1, start-1, -step):
                    for s3 in range(end-1, start-1, -step):
                        ###################################### modify the stride by your self ######################################

                        # swin-transformer 4-1 pattern
                        shape = [s0, s1, s2, s3]
                        output_strides = [(1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2])]
                        input_strides = [(1, shape[0] * shape[2] * 3, shape[0], shape[0] * shape[1] * shape[2] * 3)]
                        # end swin-transformer

                        # llama-7b 4-1 pattern
                        output_strides = [(1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2])]
                        input_strides = [(1, shape[0] * 348, shape[0], shape[0] * shape[1] * 348)]

                        output_strides = [(1, shape[0] * 348, shape[0], shape[0] * shape[1] * 348)]
                        input_strides = [(1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2])]

                        output_strides = [(1, shape[0] * 2, shape[0] * shape[1] * 2, shape[0] * shape[1] * shape[2] * 2)]
                        input_strides = [(1, shape[0] * shape[2], shape[0], shape[0] * shape[1] * shape[2])]

                        output_strides = [(1, shape[0] * 2, shape[0] * shape[1] * 2, shape[0] * shape[1] * shape[2] * 2)]
                        input_strides = [(1, shape[0] * shape[2] * 2, shape[0] * 2, shape[0] * shape[1] * shape[2] * 2)]

                        output_strides = [(1, shape[0], shape[0] * 348, shape[0] * shape[1] * 1392)]
                        input_strides = [(1, shape[0] * 2048, shape[0], shape[0] * shape[1] * 2048)]
                        # end llama-7b

                        ############################################################################################################
                        tests += 1
                        align_result = test_copy(shape=shape, dtype=dtype, test_not_align=False, output_strides=output_strides, input_strides=input_strides)
                        if align_result == Result.Error:
                            align_error_test += 1
                        elif align_result == Result.OutMemory:
                            out_memory_test += 1
                        else:
                            success_test += 1
                        tests += 1
                        not_align_result = test_copy(shape=shape, dtype=dtype, test_not_align=True, output_strides=output_strides, input_strides=input_strides)
                        if not_align_result == Result.Error:
                            not_align_error_test += 1
                        elif not_align_result == Result.OutMemory:
                            out_memory_test += 1
                        else:
                            success_test += 1
    print(f"-----------All tests:{tests}")
    print(f"-----------Success tests:{success_test}")
    print(f"-----------OutMemory tests:{out_memory_test}")
    print(f"-----------Align error test:{align_error_test}")
    print(f"-----------Not align error test:{not_align_error_test}")

def test_elementwise_4_2():
    start, end, step = 2, 200, 3
    align_error_test, not_align_error_test, tests, out_memory_test, success_test = 0, 0, 0, 0, 0
    dtypes = [torch.float16, torch.float32]
    for dtype in dtypes:
        # for s0 in range(end-1, start-1, -step):
        for s0 in [256, 32, 64, 128]:
            for s1 in range(end-1, start-1, -step):
                for s2 in range(end-1, start-1, -step):
                    for s3 in range(end-1, start-1, -step):
                        ###################################### modify the stride by your self ######################################

                        # llama-7b 4-2 pattern
                        shape = [s0, s1, s2, s3]
                        output_strides = []
                        input_strides = [(1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2]), 
                                         (1, shape[0] * shape[2], shape[0], shape[0] * shape[1] * shape[2])]
                        # end llama-7b

                        ############################################################################################################
                        tests += 1
                        align_result = test_add(shape=shape, dtype=dtype, test_not_align=False, output_strides=output_strides, input_strides=input_strides)
                        if align_result == Result.Error:
                            align_error_test += 1
                        elif align_result == Result.OutMemory:
                            out_memory_test += 1
                        else:
                            success_test += 1
                        tests += 1
                        not_align_result = test_add(shape=shape, dtype=dtype, test_not_align=True, output_strides=output_strides, input_strides=input_strides)
                        if not_align_result == Result.Error:
                            not_align_error_test += 1
                        elif not_align_result == Result.OutMemory:
                            out_memory_test += 1
                        else:
                            success_test += 1
    print(f"-----------All tests:{tests}")
    print(f"-----------Success tests:{success_test}")
    print(f"-----------OutMemory tests:{out_memory_test}")
    print(f"-----------Align error test:{align_error_test}")
    print(f"-----------Not align error test:{not_align_error_test}")


if __name__ == "__main__":
    test_elementwise_3_1()
    test_elementwise_4_1()
    test_elementwise_4_2()