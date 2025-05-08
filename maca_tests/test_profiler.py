import torch
import torch.nn as nn


input_cpu_1 = torch.rand(1,8,4,4, requires_grad=True)
input_cpu_2 = torch.rand(1,8,4,4, requires_grad=True)
input1_cuda = input_cpu_1.cuda()
input2_cuda = input_cpu_2.cuda()
flag_export_chrome_trace = False


def test_func():
    global input1_cuda
    global input2_cuda
    out = input1_cuda + input2_cuda
    a = out.sum()
    return a


def test_profiler_profile_cupti():
    def trace_handler(prof):
        print(prof.key_averages().table())
        if flag_export_chrome_trace:
            prof.export_chrome_trace("trace_maca_test_profiler_profile_cupti.json")

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=trace_handler) as prof:
        out_cuda = test_func()

    found_vectorized_elementwise_kernel = False
    found_reduce_kernel = False
    found_LaunchKernel = False
    for evt in prof.events():
        if "vectorized_elementwise_kernel" in evt.name and evt.device_type == torch.profiler.DeviceType.CUDA:
            found_vectorized_elementwise_kernel = True
        if "reduce_kernel" in evt.name and evt.device_type == torch.profiler.DeviceType.CUDA:
            found_reduce_kernel = True
        if "LaunchKernel" in evt.name and evt.device_type == torch.profiler.DeviceType.CPU:
            found_LaunchKernel = True
    return found_vectorized_elementwise_kernel and found_reduce_kernel and found_LaunchKernel


def test_autograd_profiler_profile_cupti():
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, use_kineto=True, use_cpu=True) as prof:
        out_cuda = test_func()
    
    print(prof.key_averages().table())
    if flag_export_chrome_trace:
        prof.export_chrome_trace("trace_maca_test_autograd_profiler_profile_cupti.json")

    found_vectorized_elementwise_kernel = False
    found_reduce_kernel = False
    found_LaunchKernel = False
    for evt in prof.function_events:
        if "vectorized_elementwise_kernel" in evt.name and evt.device_type == torch.profiler.DeviceType.CUDA:
            found_vectorized_elementwise_kernel = True
        if "reduce_kernel" in evt.name and evt.device_type == torch.profiler.DeviceType.CUDA:
            found_reduce_kernel = True
        if "LaunchKernel" in evt.name and evt.device_type == torch.profiler.DeviceType.CPU:
            found_LaunchKernel = True
    return found_vectorized_elementwise_kernel and found_reduce_kernel and found_LaunchKernel


def test_autograd_profiler_profile_no_cupti():
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, use_kineto=False, use_cpu=True) as prof:
        out_cuda = test_func()
    
    print(prof.key_averages().table())
    if flag_export_chrome_trace:
        prof.export_chrome_trace("trace_cuda_test_autograd_profiler_profile_no_cupti.json")

    found_aten_add = False
    found_aten_sum = False
    found_aten_as_strided = False
    found_LaunchKernel = False
    for evt in prof.function_events:
        if "aten::add" in evt.name and evt.cuda_time>0:
            found_aten_add = True
        if "aten::sum" in evt.name and evt.cuda_time>0:
            found_aten_sum = True
        if "aten::as_strided" in evt.name and evt.cuda_time>0:
            found_aten_as_strided = True
        if "LaunchKernel" in evt.name and evt.device_type == torch.profiler.DeviceType.CPU:
            found_LaunchKernel = True
    return found_aten_add and found_aten_sum and found_aten_as_strided and found_LaunchKernel


ret = True
ret = ret and test_profiler_profile_cupti()
ret = ret and test_autograd_profiler_profile_cupti()
ret = ret and test_autograd_profiler_profile_no_cupti()

if ret:
    print("test_profiler.py passed!")
    exit(0)
else:
    print("test_profiler.py failed!")
    exit(1)