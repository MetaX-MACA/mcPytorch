import numpy as np
import torch
import torch.nn as nn

def trace_handler(prof):
    print(prof.key_averages(group_by_input_shape=True,
                            group_by_stack_n=1).table(sort_by="self_cuda_time_total",
                                                      max_name_column_width=10000,
                                                      max_src_column_width=10000,
                                                      row_limit = -1))



def test_arity1():
    inp0_base = torch.rand(100000000,device="cuda",dtype=torch.float)
    out_base = torch.rand(100000000,device="cuda",dtype=torch.float)
    inp = inp0_base.as_strided((100,20,30),(1300,60,1))
    inp1 = out_base.as_strided((100,20,30),(1400,70,1))
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        on_trace_ready=trace_handler) as prof:
       # out=torch.pow(inp,exponent=2)
       # out=torch.rsqrt(inp)
       # out=torch.add(inp,1)
       # out=torch.add(1,inp)
        out=torch.add(inp,inp1)
       # out=torch.exp(inp)
       # out=torch.log(inp)
       # out = torch.div(inp,inp1,rounding_mode="trunc")
       # out = torch.div(inp,inp1,rounding_mode="floor")
       # out = torch.div(inp,0,rounding_mode="trunc")

       # out = torch.div(inp,0,rounding_mode="floor")
       # out = torch.div(inp,0)
       # out = torch.div(inp,1,rounding_mode="floor")
       # out = torch.div(inp,1)

       # out = torch.remainder(inp, 0.1)
       # out = torch.remainder(inp, inp1)
       # out = torch.fmod(inp, 0.1)
       # out = torch.fmod(inp, inp1)
       # out = torch.eq(inp,0.1)
       # out=torch.sqrt(inp)
       # out = torch.reciprocal(inp)
test_arity1()

def test_bitwise_or():
    a = torch.tensor([-1,-2,-3],dtype=torch.int8, device="cuda")
    b = torch.tensor([1,0,3],dtype=torch.int8, device="cuda")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        on_trace_ready=trace_handler) as prof:
        out = torch.bitwise_or(a,b)
        out = torch.bitwise_xor(a,b)
        out = torch.bitwise_and(a,b)

        out = torch.logical_and(a,b)
        out = torch.logical_or(a,b)
        out = torch.logical_xor(a,b)
        out = torch.logical_not(a)

#test_bitwise_or()

def test_mse():
    inp0_base = torch.rand(100000000,device="cuda",dtype=torch.float)
    out_base = torch.rand(100000000,device="cuda",dtype=torch.float)
    inp = inp0_base.as_strided((100,20,30),(1300,60,1))
    inp1 = out_base.as_strided((100,20,30),(1400,70,1))
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        on_trace_ready=trace_handler) as prof:
        out = nn.MSELoss()(inp,inp1)
        out.backward(torch.ones(out.shape,device="cuda"))

