import numpy as np
import torch
import torch.nn as nn
from kernel_map import *


def trace_handler(prof):
    print(prof.key_averages(group_by_input_shape=True,
                            group_by_stack_n=1).table(sort_by="self_cuda_time_total",
                                                      max_name_column_width=10000,
                                                      max_src_column_width=10000,
                                                      row_limit = -1))

def transform_time(time_s):
    if "ms" in time_s:
        return float(time_s[:-2])*1000
    elif "us" in time_s:
        return float(time_s[:-2])
    elif "s" in time_s:
        return float(time_s[:-1])*1000000
    else:
        print("time is error")
        exit(1)


def transform_str(time_s):
    if "ms" in time_s:
        return float(time_s[:-2])*1000
    elif "us" in time_s:
        return float(time_s[:-2])
    elif "s" in time_s:
        return float(time_s[:-1])*1000000
    elif time_s == "__":
        return 0.0
    else:
        return float(time_s)


def is_legal_elementwise_info(line):
    if not line.startswith("p_e_"):
        return False
    if line[-1] != ",":
        return False

    line_list = line.split(",")
    if len(line_list) <= 3:
        return False

    #判断len(line_list)
    dim = int(line_list[1])
    arity = int(line_list[2])
    if dim == 0 or arity == 0:
        return False

    #判断长度
    num = dim * (arity + 2) + (arity + 1) + 4
    if num != len(line_list):
        return False

    #中间必须为整数
    begin = 1
    end = 1 + dim * (arity + 2) + 2
    for i in range(begin, end):
        item = line_list[i]
        if not str(item).isdigit():
            return False

    #最后必须为数据类型
    begin = end
    end   = len(line_list)-1
    for i in range(begin, end):
        if line_list[i] not in Dtype_bytes:
            return False

    return True


def get_tensor(shape_s, stride_s, dtype, need_grads = False):
    if dtype not in Dtype_bytes:
        print("unexpect dtype in get_tensor")
        exit(1)

    shape = [int(shape_s[i]) for i in range(len(shape_s))]
    stride = [int(int(stride_s[i])/Dtype_bytes[dtype]) for i in range(len(shape_s))]
    shape = shape[::-1]
    stride = stride[::-1]

    size = 0
    for i in range(len(shape)):
        if (shape[i]*stride[i])>size:
            size = shape[i]*stride[i]
    size = int(size * 1.3) + 64

    if dtype == "Byte":
        a = torch.randint(0,127,(size,),device="cuda").to(dtype=torch.uint8)
        return a.as_strided(shape, stride)
    elif dtype == "Char":
        a = torch.randint(-127,127,(size,),device="cuda").to(dtype=torch.int8)
        return a.as_strided(shape, stride)
    elif dtype == "Short":
        a = torch.randint(-127,127,(size,),device="cuda").to(dtype=torch.int16)
        return a.as_strided(shape, stride)
    elif dtype == "Int":
        a = torch.randint(-127,127,(size,),device="cuda")
        return a.as_strided(shape, stride)
    elif dtype == "BFloat16":
        a = torch.rand(size, dtype=torch.bfloat16, device="cuda")
        if need_grads:
            a.requires_grad = True
        return a.as_strided(shape, stride)
    elif dtype == "Long":
        a = torch.randint(-127,127,(size,), device="cuda").to(dtype=torch.int64)
        return a.as_strided(shape, stride)
    elif dtype == "Half":
        a = torch.rand(size, dtype=torch.float16, device="cuda")
        if need_grads:
            a.requires_grad = True
        return a.as_strided(shape, stride)
    elif dtype == "Float":
        a = torch.rand(size, dtype=torch.float32, device="cuda")
        if need_grads:
            a.requires_grad = True
        return a.as_strided(shape, stride)
    elif dtype == "Double":
        a = torch.rand(size, dtype=torch.float64, device="cuda")
        if need_grads:
            a.requires_grad = True
        return a.as_strided(shape, stride)
    elif dtype == "Bool":
        a = torch.randint(0,2,(size,), device="cuda").to(dtype=torch.bool)
        return a.as_strided(shape, stride)
    else:
        print("unexpect dtype in get_tensor")
        exit(1)


def save_tmp_log(out_path, line_list, sort = False, sortIdx = -1, reverse = True):
    save_list = []
    for i in range(len(line_list)):
        save_list.append(line_list[i].split())

    head_list = save_list[0:1]
    other_list = save_list[1:]
    if sort:
        other_list = sorted(other_list, key = lambda x: transform_str(x[sortIdx]), reverse = reverse)


    maxlen_list=[]
    for i in range(len(save_list[0])):
        maxlen=0
        for j in range(len(save_list)):
            maxlen = max(maxlen, len(save_list[j][i]))
        maxlen_list.append(maxlen+6)

    with open(out_path,"w") as f:
        for i in range(len(head_list)):
            line=""
            for j in range(len(head_list[0])):
                line+=head_list[i][j]
                line+=(maxlen_list[j] - len(head_list[i][j]))*" "
            f.write(line+"\n")

    with open(out_path,"a") as f:
        for i in range(len(other_list)):
            line=""
            for j in range(len(other_list[0])):
                line+=other_list[i][j]
                line+=(maxlen_list[j] - len(other_list[i][j]))*" "
            f.write(line+"\n")


def get_middle_log(line, info):
    line_list = line.split(",")
    NumOfCalls = info["NumOfCalls"]

    dim = int(line_list[1])
    arity = int(line_list[2])

    #shape index
    begin = 3
    end   = 1 + dim * (arity + 2) + 2
    shape = ",".join(line_list[begin:end])

    #dtype
    begin = end
    end   = len(line_list) - 1
    dtype = ",".join(line_list[begin:end])

    eleInfo = line_list[0]

    isGetTime = "1"
    isOpt = "1"
    func = "__"
    substride = "__"
    if eleInfo.startswith("p_e_noopt"):
        isOpt = "0"


    if arity == 1:
        for art1map in [arity1_tensor_map, arity1_torch_map]:
            for k,v in art1map.items():
                klist = k.split("&&")
                isfunc = True
                for k0 in klist:
                    if k0 not in eleInfo:
                        isfunc = False
                if  isfunc:
                    func = k
                    if "substitue" in v:
                        substride = v["substitue"]
        if func == "__":
            isGetTime = "0"
        return isGetTime+"  "+str(dim)+" "+str(arity)+" "+shape+"  "+dtype+"  "+isOpt+" "+func+"  "+substride+"  "+str(NumOfCalls)+"  "+eleInfo

    if arity == 2:
        for art2map in [arity2_tensor_map, arity2_torch_map]:
            for k,v in art2map.items():
                klist = k.split("&&")
                isfunc = True
                for k0 in klist:
                    if k0 not in eleInfo:
                        isfunc = False
                if  isfunc:
                    func = k
                    if "substitue" in v:
                        substride = v["substitue"]
        if func == "__":
            isGetTime = "0"
        return isGetTime+"  "+str(dim)+" "+str(arity)+" "+shape+"  "+dtype+"  "+isOpt+" "+func+"  "+substride+"  "+str(NumOfCalls)+"  "+eleInfo


    if arity == 3:
        for art3map in [arity3_torch_map]:
            for k,v in art3map.items():
                klist = k.split("&&")
                isfunc = True
                for k0 in klist:
                    if k0 not in eleInfo:
                        isfunc = False
                if  isfunc:
                    func = k
                    if "substitue" in v:
                        substride = v["substitue"]
        if func == "__":
            isGetTime = "0"
        return isGetTime+"  "+str(dim)+" "+str(arity)+" "+shape+"  "+dtype+"  "+isOpt+" "+func+"  "+substride+"  "+str(NumOfCalls)+"  "+eleInfo

    isGetTime = "0"
    return isGetTime+"  "+str(dim)+" "+str(arity)+" "+shape+"  "+dtype+"  "+isOpt+" "+func+"  "+substride+"  "+str(NumOfCalls)+"  "+eleInfo

def resolve_time(line):
    str_val = line.strip().split()[-2]
    if "us" in str_val or "ms" in str_val:
        return str_val
    elif "ns" in str_val:
        val =  float(str_val[:-2]) / 1000
        val = str(val) + "us"
        return str_val
    elif "s" in str_val:
        val = float(str_val[:-1]) * 1000
        val = str(val) + "ms"
        return str_val
    else:
        print("resolve_time error")
        exit(1)


def resolve_kernel(line):
    begin=line.find("at::native::")
    end=line.find("<")
    if begin < 0 or begin >= len(line) or end < 0 or end > len(line):
        print("resolve_kernel error")
        exit(1)
    return line[begin+len("at::native::"):end]

def extract_kernel_time(info, key):
    key_list = key.split("&&")
    key_list.insert(0,"elementwise")
    line_list = info.split("\n")
    for line in line_list:
        ismatch = True
        for key1 in key_list:
            if key1 not in line:
                ismatch = False
        if ismatch:
            time = resolve_time(line)
            kernel = resolve_kernel(line)
            return kernel, time
    print("extract_kernel_time error")
    exit(1)

warm_ups = 200

def get_arity1_tensorFunc_kernelTime(middle_line_list):
    isGetTime, dim, arity, shapeStride, dtype, isOpt, func, substride, NumOfCalls,eleInfo = middle_line_list
    dim = int(dim)

    shape = shapeStride.split(",")[0:dim]
    outStride = shapeStride.split(",")[dim:dim*2]
    inpStride = shapeStride.split(",")[dim*2:dim*3]
    outDtype = dtype.split(",")[0]
    inpDtype = dtype.split(",")[1]

    cache = torch.empty(int(256e6 // 4), dtype = torch.int, device = "cuda")

    func_v = arity1_tensor_map[func]
    if func == "direct_copy_kernel_cuda":
        if outDtype == "Int" and inpDtype == "Long":
            outDtype = "Float"
            inpDtype = "Double"
        elif outDtype == "Long" and inpDtype == "Int":
            outDtype = "Double"
            inpDtype == "Float"

    if func_v["func"] == "backward":
        outTensor = get_tensor(shape, outStride, outDtype, True)
        inpTensor = get_tensor(shape, inpStride, inpDtype, True)

        #warpup
        for i in range(warm_ups):
            forward_out = func_v["forward_func"](inpTensor, **func_v["extra_param"])
            forward_out.backward(outTensor, retain_graph=True)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=trace_handler) as prof:
            for i in range(warm_ups):
                cache.zero_()
                forward_out = func_v["forward_func"](inpTensor, **func_v["extra_param"])
                forward_out.backward(outTensor, retain_graph=True)

        info = prof.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        key = func
        if "substitue" in func_v:
            key = func_v["substitue"]
        return extract_kernel_time(info,key)

    else:
        outTensor = get_tensor(shape, outStride, outDtype, False)
        inpTensor = get_tensor(shape, inpStride, inpDtype, False)
        f = "outTensor." + func_v["func"]
        #warpup
        for i in range(warm_ups):
            eval(f)(inpTensor, **func_v["extra_param"])

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=trace_handler) as prof:
            for i in range(warm_ups):
                cache.zero_()
                eval(f)(inpTensor, **func_v["extra_param"])

        info = prof.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        key = func
        if "substitue" in func_v:
            key = func_v["substitue"]
        return extract_kernel_time(info, key)


def get_arity1_torchFunc_kernelTime(middle_line_list):
    isGetTime, dim, arity, shapeStride, dtype, isOpt, func, substride, NumOfCalls,eleInfo = middle_line_list
    dim = int(dim)

    shape = shapeStride.split(",")[0:dim]
    outStride = shapeStride.split(",")[dim:dim*2]
    inpStride = shapeStride.split(",")[dim*2:dim*3]
    outDtype = dtype.split(",")[0]
    inpDtype = dtype.split(",")[1]

    outTensor = get_tensor(shape, outStride, outDtype, False)
    inpTensor = get_tensor(shape, inpStride, inpDtype, False)
    
    cache = torch.empty(int(256e6 // 4), dtype = torch.int, device = "cuda")

    func_v = arity1_torch_map[func]
    #warpup
    for i in range(warm_ups):
        out = func_v["func"](inpTensor, **func_v["extra_param"])

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=trace_handler) as prof:
            for i in range(warm_ups):
                cache.zero_()
                out = func_v["func"](inpTensor, **func_v["extra_param"])

    info = prof.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
    key = func
    if "substitue" in func_v:
        key = func_v["substitue"]
    return extract_kernel_time(info, key)


def get_arity2_tensorFunc_kernelTime(middle_line_list):
    isGetTime, dim, arity, shapeStride, dtype, isOpt, func, substride, NumOfCalls,eleInfo = middle_line_list
    dim = int(dim)

    shape = shapeStride.split(",")[0:dim]
    outStride = shapeStride.split(",")[dim:dim*2]
    inp0Stride = shapeStride.split(",")[dim*2:dim*3]
    inp1Stride = shapeStride.split(",")[dim*3:dim*4]
    outDtype = dtype.split(",")[0]
    inp0Dtype = dtype.split(",")[1]
    inp1Dtype = dtype.split(",")[2]
    
    cache = torch.empty(int(256e6 // 4), dtype = torch.int, device = "cuda")

    func_v = arity2_tensor_map[func]
    if func_v["func"] == "backward":
        outTensor = get_tensor(shape, outStride, outDtype, True)
        inp0Tensor = get_tensor(shape, inp0Stride, inp0Dtype, True)
        inp1Tensor = get_tensor(shape, inp1Stride, inp1Dtype, True)

        #warpup
        for i in range(warm_ups):
            forward_out = func_v["forward_func"](inp1Tensor, **func_v["extra_param"])
            forward_out.backward(inp0Tensor, retain_graph=True)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=trace_handler) as prof:
            for i in range(warm_ups):
                cache.zero_()
                forward_out = func_v["forward_func"](inp1Tensor, **func_v["extra_param"])
                forward_out.backward(inp0Tensor, retain_graph=True)

        info = prof.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        key = func
        if "substitue" in func_v:
            key = func_v["substitue"]
        return extract_kernel_time(info,key)

    else:
        outTensor = get_tensor(shape, outStride, outDtype, False)
        inp0Tensor = get_tensor(shape, inp0Stride, inp0Dtype, False)
        inp1Tensor = get_tensor(shape, inp1Stride, inp1Dtype, False)
        
        f = "inp0Tensor." + func_v["func"]
        #warpup
        for i in range(warm_ups):
            eval(f)(inp1Tensor, **func_v["extra_param"])

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=trace_handler) as prof:
            for i in range(warm_ups):
                cache.zero_()
                eval(f)(inp1Tensor, **func_v["extra_param"])

        info = prof.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        key = func
        if "substitue" in func_v:
            key = func_v["substitue"]
        return extract_kernel_time(info, key)


def get_arity2_torchFunc_kernelTime(middle_line_list):
    isGetTime, dim, arity, shapeStride, dtype, isOpt, func, substride, NumOfCalls,eleInfo = middle_line_list
    dim = int(dim)

    shape = shapeStride.split(",")[0:dim]
    outStride = shapeStride.split(",")[dim:dim*2]
    inp0Stride = shapeStride.split(",")[dim*2:dim*3]
    inp1Stride = shapeStride.split(",")[dim*3:dim*4]
    outDtype = dtype.split(",")[0]
    inp0Dtype = dtype.split(",")[1]
    inp1Dtype = dtype.split(",")[2]

    outTensor = get_tensor(shape, outStride, outDtype, False)
    inp0Tensor = get_tensor(shape, inp0Stride, inp0Dtype, False)
    inp1Tensor = get_tensor(shape, inp1Stride, inp1Dtype, False)
    
    cache = torch.empty(int(256e6 // 4), dtype = torch.int, device = "cuda")

    func_v = arity2_torch_map[func]
    #warpup
    for i in range(warm_ups):
        out = func_v["func"](inp0Tensor,inp1Tensor, **func_v["extra_param"])

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=trace_handler) as prof:
            for i in range(warm_ups):
                cache.zero_()
                out = func_v["func"](inp0Tensor,inp1Tensor, **func_v["extra_param"])

    info = prof.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
    key = func
    if "substitue" in func_v:
        key = func_v["substitue"]
    return extract_kernel_time(info, key)


def get_arity3_tensorFunc_kernelTime(middle_line_list):
    isGetTime, dim, arity, shapeStride, dtype, isOpt, func, substride, NumOfCalls,eleInfo = middle_line_list
    dim = int(dim)

    shape = shapeStride.split(",")[0:dim]
    outStride = shapeStride.split(",")[dim:dim*2]
    inp0Stride = shapeStride.split(",")[dim*2:dim*3]
    inp1Stride = shapeStride.split(",")[dim*3:dim*4]
    inp2Stride = shapeStride.split(",")[dim*4:dim*5]
    outDtype = dtype.split(",")[0]
    inp0Dtype = dtype.split(",")[1]
    inp1Dtype = dtype.split(",")[2]
    inp2Dtype = dtype.split(",")[3]
    
    cache = torch.empty(int(256e6 // 4), dtype = torch.int, device = "cuda")

    func_v = arity3_tensor_map[func]
    if func_v["func"] == "backward":
        outTensor = get_tensor(shape, outStride, outDtype, True)
        inp0Tensor = get_tensor(shape, inp0Stride, inp0Dtype, True)
        inp1Tensor = get_tensor(shape, inp1Stride, inp1Dtype, True)
        inp2Tensor = get_tensor(shape, inp2Stride, inp2Dtype, True)

        #warpup
        for i in range(warm_ups):
            forward_out = func_v["forward_func"](inp1Tensor,inp2Tensor, **func_v["extra_param"])
            forward_out.backward(inp0Tensor, retain_graph=True)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=trace_handler) as prof:
            for i in range(warm_ups):
                cache.zero_()
                forward_out = func_v["forward_func"](inp1Tensor,inp2Tensor, **func_v["extra_param"])
                forward_out.backward(inp0Tensor, retain_graph=True)

        info = prof.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        key = func
        if "substitue" in func_v:
            key = func_v["substitue"]
        return extract_kernel_time(info,key)

    else:
        outTensor = get_tensor(shape, outStride, outDtype, False)
        inp0Tensor = get_tensor(shape, inp0Stride, inp0Dtype, False)
        inp1Tensor = get_tensor(shape, inp1Stride, inp1Dtype, False)
        inp2Tensor = get_tensor(shape, inp1Stride, inp2Dtype, False)

        f = "inp0Tensor." + func_v["func"]
        #warpup
        for i in range(warm_ups):
            eval(f)(inp1Tensor,inp2Tensor, **func_v["extra_param"])

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=trace_handler) as prof:
            for i in range(warm_ups):
                cache.zero_()
                eval(f)(inp1Tensor,inp2Tensor, **func_v["extra_param"])

        info = prof.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        key = func
        if "substitue" in func_v:
            key = func_v["substitue"]
        return extract_kernel_time(info, key)


def get_arity3_torchFunc_kernelTime(middle_line_list):
    isGetTime, dim, arity, shapeStride, dtype, isOpt, func, substride, NumOfCalls,eleInfo = middle_line_list
    dim = int(dim)

    shape = shapeStride.split(",")[0:dim]
    outStride = shapeStride.split(",")[dim:dim*2]
    inp0Stride = shapeStride.split(",")[dim*2:dim*3]
    inp1Stride = shapeStride.split(",")[dim*3:dim*4]
    inp2Stride = shapeStride.split(",")[dim*4:dim*5]
    outDtype = dtype.split(",")[0]
    inp0Dtype = dtype.split(",")[1]
    inp1Dtype = dtype.split(",")[2]
    inp2Dtype = dtype.split(",")[3]

    outTensor = get_tensor(shape, outStride, outDtype, False)
    inp0Tensor = get_tensor(shape, inp0Stride, inp0Dtype, False)
    inp1Tensor = get_tensor(shape, inp1Stride, inp1Dtype, False)
    inp2Tensor = get_tensor(shape, inp2Stride, inp2Dtype, False)
    
    cache = torch.empty(int(256e6 // 4), dtype = torch.int, device = "cuda")

    func_v = arity3_torch_map[func]
    #warpup
    for i in range(warm_ups):
        out = func_v["func"](inp0Tensor,inp1Tensor,inp2Tensor, **func_v["extra_param"])

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=trace_handler) as prof:
            for i in range(warm_ups):
                cache.zero_()
                out = func_v["func"](inp0Tensor,inp1Tensor,inp2Tensor, **func_v["extra_param"])

    info = prof.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
    key = func
    if "substitue" in func_v:
        key = func_v["substitue"]
    return extract_kernel_time(info, key)

def get_kernel_time(middle_line_list):
    arity = int(middle_line_list[2])
    func = middle_line_list[6]

    if arity == 1:
        if func in arity1_tensor_map:
            kernel,Time = get_arity1_tensorFunc_kernelTime(middle_line_list)
            return kernel,Time
        elif func in arity1_torch_map:
            kernel,Time = get_arity1_torchFunc_kernelTime(middle_line_list)
            return kernel,Time
        else:
            print("get_kernel_time arity1 cannot map func")
            exit(1)
    elif arity == 2:
        if func in arity2_tensor_map:
            kernel,Time = get_arity2_tensorFunc_kernelTime(middle_line_list)
            return kernel,Time
        elif func in arity2_torch_map:
            kernel,Time = get_arity2_torchFunc_kernelTime(middle_line_list)
            return kernel,Time
        else:
            print("get_kernel_time arity2 cannot map func")
            exit(1)
    elif arity == 3:
        if func in arity3_tensor_map:
            kernel,Time = get_arity3_tensorFunc_kernelTime(middle_line_list)
            return kernel,Time
        elif func in arity3_torch_map:
            kernel,Time = get_arity3_torchFunc_kernelTime(middle_line_list)
            return kernel,Time
        else:
            print("get_kernel_time arity3 cannot map func")
            exit(1)

    print("get_kernel_time get unrecognize arity")
    exit(1)



