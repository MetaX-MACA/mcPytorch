import torch

Dtype_bytes = {"Byte":1, "Char":1, "Short":2, "Int":4,    "BFloat16":2,
               "Long":8, "Half":2, "Float":4, "Double":8, "Bool":1}


def trace_handler(prof):
    print(prof.key_averages(group_by_input_shape=True,
                            group_by_stack_n=1).table(sort_by="self_cuda_time_total",
                                                      max_name_column_width=10000,
                                                      max_src_column_width=10000,
                                                      row_limit = -1))

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


def filter_compare_nocontiguous():
    inp = "compare.txt"
    with open(inp,"r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            lines_list = line.strip().split()
            dim = int(lines_list[2])
            shapeStride = lines_list[4].split(",")
            dtype = lines_list[5].split(",")[0]

            contigusou = is_arity_contigusou(dim, shapeStride, dtype, 0)

            if not contigusou and line.split()[12]!="direct_copy_kernel_cuda":
            #if not contigusou:
                print(line)


def filter_no_time_case():
    inlog = "all_text.txt"
    outlog = "all_text1.txt"
    line_list = []
    line_map = {}

    with open(inlog, "r") as f:
        lines = f.readlines()
        for line in lines:
            # if line.strip().split()[0] == "0":
            if "p_e_noopt" in line:
                line_list.append(line)
                line_map[line.strip().split()[-1]] = 0
    print(line_map)
    with open(outlog, "w") as f:
        for line in line_list:
            f.write(line)
        f.write("|||||||||||||||\n")
        for (k,v) in line_map.items():
            f.write(k+"\n")

filter_no_time_case()

def get_case():
    dim = 1
    arity = 1
    shapeStride = "32768,2,4"
    dtypes = "Half,Float"


    shape = shapeStride.split(",")[:dim]

    outstride = shapeStride.split(",")[dim:dim*2]
    outDtype = dtypes.split(",")[0]
    out = get_tensor(shape, outstride, outDtype)

    inp1stride = shapeStride.split(",")[dim*2:dim*3]
    inp1Dtype = dtypes.split(",")[1]
    inp1 = get_tensor(shape, inp1stride, inp1Dtype)

    # inp2stride = shapeStride.split(",")[dim*3:dim*4] 
    # inp2Dtype = dtypes.split(",")[2]
    # inp2 = get_tensor(shape, inp2stride, inp2Dtype)
    
    is_cons = is_arity_contigusou(dim, shapeStride, dtypes, 0)
    print(cons)

    cache = torch.empty(int(256e6 // 4), dtype = torch.int, device = "cuda")
    for i in range(200):
        out.copy_(inp1)

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=trace_handler) as prof:
        for i in range(200):
            cache.zero_()
            out.copy_(inp1)
    
