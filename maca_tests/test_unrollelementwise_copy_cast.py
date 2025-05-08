import torch

def trace_handler(prof):
    print(prof.key_averages(group_by_input_shape=True, group_by_stack_n=1).table(sort_by="self_cuda_time_total", max_name_column_width=10000, max_src_column_width=10000, row_limit = -1))
    prof.export_chrome_trace("test.json")

dtype_list = [(torch.half,torch.float),
              (torch.float,torch.half),
              (torch.bfloat16,torch.float),
              (torch.float,torch.bfloat16),
              (torch.int,torch.float),
              (torch.half,torch.int),
              (torch.long,torch.int),
              (torch.int,torch.long),
              (torch.half,torch.bfloat16),
              (torch.bfloat16,torch.half),
              (torch.float,torch.long),
              ]

num_list = [4096, 4096+1,  65536, 65536+10, 524288, 524288+79, 4194304, 4194304+1111]

for num in num_list:
    for dtypes in dtype_list:
        dtype0 = dtypes[0]
        dtype1 = dtypes[1]
        out  = 1
        outc = 1
        inp  = 1
        if str(dtype0) == "torch.int32" or str(dtype0) == "torch.int64" or str(dtype0) == "torch.bool":
            out = torch.randint(0,1,(num,),device="cuda",dtype=dtype0)
            outc = torch.randint(0,1,(num,),dtype=dtype0)
        else:
            out = torch.rand(num, device="cuda", dtype=dtype0)
            outc = torch.rand(num, dtype=dtype0)
        if str(dtype1) == "torch.int32" or str(dtype1) == "torch.int64" or str(dtype1) == "torch.bool":
            inp = torch.randint(0,1,(num,),device="cuda",dtype=dtype1)
        else:
            inp = torch.rand(num, device="cuda", dtype=dtype1)
        out.copy_(inp)
        inpc = inp.cpu()
        outc.copy_(inpc)
        diff = torch.max(torch.abs(out.cpu()-outc))
        if diff > 0.0001:
            print("copy unroll error")
            print(diff)
            exit(1)
