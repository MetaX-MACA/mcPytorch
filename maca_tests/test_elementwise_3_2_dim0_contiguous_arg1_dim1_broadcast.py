import torch


def trace_handler(prof):
    print(prof.key_averages(group_by_input_shape=True, group_by_stack_n=1).table(sort_by="self_cuda_time_total", max_name_column_width=10000, max_src_column_width=10000, row_limit = -1))
    prof.export_chrome_trace("test.json")


def test_masked_fill():
    shape_list = [(512,128,128), (111, 444, 160), (22, 124, 180), (13, 128,64)]
    stride_list= [(128,  0,  1), (160,   0,   1), (180,  0,   1), (64, 0,   1)]

    for i in range(len(shape_list)):
        shape  = shape_list[i]
        stride = stride_list[i]
        for dtype in [torch.half, torch.float]:
            inp0 = torch.rand(shape,device="cuda",dtype=dtype)
            inp1 = torch.randint(0,1,(100000000,),device="cuda").bool().as_strided(shape, stride)
            out = torch.masked_fill(inp0,inp1,value=0)

            inp0c = inp0.cpu()
            inp1c = inp1.cpu()
            outc  = torch.masked_fill(inp0c,inp1c,value=0)

            diff = torch.max(torch.abs(out.cpu()-outc))
            if diff>0.0001:
                print("test_masked_fill is error")
                exit(1)



def test_3_2_dim0_contiguous_arg1_dim1_broadcast():
    shape_list =   [(6147, 64, 80), (6147, 64, 80), (1111, 68, 128), (222, 96, 100), (222, 100, 100)]
    stride0_list = [(5120, 80, 1),  (15360,240,1),  (8704, 256,1),   (2000,200,1),   (5002,400,1)]
    stride1_list = [(80,   0,  1),  (80,   0,  1),  (128,  0,  1),   (100, 0,  1),   (100,0,1)]


    for i in range(len(shape_list)):
        shape = shape_list[i]
        stride0 = stride0_list[i]
        stride1 = stride1_list[i]
        for dtype in [torch.float,torch.half]:
            inp0 = torch.rand(100000000,device="cuda",dtype=dtype).as_strided(shape, stride0)
            inp1 = torch.rand(100000000,device="cuda",dtype=dtype).as_strided(shape, stride1)
            out  = inp0 + inp1

            inp0c = inp0.cpu()
            inp1c = inp1.cpu()
            outc  = inp0c + inp1c

            diff = torch.max(torch.abs(out.cpu()-outc))
            if diff>0.0001:
                print("test_masked_fill is error")
                exit(1)


test_masked_fill()
test_3_2_dim0_contiguous_arg1_dim1_broadcast()
