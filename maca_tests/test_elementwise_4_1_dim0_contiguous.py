import torch
import torch.nn as nn

def test4_1_lowdim_contiguous():
    inp1_base = torch.rand(100000000,device="cuda",dtype=torch.bfloat16)

    shape_list = [(2623,2,32,64),(2623,2,32,64),(2623,2,32,128)]

    stride1_list=[(8192,4096,128,1),(8192,4096,128,1),(24576,12288,384,1)]
    stride2_list=[(4096,10743808,128,1), (2048,5371904,64,1),(4096,10743808,128,1)]

    for i in range(len(shape_list)):
        for dtype in [torch.half, torch.float, torch.bfloat16]:
            inp1_base = torch.rand(100000000,device="cuda",dtype=dtype)
            inp2_base = torch.rand(100000000,device="cuda",dtype=dtype)
            shape=shape_list[i]
            stride1=stride1_list[i]
            stride2=stride2_list[i]

            print(shape)
            print(stride1)
            print(stride2)
            inp1 = inp1_base.as_strided(shape, stride1)
            inp2 = inp2_base.as_strided(shape, stride2)
            inp1.copy_(inp2)

            inp1c = inp1.cpu()
            inp2c = inp2.cpu()
            inp1c.copy_(inp2c)

            diff=torch.max(torch.abs(inp1.cpu()-inp1c))

            if diff > 0.0001:
                print("test_32_lowdim_contigusou is error")
                exit(1)

test4_1_lowdim_contiguous()

