import torch
import torch.nn as nn

def test3_2_lowdim_contiguous():
    inp1_base = torch.rand(100000000,device="cuda",dtype=torch.bfloat16)

    shape_list = [(127,11,64),(1111,22,128),(3,223,256),(2,17,512),
                  (6,6,1024),(3,4,2048),(11,12,4096),(13,12,4096),
                  (2623, 64, 128),(2623, 2, 4096)]

    stride1_list=[(127,11,1),(222,44,1),(256,1111,1),(444,12,1),
                  (1024,4444,1),(2345,456,1),(12,4567,1),(3456,23,1),
                  (24576,384,1),(1*4096*2,1*4096,1)]
    stride2_list=[(11,127,1),(44,222,1),(1111,256,1),(12,444,1),
                  (4444,1024,1),(456,2345,1),(4567,12,1),(23,3456,1),
                  (128,0,1),  (4096,10743808,1)]

    for i in range(len(shape_list)):
        for dtype in [torch.half, torch.float, torch.bfloat16]:
            inp1_base = torch.rand(100000000,device="cuda",dtype=dtype)
            inp2_base = torch.rand(100000000,device="cuda",dtype=dtype)
            shape=shape_list[i]
            stride1=stride1_list[i]
            stride2=stride2_list[i]

            inp1 = inp1_base.as_strided(shape, stride1)
            inp2 = inp2_base.as_strided(shape, stride2)
            out = inp1+inp2

            inp1c = inp1.cpu()
            inp2c = inp2.cpu()
            outc = inp1c+inp2c

            diff=torch.max(torch.abs(out.cpu()-outc))

            if diff > 0.0001:
                print("test_32_lowdim_contigusou is error")
                exit(1)


