import torch
import torch.nn as nn

def test5_1_lowdim_contiguous():
    inp1_base = torch.rand(100000000,device="cuda",dtype=torch.bfloat16)

    shape_list = [(256,49,3,16,32),(8,8,8,8,8),(7,8,7,7,16),(20,30,40,6,32),
                  (3,4,5,6,8),(11,12,13,14,48),(2,3,4,5,480),(3,4,5,6,64)]
    dtype_list = [torch.float,torch.half,torch.bfloat16]

    for shape in shape_list:
        for dtype in dtype_list:
            inp1_base = torch.rand(100000000,device="cuda",dtype=dtype)
            inp1 = inp1_base.as_strided(shape,(shape[1]*shape[3]*shape[4],shape[4],shape[0]*shape[1]*shape[3]*shape[4],shape[1]*shape[4],1))
            inp2 = torch.rand(shape,device="cuda",dtype=dtype)

            inp1_c = inp1.cpu()
            inp2_c = inp2.cpu()

            inp2.copy_(inp1)
            inp2_c.copy_(inp1_c)
            res = torch.allclose(inp2.cpu(),inp2_c)
            if not res:
                print("test5_1_lowdim_contiguous is error")
                exit(1)

test5_1_lowdim_contiguous()
