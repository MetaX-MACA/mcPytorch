import torch
import torch.nn as nn

def test_upsample_bicubic():
    shape_list = [(1, 4096, 16, 16), (1, 1644, 16, 16), (32, 64, 49, 49), (32, 64, 31, 63), (1, 31, 16, 16)]

    for sf in [1, 2, 3]:
        m = nn.Upsample(scale_factor=sf, mode='bicubic')

        for Clast in [True, False]:
            for shape in shape_list:
                inp = torch.rand(shape,dtype=torch.float,device="cuda")
                if Clast:
                    inp = inp.to(memory_format=torch.channels_last)
                inp_c = inp.cpu()

                out = m(inp)
                out_c = m(inp_c) 
                res=torch.allclose(out.cpu(), out_c)
                if not res:
                    print("test_upsample_bicubic is error")
                    exit(1)

test_upsample_bicubic()