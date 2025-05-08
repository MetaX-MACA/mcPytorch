import torch
import torch.nn as nn

def test_upsample_trilinear():
    shape_list = [(1,17,32,200,200), (3,17,15,31,31), (2,5,1,7,7)]

    for sf in [1, 2, 3]:
        m = nn.Upsample(scale_factor=sf, mode='trilinear')

        for Clast in [True, False]:
            for shape in shape_list:
                inp = torch.rand(shape,dtype=torch.float,device="cuda")
                if Clast:
                    inp = inp.to(memory_format=torch.channels_last_3d)
                inp_c = inp.cpu()

                out = m(inp)
                out_c = m(inp_c)
                res=torch.allclose(out.cpu(), out_c, rtol=1e-04, atol=1e-06)
                if not res:
                    print("test_upsample_trilinear is error ", shape, sf, Clast)
                    exit(1)

test_upsample_trilinear()