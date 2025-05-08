import torch
import torch.nn as nn
import torch.nn.functional as F

def test_elementwise4_1_input_lowdim_continuous():
    inp_base = torch.rand(100000000,device="cuda")
    out_base = torch.rand(100000000,device="cuda")


    shape_stride_list = [[(128,12,512,88),(655360,88,1056,1)],[(128,12,512,8),(3000,122,12,1)],
                         [(127,11,13,16),(4444,378,14,1)],[(11,222,333,24),(4444,136,34,1)],
                         [(33,44,55,32),(6,4,2,1)],[(128,12,512,48),(2000,200,20,1)]]
    
    for dtype in [torch.float,torch.half,torch.bfloat16]:
        for item in shape_stride_list:
            shape,stride=item
            inp = inp_base.to(dtype=dtype).as_strided(shape,stride)
            out = out_base.to(dtype=dtype).as_strided(shape,(shape[1]*shape[2]*shape[3],shape[2]*shape[3],shape[3],1))

            inpc = inp.detach().clone().cpu()
            outc = out.detach().clone().cpu()

            out.copy_(inp)
            outc.copy_(inpc)

            res = torch.allclose(out.cpu(),outc)
            if not res:
                print("test_elementwise4_1_input_lowdim_continuous error")
                exit(1)

test_elementwise4_1_input_lowdim_continuous()


