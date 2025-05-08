import torch
import torch.nn as nn
import torch.nn.functional as F

def test_elementwise2_1_input_lowdim_continuous():
    query_base = torch.rand(100000000,device="cuda",dtype=torch.bfloat16)

    shape_stride_list=[[(128, 540672),(655360,1)], [(32, 8), (4 ,1)], [(100,16),(6 , 1)],
                       [(120,24),(300,1)], [(120,32),(70,1)], [(440,72), (40, 1)]
                      ]
    for item in shape_stride_list:
       inp = query_base.as_strided(item[0],item[1])
       out = inp*2
       inpc = inp.detach().clone().cpu()
       outc = inpc*2
       res = torch.allclose(out.cpu(),outc)
       if not res:
           print("test_elementwise2_1_input_lowdim_continuous is error")
           exit(1)
test_elementwise2_1_input_lowdim_continuous()