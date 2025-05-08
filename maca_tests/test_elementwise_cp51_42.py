import torch
import numpy as np
import torch.nn as nn
import random



def test_elem_cp_5_1():
    inpg = torch.rand(10000000).cuda()

    shape_stride_list = [[(10,9,8,7,6),(8000,889,111,13,2)],
                         [(11,10,9,8,7),(16241,1624,180,22,3)],
                         [(12,11,10,9,8),(33000,2997,299,33,4)],
                         [(13,12,11,10,9),(61566,5130,466,46,5)]
                        ]

    for item in shape_stride_list:
        inp=inpg.as_strided(item[0],item[1])
        outg=inp.contiguous()

        inpc=inp.detach().cpu()
        out=inpc.contiguous()

        res=torch.allclose(outg.cpu(),out,1e-7,1e-7)
        if not res:
            print("test_elem_cp_5_1 test error")
            exit(1)

def test_add_4_2():
    inp = torch.rand(10000000).cuda()
    inpp = torch.rand(10000000).cuda()

    shape_stride_list = [[(9,8,7,6),(889,111,13,2)],
                         [(10,9,8,7),(1624,180,22,3)],
                         [(11,10,9,8),(2997,299,33,4)],
                         [(12,11,10,9),(5130,466,46,5)]
                        ]


    for item in shape_stride_list:
       inp0=inp.as_strided(item[0],item[1])
       inp1=inpp.as_strided(item[0],item[1])

       out = inp0+inp1

       inpc0=inp0.detach().cpu()
       inpc1=inp1.detach().cpu()
       outc=inpc0+inpc1
       res=torch.allclose(out.cpu(),outc,1e-7,1e-7)
       if not res:
            print("test_add_4_2 test error")
            exit(1)

test_elem_cp_5_1()
test_add_4_2()
exit(0)

