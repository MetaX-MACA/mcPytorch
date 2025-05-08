import torch
import torch.nn as nn


def test3_2_arity2_transpose():

    shape_list = [(64,256,784), (33,24,24), (34,24,32), (100,32,64),
                  (31,256,256), (29,72,80), (44,88,96), (77,40,400),
                  (64,256,792), (33,24,32), (34,32,32), (100,36,72),
                  (64,512,196), (29,76,84), (44,92,100),(77,44,404),
                  (2,1024,225), (16,128,1025), (31,256,49),(31,256,70),
                  (64,81,81), (16,49,49), (16,70,70), (16,49,70)]

    dtype_list = [torch.half,torch.bfloat16,torch.float]

    for shape in shape_list:
        for dtype in dtype_list:
            inp2_base = torch.rand(100000000,dtype=dtype,device="cuda")
            inp1=torch.rand(shape,device="cuda",dtype=dtype)
            inp2=inp2_base.as_strided(shape,(shape[1]*shape[2], 1, shape[1]))
            inp1_c = inp1.cpu()
            inp2_c = inp2.cpu()

            out = inp1+inp2
            out_c = inp1_c+inp2_c
            res=torch.allclose(out.cpu(), out_c)
            if not res:
                print(shape)
                print(dtype)
                print("test3_2_arity2_transpose_half is error")
                exit(1)

test3_2_arity2_transpose()


