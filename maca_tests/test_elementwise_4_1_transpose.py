import argparse
import torch
torch.manual_seed(0)


def test_elementwise_transpose_4_1():
    shape_list = [(128,20,512,24), (128,512,20,24), (128,512,20,16), (128,20,512,16),
                  (128,512,12,88), (128,12,512,88), (128,512,20,32), (128,20,512,32),
                  (128,17,66,8), (64,33,55,16), (9,80,77,24), (11,23,55,24)]
    dtype_list = [torch.float,torch.half,torch.bfloat16]
    for dtype in dtype_list:
        for shape in shape_list:
            inp1_base = torch.rand(100000000,device="cuda",dtype=dtype)
            inp2_base = torch.rand(100000000,device="cuda",dtype=dtype)

            inp1 = inp1_base.as_strided(shape,(shape[1]*shape[2]*shape[3],shape[3],shape[1]*shape[3],1))
            inp2 = inp2_base.as_strided(shape,(shape[1]*shape[2]*shape[3],shape[2]*shape[3],shape[3],1))

            inp1_c = inp1.cpu()
            inp2_c = inp2.cpu()

            inp2.copy_(inp1)
            inp2_c.copy_(inp1_c)

            res = torch.allclose(inp2.cpu(), inp2_c)
            if not res:
                print("test_elementwise_transpose_4_1 is error")
                exit(1)
test_elementwise_transpose_4_1()
exit(0)
