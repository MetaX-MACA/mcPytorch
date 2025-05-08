import argparse
import torch
torch.manual_seed(0)


def test_elementwise_dilation_1_1():


    shape_list = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                  192, 384, 704, 1280, 2368, 4480 ,8460,
                  10485760, 15728640 , 34603008 , 41943040]
    for shape in shape_list:
        inp1_base = torch.rand(100000000).cuda()
        inp2_base = torch.rand(100000000).cuda()
        inp1 = inp1_base.as_strided((shape,),(1,))
        inp2 = inp2_base.as_strided((shape,),(2,))

        inp1_c = inp1.cpu()
        inp2_c = inp2.cpu()

        inp2.copy_(inp1)
        inp2_c.copy_(inp1_c)

        for i in range(100):
            inp2.copy_(inp1)

        res = torch.allclose(inp2.cpu(), inp2_c)
        if not res:
            print(shape)
            print("test_elementwise_dilation_1_1.py is error")
            eixt(1)

test_elementwise_dilation_1_1()
