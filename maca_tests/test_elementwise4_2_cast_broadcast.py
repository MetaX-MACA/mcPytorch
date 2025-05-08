import torch


def test_element_4_2_cast_broadcast():
    inp1_base = torch.randn(100000000, dtype=torch.float, device="cuda")
    inp2_base = torch.randn(100000000, dtype=torch.bfloat16, device="cuda")

    shape_list=[(128,512,20,12), (128,512,20,8), (128,512,12,44), (128,512,20,32),
                (111,12,33,4), (12,50,40,8), (13,14,15,16), (17,18,19,20)]
    for shape in shape_list:
        stride1=(2*shape[1]*shape[2]*shape[3],2*shape[2]*shape[3],2*shape[3],2)
        stride2=(0,shape[3],0,1)
        inp1 = inp1_base.as_strided(shape,stride1)
        inp2 = inp2_base.as_strided(shape,stride2)
        out = inp1 + inp2

        inp1_c = inp1.cpu()
        inp2_c = inp2.cpu()
        out_c = inp1_c + inp2_c

        res = torch.allclose(out.cpu(), out_c)
        if not res:
            print("test_element_4_2_cast_broadcast error")
test_element_4_2_cast_broadcast()

