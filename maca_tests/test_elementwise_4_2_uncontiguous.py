import torch
torch.manual_seed(0)


def test_elementwise_4_2_uncontiguous():
    shape_list = [(12,64,4096,64),(24,13,144,64),(24,13,145,64),\
                (24,13,128,128),(24,13,128,256),(24,13,64,512)]
    dtype_list = [torch.float]
    for dtype in dtype_list:
        for shape in shape_list:
            inp1_base = torch.rand(1000000000,device="cuda",dtype=dtype)
            inp2_base = torch.rand(1000000000,device="cuda",dtype=dtype)

            inp1 = inp1_base.as_strided(shape,(shape[1]*shape[2]*shape[3],shape[2]*shape[3],shape[3],1))
            inp2 = inp2_base.as_strided(shape,(shape[2],shape[2]*shape[0],1,0))

            inp1_c = inp1_base.cpu()
            inp2_c = inp2_base.cpu()

            inp1_c = inp1_c.as_strided(shape,(shape[1]*shape[2]*shape[3],shape[2]*shape[3],shape[3],1))
            inp2_c = inp2_c.as_strided(shape,(shape[2],shape[2]*shape[0],1,0))

            out = inp1 + inp2
            out_c = inp1_c + inp2_c

            res = torch.allclose(out.cpu(), out_c)
            if not res:
                print("test_elementwise_4_2_uncontiguous is error")
                exit(1)
test_elementwise_4_2_uncontiguous()
exit(0)