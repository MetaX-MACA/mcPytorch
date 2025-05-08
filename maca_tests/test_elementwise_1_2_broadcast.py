import torch

def test_1_2_broadcast():
    shape_list = [(64,), (6400,), (15360,), (134217728,)]
    dtype_list = [torch.half,torch.bfloat16,torch.float]

    for shape in shape_list:
        for dtype in dtype_list:
            inp1 = torch.rand(shape,dtype=dtype,device="cuda")
            inp2_base = torch.rand(shape,dtype=dtype,device="cuda")
            inp1_c = inp1.cpu()
            inp2_c = inp2_base.cpu()

            inp2 = inp2_base.as_strided(shape, (0,))
            inp2_c = inp2_c.as_strided(shape, (0,))

            out = inp1/inp2
            out_c = inp1_c/inp2_c
            res=torch.allclose(out.cpu(), out_c)
            if not res:
                print("test_elementwise_1_2_broadcast is error")
                exit(1)
    for shape in shape_list:
        for dtype in dtype_list:
            inp1 = torch.rand(shape,dtype=dtype,device="cuda")
            inp2 = torch.rand(shape,dtype=dtype,device="cuda")
            inp1_c = inp1.cpu()
            inp2_c = inp2.cpu()

            inp1 = inp1.as_strided(shape, (0,))
            inp1_c = inp1_c.as_strided(shape, (0,))

            out = inp1/inp2
            out_c = inp1_c/inp2_c
            res=torch.allclose(out.cpu(), out_c)
            if not res:
                print("test_elementwise_1_2_broadcast is error")
                exit(1)

test_1_2_broadcast()