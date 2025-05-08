import torch

def test_3_2_arity2_trans_dim02():
    shape_list = [(2048,32,128), (64,256,784), (24,33,24), (24,34,32),
                  (256,31,256), (72,29,80), (40,77,400), (36,100,72)]

    dtype_list = [torch.half,torch.bfloat16,torch.float]
    broad_dim_1 = [True, False]

    for shape in shape_list:
        for ifbroad in broad_dim_1:
            for dtype in dtype_list:
                inp1 = torch.rand(shape,dtype=dtype,device="cuda")
                inp2_base = torch.rand(shape,dtype=dtype,device="cuda")
                inp1_c = inp1.cpu()
                inp2_c = inp2_base.cpu()

                if ifbroad:
                    inp2 = inp2_base.as_strided(shape, (1, 0, shape[0]))
                    inp2_c = inp2_c.as_strided(shape, (1, 0, shape[0]))
                else:
                    inp2 = inp2_base.as_strided(shape, (1, shape[0]*shape[2], shape[0]))
                    inp2_c = inp2_c.as_strided(shape, (1, shape[0]*shape[2], shape[0]))

                out = inp1 * inp2
                out_c = inp1_c * inp2_c
                res=torch.allclose(out.cpu(), out_c)

                if not res:
                    print("test_elementwise_3_2_transpose_dim02 is error")
                    exit(1)

test_3_2_arity2_trans_dim02()