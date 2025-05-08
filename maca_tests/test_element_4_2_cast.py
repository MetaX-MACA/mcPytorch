import torch

def test_element_4_2_cast():
    shape_stride_list = [
      [(20,30,20,2),(5000,161,8,2)],
      [(20,32,25,4),(7000,333,13,3)],
      [(20,32,25,4),(7020,332,13,3)]
    ]

    dtype_list = [torch.float,torch.half,torch.bfloat16]
    for dtype1 in dtype_list:
        for dtype2 in dtype_list:
            for item in shape_stride_list:
                inp1_base = torch.randn(100000000, dtype=dtype1, device="cuda")
                inp2_base = torch.randn(100000000, dtype=dtype2, device="cuda")

                inp1 = inp1_base.as_strided(item[0],item[1])
                inp2 = inp2_base.as_strided(item[0],item[1])
                out = inp1 + inp2

                inp1_c = inp1.cpu()
                inp2_c = inp2.cpu()
                out_c = inp1_c + inp2_c

                res = torch.allclose(out.cpu(), out_c)
                if not res:
                    print("test_element_4_2_case error")
                    exit(1)
test_element_4_2_cast()


