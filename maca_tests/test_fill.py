import torch

def test_fill():
    value = 129

    dtype_list = [torch.half,torch.bfloat16,torch.float]
    
    shape_list = [(67108864,), (16375808,), (4276224,), (1046529,), (65536,),]

    for shape in shape_list:
        for dtype in dtype_list:
            input = torch.randn(shape, device="cuda", dtype=dtype)
            input_c = torch.randn(shape, device="cpu", dtype=dtype)

            input.fill_(value)
            input_c.fill_(value)

            res = torch.allclose(input.cpu(), input_c)
            if not res:
                print(shape)
                print(dtype)
                print("fill is error")
                exit(1)

test_fill()