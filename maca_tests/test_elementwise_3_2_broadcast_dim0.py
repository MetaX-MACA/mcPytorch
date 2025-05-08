import torch



def test_arity2_dim0_broadcast():
    shape_list = [(4532,24,128), (4096,24,128), (256,24,128), (64,33,2048), (128,77, 4096)]

    for shape in shape_list:
        for dtype in [torch.float, torch.half, torch.bfloat16]:
            inp2_base = torch.rand(100000000,device="cuda",dtype=dtype)

            inp1 = torch.rand(shape,device="cuda",dtype=dtype)
            inp2 = inp2_base.as_strided(shape, (1,shape[0],0))
            out = inp1+inp2

            inp1c = inp1.cpu()
            inp2c = inp2.cpu()
            outc = inp1c+inp2c

            diff=torch.max(torch.abs(out.cpu()-outc))
            if diff>0.0001:
                print("test_arity2_dim0_broadcast is error")
                exit(1)

test_arity2_dim0_broadcast()

