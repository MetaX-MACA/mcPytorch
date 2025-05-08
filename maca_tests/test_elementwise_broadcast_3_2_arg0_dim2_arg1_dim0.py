import torch


def test_elementwise_3_2_arg0_dim2_arg1_dim0():

    shape_list =   [(24,278528,2),(16,64,4), (28,128,2), (32,64,8), (36,256,512)]
    stride1_list = [(0,4,2),      (0,8,4),   (0,10,6),   (0,12,8),  (0,14,10)]
    stride2_list = [(557056,2,0), (444,4,0), (44,46,0),   (66,68,0), (80,20,0)]

    for i in range(len(shape_list)):
        shape=shape_list[i]
        stride1=stride1_list[i]
        stride2=stride2_list[i]

        for dtype in [torch.half, torch.float, torch.bfloat16]:
            inp1_base = torch.rand(100000000,device="cuda",dtype=dtype)
            inp2_base = torch.rand(100000000,device="cuda",dtype=dtype)
            inp1 = inp1_base.as_strided(shape, stride1)
            inp2 = inp2_base.as_strided(shape, stride2)
            out = inp1+inp2

            inp1c = inp1.cpu()
            inp2c = inp2.cpu()
            outc = inp1c+inp2c

            diff=torch.max(torch.abs(out.cpu()-outc))
            if diff>0.0001:
                print("test_elementwise_3_2_arg0_dim2_arg1_dim0 is error")
                exit(1)

test_elementwise_3_2_arg0_dim2_arg1_dim0()
