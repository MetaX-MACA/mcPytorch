import torch


def test_elementwise_4_2_braodcse_arg0_dim2_arg1_dim0():

    shape_list =   [(4352,24,64,2), (3333,28,128,4), (2222,32,256,8), (1111,36,512,16), (1000,40,1024,40)]
    stride1_list = [(256,0,4,2),    (128,0,8,4),     (64,0,16,8),   (256,0,32,16),   (128,0,64,8)]
    stride2_list = [(3072,128,2,0), (2000,256,4,0),  (1000,512,8,0),  (2048,1024,16,0), (2048,1024,32,0)]

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
                print("test_elementwise_4_2_broadcast_arg0_dim2_arg1_dim0 is error")
                exit(1)

test_elementwise_4_2_braodcse_arg0_dim2_arg1_dim0()
