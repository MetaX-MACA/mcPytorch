import torch

roll_list=[[(512, 56, 56, 128),(0,1,2,3),(0,1,2,3)],
           [(512, 56, 56, 128),(-1,-2,-3,-4),(3,2,1,0)],
           [(512, 56, 56, 128),(-10,-200,-300,-400),(0,2,1,3)],
           [(512, 56, 56, 128),(0,10,100,300),(-1,-2,-3,-4)],
           [(512, 28, 28, 256),(-1,2,-3,4),(-4,-3,-2,-1)],
           [(512, 28, 28, 256),(100,200,300,400),(2,3,1,0)]
          ]

for item_list in roll_list:
    shape,shift,dim=item_list
    x = torch.rand(512, 56, 56, 128,dtype=torch.float16)
    out0 = torch.roll(x, shifts=shift,dims=dim)
    out1 = torch.roll(x.cuda(), shifts=shift,dims=dim)

    res = torch.allclose(out0,out1.cpu(),1e-7,1e-7)
    if not res:
        print("test_roll.py is error")
        exit(1)


