import torch

shape_list = [(4,8192), (16,8192) ,(1024,1024),(5,1024,1024),(8,1024,1024),(100,128,100)]
tmp = torch.randn(1, 1, 1, 2**31 - 1, dtype=torch.float16, device="cuda")

for shape in shape_list:
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)

    inp = torch.rand(shape,dtype=torch.float32).cuda()
    result = inp.exponential_()

    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    inp1 = torch.rand(shape,dtype=torch.float32).cuda()
    result1 = inp1.exponential_()

    diff = torch.max(torch.abs(result1-result)).cpu()
    if diff > 0.0001:
        print("test exponential error")
        exit(1)

exit(0)

