import torch
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)

def test_sort():
    shape_list = [32, 64, 65, 128, 1024, 2048, 1111]
    for shape in shape_list:
        inp = torch.rand(shape).cuda()
        result = torch.sort(inp)
        inpc = inp.cpu()
        resultc = torch.sort(inpc)

        diff0 = torch.max(torch.abs(result[0].cpu()-resultc[0]))
        diff1 = torch.max(torch.abs(result[1].cpu()-resultc[1]))

        if diff0 > 0.000001 or diff1 > 0.000001:
            print("test_sort is error")
            exit(1)

test_sort()
exit(0)

