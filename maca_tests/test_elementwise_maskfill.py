import torch

def test_mask_fill_perf():
    shapes = [(128, 784, 384)]
    # mask fill broadcast
    for shape in shapes:
        shape_mask = (shape[0], 1, shape[2])
        a = torch.randn(shape, dtype=torch.float, device="cuda:0")
        mask = torch.randint(2, shape_mask, dtype=torch.bool, device="cuda:0")
        mask = mask.as_strided(shape_mask, (shape[2], 0, 1))
        value = 5.0

        b = a.masked_fill_(mask, value)
        a_cpu = a.cpu()
        mask_cpu = mask.cpu()
        b_cpu = a_cpu.masked_fill_(mask_cpu, value)
        res=torch.allclose(b.cpu(), b_cpu)
        if not res:
            print("Error!")
            exit(1)
    exit(0)

test_mask_fill_perf()