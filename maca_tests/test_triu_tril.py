import torch

sizes_list = [(10, 5), (32, 32), (35, 99), (50*1024, 50*1024), (49*1024, 51*1024), (1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)]

dtypes = [torch.bfloat16, torch.float16, torch.float32]
diagonals = [-3, -1, 0, 1, 3]

for dtype in dtypes:
    for size in sizes_list:
        for diagonal in diagonals:
            print(f"size: {size}")

            a_cpu = torch.randn(size,  dtype=dtype)
            a_cuda = a_cpu.cuda().contiguous()

            b_cpu = torch.triu(a_cpu, diagonal=diagonal)
            b_cuda = torch.triu(a_cuda, diagonal=diagonal)

            assert (b_cpu == b_cuda.cpu()).all(), f"#### {dtype}, {size} diagonal={diagonal} fail"