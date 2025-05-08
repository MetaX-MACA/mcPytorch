import torch
torch.manual_seed(0)

for shape in [
    (4096, 256)
]:  
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    for dtype in dtypes:
        input_1 = torch.randn(shape, dtype = dtype, device="cuda")
        input_2 = torch.randn((40960, 2560), dtype = dtype, device="cuda")
        output = input_2.as_strided(input_1.shape, (3584,1))
        output_cpu = output.cpu().float()
        output.copy_(input_1)
        output_cpu.copy_(input_1.cpu().float())
        if not torch.allclose(output_cpu, output.cpu().float()):
            exit(1)

for shape in [
    (4096, 3072)
]:  
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    for dtype in dtypes:
        input_1 = torch.randn(shape, dtype = dtype, device="cuda")
        input_2 = torch.randn((40960, 2560), dtype = dtype, device="cuda")
        output = input_2.as_strided(input_1.shape, (9216,1))
        output_cpu = output.cpu().float()
        output.copy_(input_1)
        output_cpu.copy_(input_1.cpu().float())
        if not torch.allclose(output_cpu, output.cpu().float()):
            exit(1)
        
for shape in [
    (4096, 3072)
]:  
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    for dtype in dtypes:
        input_2 = torch.randn((40960, 2560), dtype = dtype, device="cuda")
        output_2 = torch.randn((40960, 2560), dtype = dtype, device="cuda")
        input_1 = input_2.as_strided(shape, (3584,1))
        output = output_2.as_strided(shape, (9216,1))
        output_cpu = output.cpu().float()
        output.copy_(input_1)
        output_cpu.copy_(input_1.cpu().float())
        if not torch.allclose(output_cpu, output.cpu().float()):
            exit(1)

for shape in [
    (4096, 3072)
]:  
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    for dtype in dtypes:
        input_2 = torch.randn((40960, 2560), dtype = dtype, device="cuda")
        output_2 = torch.randn((40960, 2560), dtype = dtype, device="cuda")
        output = output_2.as_strided(shape, (3584,1))
        input_1 = input_2.as_strided(shape, (9216,1))
        output_cpu = output.cpu().float()
        output.copy_(input_1)
        output_cpu.copy_(input_1.cpu().float())
        if not torch.allclose(output_cpu, output.cpu().float()):
            exit(1)