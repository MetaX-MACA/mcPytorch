import torch

shape = (1000,300,100,100)
dtypes = [torch.bfloat16, torch.float16, torch.float32]
torch.manual_seed(0)
for dtype in dtypes:
    #Loops.cuh gpu_kernel for contiguous input test
    input = torch.rand(*shape,  dtype=dtype, device = 'cuda:0')
    leaky_relu = torch.nn.LeakyReLU(0.1)
    output = leaky_relu(input).float().cpu()
    input_cpu = input.cpu().float()
    output_golden = leaky_relu(input_cpu)
    if not torch.allclose(output_golden, output):
        print("compare failed")
        exit(1)

    #Loops.cuh gpu_kernel for noncontiguous input test
    input_tmp = torch.rand(*shape,  dtype=dtype, device = 'cuda:0')
    input = torch.as_strided(input_tmp, shape, (100, 100, 100, 1))
    leaky_relu = torch.nn.LeakyReLU(0.1)
    output = leaky_relu(input).float().cpu()
    input_cpu = input.cpu().float()
    output_golden = leaky_relu(input_cpu)
    if not torch.allclose(output_golden, output):
        print("compare failed")
        exit(1)

    #Loops.cuh gpu_kernel_maca_arity1 for contiguous input test
    input = torch.rand(*shape,  dtype=dtype, device = 'cuda:0')
    output = torch.nn.GELU()(input).float().cpu()
    input_cpu = input.cpu().float()
    output_golden = torch.nn.GELU()(input_cpu)
    if not torch.allclose(output_golden, output, atol=5e-3):
        print(torch.max(torch.abs(output_golden - output)))
        print("compare failed")
        exit(1)

    #Loops.cuh gpu_kernel_maca_arity1 for noncontiguous input test
    input_tmp = torch.rand(*shape,  dtype=dtype, device = 'cuda:0')
    input = torch.as_strided(input_tmp, shape, (100, 100, 100, 1))
    output = torch.nn.GELU()(input).float().cpu()
    input_cpu = input.cpu().float()
    output_golden = torch.nn.GELU()(input_cpu)
    if not torch.allclose(output_golden, output, atol=5e-3):
        print(torch.max(torch.abs(output_golden - output)))
        print("compare failed")
        exit(1)

    #Loops.cuh gpu_kernel_maca_arity2 for contiguous input test
    input = torch.rand(*shape,  dtype=dtype, device = 'cuda:0')
    output = torch.add(input, input).float().cpu()
    input_cpu = input.cpu().float()
    output_golden = torch.add(input_cpu, input_cpu)
    if not torch.allclose(output_golden, output):
        print(torch.max(torch.abs(output_golden - output)))
        print("compare failed")
        exit(1)

    #Loops.cuh gpu_kernel_maca_arity2 for noncontiguous input test
    input_tmp = torch.rand(*shape,  dtype=dtype, device = 'cuda:0')
    input = torch.as_strided(input_tmp, shape, (100, 100, 100, 1))
    output = torch.add(input, input).float().cpu()
    input_cpu = input.cpu().float()
    output_golden = torch.add(input_cpu, input_cpu)
    if not torch.allclose(output_golden, output):
        print(torch.max(torch.abs(output_golden - output)))
        print("compare failed")
        exit(1)

    #Loops.cuh gpu_kernel_maca_arity3 for contiguous input test
    input = torch.rand(*shape,  dtype=dtype, device = 'cuda:0')
    output = torch.nn.GroupNorm(30, 300).cuda().to(dtype)(input).float().cpu()
    input_cpu = input.cpu().float()
    output_golden = torch.nn.GroupNorm(1, 300)(input_cpu)
    if not torch.allclose(output_golden, output, atol=5e-2):
        print(torch.max(torch.abs(output_golden - output)))
        print("compare failed")
        exit(1)

    #Loops.cuh gpu_kernel_maca_arity3 for noncontiguous input test
    input_tmp = torch.rand(*shape,  dtype=dtype, device = 'cuda:0')
    input = torch.as_strided(input_tmp, shape, (100, 100, 100, 1))
    output = torch.nn.GroupNorm(30, 300).cuda().to(dtype)(input).float().cpu()
    input_cpu = input.cpu().float()
    output_golden = torch.nn.GroupNorm(1, 300)(input_cpu)
    if not torch.allclose(output_golden, output, atol=5e-2):
        print(torch.max(torch.abs(output_golden - output)))
        print("compare failed")
        exit(1)

    #Loops_copy.cuh gpu_kernel_maca_arity1
    input = torch.ones(*shape,  dtype=dtype).cuda()
    golden = torch.ones(*shape,  dtype=dtype)
    if not torch.allclose(golden, input.cpu()):
        print(torch.max(torch.abs(golden - input)))
        print("compare failed")
        exit(1)