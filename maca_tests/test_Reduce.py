import torch


error_eps = 1e-2


def check_result_cpu_cuda(input_cpu, input_cuda):
    input_cuda_cpu = input_cuda.cpu()
    real_eps = abs(input_cpu - input_cuda_cpu).sum() / input_cpu.numel()
    if real_eps < error_eps:
        return True
    else:
        return False


input_size_lists = [(3,4,5),(3,5,4),(4,3,5),(4,5,3),(5,3,4),(5,4,3),
                    (2,3,200000),(2,200000,3),(3,2,200000),(3,200000,2),(200000,2,3),(200000,3,2)]
dim_size_lists = [(0),(1),(2),(0,1),(1,2),(0,2),(0,1,2)]
result = []


for input_size in input_size_lists:
    for dim_size in dim_size_lists:
        input_cpu = torch.rand(input_size)
        output_cpu = input_cpu.sum(dim=dim_size)
        input_cuda = input_cpu.cuda()
        output_cuda = input_cuda.sum(dim=dim_size)
        result.append(check_result_cpu_cuda(output_cpu, output_cuda))

        input_cpu_t1 = input_cpu.transpose(0,1)
        output_cpu_t1 = input_cpu_t1.sum(dim=dim_size)
        input_cuda_t1 = input_cpu_t1.cuda()
        output_cuda_t1 = input_cuda_t1.sum(dim=dim_size)
        result.append(check_result_cpu_cuda(output_cpu_t1, output_cuda_t1))

        input_cpu_t2 = input_cpu.transpose(0,1)
        output_cpu_t2 = input_cpu_t2.sum(dim=dim_size)
        input_cuda_t2 = input_cpu_t2.cuda()
        output_cuda_t2 = input_cuda_t2.sum(dim=dim_size)
        result.append(check_result_cpu_cuda(output_cpu_t2, output_cuda_t2))

        input_cpu_t3 = input_cpu.transpose(0,1)
        output_cpu_t3 = input_cpu_t3.sum(dim=dim_size)
        input_cuda_t3 = input_cpu_t3.cuda()
        output_cuda_t3 = input_cuda_t3.sum(dim=dim_size)
        result.append(check_result_cpu_cuda(output_cpu_t3, output_cuda_t3))


if sum(result)<len(result):
    print("Failed: {}".format(__file__))
    exit(1)
else:
    print("Passed: {}".format(__file__))
    exit(0)