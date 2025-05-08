import torch

'''
Support data type:
INT: torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
FLOAT: torch.bfloat16, torch.float16, torch.float32, torch.float64
COMPLEX: torch.complex64, torch.complex128

Unsupport data type:
COMPLEX: torch.complex32
QINT: torch.quint8, torch.qint8, torch.qint32, torch.quint4x2
'''

datatype_list = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                 torch.bfloat16, torch.float16, torch.float32, torch.float64, 
                 torch.complex64, torch.complex128]
datatype_int_list = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
error_eps = 1e-5


def check(data_cpu, data_gpu):
    data_gpu = data_gpu.cpu()
    error = (abs(data_cpu - data_gpu)).sum() / data_cpu.numel()
    ret = True
    if error < error_eps:
        print("Test", data_gpu.dtype, "pass, error_eps:", error)
        ret = True
    else:
        print("Test", data_gpu.dtype, "failed, error_eps:", error)
        ret = False
    return ret


if __name__ == "__main__":
    ret = True
    for datatype in datatype_list:
        if datatype in datatype_int_list:
            input_cpu_1 = torch.randint(0, 127, (1024, 6), dtype=datatype)
            input_cpu_2 = torch.randint(0, 127, (1024, 6), dtype=datatype)
        else:
            input_cpu_1 = torch.rand(1024, 6, dtype=datatype)
            input_cpu_2 = torch.rand(1024, 6, dtype=datatype)
        output_cpu = torch.add(input_cpu_1, input_cpu_2)
        input_gpu_1 = input_cpu_1.cuda()
        input_gpu_2 = input_cpu_2.cuda()
        output_gpu = input_gpu_1 + input_gpu_2
        ret = ret and check(output_cpu, output_gpu)
        if ret is False:
            print("Error: Test add all data type failed!")
            break
    if ret is True:
        print("Passed: {}".format(__file__))
        exit(0)
    else:
        print("Failed: {}".format(__file__))
        exit(1)
