import torch
def test_accuracy():
    test_cases = {
      0: {"input_shape": [9216, 3072]},
      1: {"input_shape": [9216, 768]},
      2: {"input_shape": [32*64*2, 24, 768]},
      3: {"input_shape": [32*64*4, 24, 768]},
      4: {"input_shape": [32*64*5, 24, 768]},
      5: {"input_shape": [25792, 4096]}
    }
    torch.manual_seed(0)
    for id, test_case in test_cases.items():
        gpu_device = torch.device('cuda')
        x = torch.randn(test_case["input_shape"], dtype=torch.bfloat16)
        # for i in range(test_case["input_shape"][1]):
        #   x[:, i] = torch.arange(start=0.0, end=test_case["input_shape"][0],step=1)
        y_gpu, index_gpu= torch.max(x.to(gpu_device), 0)
        print("x: ", x)
        print(y_gpu)
        print(index_gpu)
        cpu_device = torch.device('cpu')
        y_cpu, index_cpu = torch.max(x.to(cpu_device), 0)
        print(y_gpu.stride())
        all_close = torch.allclose(y_gpu.cpu(), y_cpu, rtol=1e-4, atol=1e-4)
        all_close_1 = torch.allclose(index_gpu.cpu(), index_cpu, rtol=1e-4, atol=1e-4)
        print("{}: all close: {}".format(id, all_close))
        print("gpu intput{}, gpu_output{}", x, y_gpu)
        print("cpu output {}", y_cpu)
        print(y_gpu.dtype)
        if not all_close or not all_close_1:
            print("input_shape: {}".format(test_case["input_shape"]))
            print("gpu data:{}".format(y_gpu))
            print("cpu data:{}".format(y_cpu))

            print("gpu: {}".format(index_gpu))
            print("cpu: {}".format(index_cpu))
            exit(1)



if __name__ == "__main__":
    test_accuracy()
