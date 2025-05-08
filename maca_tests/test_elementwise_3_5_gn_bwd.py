import torch
import copy

def test_accuracy():
    test_cases = {
        1: {"input_shape": [1024, 8*32, 7, 7], "channels": 256, "groups": 32},
        2: {"input_shape": [1024, 8*32, 8, 8], "channels": 256, "groups": 32},
        3: {"input_shape": [1024, 10*32, 8, 8], "channels": 320, "groups": 32},
        4: {"input_shape": [256, 8*32, 8, 8], "channels": 256, "groups": 32},
        5: {"input_shape": [128, 8*32, 5, 5], "channels": 256, "groups": 32},
        6: {"input_shape": [2, 8*32, 1, 1025], "channels": 256, "groups": 32},
        7: {"input_shape": [2, 8*32, 4, 1025], "channels": 256, "groups": 32},
        8: {"input_shape": [5, 8*32, 3, 5], "channels": 256, "groups": 32},
        9: {"input_shape": [129, 8*32, 3, 5], "channels": 256, "groups": 32},
        10: {"input_shape": [129, 5*32, 3, 5], "channels": 160, "groups": 32},
    }
    torch.manual_seed(0)
    dtypes = [torch.float32]
    affines = [True, False]
    for id, test_case in test_cases.items():
        for dtype in dtypes:
            for affine in affines:
                gpu_device = torch.device('cuda')
                x = torch.randn(test_case["input_shape"], dtype=dtype)
                backward_input = torch.randn(test_case["input_shape"], dtype=dtype).to(gpu_device)
                m = torch.nn.GroupNorm(test_case["groups"], test_case["channels"], device="cuda", affine=affine).to(dtype)
                x_gpu = x.to(gpu_device)
                x_gpu.requires_grad = True
                y_gpu = m(x_gpu)
                backward_input = torch.randn_like(y_gpu).to(gpu_device)
                y_gpu.backward(backward_input)
                dx_gpu = x_gpu.grad

                cpu_device = torch.device('cpu')
                m_cpu = copy.deepcopy(m).to(torch.float32).cpu()
                x_cpu = x.detach().cpu().clone().float()
                x_cpu.requires_grad = True
                y_cpu = m_cpu(x_cpu)
                y_cpu.backward(backward_input.to(cpu_device))
                dx_cpu = x_cpu.grad
                eps=1e-4
                if dtype is torch.float16:
                    eps=1e-3
                if dtype is torch.bfloat16:
                    eps=1e-2
                
                # elementwise 3-5 only affects dx_grad
                all_close_dx_grad = torch.allclose(dx_gpu.to(torch.float32).cpu(), dx_cpu, rtol=eps, atol=eps)
                print("{}: all_close_dx_grad: {}".format(id, all_close_dx_grad))
                if not all_close_dx_grad:
                    print("all_close_dx_grad input_shape: {}".format(test_case["input_shape"]))
                    print("dx_gpu: ", dx_gpu.to(torch.float32).cpu())
                    print("dx_cpu: ", dx_cpu)
                    exit(1)
    exit(0)


if __name__ == "__main__":
    test_accuracy()
