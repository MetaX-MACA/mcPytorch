import torch
import copy

def test_accuracy():
    test_cases = {
            #  stable diffusion group norm shape, test vectorized and fused kernels
            1: {"input_shape": [1, 128, 512, 512], "channels": 128, "groups": 32},
            2: {"input_shape": [1, 256, 512, 512], "channels": 256, "groups": 32},
            3: {"input_shape": [1, 256, 256, 256], "channels": 256, "groups": 32},
            4: {"input_shape": [1, 512, 256, 256], "channels": 512, "groups": 32},
            5: {"input_shape": [1, 512, 128, 128], "channels": 512, "groups": 32},
            6: {"input_shape": [1, 512, 64, 64], "channels": 512, "groups": 32},
            7: {"input_shape": [2, 1280, 32, 32], "channels": 1280, "groups": 32},
            8: {"input_shape": [2, 1280, 16, 16], "channels": 1280, "groups": 32},
            9: {"input_shape": [2, 1280, 8, 8], "channels": 1280, "groups": 32},
            11: {"input_shape": [2, 1920, 16, 16], "channels": 1920, "groups": 32},
            12: {"input_shape": [2, 1920, 32, 32], "channels": 1920, "groups": 32},
            13: {"input_shape": [2, 2560, 16, 16], "channels": 2560, "groups": 32},
            14: {"input_shape": [2, 2560, 8, 8], "channels": 2560, "groups": 32},
            15: {"input_shape": [2, 320, 32, 32], "channels": 320, "groups": 32},
            16: {"input_shape": [2, 320, 64, 64], "channels": 320, "groups": 32},
            17: {"input_shape": [2, 640, 32, 32], "channels": 640, "groups": 32},
            18: {"input_shape": [2, 640, 32, 32], "channels": 640, "groups": 32},
            19: {"input_shape": [2, 640, 32, 32], "channels": 640, "groups": 32},
            20: {"input_shape": [2, 960, 16, 16], "channels": 960, "groups": 32},
            21: {"input_shape": [2, 960, 32, 32], "channels": 960, "groups": 32},
            # self defined case, test vectorized but not fused shape
            22: {"input_shape": [1, 512, 5, 5], "channels": 512, "groups": 32},
            23: {"input_shape": [1, 512, 7, 7], "channels": 512, "groups": 32},
            24: {"input_shape": [1, 512, 31, 31], "channels": 512, "groups": 32},
            25: {"input_shape": [1, 512, 70, 70], "channels": 512, "groups": 32},
            # self defined case, test not vectorized and not fused shape
            26: {"input_shape": [1, 640, 5, 5], "channels": 640, "groups": 64},
            27: {"input_shape": [1, 640, 7, 7], "channels": 640, "groups": 64},
            28: {"input_shape": [1, 640, 31, 31], "channels": 640, "groups": 64},
            # self defined case, test not fused shape with large batch size N > 128
            29: {"input_shape": [129, 256, 8, 8], "channels": 256, "groups": 32},
            30: {"input_shape": [256, 256, 7, 7], "channels": 256, "groups": 32},
            31: {"input_shape": [512, 256, 5, 5], "channels": 256, "groups": 32},
    }
    torch.manual_seed(0)
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
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
                
                all_close_dx_grad = torch.allclose(dx_gpu.to(torch.float32).cpu(), dx_cpu, rtol=eps, atol=eps)
                print("{}: all_close_dx_grad: {}".format(id, all_close_dx_grad))
                if not all_close_dx_grad:
                    print("all_close_dx_grad input_shape: {}".format(test_case["input_shape"]))
                    print("dx_gpu: ", dx_gpu.to(torch.float32).cpu())
                    print("dx_cpu: ", dx_cpu)
                    exit(1)
                if not affine:
                    continue
                all_close_weight_grad = torch.allclose(m.weight.grad.to(torch.float32).cpu(), m_cpu.weight.grad, rtol=1e-1, atol=1)
                all_close_bias_grad = torch.allclose(m.bias.grad.to(torch.float32).cpu(), m_cpu.bias.grad, rtol=1e-1, atol=1e-1)
                print("{}: all all_close_weight_grad: {}".format(id, all_close_weight_grad))
                print("{}: all all_close_bias_grad: {}".format(id, all_close_bias_grad))
                if not all_close_weight_grad:
                    print("all_close_weight_grad input_shape: {}".format(test_case["input_shape"]))
                    print("m.weight.grad.to(torch.float32).cpu(): ", m.weight.grad.to(torch.float32).cpu())
                    print("m_cpu.weight.grad: ", m_cpu.weight.grad)
                    exit(1)
                if not all_close_bias_grad:
                    print("all_close_bias_grad input_shape: {}".format(test_case["input_shape"]))
                    print("m.bias.grad.to(torch.float32).cpu(): ", m.bias.grad.to(torch.float32).cpu())
                    print("m_cpu.weight.grad: ", m_cpu.bias.grad)
                    exit(1)
    exit(0)


if __name__ == "__main__":
    test_accuracy()
