import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--type", default="checkin", help="<checkin|daily>")
args = parser.parse_args()

def test_accuracy():
    test_cases_checkin = {
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
            # common fused kernel test cases
            # DxHxW = 1 * warp_size * vec_size or 2 * warp_size * vec_size 
            # or 4 * warp_size * vec_size or 8 * warp_size * vec_size
            32: {"input_shape": [16, 32, 16, 16], "channels": 32, "groups": 32},
            33: {"input_shape": [16, 32, 16, 32], "channels": 32, "groups": 32},
            34: {"input_shape": [16, 32, 32, 32], "channels": 32, "groups": 32},
            35: {"input_shape": [16, 32, 32, 64], "channels": 32, "groups": 32},
            36: {"input_shape": [16, 32, 64, 64], "channels": 32, "groups": 32},
            37: {"input_shape": [16, 32, 128, 128], "channels": 32, "groups": 32},
            # rkwv
            38: {"input_shape": [8192, 768], "channels": 768, "groups": 768, "dtype": torch.bfloat16, "atol": 1e-4},
            39: {"input_shape": [1024, 4096], "channels": 4096, "groups": 4096, "dtype": torch.bfloat16, "atol": 1e-4},
            # block dim cross multiple HxW
            # read_step = block dim = smallest 2^n near DxHxW / vec_size
            # n_vec_hxw = HxW / vec_size
            # cross multiple HxW means read_step / n_vec_hxw > 1
            # to make it clear let us assume read_step = block dim = DxHxW / vec_size = 16, 32, ..., 512
            # D = 2, 3, 4, 5, 6
            40: {"input_shape": [16, 32, 256], "channels": 32, "groups": 16},
            41: {"input_shape": [16, 32, 256], "channels": 32, "groups": 8},
            42: {"input_shape": [16, 32, 256], "channels": 32, "groups": 4},
            43: {"input_shape": [16, 32, 256], "channels": 32, "groups": 2},
            44: {"input_shape": [16, 32, 256], "channels": 32, "groups": 1},
            45: {"input_shape": [16, 32, 512], "channels": 32, "groups": 16},
            46: {"input_shape": [16, 32, 1024], "channels": 32, "groups": 16}
    }

    test_cases_daily = {
        # stable diffusion 2.1 test cases
        29: {"input_shape": [16, 128, 768, 768], "channels": 128, "groups": 32},
        30: {"input_shape": [16, 256, 384, 384], "channels": 256, "groups": 32},
        31: {"input_shape": [16, 512, 192, 192], "channels": 512, "groups": 32},
    }
    torch.manual_seed(0)
    test_cases = test_cases_checkin
    if args.type == "daily":
        test_cases = test_cases_daily
    for id, test_case in test_cases.items():
        gpu_device = torch.device('cuda')
        dtype = test_case["dtype"] if "dtype" in test_case else torch.float32 
        x = torch.randn(test_case["input_shape"], dtype=dtype)
        m = torch.nn.GroupNorm(test_case["groups"], test_case["channels"], device="cuda", dtype=dtype)
        y_gpu = m(x.to(gpu_device))

        cpu_device = torch.device('cpu')
        m_cpu = m.to(cpu_device)
        y_cpu = m_cpu(x)
        atol = test_case["atol"] if "atol" in test_case else 1e-7
        rtol = test_case["rtol"] if "rtol" in test_case else 1e-4
        all_close = torch.allclose(y_gpu.cpu(), y_cpu, rtol=rtol, atol=atol)
        print("{}: all close: {}".format(id, all_close))
        if not all_close:
            print("input_shape: {}".format(test_case["input_shape"]))
            print("gpu: {}".format(y_gpu))
            print("cpu: {}".format(y_cpu))
            exit(1)
    exit(0)


if __name__ == "__main__":
    test_accuracy()
