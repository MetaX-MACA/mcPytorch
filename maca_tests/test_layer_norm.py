import torch
import copy

torch.cuda.manual_seed_all(27)
torch.manual_seed(27)

def test_accuracy():
    test_cases = {
      0: {"input_shape": [2, 4096, 320], "normalized_shape": [320]},
      1: {"input_shape": [2, 1024, 640], "normalized_shape": [640]},
      2: {"input_shape": [2, 256, 1280], "normalized_shape": [1280]},
      3: {"input_shape": [2, 64, 1280], "normalized_shape": [1280]},
      4: {"input_shape": [1, 77, 1024], "normalized_shape": [1024]},
      5: {"input_shape": [4, 78, 256], "normalized_shape": [256]},
      6: {"input_shape": [5, 100, 256], "normalized_shape": [256]},
      7: {"input_shape": [5, 100, 32], "normalized_shape": [32]},
      8: {"input_shape": [5, 100, 16], "normalized_shape": [16]},
      9: {"input_shape": [5, 10, 48], "normalized_shape": [48]},
      10: {"input_shape": [5, 10, 64], "normalized_shape": [64]},
      11: {"input_shape": [5, 10, 72], "normalized_shape": [72]},
      12: {"input_shape": [5, 10, 80], "normalized_shape": [80]},
      13: {"input_shape": [112, 320], "normalized_shape": [320]},
      14: {"input_shape": [112, 512], "normalized_shape": [512]},
      15: {"input_shape": [113, 2048], "normalized_shape": [2048]},
      16: {"input_shape": [70, 4096], "normalized_shape": [4096]},
      17: {"input_shape": [71, 4104], "normalized_shape": [4104]},
      18: {"input_shape": [141, 2560], "normalized_shape": [2560]},
      19: {"input_shape": [141, 8192], "normalized_shape": [8192]},
      20: {"input_shape": [14, 6144], "normalized_shape": [6144]},
      21: {"input_shape": [9, 6152], "normalized_shape": [6152]},
      22: {"input_shape": [9, 8200], "normalized_shape": [8200]},
      23: {"input_shape": [500, 8200], "normalized_shape": [8200]},
      24: {"input_shape": [500, 5104], "normalized_shape": [5104]},
      25: {"input_shape": [503, 264], "normalized_shape": [264]},
      26: {"input_shape": [800, 800], "normalized_shape": [800]},
      27: {"input_shape": [10, 1536], "normalized_shape": [1536]},
      28: {"input_shape": [10, 2560], "normalized_shape": [2560]},
      29: {"input_shape": [10, 3072], "normalized_shape": [3072]},
      30: {"input_shape": [10, 3584], "normalized_shape": [3584]},
      31: {"input_shape": [11, 352], "normalized_shape": [352]},
      32: {"input_shape": [1024, 640], "normalized_shape": [640]},
      33: {"input_shape": [1024, 648], "normalized_shape": [648]},
      34: {"input_shape": [1024, 656], "normalized_shape": [656]},
      35: {"input_shape": [1024, 664], "normalized_shape": [664]},
      36: {"input_shape": [102, 1032], "normalized_shape": [1032]},
      37: {"input_shape": [5000, 768], "normalized_shape": [768]},
      38: {"input_shape": [8192, 768], "normalized_shape": [768]},
      39: {"input_shape": [4000, 768], "normalized_shape": [768]},
      40: {"input_shape": [1024, 4096], "normalized_shape": [4096]},
      41: {"input_shape": [4096, 96], "normalized_shape": [96]},
      42: {"input_shape": [6400, 384], "normalized_shape": [384]},
      43: {"input_shape": [6272, 96], "normalized_shape": [96]},
      44: {"input_shape": [6400, 192], "normalized_shape": [192]},
      45: {"input_shape": [6432, 128], "normalized_shape": [128]},
      46: {"input_shape": [6464, 256], "normalized_shape": [256]},
      47: {"input_shape": [6496, 512], "normalized_shape": [512]},
      48: {"input_shape": [6496, 1024], "normalized_shape": [1024]},
      49: {"input_shape": [3136, 64], "normalized_shape": [64]},
      50: {"input_shape": [2048, 64], "normalized_shape": [64]},
      51: {"input_shape": [0], "normalized_shape": [0]},

    }
    for bias_flag in [True, False]:
        for dtype in [torch.float16, torch.float32, torch.bfloat16, torch.float64]:
            for id, test_case in test_cases.items():
                inp_g = torch.randn(test_case["input_shape"]).to(device="cuda",dtype=dtype)
                inp_g.requires_grad = True
                m_g = torch.nn.LayerNorm(test_case["normalized_shape"], elementwise_affine=bias_flag).to(device="cuda",dtype=dtype)
                out_g = m_g(inp_g)
                out_g.backward(torch.ones(out_g.shape).to(device="cuda",dtype=dtype))

                inp_c = inp_g.detach().cpu().clone().float()
                inp_c.requires_grad = True
                m_c = copy.deepcopy(m_g).to(device="cpu",dtype=torch.float)
                out_c = m_c(inp_c)
                out_c.backward(torch.ones(out_c.shape))

                eps=1e-4
                if dtype is torch.float16:
                    eps=1e-3
                if dtype is torch.bfloat16:
                    eps=1e-2
                all_close0 = torch.allclose(out_g.cpu().float(), out_c, rtol=eps, atol=eps)
                all_close1 = torch.allclose(inp_g.grad.cpu().float(), inp_c.grad, rtol=eps, atol=eps)
                all_close2 = True
                all_close3 = True
                if bias_flag:
                    all_close2 = torch.allclose(m_g.weight.grad.cpu().float(), m_c.weight.grad, rtol=eps, atol=eps)
                    all_close3 = torch.allclose(m_g.bias.grad.cpu().float(), m_c.bias.grad, rtol=eps, atol=eps)

                is_pass = all_close0 and all_close1 and all_close2 and all_close3
                print("{}: all close: {} {} {} {}".format(id, all_close0, all_close1, all_close2, all_close3))
                if not is_pass:
                    print("Failed: {}".format(__file__))
                    exit(1)

if __name__ == "__main__":
    test_accuracy()
    exit(0)
