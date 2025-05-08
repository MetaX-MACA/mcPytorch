import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import copy
import os, sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))

def cal_rel_err(infer, golden):
    diff = infer - golden
    diff_square = diff * diff
    infer_result_square_double = 2 * infer * infer
    sum_diff_square = torch.sum(diff_square)
    sum_infer_result_square_double = torch.sum(infer_result_square_double)
    result = torch.sqrt(sum_diff_square / (sum_infer_result_square_double + 1e-12))
    return result

for dtype in [torch.float, torch.float16, torch.bfloat16]:
    for shape in [[2, 64, 64, 4], [32, 256, 4, 1024], [32, 264, 4, 1024], [32, 272, 4, 1024], [32, 280, 4, 1024], [32, 288, 4, 1024], 
                  [32, 128, 100, 168], [32, 1024, 50, 84], [32, 64, 200, 336], [32, 512, 100, 168], [32, 512, 25, 42], [32, 2048, 25, 42],
                  [32, 256, 200, 336], [32, 256, 50, 80], [32, 512, 25, 40], [32, 1024, 84, 50], [32, 64, 672, 400], [1, 256, 50, 76], 
                  [1, 256, 50, 68], [1, 128, 100, 152], [1, 64, 200, 304], [1, 128, 100, 136], [1, 512, 25, 38], [1, 64, 200, 272],
                  [1, 512, 25, 34], [1, 2048, 25, 38], [8, 512, 20, 20], [8, 256, 40, 40], [8, 256, 40, 40], [8, 32, 160, 160], [8, 32, 320, 320],
                  [32, 64, 400, 672], [32, 256, 50, 84], [32, 296, 4, 1024], [256, 296, 4, 256], [257, 296, 4, 256],
                  [3, 37*2, 2, 37], [2, 39*2, 2, 39], [5, 43*2, 2, 43], [6, 51*2, 2, 51], [7, 55*2, 2, 55], [8, 101*2, 2, 101], [9, 111*2, 2, 111],
                  [4, 133*2, 2, 37], [2, 151*2, 2, 39], [5, 43*2, 2, 177], [6, 51*2, 2, 111], [7, 55*2, 2, 191], [18, 101*2, 2, 55], [9, 111*2, 2, 77]]:
        input_g = torch.randn(shape, dtype=torch.float).requires_grad_(True)
        input = input_g.clone().detach().to(dtype).cuda().requires_grad_(True)
        grad_input_g = torch.randn(shape, dtype=torch.float)
        grad_input = grad_input_g.clone().detach().to(dtype).cuda()

        m_g = torch.nn.BatchNorm2d(shape[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        m_g.weight = torch.nn.Parameter(torch.randn(shape[1], dtype=torch.float))
        m = copy.deepcopy(m_g).cuda()
        m_g.eval()
        m.eval()
        out = m(input)
        out.backward(grad_input)
        out_g = m_g(input_g)
        out_g.backward(grad_input_g)
        #status = torch.allclose(input_g.grad, input.grad.cpu().float(), 1e-4, 1e-4)
        print(torch.sum(input_g.grad), torch.sum(input.grad))
        tol = 1e-5 if dtype==torch.float else 5e-3
        status = cal_rel_err(input_g.grad, input.grad.cpu().float()) < tol
        print("error: ", cal_rel_err(input_g.grad, input.grad.cpu().float()))
        if not status:
            print(f"fail shape: {shape}, dtype: {dtype}")
            exit(1)
print("#### pass")
exit(0)
