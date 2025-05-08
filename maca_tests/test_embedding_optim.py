import torch
import torch.nn as nn
import os
import copy
from utils import perfModeEnvGuard

def test_embedding():
    test_cases = [{'shape' : (32, 128), "MACA_TORCH_PERF_MODE" : True},
                  {'shape' : (32, 128), "MACA_TORCH_PERF_MODE" : False},
                  {'shape' : (32, 256), "MACA_TORCH_PERF_MODE" : True},
                  {'shape' : (32, 256), "MACA_TORCH_PERF_MODE" : False}]
    
    emb = nn.Embedding(30522, 768)
    emb1 = copy.deepcopy(emb).cuda()
    for case in test_cases:
        shape = case['shape']
        run_perf_flag = case['MACA_TORCH_PERF_MODE']
        if run_perf_flag:
            temp_env = "embedding_kernel"
        else:
            temp_env = ""

        with perfModeEnvGuard(temp_env):
            inp = torch.randint(30000, shape)
            out = emb(inp)
            out.backward(torch.ones(out.shape))

            inp1 = inp.detach().clone().cuda()
            out1 = emb1(inp1)
            out1.backward(torch.ones(out1.shape).cuda())
            
            ret1 = torch.allclose(out1.cpu(), out, rtol=1e-7, atol=1e-7)
            ret2 = torch.allclose(emb1.weight.grad.cpu(), emb.weight.grad, rtol=1e-7, atol=1e-7)

            if ret1 is False or ret2 is False:
                exit(1)

    exit(0)

test_embedding()

   