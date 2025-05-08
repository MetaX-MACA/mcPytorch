import torch.nn as nn
import torch
import os
import copy

torch.set_printoptions(precision=5, threshold=10000)

def test_div_float():
    ori = torch.rand(10000)

    inp = ori.as_strided((2,7),(12,13))
    out = inp / 2.345

    inp1 = inp.cuda()
    out1 = inp1 / 2.345

    max_ele = torch.abs((out1.cpu()-out).max())
    min_ele = torch.abs((out1.cpu()-out).min())

    if max_ele > 1e-7 or min_ele > 1e-7:
        print("test_div_float test error")
        exit(1)

def test_div_half():
    ori = torch.rand(10000).half()

    inp = ori.as_strided((2,7),(12,13))
    out = inp / 2.345
    print(inp)
    inp1 = inp.cuda()
    out1 = inp1 / 2.345

    max_ele = torch.abs((out1.cpu()-out).max())
    min_ele = torch.abs((out1.cpu()-out).min())
    print(max_ele)
    print(min_ele)
    if max_ele > 1e-3 or min_ele > 1e-3:
        print("test_div_half test error")
        exit(1)



test_div_float()
test_div_half()
exit(0)
