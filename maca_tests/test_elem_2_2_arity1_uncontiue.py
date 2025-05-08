import torch
import sys
import time
import os
import numpy as np
import random

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))


def test_elem1():
    inpo = torch.rand(629145600,device="cuda",dtype=torch.float16)
    inp1 = inpo.as_strided((8192,1280),(1280*2,1))
    inp2 = torch.rand(8192,1280,device="cuda",dtype=torch.float16)
    out = inp1+inp2

    inp1_cpu = inp1.detach().clone().cpu()
    inp2_cpu = inp2.detach().clone().cpu()
    out_cpu = inp1_cpu+inp2_cpu

    res=out.cpu()-out_cpu
    max_ele = float(torch.abs(res.max()))
    min_ele = float(torch.abs(res.min()))
    if max_ele > 1e-6 or min_ele > 1e-6:
        print("launch_legacy_kernel_maca_2_2_interval_arity1 error")
        exit(1)


def test_elem2():
    inpo = torch.rand(629145600,device="cuda",dtype=torch.float16)
    inp1 = inpo.as_strided((8192,1280),(1280*3,1))
    inp2 = torch.rand(8192,1280,device="cuda",dtype=torch.float16)
    out = inp1+inp2

    inp1_cpu = inp1.detach().clone().cpu()
    inp2_cpu = inp2.detach().clone().cpu()
    out_cpu = inp1_cpu+inp2_cpu

    res=out.cpu()-out_cpu
    max_ele = float(torch.abs(res.max()))
    min_ele = float(torch.abs(res.min()))
    if max_ele > 1e-6 or min_ele > 1e-6:
        print("launch_legacy_kernel_maca_2_2_interval_arity1 error")
        exit(1)

def test_elem3():
    inpo = torch.rand(629145600,device="cuda",dtype=torch.bfloat16)
    inp1 = inpo.as_strided((8192,1280),(1280*3,1))
    inp2 = torch.rand(8192,1280,device="cuda",dtype=torch.bfloat16)
    out = inp1+inp2

    inp1_cpu = inp1.detach().clone().cpu()
    inp2_cpu = inp2.detach().clone().cpu()
    out_cpu = inp1_cpu+inp2_cpu

    res=out.cpu()-out_cpu
    max_ele = float(torch.abs(res.max()))
    min_ele = float(torch.abs(res.min()))
    if max_ele > 1e-6 or min_ele > 1e-6:
        print("launch_legacy_kernel_maca_2_2_interval_arity1 error")
        exit(1)

def test_elem4():
    inpo = torch.rand(629145600,device="cuda",dtype=torch.float16)
    inp1 = inpo.as_strided((333,128),(128*2,1))
    inp2 = torch.rand(333,128,device="cuda",dtype=torch.float16)
    out = inp1+inp2

    inp1_cpu = inp1.detach().clone().cpu()
    inp2_cpu = inp2.detach().clone().cpu()
    out_cpu = inp1_cpu+inp2_cpu

    res=out.cpu()-out_cpu
    max_ele = float(torch.abs(res.max()))
    min_ele = float(torch.abs(res.min()))
    if max_ele > 1e-6 or min_ele > 1e-6:
        print("launch_legacy_kernel_maca_2_2_interval_arity1 error")
        exit(1)

def test_elem5():
    inpo = torch.rand(629145600,device="cuda",dtype=torch.float16)
    inp1 = inpo.as_strided((333,64),(128*2,1))
    inp2 = torch.rand(333,64,device="cuda",dtype=torch.float16)
    out = inp1+inp2

    inp1_cpu = inp1.detach().clone().cpu()
    inp2_cpu = inp2.detach().clone().cpu()
    out_cpu = inp1_cpu+inp2_cpu

    res=out.cpu()-out_cpu
    max_ele = float(torch.abs(res.max()))
    min_ele = float(torch.abs(res.min()))
    if max_ele > 1e-6 or min_ele > 1e-6:
        print("launch_legacy_kernel_maca_2_2_interval_arity1 error")
        exit(1)


test_elem1()
test_elem2()
test_elem3()
test_elem4()
test_elem5()
exit(0)