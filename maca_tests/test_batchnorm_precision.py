#!/usr/bin/env python
import torch
import torch.nn as nn
import sys
import time
import os
import numpy as np
import random
import copy

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))

def test_half():
    shapelist=[(2 ,256 ,50 ,76),(2 ,128 ,100 ,152),(2 ,1024 ,50 ,76),(2 ,256 ,50 ,68),(2 ,64 ,200 ,304),(2 ,256 ,76 ,50),
               (2 ,512 ,100 ,152),(2 ,512 ,25 ,38),(2 ,128 ,100 ,136),(2 ,1024 ,50 ,68),(2 ,256 ,200 ,304),(2 ,2048 ,25 ,38),
               (2 ,64 ,200 ,272),(2 ,128 ,152 ,100),(2 ,1024 ,76 ,50),(2 ,512 ,100 ,136),(2 ,512 ,25 ,34),(2 ,64 ,304 ,200),
               (2 ,256 ,200 ,272),(2 ,2048 ,25 ,34),(2 ,256 ,50 ,84),(2 ,256 ,68 ,50),(2 ,512 ,152 ,100),(2 ,512 ,38 ,25),
               (2 ,256 ,304 ,200),(2 ,2048 ,38 ,25),(2 ,128 ,100 ,168),(2 ,1024 ,50 ,84),(2 ,128 ,136 ,100),(2 ,1024 ,68 ,50),
               (2 ,256 ,50 ,78),(2 ,64 ,200 ,336),(2 ,64 ,272 ,200),(2 ,512 ,100 ,168),(2 ,512 ,25 ,42),(2 ,512 ,136 ,100)]

    for shape in shapelist:
        grad_out = torch.rand(size=shape).to(memory_format=torch.channels_last)

        inp = torch.rand(size=shape)
        inp.requires_grad=True
        mod = nn.BatchNorm2d(shape[1])
        mod.training=False
        out = mod(inp)
        out.backward(grad_out)

        inp1 = inp.detach().clone().cuda().half().to(memory_format=torch.channels_last)
        inp1.requires_grad=True
        mod1 = copy.deepcopy(mod).cuda()
        mod1.training=False
        out1 = mod1(inp1)
        out1.backward(grad_out.cuda().half().to(memory_format=torch.channels_last))

        flag0 = torch.allclose(out1.cpu().float(),out,1e-3,1e-3)
        flag1 = torch.allclose(inp1.grad.cpu().float(),inp.grad,1e-3,1e-3)
        flag2 = torch.allclose(mod1.weight.grad.cpu().float(),mod.weight.grad,1e-3,1e-3)
        flag3 = torch.allclose(mod1.bias.grad.cpu().float(),mod.bias.grad,1e-3,1e-3)

        if not (flag0 and flag1 and flag2 and flag3):
            exit(1)


def test_float():
    shapelist=[(2 ,256 ,50 ,76),(2 ,128 ,100 ,152),(2 ,1024 ,50 ,76),(2 ,256 ,50 ,68),(2 ,64 ,200 ,304),(2 ,256 ,76 ,50),
               (2 ,512 ,100 ,152),(2 ,512 ,25 ,38),(2 ,128 ,100 ,136),(2 ,1024 ,50 ,68),(2 ,256 ,200 ,304),(2 ,2048 ,25 ,38),
               (2 ,64 ,200 ,272),(2 ,128 ,152 ,100),(2 ,1024 ,76 ,50),(2 ,512 ,100 ,136),(2 ,512 ,25 ,34),(2 ,64 ,304 ,200),
               (2 ,256 ,200 ,272),(2 ,2048 ,25 ,34),(2 ,256 ,50 ,84),(2 ,256 ,68 ,50),(2 ,512 ,152 ,100),(2 ,512 ,38 ,25),
               (2 ,256 ,304 ,200),(2 ,2048 ,38 ,25),(2 ,128 ,100 ,168),(2 ,1024 ,50 ,84),(2 ,128 ,136 ,100),(2 ,1024 ,68 ,50),
               (2 ,256 ,50 ,78),(2 ,64 ,200 ,336),(2 ,64 ,272 ,200),(2 ,512 ,100 ,168),(2 ,512 ,25 ,42),(2 ,512 ,136 ,100)]

    for shape in shapelist:
        grad_out = torch.rand(size=shape).to(memory_format=torch.channels_last)

        inp = torch.rand(size=shape)
        inp.requires_grad=True
        mod = nn.BatchNorm2d(shape[1])
        mod.training=False
        out = mod(inp)
        out.backward(grad_out)

        inp1 = inp.detach().clone().cuda().to(memory_format=torch.channels_last)
        inp1.requires_grad=True
        mod1 = copy.deepcopy(mod).cuda()
        mod1.training=False
        out1 = mod1(inp1)
        out1.backward(grad_out.cuda().to(memory_format=torch.channels_last))

        flag0 = torch.allclose(out1.cpu().float(),out,1e-3,1e-3)
        flag1 = torch.allclose(inp1.grad.cpu().float(),inp.grad,1e-3,1e-3)
        flag2 = torch.allclose(mod1.weight.grad.cpu().float(),mod.weight.grad,1e-3,1e-3)
        flag3 = torch.allclose(mod1.bias.grad.cpu().float(),mod.bias.grad,1e-3,1e-3)

        if not (flag0 and flag1 and flag2 and flag3):
            exit(1)

def test_batchnorm_grad_nhwc():
    shape_list = [(138,166,256),(133,198, 256),(111,241,256),
                  (172,143, 256),(205,114, 256),(142,174,256)]
    for input_shape in shape_list:
        inp = torch.rand(input_shape, device="cuda", dtype=torch.bfloat16)
        inp.requires_grad = True
        tmp = inp.transpose(1,2)
        norm = nn.BatchNorm1d(256, device="cuda", dtype=torch.float32)
        back_input = torch.rand(input_shape, device = "cuda", dtype=torch.float32).transpose(1,2)
        out = norm(tmp)
        out.backward(back_input)

        inpg = inp.detach().clone().to(device="cuda", dtype=torch.bfloat16)
        inpg.requires_grad = True
        tmpg = inpg.transpose(1,2)
        normg = copy.deepcopy(norm).cuda()
        back_inputg = back_input.detach().clone().cuda().contiguous()
        outg = normg(tmpg)
        outg.backward(back_inputg)

        res0 = torch.allclose(outg,out,1e-2,1e-3)
        res1 = torch.allclose(inpg.grad,inp.grad,1e-2,1e-3)
        res2 = torch.allclose(normg.weight.grad, norm.weight.grad,1e-2,1e-3)
        res3 = torch.allclose(normg.bias.grad, norm.bias.grad,1e-2,1e-3)

        if not (res0 and res1 and res2 and res3):
            print("test_batchnorm_nhwc.py is error")
            exit(1)


test_half()
test_float()
test_batchnorm_grad_nhwc()
exit(0)

