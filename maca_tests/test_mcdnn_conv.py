#!/usr/bin/env python
import torch
import torch.nn as nn
import sys
import os
import copy
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import MMTest, BERT_TEST_BATCHSIZE, check_close
import contextlib

torch.set_printoptions(precision=5, threshold=10000)


def retain_global_cudnn_tf32(f):
    def func(*args, **kwargs):
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        f(*args, **kwargs)
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
    return func


@retain_global_cudnn_tf32
def test_mcdnn_conv2d_noncontinuous():

    def test_help(group_, shape_, out_channel_, size_, param_, dt_):
        s = shape_[1]
        shape_[1] = shape_[1] * group_

        inp = torch.rand(*shape_).to(memory_format=torch.channels_last)
        inp.requires_grad = True
        model = nn.Conv2d(shape_[1], out_channel_*group_, kernel_size=size_,stride=param_,padding=param_,dilation=param_,groups=group_).to(memory_format=torch.channels_last)
        out = model(inp)
        out.backward(torch.ones(out.shape))
        shape_[1] = s
        
        def test_no_tf32():
            torch.backends.cudnn.allow_tf32=False  
            tol = 1e-4
            inp1 = inp.detach().clone().cuda().to(dtype=dt_)
            inp1.requires_grad = True
            model1 = copy.deepcopy(model).cuda().to(dtype=dt_)
            out1 = model1(inp1)
            out1.backward(torch.ones(out1.shape).cuda().to(dtype=dt_))

            if dt is torch.half or dt is torch.bfloat16:
                tol = 1e-2
            
            res1 = check_close(out1.float().cpu(), out, tol)
            res2 = check_close(inp1.grad.float().cpu(), inp.grad, tol)
            res3 = check_close(model1.weight.grad.float().cpu(), model.weight.grad, tol)

            if not(res1 and res2 and res3):
                print("conv2d precision error: ")
                print("torch.backends.cudnn.allow_tf32=False")
                print("input shape: ",inp.shape)
                print("in_channel: ", shape_[1] * group_)
                print("out_chanel: ", out_channel_*group_)
                print("kernel_size: ",size_)
                print("stride,padding,dilation: ",param_)
                print("group: ",group)
                exit(1)
        
        def test_tf32():
            torch.backends.cudnn.allow_tf32=True 
            tol = 1e-3
            inp1 = inp.detach().clone().cuda().to(dtype=dt_)
            inp1.requires_grad = True
            model1 = copy.deepcopy(model).cuda().to(dtype=dt_)
            out1 = model1(inp1)
            out1.backward(torch.ones(out1.shape).cuda().to(dtype=dt_))

            if dt is torch.half or dt is torch.bfloat16:
                tol = 1e-2
            
            res1 = check_close(out1.float().cpu(), out, tol)
            res2 = check_close(inp1.grad.float().cpu(), inp.grad, tol)
            res3 = check_close(model1.weight.grad.float().cpu(), model.weight.grad, tol)
            if not(res1 and res2 and res3):
                print("conv2d precision error: ")
                print("torch.backends.cudnn.allow_tf32=True")
                print("input shape: ",inp.shape)
                print("in_channel: ", shape_[1] * group_)
                print("out_chanel: ", out_channel_*group_)
                print("kernel_size: ",size_)
                print("stride,padding,dilation: ",param_)
                print("group: ",group)
                exit(1) 
        
        test_no_tf32()
        test_tf32()
    
    groups=[2,3]
    shapes = [[1,2,57, 57],[3,4,66,77]]
    out_channels = [2,6]
    sizes = [(8,9),(5,5)]
    params = [(3,4)]
    dtypes = [torch.float, torch.double, torch.half, torch.bfloat16]

    for group in groups:
        for shape in shapes:
            for out_channel in out_channels:
                for size in sizes:
                    for param in params:
                        for dt in dtypes:
                            test_help(group, shape, out_channel, size, param, dt)


def test_mcdnn_conv2d_nhwc_c1():
    shape_list=[(3,1,4,5),(10,1,20,30),(10,1,20,40)]
    for shape in shape_list:
        inp = torch.rand(*shape)
        inp.requires_grad = True
        model = nn.Conv2d(1, 2, kernel_size=(3))
        out=model(inp)
        out.backward(torch.ones(out.shape))

        inp1 = inp.detach().clone().cuda().to(dtype=torch.half)
        inp1.requires_grad = True
        model1 = copy.deepcopy(model).cuda().to(dtype=torch.half)
        out1 = model1(inp1)
        out1.backward(torch.ones(out1.shape).cuda().to(dtype=torch.half))

        res1 = check_close(out1.float().cpu(), out, 1e-3)
        res2 = check_close(inp1.grad.float().cpu(), inp.grad, 1e-3)
        res3 = check_close(model1.weight.grad.float().cpu(), model.weight.grad, 1e-3)
        print(res1)
        print(res2)
        print(res3)
        if not(res1 and res2 and res3):
            print("test_mcdnn_conv2d_nhwc_c1 error: ")
            exit(1) 


test_mcdnn_conv2d_noncontinuous()
test_mcdnn_conv2d_nhwc_c1()
exit(0)


