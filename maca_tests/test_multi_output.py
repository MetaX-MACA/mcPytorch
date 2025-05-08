#!/usr/bin/env python
import torch
import torch.nn as nn
import os
import copy

torch.set_printoptions(precision=5, threshold=10000)


def test_element_nocontiguous_out2_arity1_dim2():
    inp1 = torch.rand(20,3,5,6,4,5).cuda()*100
    inp1 = inp1.contiguous().as_strided((2,7), (2,6))
    man1,exp1=torch.frexp(inp1)

    inp = inp1.cpu()
    man,exp=torch.frexp(inp)

    man_diff=man1.cpu()-man
    exp_diff=exp1.cpu()-exp
    max_ele1 = float(torch.abs(man_diff.max()))
    min_ele1 = float(torch.abs(man_diff.min()))
    max_ele2 = float(torch.abs(exp_diff.max()))
    min_ele2 = float(torch.abs(exp_diff.min()))
    if max_ele1 > 1e-8 or min_ele1 > 1e-8 or max_ele2 > 1e-8 or min_ele2 > 1e-8:
        print("test_element_nocontiguous_out2_arity1_dim2 test error")
        exit(1)


def test_element_nocontiguous_out2_arity1_dim3():
    inp1 = torch.rand(20,3,5,6,4,5).cuda()*100
    inp1 = inp1.contiguous().as_strided((2,3,5), (2,4,4))
    man1,exp1=torch.frexp(inp1)

    inp = inp1.cpu()
    man,exp=torch.frexp(inp)
    
    man_diff=man1.cpu()-man
    exp_diff=exp1.cpu()-exp
    max_ele1 = float(torch.abs(man_diff.max()))
    min_ele1 = float(torch.abs(man_diff.min()))
    max_ele2 = float(torch.abs(exp_diff.max()))
    min_ele2 = float(torch.abs(exp_diff.min()))
    if max_ele1 > 1e-8 or min_ele1 > 1e-8 or max_ele2 > 1e-8 or min_ele2 > 1e-8:
        print("test_element_nocontiguous_out2_arity1_dim3 test error")
        exit(1)
    



def test_element_nocontiguous_out2_arity1_dim4():
    inp1 = torch.rand(20,3,5,6,4,5,dtype=torch.float64).cuda()*100
    inp1 = inp1.contiguous().as_strided((2,3,4,5), (2,4,1,3))
    man1,exp1=torch.frexp(inp1)

    inp = inp1.cpu()
    man,exp=torch.frexp(inp)

    man_diff=man1.cpu()-man
    exp_diff=exp1.cpu()-exp
    max_ele1 = float(torch.abs(man_diff.max()))
    min_ele1 = float(torch.abs(man_diff.min()))
    max_ele2 = float(torch.abs(exp_diff.max()))
    min_ele2 = float(torch.abs(exp_diff.min()))
    if max_ele1 > 1e-8 or min_ele1 > 1e-8 or max_ele2 > 1e-8 or min_ele2 > 1e-8:
        print("test_element_nocontiguous_out2_arity1_dim4 test error")
        exit(1)



def test_element_contiguous():
    inp1 = torch.rand(2,3,5,6).cuda()*100
    man1,exp1=torch.frexp(inp1)
    
    inp = inp1.cpu()
    man,exp=torch.frexp(inp)

    man_diff=man1.cpu()-man
    exp_diff=exp1.cpu()-exp
    max_ele1 = float(torch.abs(man_diff.max()))
    min_ele1 = float(torch.abs(man_diff.min()))
    max_ele2 = float(torch.abs(exp_diff.max()))
    min_ele2 = float(torch.abs(exp_diff.min()))
    if max_ele1 > 1e-8 or min_ele1 > 1e-8 or max_ele2 > 1e-8 or min_ele2 > 1e-8:
        print("test_element_nocontiguous_out2_arity1_dim4 test error")
        exit(1)



test_element_nocontiguous_out2_arity1_dim2()
test_element_nocontiguous_out2_arity1_dim3()
test_element_nocontiguous_out2_arity1_dim4()
test_element_contiguous()
exit(0)


