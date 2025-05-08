import torch
import torch.nn as nn


def test_sigmoid_float():
    a = torch.rand(100000)

    inp = a.as_strided((20,3),(3,4))
    inp.requires_grad = True
    out = nn.Sigmoid()(inp)
    out.backward(torch.ones(out.shape))

    inp1 = inp.detach().cuda()
    inp1.requires_grad = True
    out1 = nn.Sigmoid()(inp1)
    out1.backward(torch.ones(out1.shape).cuda())

    max_ele = torch.abs((out1.cpu()-out).max())
    min_ele = torch.abs((out1.cpu()-out).min())

    max_ele1= torch.abs((inp1.grad.cpu()-inp.grad).max())
    min_ele1= torch.abs((inp1.grad.cpu()-inp.grad).min())

    if max_ele > 1e-5 or min_ele > 1e-5 or max_ele1 > 1e-5 or min_ele1 > 1e-5:
        print("test_sigmoid_float error")
        exit(1)


def test_sigmoid_half():
    a = torch.rand(100000)

    inp = a.as_strided((20,3),(3,4))
    inp.requires_grad = True
    out = nn.Sigmoid()(inp)
    out.backward(torch.ones(out.shape))

    inp1 = inp.detach().cuda().half()
    inp1.requires_grad = True
    out1 = nn.Sigmoid()(inp1)
    out1.backward(torch.ones(out1.shape).cuda())

    max_ele = torch.abs((out1.float().cpu()-out).max())
    min_ele = torch.abs((out1.float().cpu()-out).min())

    max_ele1= torch.abs((inp1.grad.float().cpu()-inp.grad).max())
    min_ele1= torch.abs((inp1.grad.float().cpu()-inp.grad).min())

    if max_ele > 1e-3 or min_ele > 1e-3 or max_ele1 > 1e-3 or min_ele1 > 1e-3:
        print("test_sigmoid_half error")
        exit(1)




test_sigmoid_float()
test_sigmoid_half()
exit(0)


