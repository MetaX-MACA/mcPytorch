import torch
from torch.testing._internal.common_utils import TestCase

shape_list = [(5), (11, 17), (64, 43), (128, 55, 23), (256, 64, 32, 64), (255, 33, 55, 74)]


for shape in shape_list:
    x = torch.randn(shape)
    x_cuda = x.cuda()
    y = torch.topk(x, 3)
    y_cuda = torch.topk(x_cuda, 3)
    TestCase().assertEqual(y, y_cuda)
