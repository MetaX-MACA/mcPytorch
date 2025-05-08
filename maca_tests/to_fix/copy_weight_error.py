# Bug description:
# 1. In MACA(cmodel) environment, because the conv.weight.data.requires_grad is False,
#    when it run in the backward, the function ToCopyBackward0 is not generated.
#    (mcFramwork/pytorch/torch/csrc/autograd/generated/Variabletype_0.cpp:1299)
#    (mcFramwork/pytorch/torch/csrc/autograd/generated/Functions.cpp:840)
#    Therefore, when the net do the conv, input x is in the cpu but the weight is in the cuda.
# 2. In the cuda environment, the program does not report this error, however when we use gdb
#    to debug in cuda enviroment, the program will report this error.


# Bug demo:
import torch
import torch.nn as nn


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        x = x.cpu()
        self.conv.weight.data = self.conv.weight.data.cpu()
        print("!!!!:", self.conv.weight.data.requires_grad)
        x = self.conv(x)
        x = x.cuda()
        self.conv.weight.data = self.conv.weight.data.cuda()
        print("!!!!:", self.conv.weight.data.requires_grad)

        x = self.bn(x)
        return x


if __name__ == "__main__":
    input_tensor = torch.rand(1, 3, 224, 224, dtype=torch.float32)
    input_tensor.requires_grad_()
    input_tensor = input_tensor.cuda()
    golden_tensor = torch.rand(1, 64, 112, 112, dtype=torch.float32)
    golden_tensor = golden_tensor.cuda()
    test_module = TestNet().cuda()
    test_module.train()
    output_tensor = test_module(input_tensor)
    criterion = nn.MSELoss(reduction='none')
    loss = criterion(golden_tensor, output_tensor)
    loss = loss.sum()
    loss.backward()
    print("check out finish, loss:", loss)
