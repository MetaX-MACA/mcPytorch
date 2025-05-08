'''
This test use to watch tl.device_assert. It's disabled now during inductor codegen.
'''
import torch


@torch.compile
def t(input, index):
    return torch.gather(input, 1, index)

device = "cuda:0"
input = tensor = torch.rand((4, 128))
index = torch.tensor([[0, 1], [3, 4]])


output = t(input.to(device), index.to(device))
output_golden = torch.gather(input, 1, index)

if torch.allclose(output.cpu(), output_golden):
    print("Passed: {}".format(__file__))
    exit(0)
else:
    print("Failed: {}".format(__file__))
    exit(1)

