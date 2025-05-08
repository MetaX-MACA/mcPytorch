import torch

input = torch.rand(8)
output_golden = torch.diag(input)

input_d = input.cuda()
output_d = torch.diag(input_d)
output = output_d.cpu()

if torch.allclose(output, output_golden):
    print("Passed: {}".format(__file__))
    exit(0)
else:
    print("Failed: {}".format(__file__))
    exit(1)