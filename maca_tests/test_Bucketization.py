import torch

sorted_sequence = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
values = torch.tensor([[3, 6, 9], [3, 6, 9]])

output_golden = torch.searchsorted(sorted_sequence, values)

sorted_sequence_d = sorted_sequence.cuda()
values_d = values.cuda()
output_d = torch.searchsorted(sorted_sequence_d, values_d)
output = output_d.cpu()

if torch.allclose(output, output_golden):
    print("Passed: {}".format(__file__))
    exit(0)
else:
    print("Failed: {}".format(__file__))
    exit(1)
