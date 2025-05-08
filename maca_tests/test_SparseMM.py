import torch

indices = torch.tensor([[4, 2, 1], [2, 0, 2]])
values = torch.tensor([3, 4, 5], dtype=torch.float32)
input = torch.sparse_coo_tensor(indices=indices, values=values, size=[5, 5], dtype=torch.float32)
mat1 = torch.sparse_coo_tensor(indices=indices, values=values, size=[5, 5], dtype=torch.float32)
mat2 = torch.rand(5, 5, dtype=torch.float32)

output_golden = torch.sspaddmm(input, mat1, mat2)

input_d = input.cuda()
mat1_d = mat1.cuda()
mat2_d = mat2.cuda()
output_d = torch.sspaddmm(input_d, mat1_d, mat2_d)
output = output_d.cpu()

if torch.allclose(output, output_golden):
    exit(0)
else:
    exit(1)
