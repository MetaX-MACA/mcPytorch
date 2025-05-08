import torch
import torch.nn.functional as F
torch.manual_seed(0)

for dtype in [torch.float16, torch.bfloat16, torch.float32]:
  for rows in [1, 7, 64, 127, 128, 255, 256, 511, 512]:
    for cols in list(range(1,1024)) + [1025, 1224, 1448, 1627, 1811, 1919, 2047, 2048, 3071, 3072, 4095, 4096]:
      input_d = torch.randn(rows, cols, dtype = dtype, device='cuda:0')
      input_d.requires_grad = True
      output_d = input_d.softmax(dim=-1)
      output_d.sum().backward()

      input_cpu = input_d.detach().cpu().float()
      input_cpu.requires_grad = True
      output_golden = input_cpu.softmax(dim=-1)
      output_golden.sum().backward()
      if dtype == torch.float16:
        print("type:{}, rows:{}, cols:{}, max diff:{}".format(dtype, rows, cols, torch.max(input_d.grad.cpu().float() - input_cpu.grad)))
        if not torch.allclose(input_d.grad.cpu().float(), input_cpu.grad, rtol=5e-3, atol=5e-3):
          exit(1)
      elif dtype == torch.bfloat16:
        print("type:{}, rows:{}, cols:{}, max diff:{}".format(dtype, rows, cols, torch.max(input_d.grad.cpu().float() - input_cpu.grad)))
        if not torch.allclose(input_d.grad.cpu().float(), input_cpu.grad, rtol=5e-3, atol=5e-3):
          exit(1)
      else:
        print("type:{}, rows:{}, cols:{}, max diff:{}".format(dtype, rows, cols, torch.max(input_d.grad.cpu().float() - input_cpu.grad)))
        if not torch.allclose(input_d.grad.cpu().float(), input_cpu.grad, rtol=5e-3, atol=5e-3):
          exit(1)
exit(0)
