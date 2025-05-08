import torch
import torch.nn.functional as F


def run(input, rows, eps = 1e-3):
  def impl(input, rows):
    t = input.expand(rows, input.shape[0])
    output = torch.zeros_like(t)
    return output.copy_(t)
  
  out_g = impl(input.cuda(), rows)
  out = impl(input, rows)
  return torch.allclose(out_g.cpu(), out, rtol=eps, atol=eps)


# boardcast copy only support half, and cols must be even and >=64
for dtype in [torch.float16, torch.bfloat16, torch.float32]:
  for rows in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 64, 65, 127, 128, 256, 257, 258, 259, 4096, 5000]:
    for cols in [64, 66, 128, 256, 258, 4096, 5000, 30522, 30524, 30526, 30528]:
      input = torch.randn(cols, dtype = dtype, device="cpu")
      status = run(input, rows)
      if not status:
        print(f"$$$$ rows: {rows}, {cols}, fail")
        exit(1)
print("#### pass")
exit(0)
