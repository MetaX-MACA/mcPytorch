import torch

# the shape include in bert large
shape = [24, 384, 1024]

def test_element_3_2_cast(dtype):
  a = torch.randn(shape, dtype=dtype)
  a = torch.as_strided(a, shape, (1, 1, 0))
  if dtype == torch.float16:
    btype = torch.bfloat16
  else:
    btype = torch.float16
  b = torch.randn(shape, dtype=btype)
  b = torch.as_strided(b, shape, (0, shape[2], 0))
  ref = a + b

  ad = a.cuda()
  bd = b.cuda()
  out = ad + bd

  if not torch.allclose(ref, out.cpu()):
    return False
  return True

if __name__ == "__main__":
  for dtype in [torch.float32, torch.float16, torch.bfloat16, ]:
    if not test_element_3_2_cast(dtype):
      exit(1)
  exit(0)
