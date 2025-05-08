import torch
from torch.testing._internal.common_utils import TestCase
import argparse

def test_elementwise_2_2_broadcast():
  inp = torch.randn(4096, 8192).cuda()
  inp = torch.as_strided(inp, (4096, 8192), (1, 4096))
  inp2 = torch.randn(4096, 8192).cuda().to(torch.half)
  inp2 = torch.as_strided(inp2, (4096, 8192), (1, 0))
  out = (inp2*inp)

  inp_cpu = inp.cpu()
  inp2_cpu = inp2.cpu()

  out_cpu = (inp2_cpu*inp_cpu)

  TestCase().assertEqual(out, out_cpu)


  inp = torch.randn(4096, 8192).cuda().to(torch.half)
  inp = torch.as_strided(inp, (4096, 8192), (1, 4096))
  inp2 = torch.randn(4096, 8192).cuda()
  inp2 = torch.as_strided(inp2, (4096, 8192), (0, 1))
  out = (inp*inp2)
  inp_cpu = inp.cpu()
  inp2_cpu = inp2.cpu()

  out_cpu = (inp_cpu*inp2_cpu)

  TestCase().assertEqual(out, out_cpu)

def test_elementwise_2_2_broadcast_bool(shape):
  inp1_base = torch.rand(shape, dtype=torch.float, device="cuda")
  inp2_base = torch.rand(shape, dtype=torch.float, device="cuda") > 0.5

  inp1_c = inp1_base.cpu()
  inp2_c = inp2_base.cpu()

  inp1 = inp1_base.as_strided(shape, [shape[1], 0])
  inp2 = inp2_base.as_strided(shape, [0, 1])

  inp1_c = inp1_c.as_strided(shape, [shape[1], 0])  
  inp2_c = inp2_c.as_strided(shape, [0, 1])

  out = inp1 * inp2
  out_c = inp1_c * inp2_c

  assert torch.allclose(out.cpu(), out_c)

def test_elementwise_2_2_broadcast_template(shape, test_type="uncontiguous"):
  a = torch.randn(shape[0] * 2, shape[1] * 2, dtype=torch.float, device="cuda")
  a_c = a.cpu()
  if test_type == "contiguous":
      a = a.as_strided(shape, (1, shape[0]))
  else:
      a = a.as_strided(shape, (2, shape[0]*2))
  b = torch.randn(shape, dtype=torch.float16, device="cuda")
  b_c = b.cpu()
  b = b.as_strided(shape, (1, 0))

  if test_type == "contiguous":
      a_c = a_c.as_strided(shape, (1, shape[0]))
  else:
      a_c = a_c.as_strided(shape, (2, shape[0]*2))
  b_c = b_c.as_strided(shape, (1, 0))

  output = a * b
  output_c = a_c * b_c
  if not torch.allclose(output.cpu(), output_c):
    print(f"test_elementwise_2_2_broadcast_template error with shape:{shape}!")
    exit(1)

def test_elementwise_2_2_broadcast_template_not_align(shape, test_type="uncontiguous", i=1):
  a = torch.randn(shape[0] * 2 + i, shape[1] * 2 + i, dtype=torch.float, device="cuda")
  a_c = a.cpu()
  a = a[i:shape[0] * 2 + i, i:shape[1] * 2 + i]
  if test_type == "contiguous":
      a = a.as_strided(shape, (1, shape[0]))
  else:
      a = a.as_strided(shape, (2, shape[0]*2))
  b = torch.randn(shape, dtype=torch.float16, device="cuda")
  b_c = b.cpu()
  b = b.as_strided(shape, (1, 0))

  a_c = a_c[i:shape[0] * 2 + i, i:shape[1] * 2 + i]
  if test_type == "contiguous":
      a_c = a_c.as_strided(shape, (1, shape[0]))
  else:
      a_c = a_c.as_strided(shape, (2, shape[0]*2))
  b_c = b_c.as_strided(shape, (1, 0))

  output = a * b
  output_c = a_c * b_c
  if not torch.allclose(output.cpu(), output_c):
    print(f"test_elementwise_2_2_broadcast_template error with shape:{shape}!")
    exit(1)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--type", default="checkin", help="<checkin|daily>")

  args = parser.parse_args()

  if args.type == "checkin":
    test_elementwise_2_2_broadcast()
    shapes = [[192,84],[156,144],[52,64]]
    for shape in shapes:
      test_elementwise_2_2_broadcast_bool(shape)
    shapes = [[6144,2560], [22528,1536], [16384,2560]]
    for shape in shapes:
      test_elementwise_2_2_broadcast_template(shape)
    shapes = [[4096,2560],]
    for shape in shapes:
      test_elementwise_2_2_broadcast_template(shape, "contiguous")
  else:
    for s0 in range(4, 513, 4):
      for i in range(1, 5):
         shape = [s0, s0 + i]
         test_elementwise_2_2_broadcast_template(shape)
         test_elementwise_2_2_broadcast_template(shape, "contiguous")
         test_elementwise_2_2_broadcast_template_not_align(shape)
         test_elementwise_2_2_broadcast_template_not_align(shape, "contiguous")