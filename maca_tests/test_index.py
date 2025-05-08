import torch
import argparse

torch.manual_seed(0)
torch.cuda.manual_seed(0)


def test_index_1_2_0(shape, dtype):
  src = torch.randn(shape, device="cuda", dtype=dtype)
  index_max = min(shape)
  index = torch.randint(0, index_max, (shape[0], ), device="cuda")
  res = src[index, :]

  src_c = src.cpu()
  index_c = index.cpu()
  res_c = src_c[index_c, :]

  if not torch.allclose(res.cpu(), res_c):
      print(f"Error with shape:{shape}, dtype:{dtype} test_index_1_2_0!")
      exit(1)

def test_index_1_2_1(shape, dtype):
  src = torch.randn(shape, device="cuda", dtype=dtype)
  index_max = min(shape)
  index = torch.randint(0, index_max, (shape[1], ), device="cuda")
  res = src[:, index]
  src_c = src.cpu()
  index_c = index.cpu()
  res_c = src_c[:, index_c]

  if not torch.allclose(res.cpu(), res_c):
      print(f"Error with shape:{shape}, dtype:{dtype} test_index_1_2_1!")
      exit(1)

def test_index_1_2_0_bool(shape):
  src = torch.randn(shape, device="cuda", dtype=torch.float).to(torch.int8)
  index_max = min(shape)
  index = torch.randint(0, index_max, (shape[0], ), device="cuda")
  res = src[index, :]

  src_c = src.cpu()
  index_c = index.cpu()
  res_c = src_c[index_c, :]

  if not torch.allclose(res.cpu(), res_c):
      print(f"Error with shape:{shape}, dtype:{dtype} test_index_1_2_0!")
      exit(1)

def test_index_1_2_1_bool(shape):
  src = torch.randn(shape, device="cuda", dtype=torch.float).to(torch.int8)
  index_max = min(shape)
  index = torch.randint(0, index_max, (shape[1], ), device="cuda")
  res = src[:, index]
  src_c = src.cpu()
  index_c = index.cpu()
  res_c = src_c[:, index_c]

  if not torch.allclose(res.cpu(), res_c):
      print(f"Error with shape:{shape}, dtype:{dtype} test_index_1_2_1!")
      exit(1)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--type", default="checkin", help="<checkin|daily>")

  args = parser.parse_args()

  dtypes = [torch.float, torch.float16, torch.bfloat16]
  if args.type == "checkin":
     for s0 in range(2, 128, 3):
        for s1 in range(2, 128, 3):
          shape = [s0, s1]
          test_index_1_2_0_bool(shape)
          test_index_1_2_1_bool(shape)
          for dtype in dtypes:
            test_index_1_2_0(shape, dtype)
            test_index_1_2_1(shape, dtype)
  else:
     for s0 in range(2, 512, 5):
        for s1 in range(2, 512, 5):
          shape = [s0, s1]
          test_index_1_2_0_bool(shape)
          test_index_1_2_1_bool(shape)
          for dtype in dtypes:
            test_index_1_2_0(shape, dtype)
            test_index_1_2_1(shape, dtype)
