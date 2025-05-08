import torch
import argparse
torch.manual_seed(0)

def test_elementwise_broadcast_1_1(shape, dtype):
    input = torch.randn(shape, dtype=dtype, device="cuda")
    input_c = input.cpu()
    input = input.as_strided(shape, (0, ))
    output = torch.randn(shape, dtype=dtype, device="cuda")
    output_c = output.cpu()

    input_c = input_c.as_strided(shape, (0, ))

    output_c.copy_(input_c)

    output.copy_(input)

    if not torch.allclose(output.cpu(), output_c):
        return False

    return True


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--type", default="checkin", help="<checkin|daily>")

  args = parser.parse_args()

  dtypes = [torch.float32, torch.float16, torch.bfloat16]

  if args.type == "checkin":
    for dtype in dtypes:
      shapes = list(range(64, 1100, 64))
      shapes.extend([65, 131, 1025])
      for s in shapes:
        shape = [s]
        if not test_elementwise_broadcast_1_1(shape, dtype):
           print(f"----------Elementwise broadcast kernel error with shape:{shape}, dtype:{dtype}")
           exit(1)
  else:
    for dtype in dtypes:
      for s in range(63, 3000):
         shape = [s]
         if not test_elementwise_broadcast_1_1(shape, dtype):
           print(f"----------Elementwise broadcast kernel error with shape:{shape}, dtype:{dtype}")
           exit(1)
exit(0)
