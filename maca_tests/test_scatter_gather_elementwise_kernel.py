import torch
import numpy as np
import argparse

torch.manual_seed(0)

def gather_elementwise_kernel(shape, dtype, dim = 0):
    a = torch.randn(shape, dtype=dtype, device="cuda")
    a_c = a.cpu()
    shape_min = min(np.array(shape))
    index = torch.randint(0, shape_min, shape, device="cuda")
    index_c = index.cpu()

    output = torch.randn(shape, dtype=dtype, device="cuda")
    output_c = output.cpu()

    output = torch.gather(a, dim, index)
    output_c = torch.gather(a_c, dim, index_c)

    if not torch.allclose(output_c, output.cpu()):
        print(f"scatter_gather_elementwise_kernel error with shape:{shape}, dtype:{dtype}!")
        exit(1)

def gather_elementwise_kernel_opt(shape, dtype, dim = 0):
    a = torch.randn(shape, dtype=dtype, device="cuda")
    a_c = a.cpu()
    shape_min = min(np.array(shape))
    index = torch.randint(0, shape_min, shape, device="cuda")
    index_c = index.cpu()
    index = index.as_strided(shape, (1, 0))
    index_c = index_c.as_strided(shape, (1, 0))

    output = torch.randn(shape, dtype=dtype, device="cuda")
    output_c = output.cpu()

    output = torch.gather(a, dim, index)
    output_c = torch.gather(a_c, dim, index_c)

    if not torch.allclose(output_c, output.cpu()):
        print(f"scatter_gather_elementwise_kernel_opt error with shape:{shape}, dtype:{dtype}!")
        exit(1)

def scatter_elementwise_kernel_opt(shape, dtype, dim = 0):
    # only dim=0 can get into the scatter opt kernel
    a = torch.randn(shape, dtype=dtype, device="cuda")
    a_c = a.cpu()
    shape_min = min(np.array(shape))
    index = torch.randint(0, shape_min, shape, device="cuda")
    for i in range(shape[0]):
        index[i, ] = torch.randperm(shape[1])
    index_c = index.cpu()
    index = index.as_strided(shape, (1, 0))
    # print(index)
    index_c = index_c.as_strided(shape, (1, 0))

    output = torch.randn(shape, dtype=dtype, device="cuda")
    output_c = output.cpu()

    output.scatter_add_(dim, index, a)
    output_c.scatter_add_(dim, index_c, a_c)

    if not torch.allclose(output_c, output.cpu()):
        print(f"scatter_elementwise_kernel_opt error with shape:{shape}, dtype:{dtype}!")
        exit(1)

def scatter_elementwise_kernel_opt_assign(shape, dtype, dim = 0):
    # only dim=0 can get into the scatter opt kernel
    a = torch.randn(shape, dtype=dtype, device="cuda")
    a_c = a.cpu()
    shape_min = min(np.array(shape))
    index = torch.randint(0, shape_min, shape, device="cuda")
    for i in range(shape[0]):
      index[i, :] = torch.randperm(shape[1])
    index_c = index.cpu()
    index = index.as_strided(shape, (1, 0))
    # print(index)
    index_c = index_c.as_strided(shape, (1, 0))

    output = torch.randn(shape, dtype=dtype, device="cuda")
    output_c = output.cpu()

    output.scatter_(dim, index, a)
    output_c.scatter_(dim, index_c, a_c)
    if not torch.allclose(output_c, output.cpu()):
        print(f"scatter_elementwise_kernel_opt_assign error with shape:{shape}, dtype:{dtype}!")
        exit(1)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--type", default="checkin", help="<checkin|daily>")

  args = parser.parse_args()

  dtypes = [torch.float, torch.double, torch.half, torch.bfloat16]

  if args.type == "checkin":
    shapes = [[63, 65], [31, 129, 255], [64, 32, 31, 64]]
    for dtype in dtypes:
      for shape in shapes:
          for dim in range(len(shape)):
            gather_elementwise_kernel(shape, dtype, dim)
    for dtype in [torch.float, torch.half, torch.bfloat16]:
      for s1 in [2, 4, 6, 7, 64, 513]:
        for i in range(1, 14):
          shape = [s1, i*64]
          for dim in range(len(shape)):
            gather_elementwise_kernel_opt(shape, dtype, dim)
    for dtype in [torch.float, torch.half, torch.bfloat16]:
      shapes = [[64, 64], [96, 96], [128, 128]]
      for shape in shapes:
        scatter_elementwise_kernel_opt(shape, dtype)
    for dtype in [torch.float, torch.half, torch.bfloat16]:
      shapes = [[1024, 1024], [960, 960], [896, 896], [13, 13], [64, 64], [128, 128], [127, 127]]
      for shape in shapes:
        scatter_elementwise_kernel_opt_assign(shape, dtype)
  else:
     # dim2
      for dtype in dtypes:
          for s0 in range(2, 64, 7):
            for s1 in range(2, 256, 33):
                shape = [s0, s1]
                for dim in range(2):
                  gather_elementwise_kernel(shape, dtype, dim)
      
      # dim3
      for dtype in dtypes:
          for s0 in range(2, 64, 7):
            for s1 in range(2, 256, 33):
                for s2 in range(2, 256, 33):
                  shape = [s0, s1, s2]
                  for dim in range(3):
                    gather_elementwise_kernel(shape, dtype, dim)
      
      for dtype in [torch.float, torch.half, torch.bfloat16]:
        for s1 in range(1, 2049, 13):
          for i in range(1, 14):
            shape = [s1, i*64]
            for dim in range(len(shape)):
              gather_elementwise_kernel_opt(shape, dtype, dim)
      
      for dtype in [torch.float, torch.half, torch.bfloat16]:
        shapes = [[64, 64], [96, 96], [128, 128], [8192, 8192]]
        for shape in shapes:
          scatter_elementwise_kernel_opt(shape, dtype)

      for dtype in [torch.float, torch.half, torch.bfloat16]:
        shapes = [[1024, 1024], [960, 960], [896, 896], [13, 13], [64, 64], [128, 128], [127, 127], 
                  [1088, 1088], [1152, 1152]]
        for shape in shapes:
          scatter_elementwise_kernel_opt_assign(shape, dtype)
