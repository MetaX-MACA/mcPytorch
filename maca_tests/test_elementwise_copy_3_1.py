import torch
import copy
import argparse
import math

def test_elementwise_copy_3_1_transpose012(shape, dtype):
  input1 = torch.randn(shape[0], shape[1], shape[2] * shape[1], dtype=dtype, device='cuda')
  input2 = input1.as_strided(shape, (1, shape[1], shape[1] * shape[2]))
  input3 = input1.as_strided(shape, (1, shape[1] // 2, shape[1] * shape[2]))
  output = torch.randn(shape, dtype=dtype, device='cuda')
  output_c = copy.deepcopy(output).cpu()
  output.copy_(input2)
  # acc check
  input_c = input1.cpu()
  input_c_1 = input_c.as_strided(shape, (1, shape[1], shape[1] * shape[2]))
  input_c_2 = input_c.as_strided(shape, (1, shape[1] // 2, shape[1] * shape[2]))
  output_c.copy_(input_c_1)
  if not torch.allclose(output_c, output.cpu()):
      return False

  output.copy_(input3)
  output_c.copy_(input_c_2)
  if not torch.allclose(output_c, output.cpu()):
      return False

  return True

def test_elementwise_3_1_template(shape, output_stride, input_stride, dtype):
    shape_mul = shape[0] * shape[1]
    output_max = max(output_stride)
    output_max_index = output_stride.index(output_max)
    output_ceil = math.ceil(output_max / shape_mul)
    input_max = max(input_stride)
    input_max_index = input_stride.index(input_max)
    input_ceil = math.ceil(input_max / shape_mul)

    if input_ceil > 1:
        input = torch.randn(shape[0] * round(input_max / shape_mul) * shape[input_max_index], shape[1], shape[2], dtype=dtype, device="cuda")
    else:
        input = torch.randn(shape[0], shape[1], shape[2], dtype=dtype, device="cuda")
    input_c = input.cpu()
    input = input.as_strided(shape, input_stride)
    if output_ceil > 1:
        output = torch.randn(shape[0] * round(output_max / shape_mul) * shape[output_max_index] , shape[1], shape[2], dtype=dtype, device="cuda")
    else:
        output = torch.randn(shape[0] , shape[1], shape[2], dtype=dtype, device="cuda")
    output_c = output.cpu()
    output = output.as_strided(shape, output_stride)

    input_c = input_c.as_strided(shape, input_stride)
    output_c = output_c.as_strided(shape, output_stride)

    output_c.copy_(input_c)
    output.copy_(input)

    if not torch.allclose(output.cpu(), output_c):
        print("Error in test_elementwise_3_1!")
        exit(1)

def test_elementwise3_1_transpose_copy_8():
    inp_base = torch.rand(100000000,device="cuda",dtype=torch.bfloat16)
    out_base = torch.rand(100000000,device="cuda",dtype=torch.bfloat16)

    shape_list = [[128, 1056, 512], [128, 512, 1056], [32, 24, 24], [33, 24, 32],
                  [34,  24,    40],  [34,  24, 48], [35, 24, 56], [35, 24, 72],
                  [71,  72,   80],   [71, 40, 160], [71, 160, 24], [72, 168, 32],
                  [10,  24,  40],   [10, 240, 400], [10, 152, 264], [11, 408, 88]]

    for shape in shape_list:
        inp = inp_base.as_strided(shape,(shape[1]*shape[2],1,shape[1]))
        out = out_base.as_strided(shape,(shape[1]*shape[2],shape[2],1))

        inpc = inp.detach().clone().cpu()
        outc = out.detach().clone().cpu()

        out.copy_(inp)
        outc.copy_(inpc)

        res = torch.allclose(out.cpu(),outc)
        if not res:
            print("test_elementwise3_1_transpose_copy_8 error")
            exit(1)

def test_elementwise_3_1_dim0_contiguous():
    shape_stride_list = [[(32, 192, 80), (670720, 80, 1), (0, 240, 1)], [(32, 8192, 80), (670720, 80, 1), (1966080, 240, 1)]]
    for shape, stride_out, stride_in in shape_stride_list:
        inp_base = torch.rand(100000000,device="cuda",dtype=torch.bfloat16)
        out_base = torch.rand(100000000,device="cuda",dtype=torch.bfloat16)

        inp_base_c = inp_base.detach().clone().cpu()
        out_base_c = out_base.detach().clone().cpu()

        inp = inp_base.as_strided(shape, stride_in)
        out = out_base.as_strided(shape, stride_out)

        inpc = inp_base_c.as_strided(shape, stride_in)
        outc = out_base_c.as_strided(shape, stride_out)

        out.copy_(inp)
        outc.copy_(inpc)

        res = torch.allclose(out.cpu(),outc)
        if not res:
            print("test_elementwise_3_1_dim0_contiguous error")
            exit(1)

def test_elementwise_3_1_transpose12_half_copy(shape, dtype):
    inp_base = torch.rand(100000000,device="cuda",dtype=dtype)
    out_base = torch.rand(100000000,device="cuda",dtype=dtype)

    inp_base_c = inp_base.detach().clone().cpu()
    out_base_c = out_base.detach().clone().cpu()
    
    for alpha_stride in [1, 2, 3]:
        inp = inp_base.as_strided(shape, (shape[2], alpha_stride * shape[0] * shape[2], 1))
        out = out_base.as_strided(shape, (shape[2] * shape[1], shape[2], 1))

        input_c = inp_base_c.as_strided(shape, (shape[2], alpha_stride * shape[0] * shape[2], 1))
        output_c = out_base_c.as_strided(shape, (shape[2] * shape[1], shape[2], 1))

        output_c.copy_(input_c)
        out.copy_(inp)

        if not torch.allclose(out.cpu(), output_c):
            print("test_elementwise_3_1_transpose12_half_copy Error!")
            exit(1)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--type", default="checkin", help="<checkin|daily>")

  args = parser.parse_args()

  # 3_1_transpose012_copy only for float16 and bfloat16
  dtypes = [torch.float16, torch.bfloat16]
  if args.type == "checkin":
    for dtype in dtypes:
      for s0 in [32, 64, 96, 191]:
          for s1 in [64, 96, 191]:
              for s2 in [64, 96, 191]:
                  shape = [s0,s1,s2]
                  if not test_elementwise_copy_3_1_transpose012(shape, dtype):
                    print("Error")
                    exit(1)
  else:
    for dtype in dtypes:
      for s0 in [64, 63, ]:
          for s1 in [2, 3]:
              for s2 in [-1, 0, 1, 2, 3]:
                shape = [s0,s1,s2+s0*4*s1]
                test_elementwise_3_1_template(shape, (1, shape[0] * 4, shape[2]), (1, shape[0] * 2, shape[2]), dtype)
                test_elementwise_3_1_template(shape, (1, shape[0] * 4, shape[2] * 2), (1, shape[0] * 2, shape[2] * 2), dtype)
                test_elementwise_3_1_template(shape, (1, shape[0] * 4, shape[2] * 4), (1, shape[0] * 2, shape[2] * 4), dtype)

    for dtype in dtypes:
        for shape in [(4, 1323, 1280), (1536, 4, 1280)]:
            test_elementwise_3_1_transpose12_half_copy(shape, dtype)

  test_elementwise3_1_transpose_copy_8() 
  test_elementwise_3_1_dim0_contiguous()
  exit(0)
