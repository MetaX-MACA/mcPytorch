import torch
import torch.nn as nn
import os
import copy
import argparse
import math

torch.manual_seed(0)
torch.set_printoptions(precision=5, threshold=10000)

def test_elementwise_dim4_copy_long():
    dim0_list = [2, 4, 9, 16]
    dim1_list = [101, 516, 517, 1024]
    dim2_list = [5, 16, 17, 32]
    dim3_list = [17, 64, 128, 129]
    trans_list = [(1,2),(1,3),(0,1),(0,2)]

    for dim0 in dim0_list:
        for dim1 in dim1_list:
            for dim2 in dim2_list:
                for dim3 in dim3_list:
                    for trans in trans_list:
                        print(trans)
                        a=torch.rand(dim0,dim1,dim2,dim3).half()
                        a=a.contiguous()
                        b=a.transpose(trans[0],trans[1]).contiguous()

                        a1= a.detach().clone().cuda()
                        a1= a1.contiguous()
                        b1= a1.transpose(trans[0],trans[1]).contiguous()

                        out=b1.cpu()-b
                        max_ele = float(torch.abs(out.max()))
                        min_ele = float(torch.abs(out.min()))
                        if max_ele > 1e-8 or min_ele > 1e-8:
                            print("input size is [{},{},{},{}]".format(dim0,dim1,dim2,dim3))
                            print("trans size is ({},{})".format(trans[0],trans[1]))
                            exit(1)

def test_elementwise_dim4_copy():
    dim0_list = [2, 16]
    dim1_list = [101, 1024]
    dim2_list = [5, 32]
    dim3_list = [64, 128]
    trans_list = [(1,2)]

    for dim0 in dim0_list:
        for dim1 in dim1_list:
            for dim2 in dim2_list:
                for dim3 in dim3_list:
                    for trans in trans_list:
                        print(trans)
                        a=torch.rand(dim0,dim1,dim2,dim3).half()
                        a=a.contiguous()
                        b=a.transpose(trans[0],trans[1]).contiguous()

                        a1= a.detach().clone().cuda()
                        a1= a1.contiguous()
                        b1= a1.transpose(trans[0],trans[1]).contiguous()

                        out=b1.cpu()-b
                        max_ele = float(torch.abs(out.max()))
                        min_ele = float(torch.abs(out.min()))
                        if max_ele > 1e-8 or min_ele > 1e-8:
                            print("input size is [{},{},{},{}]".format(dim0,dim1,dim2,dim3))
                            print("trans size is ({},{})".format(trans[0],trans[1]))
                            exit(1)

def test_elementwise_broadcast_4_1(shape, dtype):
    input = torch.randn(shape, dtype=dtype, device="cuda")
    input = input.as_strided(shape, (1, shape[0] * shape[2], shape[0], shape[0] * shape[1] * shape[2]))
    output = torch.randn(shape, dtype=dtype, device="cuda")
    output = output.as_strided(shape, (1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2]))

    input_c = input.cpu()
    input_c = input_c.as_strided(shape, (1, shape[0] * shape[2], shape[0], shape[0] * shape[1] * shape[2]))
    output_c = output.cpu()
    output_c = output_c.as_strided(shape, (1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2]))

    output_c.copy_(input_c)
    output.copy_(input)

    if not torch.allclose(output.cpu(), output_c):
        print("Error in test_elementwise_broadcast_4_1!")
        exit(1)

def test_elementwise_4_1_template(shape, output_stride, input_stride, dtype):
    shape_mul = shape[0] * shape[1] * shape[2]
    output_max = max(output_stride)
    output_max_index = output_stride.index(output_max)
    output_ceil = math.ceil(output_max / shape_mul)
    input_max = max(input_stride)
    input_max_index = input_stride.index(input_max)
    input_ceil = math.ceil(input_max / shape_mul)

    if input_ceil > 1:
        input = torch.randn(shape[0] * round(input_max / shape_mul) * shape[input_max_index], shape[1], shape[2], shape[3], dtype=dtype, device="cuda")
    else:
        input = torch.randn(shape[0], shape[1], shape[2], shape[3], dtype=dtype, device="cuda")
    input_c = input.cpu()
    input = input.as_strided(shape, input_stride)
    if output_ceil > 1:
        output = torch.randn(shape[0] * round(output_max / shape_mul) * shape[output_max_index] , shape[1], shape[2], shape[3], dtype=dtype, device="cuda")
    else:
        output = torch.randn(shape[0] , shape[1], shape[2], shape[3], dtype=dtype, device="cuda")
    output_c = output.cpu()
    output = output.as_strided(shape, output_stride)

    input_c = input_c.as_strided(shape, input_stride)
    output_c = output_c.as_strided(shape, output_stride)

    output_c.copy_(input_c)
    output.copy_(input)

    if not torch.allclose(output.cpu(), output_c):
        print("Error in test_elementwise_4_1_template!")
        exit(1)

def test_elementwise_4_2(shape, dtype):
    a = torch.randn(shape, dtype=dtype, device="cuda")
    a_c = a.cpu()
    a = a.as_strided(shape, (1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2]))
    b = torch.randn(shape, dtype=dtype, device="cuda")
    b_c = b.cpu()
    b = b.as_strided(shape, (1, shape[0] * shape[2], shape[0], shape[0] * shape[1] * shape[2]))

    output = a + b

    a_c = a_c.as_strided(shape, (1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2]))
    b_c = b_c.as_strided(shape, (1, shape[0] * shape[2], shape[0], shape[0] * shape[1] * shape[2]))
    output_c = a_c + b_c

    if not torch.allclose(output_c, output.cpu()):
        print("Error in test_elementwise_4_2!")
        exit(1)

def test_elementwise_4_2_not_align(shape, dtype, i=1):
    a = torch.randn(shape[0] + i, shape[1] + i, shape[2] + i, shape[3] + i, dtype=dtype, device="cuda")
    a_c = a.cpu()
    a = a[i:shape[0]+i, i:shape[1]+i, i:shape[2]+i, i:shape[2]+i]
    a = a.as_strided(shape, (1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2]))
    b = torch.randn(shape, dtype=dtype, device="cuda")
    b_c = b.cpu()
    b = b.as_strided(shape, (1, shape[0] * shape[2], shape[0], shape[0] * shape[1] * shape[2]))

    output = a + b

    a_c = a_c[i:shape[0]+i, i:shape[1]+i, i:shape[2]+i, i:shape[2]+i]
    a_c = a_c.as_strided(shape, (1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2]))
    b_c = b_c.as_strided(shape, (1, shape[0] * shape[2], shape[0], shape[0] * shape[1] * shape[2]))
    output_c = a_c + b_c

    if not torch.allclose(output_c, output.cpu()):
        print("Error in test_elementwise_4_2_not_align!")
        exit(1)

def test_elementwise_4_1(shape, dtype):
    input = torch.randn(shape[0] * 3, shape[1], shape[2], shape[3], dtype=dtype, device="cuda")
    input_c = input.cpu()
    input = input.as_strided(shape, (1, shape[0] * shape[2] * 3, shape[0], shape[0] * shape[1] * shape[2] * 3))
    output = torch.randn(shape, dtype=dtype, device="cuda")
    output_c = output.cpu()
    output = output.as_strided(shape, (1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2]))

    input_c = input_c.as_strided(shape, (1, shape[0] * shape[2] * 3, shape[0], shape[0] * shape[1] * shape[2] * 3))
    output_c = output_c.as_strided(shape, (1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2]))

    output_c.copy_(input_c)

    output.copy_(input)

    if not torch.allclose(output.cpu(), output_c):
        print("Error in test_elementwise_4_1!")
        exit(1)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--type", default="checkin", help="<checkin|daily>")

  args = parser.parse_args()

  if args.type == "checkin":
      test_elementwise_dim4_copy()
      test_elementwise_4_2((128,32,340,4), torch.float16)
      shapes = [[32,49,16,2048], [32,49,32,512], [32,49,4,32768], [32,49,8,8192]]
      for shape in shapes:
            test_elementwise_4_1(shape, torch.float16)
  else:
      # shape no-opt opt a100
      # [64,16,512,32], 403.750us 96.500us 99.250us
      # [64,512,16,32], 404.350us 96.850us 101.950us
      # used in bert
      for dtype in [torch.float16, torch.float32, torch.bfloat16]:
        for s0 in [32, 33, 64, 96, 128]:
            for s1 in [3, 8, 32, 64, 65, 128, 512]:
                for s2 in [3, 8, 32, 64, 65, 128, 512]:
                    for s3 in [3, 8, 32]:
                        shape = [s0, s1, s2, s3]
                        test_elementwise_broadcast_4_1(shape, dtype)

      for dtype in [torch.float16, torch.float32, torch.bfloat16]:
        for s0 in [32, 33, 64]:
            for s1 in [4, 5, 8]:
                for s2 in [2, 3, 4]:
                    for s3 in [4, 5]:
                        shape = [s0, s1, s2, s3]
                        # shape,output-stride,input-stride,dtype
                        test_elementwise_4_1_template(shape, (1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2]), 
                                                      (1, shape[0] * 348, shape[0], shape[0] * shape[1] * 348), dtype)
                        test_elementwise_4_1_template(shape, (1, shape[0] * 348, shape[0], shape[0] * shape[1] * 348), 
                                                      (1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2]), dtype)
                        test_elementwise_4_1_template(shape, (1, shape[0] * 2, shape[0] * shape[1] * 2, shape[0] * shape[1] * shape[2] * 2), 
                                                      (1, shape[0] * shape[2], shape[0], shape[0] * shape[1] * shape[2]), dtype)
                        test_elementwise_4_1_template(shape, (1, shape[0] * 2, shape[0] * shape[1] * 2, shape[0] * shape[1] * shape[2] * 2), 
                                                      (1, shape[0] * shape[2] * 2, shape[0] * 2, shape[0] * shape[1] * shape[2] * 2), dtype)
                        test_elementwise_4_1_template(shape, (1, shape[0], shape[0] * 348, shape[0] * shape[1] * 1392), 
                                                      (1, shape[0] * 2048, shape[0], shape[0] * shape[1] * 2048), dtype)
      
      for dtype in [torch.float16, torch.float32, torch.bfloat16]:
        # 128,32,340,4
        for s0 in [64, 65, 128]:
            for s1 in [32, 33]:
                for s2 in [8, 9]:
                    for s3 in [4, 5]:
                        shape = [s0, s1, s2, s3]
                        test_elementwise_4_2(shape, dtype)
                        test_elementwise_4_2_not_align(shape, dtype)
        for s0 in [32, 64, 33]:
            for s1 in range(2, 5):
                for s2 in range(2, 5):
                    for s3 in range(2, 100, 13):
                        shape = [s0, s1, s2, s3]
                        # shape,output-stride,input-stride,dtype
                        test_elementwise_4_1_template(shape, (1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2]), 
                                                      (1, shape[0] * shape[2] * 3, shape[0], shape[0] * shape[1] * shape[2] * 3), dtype)
        shapes = [[32, 300, 2, 3], [64, 301, 2, 3], [32, 2, 301, 3], [64, 2, 301, 3]]
        for shape in shapes:
            test_elementwise_4_1_template(shape, (1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2]), 
                                                      (1, shape[0] * shape[2] * 3, shape[0], shape[0] * shape[1] * shape[2] * 3), dtype)
  exit(0)

