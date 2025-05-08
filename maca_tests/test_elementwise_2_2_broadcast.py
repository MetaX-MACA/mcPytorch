import torch
import argparse
import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../".format(cur_dir))
sys.path.append("{}/../maca_tests".format(cur_dir))
from utils import test_add, test_sub, Result

DTYPES = [torch.float16, torch.bfloat16, torch.float]

def test_elementwise_2_2_broadcast(test_type="checkin"):
  if test_type == "checkin":
    s0_list = [64, 67, 128, 192, 256, 320, 384, 448, 512, 576, 640, 1024]
    s1_list = [512, 576, 640, 513]
  else:
    s0_list = list(i for i in range(64, 513, 23))
    s1_list = list(i for i in range(64, 1025, 33))
  for dtype in DTYPES:
    for s0 in s0_list:
      for s1 in s1_list:
        shape = [s0, s1]
        for stride in [1, 2, shape[0], shape[0] - 1, shape[0] - 2, 3 * shape[0]]:
          output_strides = [(1, shape[0])]
          input_strides = [(1, stride), (1, 0)]
          align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=False, output_strides=output_strides, input_strides=input_strides)
          not_align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=True, output_strides=output_strides, input_strides=input_strides)
          if align_result == Result.Error or not_align_result == Result.Error:
            exit(1)
          
          output_strides = [(1, shape[0])]
          input_strides = [(1, stride), (0, 1)]
          align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=False, output_strides=output_strides, input_strides=input_strides)
          not_align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=True, output_strides=output_strides, input_strides=input_strides)
          if align_result == Result.Error or not_align_result == Result.Error:
            exit(1)
          
          output_strides = [(1, shape[0])]
          input_strides = [(1, 0), (1, stride)]
          align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=False, output_strides=output_strides, input_strides=input_strides)
          not_align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=True, output_strides=output_strides, input_strides=input_strides)
          if align_result == Result.Error or not_align_result == Result.Error:
            exit(1)
          
          output_strides = [(1, shape[0])]
          input_strides = [(0, 1), (1, stride)]
          align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=False, output_strides=output_strides, input_strides=input_strides)
          not_align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=True, output_strides=output_strides, input_strides=input_strides)
          if align_result == Result.Error or not_align_result == Result.Error:
            exit(1)

def test_elementwise_2_2_template(test_type="checkin"):
  if test_type == "checkin":
    s0_list = list(i for i in range(3, 64, 12))
    s1_list = list(i for i in range(3, 64, 15))
  else:
    s0_list = list(i for i in range(3, 513, 23))
    s1_list = list(i for i in range(3, 1025, 33))

  for dtype in DTYPES:
    for s0 in s0_list:
      for s1 in s1_list:
        shape = [s0, s1]
        stride_list = list(i for i in range(1, s0, s0//3))
        stride_list.append(s0)
        for stride in stride_list:
          output_strides = [(1, shape[0])]
          input_strides = [(1, stride), (1, stride)]
          align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=False, output_strides=output_strides, input_strides=input_strides)
          not_align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=True, output_strides=output_strides, input_strides=input_strides)
          if align_result == Result.Error or not_align_result == Result.Error:
            exit(1)

          input_strides = [(stride, 1), (1, stride)]
          align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=False, output_strides=output_strides, input_strides=input_strides)
          not_align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=True, output_strides=output_strides, input_strides=input_strides)
          if align_result == Result.Error or not_align_result == Result.Error:
            exit(1)
          
          input_strides = [(stride, 1), (stride, 1)]
          align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=False, output_strides=output_strides, input_strides=input_strides)
          not_align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=True, output_strides=output_strides, input_strides=input_strides)
          if align_result == Result.Error or not_align_result == Result.Error:
            exit(1)

          input_strides = [(1, stride), (1, stride)]
          align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=False, output_strides=output_strides, input_strides=input_strides)
          not_align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=True, output_strides=output_strides, input_strides=input_strides)
          if align_result == Result.Error or not_align_result == Result.Error:
            exit(1)
          
          input_strides = [(1, stride), (stride, 1)]
          align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=False, output_strides=output_strides, input_strides=input_strides)
          not_align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=True, output_strides=output_strides, input_strides=input_strides)
          if align_result == Result.Error or not_align_result == Result.Error:
            exit(1)
          
          input_strides = [(1, stride), (stride, stride - 1)]
          align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=False, output_strides=output_strides, input_strides=input_strides)
          not_align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=True, output_strides=output_strides, input_strides=input_strides)
          if align_result == Result.Error or not_align_result == Result.Error:
            exit(1)
          
          input_strides = [(stride, 1), (stride, stride - 1)]
          align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=False, output_strides=output_strides, input_strides=input_strides)
          not_align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=True, output_strides=output_strides, input_strides=input_strides)
          if align_result == Result.Error or not_align_result == Result.Error:
            exit(1)
          
          input_strides = [(stride, stride - 1), (1, stride)]
          align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=False, output_strides=output_strides, input_strides=input_strides)
          not_align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=True, output_strides=output_strides, input_strides=input_strides)
          if align_result == Result.Error or not_align_result == Result.Error:
            exit(1)
          
          input_strides = [(stride, stride - 1), (stride, 1)]
          align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=False, output_strides=output_strides, input_strides=input_strides)
          not_align_result = test_sub(shape=shape, dtypes=[dtype,], test_not_align=True, output_strides=output_strides, input_strides=input_strides)
          if align_result == Result.Error or not_align_result == Result.Error:
            exit(1)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--type", default="checkin", help="<checkin|daily>")

  args = parser.parse_args()

  test_elementwise_2_2_broadcast(args.type)
  test_elementwise_2_2_template(args.type)
  