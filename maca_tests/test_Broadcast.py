import torch
import torch.nn.functional as F
import argparse

# dim 2-1
def launch_2_1(shape):
    for dtype in [torch.float16, torch.float32, torch.bfloat16]:
        input = torch.randn(shape, dtype = dtype, device="cpu")
        input_d = input.cuda()
        input_1 = input.as_strided(input.shape, (0,1))
        input_2 = input.as_strided(input.shape, (input.shape[1], 0))
        input_d_1 = input_d.as_strided(input.shape, (0,1))
        input_d_2 = input_d.as_strided(input.shape, (input.shape[1], 0))
        output_d_1 = input_d_1.contiguous()
        output_d_2 = input_d_2.contiguous()
        output_golden_1 = input_1.contiguous()
        output_golden_2 = input_2.contiguous()
        print("type:{}, shape:{}, stride:{}, max diff:{}".format(dtype, shape, (0,1), torch.max(output_d_1.cpu() - output_golden_1)))
        if not torch.allclose(output_d_1.cpu(), output_golden_1):
            print("error")
            exit(1)
        print("type:{}, shape:{}, stride:{}, max diff:{}".format(dtype, shape, (input.shape[1],0), torch.max(output_d_2.cpu() - output_golden_2)))
        if not torch.allclose(output_d_2.cpu(), output_golden_2):
            print("error")
            exit(1)


# dim 2-2
def launch_2_2(shape):
    for dtype in [torch.float16, torch.float32, torch.bfloat16]:
        input = torch.randn(shape, dtype = dtype, device="cpu")
        input_1 = torch.randn(shape, dtype = dtype, device="cpu")
        input_2 = torch.randn(shape, dtype = dtype, device="cpu")
        input_d = input.cuda()
        input_1_d = input_1.cuda()
        input_2_d = input_2.cuda()
        input_1 = input_1.as_strided(input_1.shape, (0,1))
        input_2 = input_2.as_strided(input_2.shape, (1,0))
        input_1_d = input_1_d.as_strided(input_1_d.shape, (0,1))
        input_2_d = input_2_d.as_strided(input_2_d.shape, (1,0))
        output_d = input_d + input_1_d  # arg1
        output_d_2 = input_d + input_2_d  # arg1 uncontiguous
        output_d_3 = input_1_d + input_d  # arg0
        output_d_4 = input_2_d + input_d  # arg0 uncotiguous
        output_golden = input + input_1
        output_golden_2 = input + input_2
        print("type:{}, shape:{}, stride:{}, max diff:{}".format(dtype, shape, (0,1), torch.max(output_d.cpu() - output_golden)))
        if not torch.allclose(output_d.cpu(), output_golden):
          exit(1)
        print("type:{}, shape:{}, stride:{}, max diff:{}".format(dtype, shape, (1,0), torch.max(output_d_2.cpu() - output_golden_2)))
        if not torch.allclose(output_d_2.cpu(), output_golden_2):
          exit(1)
        print("broadcast arg0 type:{}, shape:{}, stride:{}, max diff:{}".format(dtype, shape, (0,1), torch.max(output_d_3.cpu() - output_golden)))
        if not torch.allclose(output_d_3.cpu(), output_golden):
          exit(1)
        print("broadcast arg0 type:{}, shape:{}, stride:{}, max diff:{}".format(dtype, shape, (1,0), torch.max(output_d_4.cpu() - output_golden_2)))
        if not torch.allclose(output_d_4.cpu(), output_golden_2):
          exit(1)

#dim 3-2
def launch_3_2(shape):
    for dtype in [torch.float16, torch.float32, torch.bfloat16]:
        input = torch.randn(shape, dtype = dtype, device="cpu")
        input_1 = torch.randn(shape, dtype = dtype, device="cpu")
        input_d = input.cuda()
        input_1_d = input_1.cuda()
        input_1 = input_1.as_strided(input_1.shape, (0,1,shape[1]))
        input_1_d = input_1_d.as_strided(input_1_d.shape, (0,1,shape[1]))
        output_d = input_d + input_1_d
        output_golden = input + input_1
        print("type:{}, shape:{}, stride:{}, max diff:{}".format(dtype, shape, (0,1,shape[1]), torch.max(output_d.cpu() - output_golden)))
        if not torch.allclose(output_d.cpu(), output_golden):
          exit(1)

def test(t):
    if type == "checkin" or type == "daily":
        for shape in [(44416, 256), (693, 256), (693, 45)]:
            launch_2_2(shape)
            launch_2_1(shape)
        for shape in [(4, 7, 9), (3, 255, 825)]:
            launch_3_2(shape)
    if type == "daily":
        for shape in [(32, 4194304), (64, 4194304), (2048, 103168), (2048, 32000), (16384, 512), (16383, 512), (16384, 511), (16383, 511)]:
            launch_2_2(shape)
            launch_2_1(shape)
        for shape in [(512, 256, 888), (3, 256, 512), (5, 256, 811), (3, 233, 812)]:
            launch_3_2(shape)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--type", default="checkin", help="<checkin|daily>")

  args = parser.parse_args()
  test(args.type)
