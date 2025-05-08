import torch
import argparse

def test_transpose(type):
  dtypes = [torch.float32, torch.bfloat16, torch.half]
  for dtype in dtypes:
      if type == "checkin" or type == "daily":
        shape_list = [(4, 258), (15, 269), (64, 1024), (128, 65536),(127,65536), (1024, 511), (1024, 512), (2048, 2047), (2048, 4096), (1024, 693), (256, 44352), (256, 5544), (256, 693), (2838528, 128), (2838528, 64), (354816, 128), (354816, 256), (44352, 256), (44352, 512), (512, 44352), (512, 5544), (512, 693), (5544, 256), (5544, 512), (64, 27),(693, 256), (693, 512), (768, 5544), (768, 693), (8, 44352)]
        for shape in shape_list:
            input1 = torch.randn(shape[0], shape[1], dtype=dtype, device='cuda')
            input2 = torch.transpose(input1, 0, 1)
            output = torch.randn(shape[1], shape[0], dtype=dtype, device='cuda')
            output.copy_(input2)
            input2_cpu = input2.cpu()
            output_cpu = torch.randn(shape[1], shape[0], dtype=dtype, device='cpu')
            output_cpu.copy_(input2_cpu)
            if not torch.allclose(output.cpu(), output_cpu):
                exit(1)

        shape_list = [(1, 24, 128), (16, 128, 64), (259, 128, 256), (259, 127, 256), (1024, 128, 512), (1024, 1025, 511), (2049, 127, 1023)]
        for shape in shape_list:
            input1 = torch.randn(shape[0], shape[1], shape[2], dtype=dtype, device='cuda')
            input2 = torch.permute(input1, (0, 2, 1))
            output = torch.randn(shape[0], shape[2], shape[1], dtype=dtype, device='cuda')
            output.copy_(input2)
            input2_cpu = input2.cpu()
            output_cpu = torch.randn(shape[0], shape[2], shape[1], dtype=dtype, device='cpu')
            output_cpu.copy_(input2_cpu)
            if not torch.allclose(output.cpu(), output_cpu):
                exit(1)

        shape_list = [(256, 8, 127, 32), (256, 31, 256, 32), (256, 5, 256, 32), (129, 8, 256, 32), (256, 256, 32, 32), (256, 256, 4, 32), (256, 128, 8, 32), (128, 256, 8, 32)]
        for shape in shape_list:
            input1 = torch.randn(shape[0], shape[2], shape[1], shape[3], dtype=dtype, device='cuda')
            input2 = torch.permute(input1, (0, 2, 1, 3))
            output = torch.randn(shape[0], shape[1], shape[2], shape[3], dtype=dtype, device='cuda')
            output.copy_(input2)
            input2_cpu = input2.cpu()
            output_cpu = torch.randn(shape[0], shape[1], shape[2], shape[3], dtype=dtype, device='cpu')
            output_cpu.copy_(input2_cpu)
            if not torch.allclose(output.cpu(), output_cpu):
                exit(1)

        shape_list = [(1, 24, 128), (16, 128, 64), (259, 128, 256), (259, 127, 256), (1024, 128, 512), (1024, 1025, 511), (2049, 127, 1023)]
        for shape in shape_list:
            input1 = torch.randn(shape[1], shape[0], shape[2], dtype=dtype, device='cuda')
            input2 = torch.permute(input1, (1, 0, 2))
            output = torch.randn(shape[0], shape[1], shape[2], dtype=dtype, device='cuda')
            output.copy_(input2)
            input2_cpu = input2.cpu()
            output_cpu = torch.randn(shape[0], shape[1], shape[2], dtype=dtype, device='cpu')
            output_cpu.copy_(input2_cpu)
            if not torch.allclose(output.cpu(), output_cpu):
                exit(1)
      if type == "daily":
        # elementwise_kernel_3_2_transpose
        shape_list = [((2, 256, 3800), (256*3800, 1, 256)), ((2, 270, 15200), (15200*270, 1, 270)), ((32, 256, 1050), (1050*256, 1, 256)), 
                      ((16, 256, 67200), (256*67200, 1, 256)), ((8, 3, 4), (3*4, 1, 3)), ((4, 6, 784), (6 * 784, 1, 6)), 
                      ((2, 256, 825), (256 * 825, 1, 256)), ((2, 257, 825), (257 * 825, 1, 257)), ((2, 258, 827), (257*825, 1, 257))]
        for (shape, stride) in shape_list:
            input1 = torch.randn(shape, dtype=dtype, device='cuda')
            input2 = input1.as_strided(shape, stride)
            output = input1 + input2
            input1_cpu = input1.cpu()
            input2_cpu = input1_cpu.as_strided(shape, stride)
            output_cpu = input1_cpu + input2_cpu
            if not torch.allclose(output.cpu(), output_cpu):
                exit(1)

        # elementwise_kernel_3_2_transpose
        shape_list = [[380, 256, 2], [1520, 270, 2], [1050, 256, 32], [672, 256, 16], [8, 4, 3], [784, 6, 4], 
                      [825, 256, 2], [825, 257, 2], [827, 258, 2], [256, 554, 2],[512, 693, 2]]
        for shape in shape_list:
            a = torch.randn((1, shape[0], shape[0] * shape[1] * 2), dtype=dtype)
            b = torch.randn((shape[1], 1, shape[0] * shape[1]), dtype=dtype)
            a_c = a.as_strided(shape, (1, shape[0], shape[0] * shape[1] * 2))
            b_c = b.as_strided(shape, (shape[1], 1, shape[0] * shape[1]))
            ref = a_c + b_c

            a_d = a.cuda()
            b_d = b.cuda()
            a_d = a_d.as_strided(shape, (1, shape[0], shape[0] * shape[1] * 2))
            b_d = b_d.as_strided(shape, (shape[1], 1, shape[0] * shape[1]))
            out_d = a_d + b_d
            if not torch.allclose(out_d.cpu(), ref):
                exit(1)
        
        shape_list = [[1000,256,2],[1050,256,2],[12800,256,2],[13600,256,2],[14000,256,2],[14800,256,2],[15200,256,2],[16800,256,2],
                      [3200,256,2],[3400,256,2],[3500,256,2],[3700,256,2],[3800,256,2],[3900,256,2],[4000,256,2],
                      [4200,256,2],[51200,256,2],[54400,256,2],[59200,256,2],[60800,256,2],[62400,256,2],[64000,256,2],
                      [67200,256,2],[62399,256,2]]
        for shape in shape_list:
            a = torch.randn(shape, dtype=dtype)
            b = torch.randn(shape, dtype=dtype)
            a_c = a.as_strided(shape, (1, shape[0], shape[0] * shape[1]))
            b_c = b.as_strided(shape, (shape[1], 1, shape[0] * shape[1]))
            ref = a_c + b_c

            a_d = a.cuda()
            b_d = b.cuda()
            a_d = a_d.as_strided(shape, (1, shape[0], shape[0] * shape[1]))
            b_d = b_d.as_strided(shape, (shape[1], 1, shape[0] * shape[1]))
            out_d = a_d + b_d
            if not torch.allclose(out_d.cpu(), ref):
              print("Error")
              exit(1)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--type", default="checkin", help="<checkin|daily>")

  args = parser.parse_args()
  test_transpose(args.type)
