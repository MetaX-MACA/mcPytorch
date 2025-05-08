import torch
import torch.nn.functional as F

for dtype in [torch.float16, torch.float32, ]:
  for rows in [1, 7, 12*77, 24*77, 16*64, 16*100, 32*64, 32*144, 32*100, 64, 92, 257, 512, 1023, 2048, 2049, 16*256, 16*400, 16*576, 16*1024, 16*1600, 16*4096, 16*9216, 32*4096, 32*2304, 32*576]:
    for cols in [1, 3, 7, 12, 25, 48, 63, 64, 65, 78, 81, 95, 101, 119, 123, 124]:
      # 128, 256, 512, 1024 will be optimized, others will not
      # TODO(): add test from 1 to 1024 cols if we run checkin on chip
      input = torch.randn(rows, cols, dtype = dtype, device="cpu")
      input_d = input.cuda()
      output_d = F.softmax(input_d, dtype=torch.float32, dim=-1)
      if dtype == torch.float16:
        output_golden = F.softmax(input.float(), dtype=torch.float32, dim=-1)
        print("type:{}, rows:{}, cols:{}, max diff:{}".format(dtype, rows, cols, torch.max(output_d.cpu().float() - output_golden)))
        if not torch.allclose(output_d.cpu().float(), output_golden, rtol=5e-4, atol=5e-4):
          exit(1)
      else:
        output_golden = F.softmax(input, dim=-1)
        print("type:{}, rows:{}, cols:{}, max diff:{}".format(dtype, rows, cols, torch.max(output_d.cpu() - output_golden)))
        if not torch.allclose(output_d.cpu().float(), output_golden):
          exit(1)
exit(0)
