import torch
import torch.nn.functional as F
torch.manual_seed(0)

for dtype in [torch.float16, torch.bfloat16, torch.float32]:
  for rows in [1, 7, 64, 256, 257, 512, 513, 1023, 1024]:
    for cols in list(range(1,1024)) + [2048, 2059, 4096]:
      input = torch.randn(rows, cols, dtype = dtype, device="cpu")
      input_d = input.cuda()
      is_log_softmax_list = [True, False]
      for is_log_softmax_flag in is_log_softmax_list:
        if is_log_softmax_flag:
          output_d = F.softmax(input_d, dim=-1)
          if dtype == torch.float16:
            output_golden = F.softmax(input.float(), dim=-1)
            print("type:{}, rows:{}, cols:{}, max diff:{}".format(dtype, rows, cols, torch.max(output_d.cpu().float() - output_golden)))
            if not torch.allclose(output_d.cpu().float(), output_golden, rtol=5e-4, atol=5e-4):
              exit(1)
          elif dtype == torch.bfloat16:
            output_golden = F.softmax(input.float(), dim=-1)
            print("type:{}, rows:{}, cols:{}, max diff:{}".format(dtype, rows, cols, torch.max(output_d.cpu().float() - output_golden)))
            if not torch.allclose(output_d.cpu().float(), output_golden, rtol=5e-3, atol=5e-3):
              exit(1)
          else:
            output_golden = F.softmax(input, dim=-1)
            print("type:{}, rows:{}, cols:{}, max diff:{}".format(dtype, rows, cols, torch.max(output_d.cpu() - output_golden)))
            if not torch.allclose(output_d.cpu().float(), output_golden):
              exit(1)
        else:
          output_d = F.log_softmax(input_d, dim=-1)
          if dtype == torch.float16:
            output_golden = F.log_softmax(input.float(), dim=-1)
            print("type:{}, rows:{}, cols:{}, max diff:{}".format(dtype, rows, cols, torch.max(output_d.cpu().float() - output_golden)))
            if not torch.allclose(output_d.cpu().float(), output_golden, rtol=5e-4, atol=5e-4):
              exit(1)
          elif dtype == torch.bfloat16:
            output_golden = F.log_softmax(input.float(), dim=-1)
            print("type:{}, rows:{}, cols:{}, max diff:{}".format(dtype, rows, cols, torch.max(output_d.cpu().float() - output_golden)))
            if not torch.allclose(output_d.cpu().float(), output_golden, rtol=5e-3, atol=5e-3):
              exit(1)
          else:
            output_golden = F.log_softmax(input, dim=-1)
            print("type:{}, rows:{}, cols:{}, max diff:{}".format(dtype, rows, cols, torch.max(output_d.cpu() - output_golden)))
            if not torch.allclose(output_d.cpu().float(), output_golden):
              exit(1)
exit(0)
