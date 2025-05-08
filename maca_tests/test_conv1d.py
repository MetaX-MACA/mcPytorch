import torch
import torch.nn as nn
import copy

for dtype in [torch.float32, torch.float16]:
    for (shape, out_c, k, s, p, d, g) in [((256, 1, 3600), 4, 5, 1, 2, 1, 1), ((256, 4, 3600), 16, 5, 1, 2, 1, 1),\
                                          ((256, 16, 3600), 512, 19, 5, 9, 1, 1), ((256, 512, 720), 1024, 1, 1, 0, 1, 1),\
                                          ((256, 512, 720), 512, 15, 1, 7, 1, 512), ((256, 512, 720), 512, 1, 1, 0, 1, 1)]:
        input = torch.randn(shape, dtype = dtype, device="cpu")
        input_d = input.clone().cuda()
        m = nn.Conv1d(shape[1], out_c, k, stride = s, padding = p, dilation = d, groups = g, dtype = dtype)
        m_d = copy.deepcopy(m)
        m_d = m_d.to("cuda")
        output = m_d(input_d)
        if dtype == torch.float16 or dtype == torch.bfloat16:
          m = m.float()
          output_golden = m(input.float())
          print("type:{}, shape:{}, max diff:{}".format(dtype, shape, torch.max(output.cpu().float() - output_golden)))
          if not torch.allclose(output.cpu().float(), output_golden, rtol=5e-4, atol=5e-4):
            exit(1)
        else:
          output_golden = m(input)
          print("type:{}, shape:{}, max diff:{}".format(dtype, shape, torch.max(output.cpu() - output_golden)))
          if not torch.allclose(output.cpu().float(), output_golden, rtol=5e-4, atol=5e-4):
            exit(1)
exit(0)
