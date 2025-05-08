import torch
import torch.nn as nn 
import copy
from maca_tools import accuracy

#for dtype in [torch.float32, torch.float16]:
for dtype in [torch.float16]:
    for (shape, out_c, k, s, p, d, g) in [((256, 1, 3600), 4, 5, 1, 2, 1, 1)]:
        input = torch.randn(shape, dtype = dtype, device="cpu")
        input_d = input.clone().cuda()
        m = nn.Conv1d(shape[1], out_c, k, stride = s, padding = p, dilation = d, groups = g, dtype = dtype)
        m_d = copy.deepcopy(m)
        m_d = m_d.to("cuda")
        accuracy.start_record(record_level=2)
        output = m_d(input_d)
        accuracy.record_switch(False)
        if dtype == torch.float16 or dtype == torch.bfloat16:
          m = m.float()
          output_golden = m(input.float())
          print("type:{}, shape:{}, max diff:{}".format(dtype, shape, torch.max(output.cpu().float() - output_golden)))
          if not torch.allclose(output.cpu().float(), output_golden, rtol=5e-4, atol=5e-4):
            print("error")
            exit(1)
        else:
          output_golden = m(input)
          print("type:{}, shape:{}, max diff:{}".format(dtype, shape, torch.max(output.cpu() - output_golden)))
          if not torch.allclose(output.cpu().float(), output_golden, rtol=5e-4, atol=5e-4):
            print("error")
            exit(1)
exit(0)

