import torch
import torch.nn as nn
import copy

for dtype in [torch.float32, torch.float16]:
    for (shape, in_c, out_c, k, s) in [((1, 2, 4, 5, 4), 2, 4, (2, 3, 2), (1, 1, 1))]:
        input = torch.randn(shape, dtype = dtype)
        input_d = input.cuda()
        m = nn.ConvTranspose3d(2, 4, kernel_size=(2, 3, 2), stride=(1, 1, 1), dtype = dtype)
        m_d = copy.deepcopy(m)
        m_d = m_d.to("cuda")
        output = m_d(input_d)
        m = m.float()
        output_golden = m(input.float())
        if not torch.allclose(output.cpu().float(), output_golden, rtol=5e-4, atol=5e-4):
            print("error")
            exit(1)
exit(0)


