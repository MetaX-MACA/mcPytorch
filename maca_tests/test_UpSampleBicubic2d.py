import torch
import torch.nn as nn

m = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)

input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
output_golden = m(input)

input_d = input.cuda()
output_d = m(input_d)
output = output_d.cpu()

if torch.allclose(output, output_golden):
    print("Passed: {}".format(__file__))
    exit(0)
else:
    print("Failed: {}".format(__file__))
    exit(1)


def test_upsample_bicubic2d():

    shape_list = [(1,3,576,1024),(2,3,512,512),(1,4,1024,512)]
    for shape in shape_list:
        inp = torch.rand(shape,device="cuda",dtype=torch.float16)
        m = torch.nn.Upsample(scale_factor=2,mode="bicubic")
        out = m(inp)

        inpc = inp.cpu()
        outc = m(inpc)

        diff = torch.max(torch.abs(out.cpu()-outc))
        if diff > 0.0001:
            print("test_upsample_bicubic2d is error")
            exit(1)

test_upsample_bicubic2d()
