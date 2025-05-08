import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import copy
import os, sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))

for dtype in [torch.float, torch.float16, torch.bfloat16]:
    for size0 in [1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,33,49,111,127,128,129,130,395,396,397,512,513,514]:
        for size1 in [1,2,32, 64,65,66,67,255,256,257,258,259]: # size0 * size1 should be multiple of warp_size
            for size2 in [1,2,3,4,32,33,34,99,133,198,256,257,258]:
                input_shape=(1, size1, size0)
                out_shape=(size2, size1, size0)
                input_d = torch.randn(input_shape, dtype=dtype, device="cuda")
                input_d_1 = input_d.as_strided(input_shape, (0, 1, size1))
                # test broadcast
                output_d = torch.zeros(out_shape, dtype = dtype, device="cuda")
                output_d.copy_(input_d_1)

                output_golden = torch.zeros(out_shape, dtype = dtype, device="cpu")
                output_golden.copy_(input_d_1.cpu())

                if not torch.allclose(output_d.cpu(), output_golden):
                    print(f"broadcast 3-1 fail shape: {size2, size1, size0}, dtype: {dtype}")
                    exit(1)

print("#### pass")
exit(0)
