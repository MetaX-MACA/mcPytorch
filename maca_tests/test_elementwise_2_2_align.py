import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import copy
import os, sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))

def cal_rel_err(infer, golden):
    diff = infer - golden
    diff_square = diff * diff
    infer_result_square_double = 2 * infer * infer
    sum_diff_square = torch.sum(diff_square)
    sum_infer_result_square_double = torch.sum(infer_result_square_double)
    result = torch.sqrt(sum_diff_square / (sum_infer_result_square_double + 1e-12))
    return result

for dtype in [torch.float, torch.float16, torch.bfloat16]:
    tol = 1e-5 if dtype==torch.float else 5e-3
    for shape in [[(2048, 100000), (2048, 10000)], 
                #   [(2048, 99999), (2048, 9999)],
                  ]:   # when shape is (2048, 99999) only float would run into 2_2_align
        a = torch.randn(shape[0], dtype=dtype, device="cuda")
        b = torch.randn(shape[1], dtype=dtype, device="cuda")
        for size0_s in range(1):
            for size1_s in range(10):
                for size0 in [8, 9, 10, 16, 32, 48, 64, 512, 513, 1024]: # size0 should be multiple of 8 to into 2-2 align kernel
                    for size1 in [1, 3, 7, 9, 16, *range(128, 500, 33)]:
                        # print(f"dtype: {dtype}, shape: {shape}, size0_s: {size0_s}, size1_s: {size1_s}, size0: {size0}, size1: {size1}")
                        aa = a[size1_s:size1_s + size1, size0_s:size0_s + size0]
                        bb = b[size1_s:size1_s + size1, size0_s:size0_s + size0]

                        out = aa + bb
                        golden = aa.float().cpu() + bb.float().cpu()
                        rel_err = cal_rel_err(golden, out.float().cpu())
                        status = rel_err < tol
                        # print("rel_err: ", rel_err)
                        if not status:
                            print(f"fail rel_err: {rel_err}, dtype: {dtype}, shape: {shape}, size0_s: {size0_s}, size1_s: {size1_s}, size0: {size0}, size1: {size1}")
                            exit(1)

# test broadcast
for dtype in [torch.float16, torch.float, torch.bfloat16]:
    for s0 in range(0, 64):
        for s1 in [8*i for i in range(1, 10)]:
            shape = (s0, s1)
            tol = 1e-5 if dtype==torch.float else 5e-3
            inpo = torch.rand(629145600,device="cuda",dtype=dtype)
            inp1 = inpo.as_strided(shape, (0, 1))
            inp2 = torch.rand(shape, device="cuda",dtype=dtype)
            out = inp1+inp2

            out_g = inp1.float().cpu() + inp2.float().cpu()

            rel_err = cal_rel_err(out_g, out.float().cpu())
            status = rel_err < tol
            # print("rel_err: ", rel_err)
            if not status:
                print(f"fail rel_err: {rel_err}, dtype: {dtype}, shape: {shape}")
                exit(1)

print("#### pass")
exit(0)
