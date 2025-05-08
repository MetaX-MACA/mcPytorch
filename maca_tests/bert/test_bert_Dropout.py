#!/usr/bin/env python
import torch
import numpy as np
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import BERT_TEST_BATCHSIZE, GOLDEN_DIR

seed = 0
GOLDEN_DIR = GOLDEN_DIR + "bert/dropout/"

HWC = [[512, 768]]
result = []
for bs in BERT_TEST_BATCHSIZE:
    for hwc in HWC:
        torch.manual_seed(seed)
        shape = [bs, *hwc]
        dp = torch.nn.Dropout(p=0.1).cuda()

        input = torch.from_numpy(np.load(GOLDEN_DIR + f"input_bs_{bs}.npy")).float().cuda().requires_grad_(True)
        output_golden = torch.from_numpy(np.load(GOLDEN_DIR + f"output_golden_bs_{bs}.npy")).float()
        backward_input = torch.from_numpy(np.load(GOLDEN_DIR + f"backward_input_bs_{bs}.npy")).float().cuda()
        grad_golden = torch.from_numpy(np.load(GOLDEN_DIR + f"grad_golden_bs_{bs}.npy")).float()

        output = dp(input)
        diff = output.cpu() - output_golden
        print("### forward max diff\n", torch.max(diff))

        output.backward(backward_input)
        fw_status = torch.allclose(output.cpu(), output_golden)
        bw_status = torch.allclose(input.grad.cpu(), grad_golden)
        diff = input.grad.cpu() - grad_golden
        print("### backward max diff\n", torch.max(diff))
        print("$$$", fw_status, bw_status)
        result.append(fw_status and bw_status)

sum_result = sum(result)
print("###", result)
if sum_result < len(result):
    exit(1)
else:
    exit(0)
