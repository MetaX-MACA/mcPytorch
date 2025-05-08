#!/usr/bin/env python
import os
import torch
import numpy as np
from functools import reduce
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import BERT_TEST_BATCHSIZE, GOLDEN_DIR


seed = 0

GOLDEN_DIR += "bert/dropout/"

MODE = "maca"   # verify data on maca platform
# MODE = "cuda" # save golden data on cuda platform

HWC = [[512, 768], [1, 12, 512, 512]]
result = []

if "PYTORCH_ENABLE_SAME_RAND_A100" not in os.environ:
    has_env = False
else:
    env_old = os.environ["PYTORCH_ENABLE_SAME_RAND_A100"]
os.environ["PYTORCH_ENABLE_SAME_RAND_A100"] = "1"

for bs in BERT_TEST_BATCHSIZE:
    for hwc in HWC:
        shape = [bs, *hwc]
        input = torch.randn(shape, dtype=torch.float32, device="cuda").requires_grad_(True)
        backward_input = torch.randn(shape, dtype=torch.float32, device="cuda")
        torch.manual_seed(seed)
        dp = torch.nn.Dropout(p=0.1).cuda()
        shape_name = reduce(lambda x, y: x + "_" + y, [str(x) for x in shape])
        print(shape_name)
        if MODE == "maca":
            input = torch.from_numpy(
                np.load(GOLDEN_DIR + f"input_{shape_name}.npy")).float().cuda().requires_grad_(True)
            output_golden = torch.from_numpy(np.load(GOLDEN_DIR + f"output_golden_{shape_name}.npy")).float()
            backward_input = torch.from_numpy(np.load(GOLDEN_DIR + f"backward_input_{shape_name}.npy")).float().cuda()
            grad_golden = torch.from_numpy(np.load(GOLDEN_DIR + f"grad_golden_{shape_name}.npy")).float()
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

        else:   # "cuda" mode, save golden data
            output = dp(input)

            np.save(GOLDEN_DIR + f"input_{shape_name}.npy",
                    input.detach().cpu().numpy(), allow_pickle=False, fix_imports=False)
            np.save(GOLDEN_DIR + f"output_golden_{shape_name}.npy",
                    output.detach().cpu().numpy(), allow_pickle=False, fix_imports=False)
            np.save(GOLDEN_DIR + f"backward_input_{shape_name}.npy",
                    backward_input.detach().cpu().numpy(), allow_pickle=False, fix_imports=False)

            output.backward(backward_input)

            np.save(GOLDEN_DIR + f"grad_golden_{shape_name}.npy",
                    input.grad.detach().cpu().numpy(), allow_pickle=False, fix_imports=False)

if has_env is False:
    del os.environ["PYTORCH_ENABLE_SAME_RAND_A100"]
else:
    os.environ["PYTORCH_ENABLE_SAME_RAND_A100"] = env_old

if MODE == "maca":

    sum_result = sum(result)
    print("###", result)


    if sum_result < len(result):
        exit(1)
    else:
        exit(0)
