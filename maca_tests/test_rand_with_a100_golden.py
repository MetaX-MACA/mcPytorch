import torch
import numpy as np
import random
import os

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def cal_rel_err(infer, golden):
    diff = infer - golden
    diff_square = diff * diff
    infer_result_square_double = 2 * infer * infer
    sum_diff_square = torch.sum(diff_square)
    sum_infer_result_square_double = torch.sum(infer_result_square_double)
    result = torch.sqrt(sum_diff_square / (sum_infer_result_square_double + 1e-12))

    infer = infer.flatten()
    golden = golden.flatten()
    diff = torch.abs(golden.float().cpu() - infer.float().cpu())
    max_val = torch.max(diff)
    idx = torch.argmax(diff)
    print(f"relative error: {result}, abs max diff: {max_val}, where infer is {infer[idx]}, golden is {golden[idx]}")

    return result


gold_path = r"/netapp/pytorch/golden/golden_mcrand/a100/"
os.environ["PYTORCH_ENABLE_SAME_RAND_A100"] = "1"
if os.getenv("PYTORCH_ENABLE_SAME_RAND_A100")=='1':
    print("### already set PYTORCH_ENABLE_SAME_RAND_A100=1")
else:
    assert 0, "please set PYTORCH_ENABLE_SAME_RAND_A100=1"


seed = 31
sizes = [2*8192*96*128]
dtypes = [torch.float, torch.half, torch.bfloat16]


for size in sizes:
    for dtype in dtypes:
            # check rand
            print(50*"=")
            golden = torch.load(gold_path+"a100_rand_"+str(size)+"_"+str(dtype)+".pt")
            setup_seed(seed)
            out1 = torch.rand(size, dtype=dtype, device="cuda")
            if dtype == torch.float32: 
                rel_err = cal_rel_err(golden, out1)
                if rel_err > 1e-6:
                    exit(1)
            else:
                diff = torch.abs(golden - out1)
                v, i = diff.gather(0, torch.where(diff!=0)[0]).sort()
                # print(f"diff tensor: {v}")
                d = torch.where(diff!=0)[0].shape[0]
                t = golden.shape[0]
                print(f"* rand {dtype} {d} elemens diff in {t} elements, percent: {d/t}")
                if d/t > 0.003:
                    exit(1)

            # check randn
            print(50*"=")
            golden = torch.load(gold_path+"a100_randn_"+str(size)+"_"+str(dtype)+".pt")
            setup_seed(seed)
            out1 = torch.randn(size, dtype=dtype, device="cuda")*0.02
            if dtype == torch.float32: 
                rel_err = cal_rel_err(golden, out1)
                if rel_err > 1e-6:
                    exit(1)
            else:
                diff = torch.abs(golden - out1)
                v, i = diff.gather(0, torch.where(diff!=0)[0]).sort()
                # print(f"diff tensor: {v}")
                d = torch.where(diff!=0)[0].shape[0]
                t = golden.shape[0]
                print(f"* randn {dtype} {d} elemens diff in {t} elements, percent: {d/t}")
                if d/t > 0.001:
                    exit(1)

            # check normal_
            print(50*"=")
            golden = torch.load(gold_path+"a100_normal_"+str(size)+"_"+str(dtype)+".pt")
            setup_seed(seed)
            out1 = torch.empty(size, dtype=dtype, device="cuda")
            torch.nn.init.normal_(out1, mean=0.0, std=0.02)
            if dtype == torch.float32: 
                rel_err = cal_rel_err(golden, out1)
                if rel_err > 1e-6:
                    exit(1)
            else:
                diff = torch.abs(golden - out1)
                v, i = diff.gather(0, torch.where(diff!=0)[0]).sort()
                # print(f"diff tensor: {v}")
                d = torch.where(diff!=0)[0].shape[0]
                t = golden.shape[0]
                print(f"* normal_ {dtype} {d} elemens diff in {t} elements, percent: {d/t}")
                if d/t > 0.002:
                    exit(1)
print("### test pass")
exit(0)
