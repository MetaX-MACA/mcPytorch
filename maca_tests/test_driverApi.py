import os
import torch
import torch.nn as nn

old_env = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
old_env_small_page = os.environ.get("MACA_SMALL_PAGESIZE_ENABLE")
os.environ["MACA_SMALL_PAGESIZE_ENABLE"] = "1"

inp0 = torch.rand(128, 128, device = "cuda")
inp1 = torch.rand(128, 128, device = "cuda")
out = inp0 + inp1

inp0c = inp0.cpu()
inp1c = inp1.cpu()
outc = inp0c + inp1c

diff = torch.max(torch.abs(out.cpu() - outc))
if diff > 0.0001:
    print("test_driveApi is error")
    exit(1)

if old_env:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = old_env
else:
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF")
if old_env_small_page:
    os.environ["MACA_SMALL_PAGESIZE_ENABLE"] = old_env_small_page
else:
    os.environ.pop("MACA_SMALL_PAGESIZE_ENABLE")

