import torch
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import check_close
query_layer_32 = torch.randn(1, 12, 512, 64)
mixed_key_layer = torch.randn(1, 12, 512, 64)
key_layer_32 = mixed_key_layer.transpose(-1, -2).contiguous()

attention_scores_gpu = torch.matmul(
    query_layer_32.cuda(),
    key_layer_32.cuda()).cpu()
attention_scores_cpu = torch.matmul(query_layer_32.cpu(), key_layer_32.cpu())
diff = torch.abs(attention_scores_gpu - attention_scores_cpu)
result = check_close(attention_scores_gpu, attention_scores_cpu, 1e-4)
if result:
    exit(0)
else:
    exit(1)
