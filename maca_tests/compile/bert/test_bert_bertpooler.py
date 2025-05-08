import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformers
import copy
from typing import Tuple
import os, sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import *


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 768, 768
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]    # 1, 768
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


device = "cuda"

# model
model = BertPooler(transformers.BertConfig()).to(device)
hidden_states = torch.randn([1, 100, 768]).float().to(device).requires_grad_(True)
hidden_stats_g = hidden_states.clone().detach().requires_grad_(True)
model.train()
golden =  model(hidden_stats_g)
backward_input =  torch.randn(golden.shape).float().to(device)
golden.backward(backward_input)
grad_golden = hidden_stats_g.grad

# model_com
ret = None
model_com = torch.compile(copy.deepcopy(model).to(device), mode="max-autotune", backend="inductor")

for i in range(3):
    ret = timed(lambda: model_com(hidden_states))

    if hidden_states.grad != None:
        hidden_states.grad.zero_()
    print(f"Iter: {i+1}")
    print("     forward time(ms): ", ret[1])
    _, g_time = timed(lambda: ret[0].backward(backward_input))
    print("     backeard time(ms): ", g_time)
    ret = ret[0]

fw_status = check_close(ret, golden)
bw_status = check_close(hidden_states.grad, grad_golden)

if fw_status and bw_status:
    print("##### success")
    exit(0)
else:
    print(f"##### fail: {fw_status, bw_status}")
    exit(1)