#!/usr/bin/env python
import torch
import torch.nn as nn
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import BERT_TEST_BATCHSIZE
import copy


def run_test(m, input):
    input = input
    input_cuda = input.detach().clone().cuda()
    m_cuda = copy.deepcopy(m).cuda()

    assert torch.equal(input, input_cuda.cpu()), "$$$ input_cuda is not equal to input"
    assert torch.equal(m.weight, m_cuda.weight.cpu()), "$$$weight_cuda is not equal to weight"

    output_golden = m(input)
    output = m_cuda(input_cuda)

    backward_input = torch.randn(output_golden.shape, dtype=output_golden.dtype)
    backward_input_cuda = backward_input.detach().clone().cuda()

    output_golden.backward(backward_input)
    output.backward(backward_input_cuda)

    fw_status = torch.allclose(output.cpu(), output_golden)
    g_g = m_cuda.weight.grad.cpu()
    g_c = m.weight.grad
    bw_status = torch.allclose(g_g, g_c, atol=1e-4)
    return (fw_status and bw_status)


HWC = [[512]]
result = []
for bs in BERT_TEST_BATCHSIZE:
    for hwc in HWC:
        shape = [bs, *hwc]

        input_token = torch.zeros(shape, dtype=torch.long, device="cpu")
        input_word = torch.tensor(list(x for x in range(512)) * bs, dtype=torch.long,
                                  device="cpu").reshape(shape)
        input_position = torch.tensor(list(x for x in range(512)) * bs, dtype=torch.long,
                                      device="cpu").reshape(shape)

        m_token = nn.Embedding(2, 768)
        m_word = nn.Embedding(30522, 768, 0)
        m_position = nn.Embedding(512, 768)

        result.append(run_test(m_token, input_token))
        result.append(run_test(m_word, input_word))
        result.append(run_test(m_position, input_position))

sum_result = sum(result)
print("###", result)
if sum_result < len(result):
    exit(1)
else:
    exit(0)
