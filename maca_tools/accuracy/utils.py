import torch

import yaml
import os
import time
import logging
from datetime import datetime

_tensors_per_pt = []

class TensorSummary(object):
    def __init__(self, dtype, size, stride, min_val, max_val):
        self.dtype = dtype
        self._size = size
        self.stride = stride
        self.min_val = min_val
        self.max_val = max_val

    # Just to be consistent with the usage of tensor
    def size(self):
        return self._size

def add_tensor_info(tensor_info):
    _tensors_per_pt.append(tensor_info)

def get_tensor_infos():
    global _tensors_per_pt
    res = _tensors_per_pt
    _tensors_per_pt = []
    return res

def add_time_as_suffix(name):
    if name.endswith('_'):
        return '{}{}.csv'.format(name, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))
    else:
        return '{}_{}.csv'.format(name, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))

@torch.no_grad()
def get_min_val(tensor):
    return torch._C._TensorBase.item(torch._C._VariableFunctions.min(tensor)) \
        if tensor.numel() > 0 else None

@torch.no_grad()
def get_max_val(tensor):
    return torch._C._TensorBase.item(torch._C._VariableFunctions.max(tensor)) \
        if tensor.numel() > 0 else None


def peek(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return first


def get_log_name():
    now = datetime.now()
    tm = f"{now.strftime('%Y')}-{now.strftime('%m')}-{now.strftime('%d')}-{now.strftime('%H')}-{now.strftime('%M')}-{now.strftime('%S')}"
    return f"pytorch_accuracy_log_{tm}.log"


def log(info, log_name):
    '''
    save important infomation in the save_log_path
    '''
    logging.basicConfig(filename=log_name,
                            format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    logging.info(info)
    if not os.getenv("PYTORCH_ACC_DISABLE_VISIABLE", False):
        print(info, flush=True)


def cal_rel_err(infer, golden):
    diff = infer - golden
    diff_square = diff * diff
    infer_result_square_double = 2 * infer * infer
    sum_diff_square = torch.sum(diff_square)
    sum_infer_result_square_double = torch.sum(infer_result_square_double)
    result = torch.sqrt(sum_diff_square / (sum_infer_result_square_double + 1e-12))
    return result


def get_tensor_device(t:torch.tensor):
    d = str(t.device).split(":")
    return d[0] if len(d) == 1 else d[1]