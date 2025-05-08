import torch
import numpy as np


# get time in ms
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)


def load_data(data_path):
    input = np.load(data_path, allow_pickle=True)
    return input


def check_close(infer_result_data, golden_data, eps=1e-4):
    diff = infer_result_data - golden_data
    diff_square = diff * diff
    infer_result_square_double = 2 * infer_result_data * infer_result_data
    sum_diff_square = torch.sum(diff_square)
    sum_infer_result_square_double = torch.sum(infer_result_square_double)
    result = torch.sqrt(sum_diff_square / (sum_infer_result_square_double + 1e-12))
    print(f"check close error: {result}")

    return result < eps