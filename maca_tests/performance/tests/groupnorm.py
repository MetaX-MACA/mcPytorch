import argparse
import torch

from utils.pt_reporter import PyTorchReporter
from utils.utils import launch_prof, get_time_s, accuracy_check
from utils.perf_data_statistics import *


parser = argparse.ArgumentParser()
parser.add_argument('-platform', type=str, metavar='platform', help='chip platform')
parser.add_argument('-branch', type=str, metavar='branch', help='git branch')
parser.add_argument('-path', type=str, metavar='path', help='output json path')
parser.add_argument('--is_optim', action='store_true', help='test opt kernel')
args = parser.parse_args()

from_model_testcases = [{"input_shape": [1, 128, 512, 512], "channels": 128, "groups": 32},
                        {"input_shape": [16, 128, 768, 768], "channels": 128, "groups": 32},
                        {"input_shape": [16, 256, 384, 384], "channels": 256, "groups": 32},
                        {"input_shape": [16, 512, 192, 192], "channels": 512, "groups": 32}]

def common_test_generator():
    HxW = [32*32, 64*64, 128*128, 256*256, 512*512]
    D = [1, 2, 4, 8]
    G = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    N = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    max_size = 65536 * 65536
    for n in N:
        for g in G:
            for d in D:
                for hxw in HxW:
                    if n * g * d * hxw > max_size:
                        continue 
                    shape = (n, g*d, hxw)
                    yield shape, g*d, g
def light_common_test_generator():
    HxW = [32*32, 128*128, 512*512]
    D = [1, 2, 4, 8]
    G = [1, 4, 16, 64, 256]
    N = [1, 4, 16, 64, 256]
    max_size = 128 * 512*512
    for n in N:
        for g in G:
            for d in D:
                for hxw in HxW:
                    if n * g * d * hxw > max_size:
                        continue 
                    shape = (n, g*d, hxw)
                    yield shape, g*d, g

class GroupNormPerfTest(PyTorchReporter):
    def __init__(self, platform, branch):
        super().__init__()
        self.function = "groupnorm_test"
        self.platform = platform
        self.branch = branch
        self.record_log_name = "groupnorm"
        reset_perf_data(self.record_log_name, self.platform)

    @launch_prof(args.path)
    def test_forward(self, shape, num_groups, num_channels, dtype=None, is_optim=False, func_name="", feature=""):
        self.feature = feature
        input = torch.randn(size=shape, dtype=dtype, device="cuda")
        m = torch.nn.GroupNorm(num_groups, num_channels, device="cuda").to(dtype)
        tol = 1e-4 if dtype==torch.float else 5e-3
        assert accuracy_check(m, [input], fwd_tol=tol), f"{m}, {dtype} {shape} accuracy check fail"

        torch.cuda.synchronize()    # theoretically we no need call it there, but we need call it in maca
        # we can also control the warmup and active number by profiler.schedule, but it looks like with bug in maca
        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                            record_shapes=False,
                        ) as profiler:
            m(input)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        # print(info)
        t =  get_time_s(info)
        return t

    @launch_prof(args.path)
    def test_backward(self, shape, num_groups, num_channels, dtype=None, is_optim=False, func_name="", feature=""):
        self.feature = feature

        input = torch.randn(size=shape, dtype=dtype, device="cuda").requires_grad_(True)
        g_y = torch.randn(size=shape, dtype=dtype, device="cuda")
        m = torch.nn.GroupNorm(num_groups, num_channels, device="cuda").to(dtype)
        tol = 1e-4 if dtype==torch.float else 5e-3
        assert accuracy_check(m, [input], g_y, fwd_tol=tol, bwd_tol=tol), f"{m}, {dtype} {shape} accuracy check fail"

        out = m(input)
        torch.cuda.synchronize()    # theoretically we no need call it there, but we need call it in maca
        # we can also control the warmup and active number by profiler.schedule, but it looks like with bug in maca
        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                            record_shapes=False,
                        ) as profiler:
            out.backward(g_y)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t =  get_time_s(info)
        return t

    def run(self):
        print("[PyTorch Perf Start] groupnorm")
        for dtype in [torch.float16, torch.float32, torch.bfloat16]:
            for case in from_model_testcases:
                mean_time = self.test_forward(shape=case["input_shape"], num_groups=case["groups"], num_channels=case["channels"], dtype=dtype, is_optim=True, func_name="groupnorm_forward_test", feature="from_model;common")
                record_test_case_name = "forward_" + "shape_" + str(case["input_shape"]) + "_groups_" + str(case["groups"]) + "_channels_" + str(case["channels"]) + "_dtype_" + str(dtype)
                record_perf_data(self.record_log_name, self.platform, record_test_case_name, 0, 0, mean_time)

                mean_time = self.test_backward(shape=case["input_shape"], num_groups=case["groups"], num_channels=case["channels"], dtype=dtype, is_optim=True, func_name="groupnorm_backward_test", feature="from_model;common")
                record_test_case_name = "backward_" + "shape_" + str(case["input_shape"]) + "_groups_" + str(case["groups"]) + "_channels_" + str(case["channels"]) + "_dtype_" + str(dtype)
                record_perf_data(self.record_log_name, self.platform, record_test_case_name, 0, 0, mean_time)
            
            for shape, channels, groups in light_common_test_generator():
                mean_time = self.test_forward(shape=shape, num_groups=groups, num_channels=channels, dtype=dtype, is_optim=True, func_name="groupnorm_forward_test", feature="common")
                record_test_case_name = "forward_" + "shape_" + str(shape) + "_groups_" + str(groups) + "_channels_" + str(channels) + "_dtype_" + str(dtype)
                record_perf_data(self.record_log_name, self.platform, record_test_case_name, 0, 0, mean_time)

                mean_time = self.test_backward(shape=shape, num_groups=groups, num_channels=channels, dtype=dtype, is_optim=True, func_name="groupnorm_backward_test", feature="common")
                record_test_case_name = "backward_" + "shape_" + str(shape) + "_groups_" + str(groups) + "_channels_" + str(channels) + "_dtype_" + str(dtype)
                record_perf_data(self.record_log_name, self.platform, record_test_case_name, 0, 0, mean_time)
        print("[PyTorch Perf End] groupnorm")


if __name__ == '__main__':
    f = GroupNormPerfTest(args.platform, args.branch)
    f.run()
