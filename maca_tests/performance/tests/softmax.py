import argparse
from datetime import datetime
import time
import torch
import torch.nn.functional as F
import re
import itertools
from utils.pt_reporter import PyTorchReporter
from utils.utils import launch_prof, get_time_s, accuracy_check


parser = argparse.ArgumentParser()
parser.add_argument('-platform', type=str, metavar='platform', help='chip platform')
parser.add_argument('-branch', type=str, metavar='branch', help='git branch')
parser.add_argument('-path', type=str, metavar='path', help='output json path')
args = parser.parse_args()

class SoftMaxPerfTest(PyTorchReporter):
    def __init__(self, platform, branch, is_log_softmax=False):
        super().__init__()
        self.platform = platform
        self.branch = branch
        self.is_log_softmax = is_log_softmax

    @launch_prof(args.path)
    def launch_forward(self, shape=None, dtype=None, is_optim=False, feature="", func_name="", dim=-1):
        input = torch.randn(shape, dtype = dtype, device="cuda")
        tol = 1e-4 if dtype==torch.float else 5e-3
        if self.is_log_softmax:
            output = F.log_softmax(input, dim=dim)
            assert accuracy_check(F.log_softmax, [input], fwd_tol=tol), f"F.log_softmax forward, {dtype} {shape} {dim} accuracy check fail"
        else:
            output = F.softmax(input, dim=dim)
            assert accuracy_check(F.softmax, [input], fwd_tol=tol), f"F.softmax forward, {dtype} {shape} {dim} accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                            record_shapes=False,
                        ) as profiler:
            if self.is_log_softmax:
                output = F.log_softmax(input, dim=dim)
            else:
                output = F.softmax(input, dim=dim)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t = get_time_s(info)
        return t

    @launch_prof(args.path)
    def launch_backward(self, shape=None, dtype=None, is_optim=False, feature="", func_name="", dim=-1):
        input = torch.randn(shape, dtype = dtype, device="cuda").requires_grad_(True)
        g_y = torch.randn(size=shape, dtype=dtype, device="cuda")
        
        tol = 1e-4 if dtype==torch.float else 5e-3
        if self.is_log_softmax:
            output = F.log_softmax(input, dim=dim)
            assert accuracy_check(F.log_softmax, [input], g_y, fwd_tol=tol, bwd_tol=tol), f"F.log_softmax backward, {dtype} {shape} {dim} accuracy check fail"
        else:
            output = F.softmax(input, dim=dim)
            assert accuracy_check(F.softmax, [input], g_y, fwd_tol=tol, bwd_tol=tol), f"F.softmax backward, {dtype} {shape} {dim} accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                            record_shapes=False,
                        ) as profiler:
            output.backward(g_y)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t = get_time_s(info)
        return t

    def run(self):
        for dtype in [torch.float16, torch.float32, torch.bfloat16]:
            #shape in real networks
            for shape in [
                (2048,16,49,49), (8192,8,49,49), (32768,4,49,49), (16,64,64), (32,64,64), (16,64,77), (32,144,77), (16,100,77), (32,64,77), (16,144,77), (32,100,77), (16,400,77), (16,9216,77), (16,2304,77), (16,576,77), (32,4096,77), \
                (32,1024,77), (32,256,77), (32,6400,77), (32,1600,77), (32,400,77), (32,9216,77), (32,2304,77), (32,576,77), (12,77,77), (24,77,77), (16,4096,77), (16,1024,77), (16,256,77), \
                (16,6400,77), (16,1600,77), (16,100,100), (32,100,100),  (16,144,144), (32,144,144),(16,256,256), (32,256,256), (4,32,288,296), (4,32,340,348),(4,32,344,352), (4,32,364,372),(64,16,384,384), (128,16,384,384), \
                (4,32,380,388),(16,400,400),(32,400,400), (4,32,424,432), (4,32,428,436), (4,32,444,452), (4,32,448,456), (4,32,464,472), (4,32,480,488), (20,16,512,512), (4,32,508,516), (4,32,512,520), (4,32,512,520), (16,576,576), \
                (32,576,576), (16, 2316, 2316), (16,12,384,384), (24,12,384,384), (32,12,384,384), (64,12,384,384), (128,12,384,384), (16,16,384,384), (24,16,384,384), (32,16,384,384), (16,1024,1024), (32,1024,1024),(16,2304,2304),(32,2304,2304), \
                (16,1600,1600), (32,1600,1600), (2, 32, 2048, 2048), (2,4096,4096), (2, 5, 4096, 4096), (32,4096,4096), (1,4096,4096), (16,4096,4096), (1,6400,6400), (1,9216,9216), (2,9216,9216), (10240,30522)
                ]:
                if self.is_log_softmax:
                    function_name = "log_softmax_test_fwd"
                else:
                    function_name = "softmax_test_fwd"
                self.launch_forward(shape=shape, dtype=dtype, is_optim=True, feature="from_model", func_name=function_name, dim=-1)

                if self.is_log_softmax:
                    function_name = "log_softmax_test_bwd"
                else:
                    function_name = "softmax_test_bwd"
                self.launch_backward(shape=shape, dtype=dtype, is_optim=True, feature="from_model", func_name=function_name, dim=-1)

            #shape for combine load
            rows = [1000, 10001, 50000]
            cols = [
                520, 528, 536, 544, 552, 560, 568, 576, 584, 592, 600, 608, 616, 624, 632, 640, 648, 656, 664, 672, 680, 688, 696, \
                704, 712, 720, 728, 736, 744, 752, 760, 768, 776, 784, 792, 800, 808, 816, 824, 832, 840, 848, 856, 864, 872, 880, \
                888, 896, 904, 912, 920, 928, 936, 944, 952, 960, 968, 976, 984, 992, 1000, 1008, 1016
            ]
            for row, col in itertools.product(rows, cols):
                shape = (row, col)
                if self.is_log_softmax:
                    function_name = "log_softmax_test_fwd"
                else:
                    function_name = "softmax_test_fwd"
                self.launch_forward(shape=shape, dtype=dtype, is_optim=True, feature="common", func_name=function_name, dim=-1)

                if self.is_log_softmax:
                    function_name = "log_softmax_test_bwd"
                else:
                    function_name = "softmax_test_bwd"
                self.launch_backward(shape=shape, dtype=dtype, is_optim=True, feature="common", func_name=function_name, dim=-1)

            '''
            #shape generated randomly
            rows = [1000, 1001, 10000, 10001, 50000, 50001]
            cols = list(range(101, 5000, 1000)) + list(range(100, 5000, 1000))
            for row, col in itertools.product(rows, cols):
                shape = (row, col)
                if self.is_log_softmax:
                    function_name = "log_softmax_test_fwd"
                else:
                    function_name = "softmax_test_fwd"
                self.launch_forward(shape=shape, dtype=dtype, is_optim=False, feature="uncommon", func_name=function_name, dim=-1)

                if self.is_log_softmax:
                    function_name = "log_softmax_test_bwd"
                else:
                    function_name = "softmax_test_bwd"
                self.launch_backward(shape=shape, dtype=dtype, is_optim=False, feature="uncommon", func_name=function_name, dim=-1)

            #shape generated randomly but dim is not last dim
            shape_0_list = list(range(101, 500, 100)) + list(range(100, 500, 100))
            shape_1_list = list(range(101, 500, 100)) + list(range(100, 500, 100))
            shape_2_list = list(range(101, 500, 100)) + list(range(100, 500, 100))
            shape_3_list = list(range(101, 500, 100)) + list(range(100, 500, 100))
            for shape_0, shape_1, shape_2, shape_3 in itertools.product(shape_0_list, shape_1_list, shape_2_list, shape_3_list):
                shape = (shape_0, shape_1, shape_2, shape_3)
                for softmax_dim in range(0, 3):
                    self.launch_forward(shape, dtype, False, "random_generated_but_not_last_dim", 10, softmax_dim)
                    self.launch_backward(shape, dtype, False, "random_generated_but_not_last_dim", 10, softmax_dim)
            '''


if __name__ == '__main__':
    f_softmax = SoftMaxPerfTest(args.platform, args.branch)
    f_softmax.run()
    f_log_softmax = SoftMaxPerfTest(args.platform, args.branch, True)
    f_log_softmax.run()


