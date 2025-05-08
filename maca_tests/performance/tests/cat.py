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

class CatPerfTest(PyTorchReporter):
    def __init__(self, platform, branch):
        super().__init__()
        self.platform = platform
        self.branch = branch

    @launch_prof(args.path)
    def launch(self, shape=None, dtype=None, is_optim=False, feature="", func_name="", dim=-1, all_contiguous=True):
        input_list = []
        for shape_tmp in shape:
            print("shape_tmp:", shape_tmp)
            if all_contiguous:
                input = torch.randn(shape_tmp, dtype = dtype, device="cuda")
            else:
                input_conti = torch.randn(shape_tmp, dtype = dtype, device="cuda")
                input_tmp = torch.randn([int(item * 1.2) for item in shape_tmp], dtype = dtype, device="cuda")
                print("input_tmp.shape:", input_tmp.shape)
                input = torch.as_strided(input_tmp, shape_tmp, [int(item / 2) + 1 for item in input_conti.stride()])
                print("input.shape:", input.shape)

            input_list.append(input)
        tol = 1e-4 if dtype==torch.float else 5e-3
        assert accuracy_check(torch.cat, input_list, fwd_tol=tol), f"torch.cat, {dtype} {shape_tmp} {dim} accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                            record_shapes=False,
                        ) as profiler:
            output = torch.cat(input_list, dim=dim)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t = get_time_s(info)
        return t

    def run(self):
        for dtype in [torch.float16, torch.float32, torch.bfloat16]:
            for dim in [0, 1, 2, 3]:
                for all_contiguous in [True, False]:
                    if dim == 0:
                        pass
                    elif dim == 1:
                        shapes_list = [[(320,112,14,14), (320,56,14,14)], [(320,392,14,14), (320,56,14,14)],
                                    [(32,280,14,14), (32,56,14,14)], [(32,280,14,14), (32,56,14,14)],
                                    [(32,78,56,56), (32,26,56,56)], [(32,1456,7,7), (32,208,7,7)],
                                    [(32,1040,7,7), (32,208,7,7)], [(32,832,7,7), (32,208,7,7)], [(262272,1), (262272,1)], 
                                    [(262272,12),(262272,12),(262272,12)], [(262272,36), (262272,32), (262272,7)], [(262272,16), (262272,64),(262272,4)],
                                    [(154880,1), (154880,1)], [(154880,12), (154880,12)], [(154880,36), (154880,32), (154880,7)], [(154880,16), (154880,64), (154880,4)],
                                    [(141056,1), (141056,1)], [(141056,12), (141056,12),(141056,12)], [(141056,36), (141056,32), (141056,7)], [(141056,16), (141056,64), (141056,4)],
                                    [(137472,1), (137472,1)], [(137472,12), (137472,12), (137472,12)], [(137472,36), (137472,32), (137472,7)], [(137472,16), (137472,64), (137472,4)],
                                    [(133248,1), (133248,1)], [(133248,12), (133248,12), (133248,12)], [(190592,16), (190592,64),(190592,4)], [(239872,1), (239872,1)],
                                    [(239872,12), (239872,12), (239872,12)], [(239872,36), (239872,32), (239872,7)], [(239872,16), (239872,64), (239872,4)]]
                        function_name = f"cat_on_dim_{dim}_all_contiguous_{all_contiguous}"
                        for shapes in shapes_list:
                            self.launch(shape=shapes, dtype=dtype, is_optim=True, feature="from_model", func_name=function_name, dim=dim)
                    elif dim == 2:
                        shapes_list = [[(4096,1,11008), (4096,1,11008)], [(128, 4096, 512), (128, 4096, 512)], [(1,262144,1), (1,262144,1), (1,262144,1)]]
                        function_name = f"cat_on_dim_{dim}_all_contiguous_{all_contiguous}"
                        for shapes in shapes_list:
                            self.launch(shape=shapes, dtype=dtype, is_optim=True, feature="from_model", func_name=function_name, dim=dim)
                    elif dim == 3:
                        shapes_list = [[(256, 8, 720, 1), (256, 8, 720, 720)], [(4096,1,32,128),(4096,1,32,1)], [(4096,1,32,128), (4096,1,32,128)],  [(4096,1,32,128), (4096,1,32,96)],  [(4096,1,32,128), (4096,1,32,128),(4096,1,32,128)]]
                        function_name = f"cat_on_dim_{dim}_all_contiguous_{all_contiguous}"
                        for shapes in shapes_list:
                            self.launch(shape=shapes, dtype=dtype, is_optim=True, feature="from_model", func_name=function_name, dim=dim, all_contiguous=all_contiguous)

if __name__ == '__main__':
    f_cat = CatPerfTest(args.platform, args.branch)
    f_cat.run()