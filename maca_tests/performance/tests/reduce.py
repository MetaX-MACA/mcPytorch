import argparse
from datetime import datetime
import time
import torch
import re
import random
from random import randint
from utils.pt_reporter import PyTorchReporter
from utils.utils import launch_prof, get_time_s, accuracy_check
from utils.perf_data_statistics import *


parser = argparse.ArgumentParser()
parser.add_argument('-platform', type=str, metavar='platform', help='chip platform')
parser.add_argument('-branch', type=str, metavar='branch', help='git branch')
parser.add_argument('-path', type=str, metavar='path', help='output json path')
args = parser.parse_args()

class ReducePerfTest(PyTorchReporter):
    def __init__(self, platform, branch):
        super().__init__()
        self.platform = platform
        self.branch = branch
        self.record_log_name = "reduce"
        reset_perf_data(self.record_log_name, self.platform)

    @launch_prof(args.path)
    def launch_op(self, op, inputs, dim):
        tol = 1e-4 if inputs.dtype==torch.float else 5e-3
        assert accuracy_check(op, [inputs], fwd_tol=tol), f"{op}, {inputs.dtype} accuracy check fail"

        torch.cuda.synchronize()    # theoretically we no need call it there, but we need call it in maca
        # we can also control the warmup and active number by profiler.schedule, but it looks like with bug in maca
        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                            record_shapes=False,
                        ) as profiler:
            op(inputs, dim)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t =  get_time_s(info)
        return t  
    
    def run_op(self, op, dtype, shape, dim, is_optim, function, new_strides=None, new_shape=None, is_rand=False, feature="common"):
        if "uncommon" in feature:
            return
        self.testgroup = str(dtype)[6:]
        input = torch.randn(shape, dtype=dtype, device="cuda") if dtype != torch.int64 else (torch.randn(shape, dtype=torch.float, device="cuda") * 100).to(dtype)
        if new_strides is not None and new_shape is not None:
            input.as_strided(new_shape, new_strides)
            shape = new_shape

        assert len(shape) >= 2
        dim0 = shape[0]
        dim1 = shape[1]
        shape_2d = True
        if len(shape) > 2:
            shape_2d = False
            if isinstance(dim, list):
                for dim_i in dim:
                    dim0 *= dim_i
                for i, dim_i in enumerate(shape):
                    if i not in dim:
                        dim1 *= dim_i
            else :
                dim0 = shape[dim]
                dim1 = 1
                for i, dim_i in enumerate(shape):
                    if i != dim:
                        dim1 *= dim_i
        record_test_case_name = op.__name__ + "_" + self.testgroup + "_"  + str(dim) + "_" + str(is_optim) + "_" + str(is_rand) + "_" + str(new_strides) + "_" + feature + "_" + str(shape_2d)

        self.testcase = op.__name__ + "_" + str(shape) + "_dim_" + str(dim) + "_new_stride_" + str(new_strides)
        self.feature = feature
        self.function = function
        # json_path = args.path + "/" + self.function + "_" + self.testgroup + "_" + self.testcase + ".json"
        # print("json_path: ", json_path)
        # import os
        # if os.path.exists(json_path):
        #     print("skip: ", json_path)
        #     import json
        #     with open(json_path) as f:
        #         data = json.load(f)
        #         mean_time = data["performance"]["metrics"]["second"]
        #         record_perf_data(self.record_log_name, self.platform, record_test_case_name, dim0, dim1, mean_time)
        #     return

        mean_time = self.launch_op(op, input, dim)        
        record_perf_data(self.record_log_name, self.platform, record_test_case_name, dim0, dim1, mean_time)
    
    def rand_shape_generator(self, start=6, length=11, pow_base=2):
        for i in range(start, start+length):
            random.seed(1)
            start0 = 1 if i == 0 else pow(pow_base, i-1)
            end0 = pow(pow_base, i)
            dim0 = randint(start0, end0)
            for j in range(start, start+length):
                random.seed(2)
                start1 = 1 if j == 0 else pow(pow_base, j-1)
                end1 = pow(pow_base, j)
                dim1 = randint(start1, end1)
                yield (dim0, dim1)    

    def normal_shape_generator(self, start=6, length=11, pow_base=2):
        for i in range(start, start+length):
            dim0 = pow(pow_base, i)
            for j in range(start, start+length):
                dim1 = pow(pow_base, j)
                yield (dim0, dim1)

    def normal_shape_generator_3d(self, start=1, length=4, pow_base=16):
        max_size = 65536 * 65536
        for i in range(start, start+length):
            dim0 = pow(pow_base, i)
            for j in range(start, start+length):
                dim1 = pow(pow_base, j)
                for k in range(start, start+length):
                    dim2 = pow(pow_base, k)
                    if dim0 * dim1 * dim2 > max_size:
                        break
                    yield (dim0, dim1, dim2)   

    def stride_generator(self, shape):
        max_stride1 = 4
        max_stride0_div_size1 = 3
        assert len(shape) == 2 
        assert shape[1] >= max_stride1 and shape[0] >= max_stride0_div_size1
        for stride1 in range(2, max_stride1+1):
            new_shape = list(shape)
            new_shape[1] = int(shape[1] / stride1)
            new_stride = [shape[1], stride1]
            yield  new_stride, new_shape
        for i in range(1, max_stride0_div_size1+1):
            for j in range(1, max_stride1):
                stride0 = i * shape[1] + j
                new_shape = list(shape)
                new_shape[0] = int((shape[0] * shape[1]) / stride0)
                new_stride = [stride0, 1]
                yield new_stride, new_shape
        for stride1 in range(2, max_stride1+1):
            for i in range(1, max_stride0_div_size1+1):
                for j in range(1, max_stride1):
                    stride0 = i * shape[1] + j
                    new_shape = list(shape)
                    new_shape[0] = int((shape[0] * shape[1]) / stride0)
                    new_shape[1] = int(shape[1] / stride1)
                    new_stride = [stride0, stride1]
                    yield new_stride, new_shape              

    def run(self, niter=1):
        official_test_cases = [
            # Intern-LLM-7B
            {"op": torch.argmax, "dtype": torch.bfloat16, "shape": [2048, 103168], "dim": 1, "optimized": True, "new_strides": None, "new_shape": None}, 
            {"op": torch.sum, "dtype": torch.bfloat16, "shape": [2048, 103168], "dim": 1, "optimized": True, "new_strides": None, "new_shape": None},
            # Intern-LLM65B
            {"op": torch.argmax, "dtype": torch.bfloat16, "shape": [2048, 32000], "dim": 1, "optimized": True, "new_strides": None, "new_shape": None}, 
            {"op": torch.sum, "dtype": torch.bfloat16, "shape": [2048, 32000], "dim": 1, "optimized": True, "new_strides": None, "new_shape": None},   
            # Bert
            {"op": torch.sum, "dtype": torch.half, "shape": [9216, 768], "dim": 0, "optimized": True, "new_strides": None, "new_shape": None},     
            {"op": torch.sum, "dtype": torch.half, "shape": [9216, 3072], "dim": 0, "optimized": True, "new_strides": None, "new_shape": None}, 
            # transformer-decoder long tail
            {"op": torch.max, "dtype": torch.float, "shape": [4000, 4000], "dim": 0, "optimized": True, "new_strides": None, "new_shape": None},  
            # Bert
            {"op": torch.sum, "dtype": torch.half, "shape": [24, 3072, 384], "dim": [0, 2], "optimized": True, "new_strides": None, "new_shape": None},
            # ChatGLM
            {"op": torch.sum, "dtype": torch.half, "shape": [8, 16, 128, 2048], "dim": 1, "optimized": True, "new_strides": None, "new_shape": None},         
            {"op": torch.sum, "dtype": torch.half, "shape": [8, 16, 2048, 128], "dim": 1, "optimized": True, "new_strides": None, "new_shape": None},   
            # MaskRCNN
            {"op": torch.sum, "dtype": torch.half, "shape": [2, 256, 15200], "dim": [0, 2], "optimized": True, "new_strides": None, "new_shape": None},  
            # Bert-large 
            {"op": torch.sum, "dtype": torch.half, "shape": [10240, 1024], "dim": 0, "optimized": True, "new_strides": None, "new_shape": None},               
            {"op": torch.sum, "dtype": torch.half, "shape": [10240, 4096], "dim": 0, "optimized": True, "new_strides": None, "new_shape": None}, 
            # MMLab-MMPre   
            {"op": torch.max, "dtype": torch.int64, "shape": [1, 1000], "dim": 0, "optimized": True, "new_strides": None, "new_shape": None},               
            {"op": torch.max, "dtype": torch.half, "shape": [12845056, 9], "dim": 1, "optimized": True, "new_strides": None, "new_shape": None},   
            # OpenSora         
            {"op": torch.sum, "dtype": torch.float32, "shape": [128, 900, 16, 72], "dim": 3, "new_shape": [128, 16, 900, 72], "new_strides": [103680, 72, 1152, 1],  "optimized": True}, 
            #YOLO     
            {"op": torch.sum, "dtype": torch.half, "shape": [3200, 255], "dim": 0, "optimized": True, "new_strides": None, "new_shape": None}, 
            #SSD
            {"op": torch.sum, "dtype": torch.half, "shape": [800, 361, 486], "dim": [0, 1], "optimized": True, "new_strides": [707292, 486, 1], "new_shape": [8, 361, 486]},
        ]
        for case in official_test_cases:
            self.run_op(case["op"], case["dtype"], case["shape"], case["dim"], case["optimized"], "reduce_test_from_model", case["new_strides"], case["new_shape"], feature="common;from_model")
        
        for op in [torch.sum, torch.max, torch.argmax, torch.mean]:
            for dtype in [torch.float, torch.half, torch.bfloat16]:
                for dim in [0, 1]:
                    function = "reduce_ndim_2_along_dim_" + str(dim)
                    for shape in self.rand_shape_generator():   
                        self.run_op(op, dtype, shape, dim, False, function, is_rand=True, feature="uncommon")
                    for shape in self.normal_shape_generator():  
                        feature = "common"
                        if shape[0] <= 256 and shape[1] <= 256:
                            feature = "uncommon"
                        self.run_op(op, dtype, shape, dim, False, function, feature=feature)
        
        # test abnormal strides
        op = torch.sum
        dtype = torch.bfloat16
        function = "reduce_ndim_2_abnormal_strides"
        for dim in [0, 1]:
            for shape in self.normal_shape_generator(start=1, length=4, pow_base=16):  
                for new_stride, new_shape in self.stride_generator(shape): 
                    self.run_op(op, dtype, shape, dim, False, function, new_stride, new_shape, feature="uncommon")
        
        # 3d reduce and reduce dim1
        for shape in self.normal_shape_generator_3d():
            for reduce_dim in [[1], [0, 2]]:
                if reduce_dim == [1]:
                    function = "reduce_ndim_3_along_dim_1"
                else:
                    function = "reduce_ndim_3_along_dim_0_2"
                self.run_op(op, dtype, shape, reduce_dim, False, function, feature="uncommon")
        
        statistic_perf_data(self.record_log_name)


if __name__ == '__main__':
    f = ReducePerfTest(args.platform, args.branch)
    f.run(niter=20)


