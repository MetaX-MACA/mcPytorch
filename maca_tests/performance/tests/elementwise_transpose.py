
import argparse
from datetime import datetime
import time
import torch
import torch.nn.functional as F
import re
import copy
import math

from utils.pt_reporter import PyTorchReporter
from utils.utils import launch_prof, get_time_s, accuracy_check, make_pattern_tensor
torch.manual_seed(107)

parser = argparse.ArgumentParser()
parser.add_argument('-platform', type=str, metavar='platform', help='chip platform')
parser.add_argument('-branch', type=str, metavar='branch', help='git branch')
parser.add_argument('-path', type=str, metavar='path', help='output json path')
args = parser.parse_args()


class TransposePerfTest(PyTorchReporter):
    def __init__(self, platform, branch):
        super().__init__()
        self.function = "elementwise_transpose_test"
        self.platform = platform
        self.branch = branch

    @launch_prof(args.path, warmup_num=20, active_num=70)
    def elementwise_transpose_test_3_2(self, shape, stride, dtype, is_optim=True, func_name="", feature=""):
        input = torch.randn(shape, dtype = dtype, device = "cuda")
        input_1 = input.as_strided(shape, stride)

        m = torch.add

        tol = 1e-4 if dtype==torch.float else 5e-3
        assert accuracy_check(m, [input, input_1], fwd_tol=tol), f"{func_name}, {dtype} {shape} accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                    record_shapes=False,
                ) as profiler:
            output = m(input, input_1)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t =  get_time_s(info)
        return t

    @launch_prof(args.path, warmup_num=20, active_num=70)
    def elementwise_transpose_test_3_2_maskrcnn(self, shape, dtype, is_optim=True, func_name="", feature=""):
        a = torch.randn(shape, dtype=dtype, device="cuda")
        b = torch.randn(shape, dtype=dtype, device="cuda")

        m = torch.add

        a_d = a.as_strided(shape, (1, shape[0], shape[0] * shape[1]))
        b_d = b.as_strided(shape, (shape[1], 1, shape[0] * shape[1]))

        tol = 1e-4 if dtype==torch.float else 5e-3
        assert accuracy_check(m, [a_d, b_d], fwd_tol=tol), f"{func_name}, {dtype} {shape} accuracy check fail"
        
        torch.cuda.synchronize()
        with torch.profiler.profile(
                      activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                      record_shapes=False,
                  ) as profiler:
            out_d = m(a_d, b_d)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t =  get_time_s(info)
        return t

    @launch_prof(args.path, warmup_num=20, active_num=70)
    def elementwise_transpose_test_2_1(self, shape, dtype, is_optim=True, func_name="", feature=""):
        input1 = torch.randn(shape[0], shape[1], dtype=dtype, device='cuda')
        input2 = torch.transpose(input1, 0, 1)
        output = torch.randn(shape[1], shape[0], dtype=dtype, device='cuda')
        output_c = copy.deepcopy(output).cpu()

        # warm up
        output.copy_(input2)

        # acc check
        input_c = input1.cpu()
        input_c_1 = torch.transpose(input_c, 0, 1)
        output_c.copy_(input_c_1)
        assert torch.allclose(output_c, output.cpu()), f"{func_name}, {dtype} {shape} accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                      activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                      record_shapes=False,
                  ) as profiler:
            output.copy_(input2)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t =  get_time_s(info)
        return t
    
    @launch_prof(args.path, warmup_num=20, active_num=70)
    def elementwise_transpose_test_3_1(self, shape, dtype, is_optim=True, func_name="", feature=""):
        input1 = torch.randn(shape[0], shape[1], shape[2], dtype=dtype, device='cuda')
        input2 = torch.permute(input1, (0, 2, 1))
        output = torch.randn(shape[0], shape[2], shape[1], dtype=dtype, device='cuda')
        output_c = copy.deepcopy(output).cpu()

        # warm up
        output.copy_(input2)

        # acc check
        input_c = input1.cpu()
        input_c_1 = torch.permute(input_c, (0, 2, 1))
        output_c.copy_(input_c_1)
        assert torch.allclose(output_c, output.cpu()), f"{func_name}, {dtype} {shape} accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                      activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                      record_shapes=False,
                  ) as profiler:
            output.copy_(input2)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t = get_time_s(info)
        return t

    @launch_prof(args.path, warmup_num=20, active_num=70)
    def elementwise_transpose_test_4_1(self, shape, dtype, is_optim=True, func_name="", feature=""):
        input1 = torch.randn(shape[0], shape[2], shape[1], shape[3], dtype=dtype, device='cuda')
        input2 = torch.permute(input1, (0, 2, 1, 3))
        output = torch.randn(shape[0], shape[1], shape[2], shape[3], dtype=dtype, device='cuda')
        output_c = copy.deepcopy(output).cpu()

        # warm up
        output.copy_(input2)

        # acc check
        input_c = input1.cpu()
        input_c_1 = torch.permute(input_c, (0, 2, 1, 3))
        output_c.copy_(input_c_1)
        assert torch.allclose(output_c, output.cpu()), f"{func_name}, {dtype} {shape} accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                      activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                      record_shapes=False,
                  ) as profiler:
            output.copy_(input2)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t = get_time_s(info)
        return t

    @launch_prof(args.path, warmup_num=20, active_num=70)
    def elementwise_transpose_test_copy_template(self, shape, output_strides, input_strides, dtype, is_optim=True, func_name="", feature=""):
        outputs, inputs = make_pattern_tensor(shape, dtype, output_strides, input_strides)
        
        output, output_c = outputs[0], outputs[0].cpu()
        input, input_c = inputs[0], inputs[0].cpu()

        output_c.copy_(input_c)
        output.copy_(input)

        assert torch.allclose(output_c, output.cpu()), f"{func_name}, {dtype} {shape} accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                      activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                      record_shapes=False,
                  ) as profiler:
            output.copy_(input)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t = get_time_s(info)
        return t

    @launch_prof(args.path, warmup_num=20, active_num=70)
    def elementwise_transpose_test_add_template(self, shape, output_strides, input_strides, dtype, is_optim=True, func_name="", feature=""):
        outputs, inputs = make_pattern_tensor(shape, dtype, output_strides, input_strides)
        
        a_d = inputs[0]
        b_d = inputs[1]

        m = torch.add

        tol = 1e-4 if dtype==torch.float else 5e-3
        assert accuracy_check(m, [a_d, b_d], fwd_tol=tol), f"{func_name}, {dtype} {shape} accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                      activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                      record_shapes=False,
                  ) as profiler:
            out_d = m(a_d, b_d)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t =  get_time_s(info)
        return t

    def run_from_model(self, is_optim=True):
        self.feature = "from_model,common"
        self.is_optim = is_optim

        # benchmark for elementwise_kernel_3_2_transpose
        # self.function = "elementwise_transpose_test_3_2"
        # shape_lst = [((2, 256, 3800), (256*3800, 1, 256)), ((2, 270, 15200), (15200*270, 1, 270)), ((32, 256, 1050), (1050*256, 1, 256)), ((16, 256, 67200), (256*67200, 1, 256))]
        shape_lst = [((32, 256, 1050), (1050*256, 1, 256)), ((16, 256, 67200), (256*67200, 1, 256))]
        for dtype in [torch.float16, torch.float, torch.bfloat16]:
            for (shape, stride) in shape_lst:
                self.elementwise_transpose_test_3_2(shape=shape, stride=stride, dtype=dtype, is_optim=is_optim, func_name="elementwise_transpose_test_3_2", feature=self.feature)
        
        # benchmark for elementwise_kernel_3_2_transpose in maskrcnn
        # self.function = "elementwise_transpose_test_3_2"
        # shape_list = [[1000,256,2],[1050,256,2],[12800,256,2],[13600,256,2],[14000,256,2],[14800,256,2],[15200,256,2],[16800,256,2],
        #               [3200,256,2],[3400,256,2],[3500,256,2],[3700,256,2],[3800,256,2],[3900,256,2],[4000,256,2],
        #               [4200,256,2],[51200,256,2],[54400,256,2],[59200,256,2],[60800,256,2],[62400,256,2],[64000,256,2],
        #               [67200,256,2],[62399,256,2]]
        shape_list = [[62400,256,2],[64000,256,2],[67200,256,2],[62399,256,2]]
        # bigtransfer_shape_list = [[14400,128,16],[14400,256,16],[14400,512,16],[225,1024,16],[225,4096,16],
        #                           [3600,1024,16],[3600,256,16],[3600,512,16],[900,1024,16],[900,2048,16],[900,512,16],[128,340,128]]
        bigtransfer_shape_list = [[14400,128,16],[14400,256,16],[14400,512,16]]
        shape_list.extend(bigtransfer_shape_list)
        for dtype in [torch.float16, torch.float, torch.bfloat16]:
          for shape in shape_list:
              self.elementwise_transpose_test_3_2_maskrcnn(shape=shape, dtype=dtype, is_optim=is_optim, func_name="elementwise_transpose_test_3_2", feature=self.feature)

        # benchmark for elementwise_kernel_2_1_transpose
        # self.function = "elementwise_transpose_test_2_1"
        # shape_list = [(64, 1024), (128, 65536),(127,65536), (1024, 511), (1024, 512), (2048, 2047), (2048, 4096), (1024, 693), (256, 44352), (256, 693), (2838528, 128), (2838528, 64), (354816, 128), (354816, 256)]
        shape_list = [(2838528, 128), (2838528, 64), (354816, 128), (354816, 256)]
        for dtype in [torch.float16, torch.float, torch.bfloat16]:
          for shape in shape_list:
              self.elementwise_transpose_test_2_1(shape=shape, dtype=dtype, is_optim=is_optim, func_name="elementwise_transpose_test_2_1", feature=self.feature)
        
        # self.function = "elementwise_transpose_test_3_1"
        # shape_list = [(16, 128, 64), (259, 128, 256), (259, 127, 256), (1024, 128, 512), (1024, 1025, 511), (2049, 127, 1023), (131072, 64, 2), (131072, 2, 64), (25600, 17, 210), (25600, 210, 17)]
        shape_list = [(1024, 128, 512), (131072, 2, 64), (25600, 17, 210)]
        for dtype in [torch.float16, torch.float, torch.bfloat16]:
          for shape in shape_list:
              self.elementwise_transpose_test_3_1(shape=shape, dtype=dtype, is_optim=is_optim, func_name="elementwise_transpose_test_3_1", feature=self.feature)

        # self.function = "elementwise_transpose_test_4_1"
        # shape_list = [(256, 8, 127, 32), (256, 31, 256, 32), (256, 5, 256, 32), (129, 8, 256, 32), (256, 256, 32, 32), (256, 256, 4, 32), (256, 128, 8, 32), (128, 256, 8, 32)]
        shape_list = [(256, 8, 127, 32), (256, 256, 32, 32), (256, 256, 4, 32), (256, 128, 8, 32)]
        for dtype in [torch.float16, torch.float, torch.bfloat16]:
          for shape in shape_list:
              self.elementwise_transpose_test_4_1(shape=shape, dtype=dtype, is_optim=is_optim, func_name="elementwise_transpose_test_4_1", feature=self.feature)
        
        # self.function = "elementwise_transpose_test_3_1"
        shapes = []
        swin_transformer_shapes = [[[128,28,14336],(1,512,14336),(1,256,14336)], [[256,14,7168],(1,1024,14336),(1,512,14336)], 
                                   [[512,7,3584],(1,2048,14336),(1,1024,14336)]]
        # longformer_shapes = [[[256,256,13],(1,513,262656),(1,513,262656)], [[256,256,156],(1,513,262656),(1,513,262656)], 
        #                      [[256,256,16],(1,513,262656),(1,513,262656)], [[256,256,192],(1,513,262656),(1,513,262656)], 
        #                      [[49152,2,156],(1,49152,49152*2),(1,16384,65536)], [[49152,2,192],(1,49152,49152*2),(1,16384,65536)], 
        #                      [[64,156,512],(1,64,64*156),(1,64*512,64)], [[64,192,512],(1,64,64*192),(1,64*512,64)], 
        #                      [[768,13,512],(1,768,768*13),(1,768*512,768)], [[768,16,512],(1,768,768*16),(1,768*512,768)], 
        #                      [[768,512,13],(1,768,768*512),(1,768*13,768)], [[768,512,16],(1,768,768*512),(1,768*16,768)],
        #                      [[64,512,156],(1,64,64*512*2),(1,64*156,64)], [[64,512,192],(1,64,64*512*2),(1,64*192,64)]]
        longformer_shapes = [[[256,256,156],(1,513,262656),(1,513,262656)], [[256,256,16],(1,513,262656),(1,513,262656)],
                             [[49152,2,156],(1,49152,49152*2),(1,16384,65536)], [[768,16,512],(1,768,768*16),(1,768*512,768)]]
        shapes.extend(swin_transformer_shapes)
        shapes.extend(longformer_shapes)
        for dtype in [torch.float16, torch.float, torch.bfloat16]:
          for item in shapes:
            self.elementwise_transpose_test_copy_template(shape=item[0], output_strides=[item[1]], input_strides=[item[2]], dtype=dtype, is_optim=is_optim, func_name="elementwise_transpose_test_3_1", feature=self.feature)
        
        # self.function = "elementwise_transpose_test_4_1"
        shapes = []
        # swin_transformer_shapes = [[[32,49,16,2048],(1,32,32*49,32*49*16),(1,32*16*3,32,32*49*16*3)], [[32,49,32,512],(1,32,32*49,32*49*32),(1,32*32*3,32,32*49*32*3)], 
        #                            [[32,49,4,32768],(1,32,32*49,32*49*4),(1,32*4*3,32,32*49*4*3)], [[32,49,8,8192],(1,32,32*49,32*49*8),(1,32*8*3,32,32*49*8*3)]]
        swin_transformer_shapes = [[[32,49,4,32768],(1,32,32*49,32*49*4),(1,32*4*3,32,32*49*4*3)], [[32,49,8,8192],(1,32,32*49,32*49*8),(1,32*8*3,32,32*49*8*3)]]
        # llama_7B_shapes = [[[128,32,340,4],(1,128,128*32,128*32*340),(1,128*348,128,128*32*348)], [[64,340,32,4],(1,64*2,64*340*2,64*340*32*2),(1,64*32,64,64*340*32)], 
        #                    [[128,8,32,4],(1,128,128*348,128*8*1392),(1,128*2048,128,128*8*2048)]]
        llama_7B_shapes = [[[128,32,340,4],(1,128,128*32,128*32*340),(1,128*348,128,128*32*348)], 
                           [[128,8,32,4],(1,128,128*348,128*8*1392),(1,128*2048,128,128*8*2048)]]
        cpm_shapes = [[[64,200,16,128],(1,64,64*200,64*200*16),(1,64*16*3,64,64*200*16*3)], [[64,200,16,32],(1,64,64*200,64*200*16),(1,64*16*3,64,64*200*16*3)], ]
        shapes.extend(swin_transformer_shapes)
        shapes.extend(llama_7B_shapes)
        shapes.extend(cpm_shapes)
        for dtype in [torch.float16, torch.float, torch.bfloat16]:
          for item in shapes:
            self.elementwise_transpose_test_copy_template(shape=item[0], output_strides=[item[1]], input_strides=[item[2]], dtype=dtype, is_optim=is_optim, func_name="elementwise_transpose_test_4_1", feature=self.feature)

        # self.function = "elementwise_transpose_test_4_2"
        shapes = []
        llama_7b_shapes = [[[128,32,340,4],[],[(1,128,128*32,128*32*340),(1,128*340,128,128*32*340)]], ]
        shapes.extend(llama_7b_shapes)
        for dtype in [torch.float16, torch.float, torch.bfloat16]:
          for item in shapes:
            self.elementwise_transpose_test_add_template(shape=item[0], output_strides=item[1], input_strides=item[2], dtype=dtype, is_optim=is_optim, func_name="elementwise_transpose_test_4_2", feature=self.feature)

    def run_uncommon(self, is_optim=True):
        self.feature = "uncommon"
        self.is_optim = is_optim
        # self.function = "elementwise_transpose_test_3_2"
        shape_lst = [((8, 3, 4), (3*4, 1, 3)), ((4, 6, 784), (6 * 784, 1, 6)), 
                      ((2, 256, 825), (256 * 825, 1, 256)), ((2, 257, 825), (257 * 825, 1, 257)), ((2, 258, 827), (257*825, 1, 257))]
        for dtype in [torch.float16, torch.float]:
            for (shape, stride) in shape_lst:
                self.elementwise_transpose_test_3_2(shape=shape, stride=stride, dtype=dtype, is_optim=is_optim, func_name="elementwise_transpose_test_3_2", feature=self.feature)


if __name__ == '__main__':
    f = TransposePerfTest(args.platform, args.branch)
    f.run_from_model()
    # f.run_uncommon()
