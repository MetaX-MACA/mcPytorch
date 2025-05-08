
import argparse
from datetime import datetime
import time
import torch
import torch.nn.functional as F
import re
import copy

from utils.pt_reporter import PyTorchReporter
from utils.utils import launch_prof, get_time_s, accuracy_check, make_pattern_tensor
torch.manual_seed(107)

parser = argparse.ArgumentParser()
parser.add_argument('-platform', type=str, metavar='platform', help='chip platform')
parser.add_argument('-branch', type=str, metavar='branch', help='git branch')
parser.add_argument('-path', type=str, metavar='path', help='output json path')
args = parser.parse_args()


class ElementwiseBroadcastPerfTest(PyTorchReporter):
    def __init__(self, platform, branch):
        super().__init__()
        self.function = "elementwise_broadcast_test_"
        self.platform = platform
        self.branch = branch
    
    @launch_prof(args.path, warmup_num=5, active_num=40)
    def elementwise_broadcast_test_2_1(self, shape, dtype, is_optim=True, func_name="", feature=""):
        input = torch.randn(shape, dtype = dtype, device="cpu")
        input_d = input.cuda()
        input_1 = input.as_strided(input.shape, (0, 1))
        input_d_1 = input_d.as_strided(input.shape, (0, 1))
        output_d_1 = input_d_1.contiguous()
        output_golden_1 = input_1.contiguous()

        # acc check
        assert torch.allclose(output_golden_1, output_d_1.cpu()), f"{func_name}, {dtype} {shape} accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                            record_shapes=False,
                        ) as profiler:
            output_d_1 = input_d_1.contiguous()
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t =  get_time_s(info)
        return t
    
    @launch_prof(args.path, warmup_num=50, active_num=100)
    def elementwise_broadcast_test_3_1(self, shape, dtype, is_optim=True, func_name="", feature=""):
        input_shape=(1, shape[1], shape[0])
        out_shape=(shape[2], shape[1], shape[0])
        input_d = torch.randn(input_shape, dtype=dtype, device="cuda")
        input_d_1 = input_d.as_strided(input_shape, (0, 1, shape[1]))
        # test broadcast
        output_d = torch.zeros(out_shape, dtype = dtype, device="cuda")

        # warm up
        output_d.copy_(input_d_1)

        # acc check
        input_c = input_d.cpu()
        input_c_1 = input_c.as_strided(input_shape, (0, 1, shape[1]))
        output_c = torch.zeros(out_shape, dtype = dtype)
        output_c.copy_(input_c_1)
        assert torch.allclose(output_c, output_d.cpu()), f"{func_name}, {dtype} {input_shape} accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                            record_shapes=False,
                        ) as profiler:
            output_d.copy_(input_d_1)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t =  get_time_s(info)
        return t

    @launch_prof(args.path, warmup_num=20, active_num=60)
    def elementwise_broadcast_test_3_3(self, shape, dtype, is_optim=True, func_name="", feature=""):
        input = torch.randn(shape, dtype=dtype, device="cuda").requires_grad_(True)
        grad_input = torch.randn(shape, dtype=dtype, device="cuda")

        input_c = copy.deepcopy(input).cpu().requires_grad_(True)
        grad_input_c = copy.deepcopy(grad_input).cpu()

        m_g = torch.nn.BatchNorm2d(shape[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        m_g.weight = torch.nn.Parameter(torch.randn(shape[1], dtype=torch.float))
        m = m_g.cuda()
        m.eval()
        out = m(input)

        # warm up
        out.backward(grad_input, retain_graph=True)

        if dtype == torch.float:
          # acc check
          m_c = copy.deepcopy(m_g).cpu()
          m_c.eval()
          out_c = m_c(input_c)
          assert torch.allclose(out_c, out.cpu()), f"{func_name}, {dtype} {shape} out accuracy check fail"
          out_c.backward(grad_input_c, retain_graph=True)
          assert torch.allclose(grad_input_c, grad_input.cpu()), f"{func_name}, {dtype} {shape} grad accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                            record_shapes=False,
                        ) as profiler:
            out.backward(grad_input, retain_graph=True)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t =  get_time_s(info)
        return t
    
    @launch_prof(args.path, warmup_num=20, active_num=60)
    def elementwise_3_2_broadcast_dim2_arg0_contiguous(self, shape, dtype, is_optim=True, func_name="", feature=""):
        a = torch.randn(shape, dtype=dtype, device="cuda")
        b = torch.randn(shape, dtype=dtype, device="cuda")

        a_d = a.as_strided(shape, (1, shape[0], shape[0] * shape[1]))
        b_d = b.as_strided(shape, (0, 1, 0))

        m = torch.add

        tol = 1e-4 if dtype==torch.float else 5e-3
        assert accuracy_check(m, [a_d, b_d], fwd_tol=tol), f"{func_name}, {dtype} {shape} accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                            record_shapes=False,
                        ) as profiler:
            output = m(a_d, b_d)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t =  get_time_s(info)
        return t
    
    @launch_prof(args.path, warmup_num=20, active_num=60)
    def elementwise_3_2_broadcast_dim2_uncontiguous(self, shape, dtype, is_optim=True, func_name="", feature=""):
        a = torch.randn(shape[0] * shape[1], 2, shape[0], dtype=dtype)
        a = a.transpose(0, 2)

        b = torch.randn(shape, dtype=dtype)
        b = b.transpose(0, 2)

        a_d = copy.deepcopy(a).cuda()
        b_d = copy.deepcopy(b).cuda()
        a_d = a_d.as_strided(b.shape, (2, b.shape[2], b.shape[1] * b.shape[2]))
        b_d = b_d.as_strided(b.shape, (2, b.shape[0] * 2, 0))

        m = torch.add
        
        tol = 1e-4 if dtype==torch.float else 5e-3
        assert accuracy_check(m, [a_d, b_d], fwd_tol=tol), f"{func_name}, {dtype} {shape} accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                            record_shapes=False,
                        ) as profiler:
            output = m(a_d, b_d)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t =  get_time_s(info)
        return t
    
    @launch_prof(args.path, warmup_num=20, active_num=60)
    def elementwise_3_2_broadcast_dim1_uncontiguous(self, shape, dtype, is_optim=True, func_name="", feature=""):
        a = torch.randn(shape[0] * shape[1], shape[1], shape[2] * 4, dtype=dtype)
        a = a.transpose(0, 2)

        b = torch.randn(shape, dtype=dtype)
        b = b.transpose(0, 2)

        a_d = copy.deepcopy(a).cuda()
        b_d = copy.deepcopy(b).cuda()
        a_d = a_d.as_strided(b.shape, (2, b.shape[0] * 4, b.shape[2] // 2))
        b_d = b_d.as_strided(b.shape, (2, 0, b.shape[0] * 2))

        m = torch.add

        tol = 1e-4 if dtype==torch.float else 5e-3
        assert accuracy_check(m, [a_d, b_d], fwd_tol=tol), f"{func_name}, {dtype} {shape} accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                            record_shapes=False,
                        ) as profiler:
            output = m(a_d, b_d)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t =  get_time_s(info)
        return t

    @launch_prof(args.path, warmup_num=20, active_num=60)
    def elementwise_broadcast_test_3_2_cast(self, shape, dtype, dtype2, is_optim=True, func_name="", feature=""):
        a = torch.randn(shape, dtype=dtype, device="cuda")
        a = a.transpose(0, 2)

        b = torch.randn(shape, dtype=dtype2, device="cuda")
        b = b.transpose(0, 2)

        b = b.as_strided(b.shape, (1, 0, b.shape[0]))
        m = torch.add

        tol = 1e-4 if dtype==torch.float else 5e-3
        assert accuracy_check(m, [a, b], fwd_tol=tol), f"{func_name}, {dtype} {shape} accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                            record_shapes=False,
                        ) as profiler:
            output = m(a, b)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t =  get_time_s(info)
        return t
    
    @launch_prof(args.path, warmup_num=100, active_num=100)
    def elementwise_broadcast_test_2_2_sd(self, shape, stride, dtype, is_optim=True, func_name="", feature=""):
        inpo = torch.rand(629145600,device="cuda",dtype=dtype)
        inp1 = inpo.as_strided(shape,stride)
        inp2 = torch.rand(shape,device="cuda",dtype=dtype)
        m = torch.add

        tol = 2e-4 if dtype==torch.float else 5e-3
        assert accuracy_check(m, [inp1, inp2], fwd_tol=tol), f"{func_name}, {dtype} {shape} accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                            record_shapes=False,
                        ) as profiler:
            out = m(inp1, inp2)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t =  get_time_s(info)
        return t
    
    @launch_prof(args.path, warmup_num=5, active_num=40)
    def elementwise_broadcast_test_2_2_internlm(self, shape, dtype, is_optim=True, func_name="", feature=""):
        input = torch.randn(shape, dtype = dtype, device="cpu")
        input_1 = torch.randn(shape, dtype = dtype, device="cpu")

        input_d = input.cuda()
        input_1_d = input_1.cuda()

        # input_1 = input_1.as_strided(input_1.shape, (0,1))
        input_1_d = input_1_d.as_strided(input_1_d.shape, (0,1))

        m = torch.add
        tol = 1e-4 if dtype==torch.float else 5e-3
        assert accuracy_check(m, [input_d, input_1_d], fwd_tol=tol), f"{func_name}, {dtype} {shape} arg1 accuracy check fail"
        # assert accuracy_check(m, [input_1_d, input_d], fwd_tol=tol), f"{func_name}, {dtype} {shape} arg0 accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                            record_shapes=False,
                        ) as profiler:
            output_d = m(input_d, input_1_d)  # arg1
            # output_d_3 = m(input_1_d, input_d)  # arg0
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t =  get_time_s(info)
        return t
    
    @launch_prof(args.path, warmup_num=5, active_num=40)
    def elementwise_broadcast_test_2_2_internlm_uncontiguous(self, shape, dtype, is_optim=True, func_name="", feature=""):
        input = torch.randn(shape, dtype = dtype, device="cpu")
        input_2 = torch.randn(shape, dtype = dtype, device="cpu")

        input_d = input.cuda()
        input_2_d = input_2.cuda()

        # input_2 = input_2.as_strided(input_2.shape, (1,0))
        input_2_d = input_2_d.as_strided(input_2_d.shape, (1,0))

        m = torch.add
        tol = 1e-4 if dtype==torch.float else 5e-3
        assert accuracy_check(m, [input_d, input_2_d], fwd_tol=tol), f"{func_name}, {dtype} {shape} arg1 accuracy check fail"
        # assert accuracy_check(m, [input_2_d, input_d], fwd_tol=tol), f"{func_name}, {dtype} {shape} arg0 accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                            record_shapes=False,
                        ) as profiler:
            output_d_2 = m(input_d, input_2_d)  # arg1 uncontiguous
            # output_d_4 = m(input_2_d, input_d)  # arg0 uncotiguous
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t =  get_time_s(info)
        return t
    
    @launch_prof(args.path, warmup_num=20, active_num=100)
    def elementwise_3_2_broadcast_dim1_contiguous_s(self, shape, dtype, is_optim=True, func_name="", feature=""):
        a = torch.randn(shape, dtype=dtype)
        b = torch.randn(shape, dtype=dtype)
        a_d = a.cuda()
        b_d = b.cuda()
        a_d = a_d.as_strided(shape, (1, 0, shape[0]))
        b_d = b_d.as_strided(shape, (1, shape[0], 0))

        m = torch.add

        tol = 1e-4 if dtype==torch.float else 5e-3
        assert accuracy_check(m, [a_d, b_d], fwd_tol=tol), f"{func_name}, {dtype} {shape} accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                            record_shapes=False,
                        ) as profiler:
            output = m(a_d, b_d)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t =  get_time_s(info)
        return t

    @launch_prof(args.path, warmup_num=50, active_num=100)
    def elementwise_broadcast_test_4_1(self, shape, dtype, is_optim=True, func_name="", feature=""):
        input = torch.randn(shape, dtype=dtype, device="cuda")
        input = input.as_strided(shape, (1, shape[0] * shape[2], shape[0], shape[0] * shape[1] * shape[2]))
        output = torch.randn(shape, dtype=dtype, device="cuda")
        output = output.as_strided(shape, (1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2]))

        input_c = input.cpu()
        input_c = input_c.as_strided(shape, (1, shape[0] * shape[2], shape[0], shape[0] * shape[1] * shape[2]))
        output_c = output.cpu()
        output_c = output_c.as_strided(shape, (1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2]))

        output_c.copy_(input_c)
        output.copy_(input)

        assert torch.allclose(output_c, output.cpu()), f"{func_name}, {dtype} {shape} out accuracy check fail"

        torch.cuda.synchronize()
        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                            record_shapes=False,
                        ) as profiler:
            output.copy_(input)
        info = profiler.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", max_name_column_width=1000, row_limit=-1)
        t =  get_time_s(info)
        return t

    @launch_prof(args.path, warmup_num=30, active_num=100)
    def elementwise_broadcast_test_copy_template(self, shape, output_strides, input_strides, dtype, is_optim=True, func_name="", feature=""):
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

    def run_common(self, is_optim=True):
        self.feature = "common"
        self.is_optim = is_optim
        # benchmark for 3_2 cast broadcast in bert
        # different vec num
        # vec_shapes = [[24, 4096, 64], [24, 4096, 128], [24, 4096, 192], [24, 4096, 256], [24, 4096, 320]]
        vec_shapes = [[24, 4096, 192], [24, 4096, 256], [24, 4096, 320]]
        dtype1, dtype2 = torch.float16, torch.float
        for shape in vec_shapes:
            self.elementwise_broadcast_test_3_2_cast(shape=shape, dtype=dtype1, dtype2=dtype2, is_optim=is_optim, func_name="elementwise_broadcast_test_3_2_cast", feature=self.feature)

        # benchmark for 3_1 broadcast
        for dtype in [torch.float, torch.float16, torch.bfloat16]:
          for size0 in [128,256,512]:
              for size1 in [128,256]: # size0 * size1 should be multiple of warp_size
                  for size2 in [128,256]:
                      shape = [size0, size1, size2]
                      self.elementwise_broadcast_test_3_1(shape=shape, dtype=dtype, is_optim=is_optim, func_name="elementwise_broadcast_test_3_1", feature=self.feature)
        
        # benchmark for 3_3 broadcast bn eval backward
        for dtype in [torch.float, torch.float16, torch.bfloat16]:
          # for shape in [[32, 256, 4, 1024], [32, 264, 4, 1024], [32, 272, 4, 1024], [32, 280, 4, 1024], [32, 288, 4, 1024], 
          #               [32, 128, 100, 168], [32, 1024, 50, 84], [32, 64, 200, 336], [32, 512, 100, 168], [32, 512, 25, 42], [32, 2048, 25, 42],
          #               [32, 256, 200, 336], [32, 256, 50, 80], [32, 512, 25, 40], [32, 1024, 84, 50], [32, 64, 672, 400], 
          #               [8, 512, 20, 20], [8, 256, 40, 40], [8, 256, 40, 40], [8, 32, 160, 160], [8, 32, 320, 320],
          #               [32, 64, 400, 672], [32, 256, 50, 84], [32, 296, 4, 1024], [256, 296, 4, 256]]:
          for shape in [[32, 256, 4, 1024], [32, 128, 100, 168], [32, 512, 64, 64]]:
              self.elementwise_broadcast_test_3_3(shape=shape, dtype=dtype, is_optim=is_optim, func_name="elementwise_broadcast_test_3_3", feature=self.feature)


    def run_from_model(self, is_optim=True):
        self.feature = "from_model"
        self.is_optim = is_optim
        # benchmark for 3_2 cast broadcast in bert
        # shapes = [[24, 4608, 384], [32, 4608, 384], [64, 4608, 384], [128, 4608, 384], [21, 4608, 384]]
        shapes = [[32, 4608, 384], [64, 4608, 384], [128, 4608, 384]]

        dtype1, dtype2 = torch.float16, torch.float

        for shape in shapes:
            self.elementwise_broadcast_test_3_2_cast(shape=shape, dtype=dtype1, dtype2=dtype2, is_optim=is_optim, func_name="elementwise_broadcast_test_3_2_cast", feature=self.feature)

        # benchmark for 2_2 in stablediffusion
        # self.function = "elementwise_broadcast_test_2_2"
        # shape_lst = [((8192,1280),(1280*2,1),torch.float16), ((8192,1280),(1280*3,1),torch.float16), ((8192,1280),(1280*3,1),torch.bfloat16), 
        #              ((333,128),(128*2,1),torch.float16), ((333,64),(128*2,1),torch.float16)]
        shape_lst = [((8192,1280),(1280*2,1),torch.float16), ((8192,1280),(1280*3,1),torch.bfloat16)]
        for (shape, stride, dtype) in shape_lst:
            self.elementwise_broadcast_test_2_2_sd(shape=shape, stride=stride, dtype=dtype, is_optim=is_optim, func_name="elementwise_broadcast_test_2_2", feature=self.feature)

        # benchmark for 2_2 broadcast in internlm
        # self.function = "elementwise_broadcast_test_2_2"
        # dim 2
        for dtype in [torch.float16, torch.float, torch.bfloat16]:
            # for shape in [(32, 4194304), (64, 4194304), (2048, 103168), (2048, 32000), (16384, 512), ]:
            for shape in [(2048, 103168), (2048, 32000), (16384, 512), ]:
                self.elementwise_broadcast_test_2_2_internlm(shape=shape, dtype=dtype, is_optim=is_optim, func_name="elementwise_broadcast_test_2_2", feature=self.feature)
                
        # benchmark for 2_2 broadcast in internlm uncontiguous
        # self.function = "elementwise_broadcast_test_2_2"
        # dim 2
        # for dtype in [torch.float16, torch.float, torch.bfloat16]:
        #     for shape in [(32, 4194304), (64, 4194304), (2048, 103168), (2048, 32000), (16384, 512), ]:
        #         self.elementwise_broadcast_test_2_2_internlm_uncontiguous(shape=shape, dtype=dtype, is_optim=is_optim, func_name="elementwise_broadcast_test_2_2", feature=self.feature)

        # benchmark for 3_2 broadcast in 3ddifussion 3_2_broadcast_dim2_arg0_contiguous
        # shapes = [[1146880, 64, 2], [143360, 32, 2], [2838528, 64, 2]]
        shapes = [[143360, 32, 2], ]
        for dtype in [torch.float16, torch.float, torch.bfloat16]:
          for shape in shapes:
            self.elementwise_3_2_broadcast_dim2_arg0_contiguous(shape=shape, dtype=dtype, is_optim=is_optim, func_name="elementwise_broadcast_test_3_2", feature=self.feature)
        
        # benchmark for 3_2 broadcast in chatglm 3_2_broadcast_dim2_uncontiguous
        # shapes = [[128, 2048, 64], [128, 2048, 32], [128, 2048, 8]]
        shapes = [[128, 2048, 64], [128, 2048, 32]]
        for dtype in [torch.float16, torch.float, torch.bfloat16]:
          for shape in shapes:
            self.elementwise_3_2_broadcast_dim2_uncontiguous(shape=shape, dtype=dtype, is_optim=is_optim, func_name="elementwise_broadcast_test_3_2", feature=self.feature)

        # benchmark for 3_2 broadcast in chatglm 3_2_broadcast_dim1_uncontiguous
        shapes = [[2048, 8, 32], [2048, 8, 64]]
        for dtype in [torch.float16, torch.bfloat16]:
          for shape in shapes:
            self.elementwise_3_2_broadcast_dim1_uncontiguous(shape=shape, dtype=dtype, is_optim=is_optim, func_name="elementwise_broadcast_test_3_2", feature=self.feature)
        
        # benchmark for 3_2 broadcast in maskrcnn test_elementwise_3_2_broadcast_dim1_contiguous_s
        # contiguous dim < 64
        # shapes = [[4,14000,3],[4,3700,3],[4,51200,3],[4,54400,3],[4,56000,3],[4,59200,3],[4,60800,3],[4,62400,3],[4,67200,3],[4,800,3]]
        shapes = [[4,59200,3],[4,60800,3],[4,62400,3],[4,67200,3]]
        # self.function = "elementwise_broadcast_test_3_2"
        for dtype in [torch.float16, torch.bfloat16, torch.float]:
          for shape in shapes:
            self.elementwise_3_2_broadcast_dim1_contiguous_s(shape=shape, dtype=dtype, is_optim=is_optim, func_name="elementwise_broadcast_test_3_2", feature=self.feature)
        
        # benchmark for 2_1 broadcast
        # shapes = [(32, 4194304), (64, 4194304), (2048, 103168), (2048, 32000), (16384, 512), (44416, 256)]
        shapes = [(32, 4194304), (2048, 32000), (16384, 512), (44416, 256)]
        for dtype in [torch.float16, torch.bfloat16, torch.float]:
          for shape in shapes:
            self.elementwise_broadcast_test_2_1(shape=shape, dtype=dtype, is_optim=is_optim, func_name="elementwise_broadcast_test_2_1", feature=self.feature)

        # benchmark for 4_1 broadcast
        shapes = [[64,16,512,32], [64,512,16,32]]
        for dtype in [torch.float16, torch.bfloat16, torch.float]:
          for shape in shapes:
             self.elementwise_broadcast_test_4_1(shape=shape, dtype=dtype, is_optim=is_optim, func_name="elementwise_broadcast_test_4_1", feature=self.feature)

        # benchmark for 1_1 broadcast
        # shapes = [[4669440],[5586944],[5832704],[7598080],[7618560],[7669760],[7813120],[7833600],[7915520],[7976960],[7997440],
        #           [8036352],[8099840],[8208384],[8257536],[8437760],[8443904],[8454144],[8650752],[8699904],[8712192],[8785920]]
        shapes = [[8443904],[8454144],[8650752],[8699904],[8712192],[8785920]]
        for dtype in [torch.float16, torch.bfloat16, torch.float]:
          for shape in shapes:
             self.elementwise_broadcast_test_copy_template(shape=shape, output_strides=[(1, )], input_strides=[(0, )], dtype=dtype, is_optim=is_optim, func_name="elementwise_broadcast_test_1_1", feature=self.feature)
    
    def run_uncommon(self, is_optim=True):
        self.feature = "uncommon"
        self.is_optim = is_optim

        # benchmark for 3_1 broadcast uncommon
        for dtype in [torch.float, torch.float16, torch.bfloat16]:
          for size0 in [8,33,49,111,129,514]:
              for size1 in [65,66,67,257]: # size0 * size1 should be multiple of warp_size
                  for size2 in [34,99,133,198,257]:
                      shape = [size0, size1, size2]
                      self.elementwise_broadcast_test_3_1(shape=shape, dtype=dtype, is_optim=is_optim, func_name="elementwise_broadcast_test_3_1", feature=self.feature)
        
        # benchmark for 3_3 broadcast bn eval backward uncommon
        for dtype in [torch.float, torch.float16, torch.bfloat16]:
          for shape in [[2, 64, 64, 4], [1, 256, 50, 68], [1, 128, 100, 152], [1, 64, 200, 304], [1, 128, 100, 136], [1, 512, 25, 38], [1, 64, 200, 272],
                        [1, 512, 25, 34], [1, 2048, 25, 38], [257, 296, 4, 256], [1, 256, 50, 76]]:
              self.elementwise_broadcast_test_3_3(shape=shape, dtype=dtype, is_optim=is_optim, func_name="elementwise_broadcast_test_3_3", feature=self.feature)

        # benchmark for 3_2 broadcast in 3ddifussion 3_2_broadcast_dim2_arg0_contiguous
        shapes = [[1146880, 69, 5], [143359, 32, 2], 
                      [143360, 35, 2], [1433657, 32, 7], [2838529, 64, 2], [2838528, 61, 2], 
                      [2838528, 64, 3], [2838527, 23, 7], [192, 4, 384], [192, 5, 384], [192, 7, 512]]
        # self.function = "elementwise_broadcast_test_3_2"
        for dtype in [torch.float16, torch.float, torch.bfloat16]:
          for shape in shapes:
            self.elementwise_3_2_broadcast_dim2_arg0_contiguous(shape=shape, dtype=dtype, is_optim=is_optim, func_name="elementwise_broadcast_test_3_2", feature=self.feature)
        
        # benchmark for 3_2 broadcast in chatglm 3_2_broadcast_dim2_uncontiguous
        shapes = [[63, 128, 6], [63, 129, 4], [17, 1021, 32], [33, 511, 64], [65, 255, 128]]
        # self.function = "elementwise_broadcast_test_3_2"
        for dtype in [torch.float16, torch.float, torch.bfloat16]:
          for shape in shapes:
            self.elementwise_3_2_broadcast_dim2_uncontiguous(shape=shape, dtype=dtype, is_optim=is_optim, func_name="elementwise_broadcast_test_3_2", feature=self.feature)

        # benchmark for 3_2 broadcast in chatglm 3_2_broadcast_dim1_uncontiguous
        shapes = [[32, 1023, 8], [31, 1024, 16], [33, 1025, 2], [1023, 17, 64], [511, 33, 128], [1029, 3, 256]]
        # self.function = "elementwise_broadcast_test_3_2"
        for dtype in [torch.float16, torch.bfloat16]:
          for shape in shapes:
            self.elementwise_3_2_broadcast_dim1_uncontiguous(shape=shape, dtype=dtype, is_optim=is_optim, func_name="elementwise_broadcast_test_3_2", feature=self.feature)


if __name__ == '__main__':
    f = ElementwiseBroadcastPerfTest(args.platform, args.branch)
    f.run_from_model()
    f.run_common()
    # f.run_uncommon()

