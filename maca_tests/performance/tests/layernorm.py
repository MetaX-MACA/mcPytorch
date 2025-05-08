import argparse
import torch

from utils.pt_reporter import PyTorchReporter
from utils.utils import launch_prof, get_time_s, accuracy_check


parser = argparse.ArgumentParser()
parser.add_argument('-platform', type=str, metavar='platform', help='chip platform')
parser.add_argument('-branch', type=str, metavar='branch', help='git branch')
parser.add_argument('-path', type=str, metavar='path', help='output json path')
parser.add_argument('--is_optim', action='store_true', help='test opt kernel')
args = parser.parse_args()


regular_shape = [(65, 32), (256, 32), (784, 32), (4097, 32), (4896, 32), (8192, 32), (16384, 32), (65535, 32), (144000, 32), 
                 (524288, 32), (100352, 64), (1, 128), (65, 128), (256, 128), (784, 128), (4097, 128), (8192, 128),(10240,1024), 
                 (16384, 128), (65535, 128), (144000, 128), (614400, 128), (1228800, 128), (36000, 192), (1, 256), (65, 256), 
                 (256, 256), (784, 256), (2816, 256), (4097, 256), (8192, 256), (16384, 256), (65535, 256), (144000, 256), (12544, 320), 
                 (2784, 384), (2944, 384), (48000, 384), (2816, 512), (32768, 512), (1, 768), (8, 768), (18, 768), (65, 768), (77, 768), 
                 (128, 768), (256, 768), (784, 768), (2784, 768), (2848, 768), (2944, 768), (3137, 768), (3808, 768), (4097, 768), (6304, 768), 
                 (8192, 768), (9344, 768), (9712, 768), (11008, 768), (12544, 768), (16384, 768), (50432, 768), (65535, 768), (144000, 768), (32, 1024), 
                 (86, 1024), (1920, 1024), (2592, 1024), (2720, 1024), (2752, 1024), (2784, 1024), (3136, 1024), (18464, 1024), (2688, 1280), 
                 (3200, 1280), (2720, 1536), (1, 1600), (65, 1600), (256, 1600), (784, 1600), (1024, 1600), (4097, 1600), (8192, 1600), (16384, 1600), 
                 (65535, 1600), (144000, 1600), (1, 4096), (65, 4096), (256, 4096), (784, 4096), (4097, 4096), (4416, 4096), (8192, 4096), (16384, 4096), 
                 (65535, 4096), (144000, 4096), (1, 7200), (65, 7200), (256, 7200), (784, 7200), (4097, 7200), (8192, 7200), (16384, 7200), (65535, 7200), 
                 (144000, 7200), (1, 9600), (65, 9600), (256, 9600), (784, 9600), (4097, 9600), (8192, 9600), (16384, 9600), (65535, 9600), (144000, 9600),
                 (512, 401408), (512, 200704), (512, 100352), (512, 50176), (32, 1310720), (32, 655360), (32, 327680), (32, 81920)]

not_align_shape = [(1, 15), (65, 15), (256, 15), (784, 15), (4097, 15), (8192, 15), (16384, 15), (65535, 15), (144000, 15), (1, 63), (65, 63), 
                   (256, 63), (784, 63), (4097, 63), (8192, 63), (16384, 63), (65535, 63), (144000, 63), (1, 129), (65, 129), (256, 129), (784, 129), 
                   (4097, 129), (8192, 129), (16384, 129), (65535, 129), (144000, 129), (1, 255), (65, 255), (256, 255), (784, 255), (4097, 255), 
                   (8192, 255), (16384, 255), (65535, 255), (144000, 255), (1, 769), (65, 769), (256, 769), (784, 769), (4097, 769), (8192, 769), 
                   (16384, 769), (65535, 769), (144000, 769), (1, 1599), (65, 1599), (256, 1599), (784, 1599), (4097, 1599), (8192, 1599), (16384, 1599), 
                   (65535, 1599), (144000, 1599), (1, 4099), (65, 4099), (256, 4099), (784, 4099), (4097, 4099), (8192, 4099), (16384, 4099), 
                   (65535, 4099), (144000, 4099), (1, 7205), (65, 7205), (256, 7205), (784, 7205), (4097, 7205), (8192, 7205), (16384, 7205), 
                   (65535, 7205), (144000, 7205), (1, 9609), (65, 9609), (256, 9609), (784, 9609), (4097, 9609), (8192, 9609), (16384, 9609), 
                   (65535, 9609), (144000, 9609)]
small_shape = [(1, 4), (5, 4), (16, 4), (31, 4), (64, 4), (88, 4), (104, 4), (112, 4), (127, 4), (1, 63), (5, 63), (16, 63), (31, 63), (64, 63), (88, 63), 
               (104, 63), (112, 63), (127, 63), (1, 128), (5, 128), (16, 128), (31, 128), (64, 128), (88, 128), (104, 128), (112, 128), (127, 128), 
               (1, 512), (5, 512), (16, 512), (31, 512), (64, 512), (88, 512), (104, 512), (112, 512), (127, 512), (1, 760), (5, 760), (16, 760), (31, 760), 
               (64, 760), (88, 760), (104, 760), (112, 760), (127, 760), (1, 1601), (5, 1601), (16, 1601), (31, 1601), (64, 1601), (88, 1601), (104, 1601), 
               (112, 1601), (127, 1601), (1, 4096), (5, 4096), (16, 4096), (31, 4096), (64, 4096), (88, 4096), (104, 4096), (112, 4096), (127, 4096), 
               (1, 7210), (5, 7210), (16, 7210), (31, 7210), (64, 7210), (88, 7210), (104, 7210), (112, 7210), (127, 7210), (1, 9600), (5, 9600), (16, 9600), 
               (31, 9600), (64, 9600), (88, 9600), (104, 9600), (112, 9600), (127, 9600)]
large_not_align_shape = [(129, 4), (577, 4), (1080, 4), (4000, 4), (7200, 4), (9500, 4), (12803, 4), (65535, 4), (96001, 4), (129, 63), (577, 63), (1080, 63), 
                         (4000, 63), (7200, 63), (9500, 63), (12803, 63), (65535, 63), (96001, 63), (129, 128), (577, 128), (1080, 128), (4000, 128), 
                         (7200, 128), (9500, 128), (12803, 128), (65535, 128), (96001, 128), (129, 512), (577, 512), (1080, 512), (4000, 512), (7200, 512), 
                         (9500, 512), (12803, 512), (65535, 512), (96001, 512), (129, 760), (577, 760), (1080, 760), (4000, 760), (7200, 760), (9500, 760), 
                         (12803, 760), (65535, 760), (96001, 760), (129, 1601), (577, 1601), (1080, 1601), (4000, 1601), (7200, 1601), (9500, 1601), 
                         (12803, 1601), (65535, 1601), (96001, 1601), (129, 4096), (577, 4096), (1080, 4096), (4000, 4096), (7200, 4096), (9500, 4096), 
                         (12803, 4096), (65535, 4096), (96001, 4096), (129, 7210), (577, 7210), (1080, 7210), (4000, 7210), (7200, 7210), (9500, 7210), 
                         (12803, 7210), (65535, 7210), (96001, 7210), (129, 9600), (577, 9600), (1080, 9600), (4000, 9600), (7200, 9600), (9500, 9600), 
                         (12803, 9600), (65535, 9600), (96001, 9600)]
irregular_shape = not_align_shape + small_shape + large_not_align_shape


class LayerNormPerfTest(PyTorchReporter):
    def __init__(self, platform, branch):
        super().__init__()
        self.function = "laynorm_test"
        self.platform = platform
        self.branch = branch

    @launch_prof(args.path)
    def test_forward(self, shape=None, dtype=None, is_optim=False, func_name="", feature=""):
        self.feature = feature

        input = torch.randn(size=shape, dtype=dtype, device="cuda")
        m = torch.nn.LayerNorm(shape[1], elementwise_affine=False, device="cuda").to(dtype)
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
    def test_backward(self, shape=None, dtype=None, is_optim=False, func_name="", feature=""):
        self.feature = feature

        input = torch.randn(size=shape, dtype=dtype, device="cuda").requires_grad_(True)
        g_y = torch.randn(size=shape, dtype=dtype, device="cuda")
        m = torch.nn.LayerNorm(shape[1], elementwise_affine=False, device="cuda").to(dtype)
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
        # print(info)
        t =  get_time_s(info)
        return t

    def run(self):
        print("[PyTorch Perf Start] layernorm")
        for dtype in [torch.float16, torch.float32, torch.bfloat16]:
            for shape in regular_shape:
                self.test_forward(shape=shape, dtype=dtype, is_optim=True, func_name="layernorm_forward_test", feature="from_model;common")

            for shape in regular_shape:
                self.test_backward(shape=shape, dtype=dtype, is_optim=True, func_name="layernorm_backward_test", feature="from_model;common")

   #         for shape in irregular_shape:
   #             self.test_forward(shape=shape, dtype=dtype, is_optim=True, func_name="layernorm_forward_test", feature="uncommon")

   #         for shape in irregular_shape:
   #             self.test_backward(shape=shape, dtype=dtype, is_optim=True, func_name="layernorm_backward_test", feature="uncommon")
        print("[PyTorch Perf End] layernorm")


if __name__ == '__main__':
    f = LayerNormPerfTest(args.platform, args.branch)
    f.run()
