import os
import torch

def test_elementwise_2_1_copy():
    offsets = [0, 16, 32, 48, 80, 160, 20]
    dtypes = [torch.float, torch.float16]

    for off in offsets:
        shape_stride_list = [[(245760, 40), (80, 1), (40, 1), torch.bfloat16], [(393408, 40), (80, 1), (40, 1), torch.bfloat16],\
                            [(245760, 80), (240, 1), (80, 1), torch.bfloat16], [(245760, 40), (80, 1), (240, 1), torch.bfloat16],\
                            [(19600, 40), (80, 1), (80, 1), torch.float], [(19600, 40), (80, 1), (40, 1), torch.float], \
                            [(57232,40),(80, 1),(40, 1), torch.float]]

        for shape, stride_out, stride_in, dtype in shape_stride_list:
            inp_base = torch.rand(100000000, device="cuda",dtype=dtype)
            out_base = torch.rand(100000000, device="cuda",dtype=dtype)
            out_off = out_base[off:]
            inp_base_c = inp_base.detach().clone().cpu()
            out_base_c = out_off.detach().clone().cpu()

            inp = inp_base.as_strided(shape, stride_in)
            out = out_off.as_strided(shape, stride_out)

            inpc = inp_base_c.as_strided(shape, stride_in)
            outc = out_base_c.as_strided(shape, stride_out)

            out.copy_(inp)
            outc.copy_(inpc)

            flat_shape = (shape[0], stride_out[0])
            out1 = out.as_strided(flat_shape, stride_out)
            outc = outc.as_strided(flat_shape, stride_out)

            res = torch.allclose(out1.cpu(),outc)
            if not res:
                print("error")
                exit(1)

def test_elementwise_4_1_copy():
    offsets = [0, 16, 32, 48, 80, 160, 20]

    for off in offsets:
        inp_base = torch.rand(100000000, device="cuda",dtype=torch.bfloat16)
        out_base = torch.rand(100000000, device="cuda",dtype=torch.bfloat16)
        out_off = out_base[off:]
        inp_base_c = inp_base.detach().clone().cpu()
        out_base_c = out_off.detach().clone().cpu()

        shape_stride_list = [[(2048, 5, 24, 40), (9600, 1920, 80, 1), (1920, 3932160, 80, 1)], \
                            [(2048, 5, 24, 80), (28800, 5760, 240, 1), (1920, 3932160, 80, 1)], \
                            [(2048, 5, 24, 40), (9600, 1920, 80, 1), (960, 1966080, 40, 1)]]

        for shape, stride_out, stride_in in shape_stride_list:
            inp = inp_base.as_strided(shape, stride_in)
            out = out_off.as_strided(shape, stride_out)

            inpc = inp_base_c.as_strided(shape, stride_in)
            outc = out_base_c.as_strided(shape, stride_out)

            out.copy_(inp)
            outc.copy_(inpc)

            flat_shape = (shape[0], shape[1], shape[2], stride_out[2])
            out1 = out.as_strided(flat_shape, stride_out)
            outc = outc.as_strided(flat_shape, stride_out)

            res = torch.allclose(out1.cpu(),outc)
            if not res:
                print("error")
                exit(1)

os.environ["PYTORCH_ENABLE_ELEMENTWISE_N_1_DIM0_PAD"] = "1"
test_elementwise_2_1_copy()
test_elementwise_4_1_copy()
os.unsetenv("PYTORCH_ENABLE_ELEMENTWISE_N_1_DIM0_PAD")
exit(0)