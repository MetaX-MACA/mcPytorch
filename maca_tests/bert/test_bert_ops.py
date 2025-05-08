#!/usr/bin/env python
import torch
import torch.nn as nn
import sys
import os
import time
import argparse
import numpy as np
import copy
from functools import reduce
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import RunCpuAndGpuTest, RunSimpleGpuTest, RunSimpleMethodTest, check_close, MMTest, GOLDEN_DIR, perfModeEnvGuard

dtype = torch.float32

def test_add(case_id, batch_size, only_run, only_fwd):
    test_cases = {
        0: {"input_shape_1": (batch_size, 12, 512, 512), "input_shape_2":(batch_size, 1, 1, 512)},
        1: {"input_shape_1": (batch_size, 12, 512, 512), "input_shape_2":(batch_size, 1, 1, 512), "MACA_TORCH_PERF_MODE":True},
        2: {"input_shape_1": (batch_size, 512, 768), "input_shape_2":(batch_size, 512, 768)}
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        run_perf_flag = case.get("MACA_TORCH_PERF_MODE", False)
        if run_perf_flag:
            temp_env = "elementwise_kernel"
        else:
            temp_env = ""
        with perfModeEnvGuard(temp_env):
            time_s = time.time()
            m = torch.add
            input_1 = torch.rand(case["input_shape_1"], dtype=dtype)
            input_2 = torch.rand(case["input_shape_2"], dtype=dtype)
            if only_run:
                RunSimpleGpuTest(m, input_1, input_2, backward=(not only_fwd))
            elif not RunCpuAndGpuTest(m, input_1, input_2, backward=(not only_fwd), loop=True):
                print("!!!!!! Error raise with add case:", case)
                ret = False
            duration = time.time() - time_s
            print(case, "time: ", duration)
    return ret

def test_arange(case_id, only_run):
    test_cases = {
        0: {"input_length":512}
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        output_golden = torch.arange(case["input_length"], device="cpu")
        output = torch.arange(case["input_length"], device="cuda").cpu()
        if not only_run:
            if not torch.allclose(output, output_golden):
                print("!!!!!! Error raise with arange case:", case)
                ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret

def test_contiguous(case_id, batch_size, only_run):
    test_cases = {
        0:{"input_shape":(batch_size, 512, 12, 64)}
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        input = torch.rand(case["input_shape"])
        if only_run:
            RunSimpleMethodTest("contiguous", input)
        elif not RunCpuAndGpuTest("contiguous", input, loop=True):
            print("!!!!!! Error raise with contiguous case:", case)
            ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret

def test_dividescalar(case_id, batch_size, only_run):
    test_cases = {
        0:{"input_shape":(batch_size, 12, 512, 512)}
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        input = torch.rand(case["input_shape"], dtype=dtype)
        scalar = 4.0
        output_golden = input / scalar
        output = input.cuda() / scalar
        if only_run:
            pass
        elif not torch.allclose(output.cpu(), output_golden):
            print("!!!!!! Error raise with dividescalar case:", case)
            ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret

def test_dropout(case_id, batch_size, only_run, only_fwd):
    test_cases = {
        0:{"input_shape":(batch_size, 512, 768)},
        1:{"input_shape":(batch_size, 1, 12, 512, 512)}
    }
    case_id = int(case_id)
    seed = 0
    golden_dir = GOLDEN_DIR + "bert/dropout/"
    has_env = True
    if "PYTORCH_ENABLE_SAME_RAND_A100" not in os.environ:
        has_env = False
    else:
        env_old = os.environ["PYTORCH_ENABLE_SAME_RAND_A100"]
    os.environ["PYTORCH_ENABLE_SAME_RAND_A100"] = "1"
    os.environ["PYTORCH_DISABLE_FAST_FUSED_DROPOUT"] = "1"

    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        shape = case["input_shape"]
        input = torch.rand(case["input_shape"], dtype=dtype)
        input = torch.randn(shape, dtype=torch.float32, device="cuda").requires_grad_(True)
        backward_input = torch.randn(shape, dtype=torch.float32, device="cuda")
        torch.manual_seed(seed)
        dp = torch.nn.Dropout(p=0.1).cuda()
        shape_name = reduce(lambda x, y: x + "_" + y, [str(x) for x in shape])
        input = torch.from_numpy(
            np.load(golden_dir + f"input_{shape_name}.npy")).float().cuda().requires_grad_(True)
        output_golden = torch.from_numpy(np.load(golden_dir + f"output_golden_{shape_name}.npy")).float()
        backward_input = torch.from_numpy(np.load(golden_dir + f"backward_input_{shape_name}.npy")).float().cuda()
        grad_golden = torch.from_numpy(np.load(golden_dir + f"grad_golden_{shape_name}.npy")).float()
        output = dp(input)
        if not only_fwd:
            output.backward(backward_input)
        if not only_run:
            fw_status = True
            bw_status = True
            fw_status = torch.allclose(output.cpu(), output_golden)
            if not only_fwd:
                bw_status = torch.allclose(input.grad.cpu(), grad_golden)
            print("$$$", fw_status, bw_status)
            if not fw_status or not bw_status:
                print("!!!!!! Error raise with dropout case:", case)
                ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    del os.environ["PYTORCH_DISABLE_FAST_FUSED_DROPOUT"]
    if has_env is False:
        del os.environ["PYTORCH_ENABLE_SAME_RAND_A100"]
    else:
        os.environ["PYTORCH_ENABLE_SAME_RAND_A100"] = env_old
    return ret

def run_test(m, input, do_acc_check, backward):
    input = input
    input_cuda = input.detach().clone().cuda()
    m_cuda = copy.deepcopy(m).cuda()
    output_golden = m(input)
    output = m_cuda(input_cuda)
    if backward:
        backward_input = torch.randn(output_golden.shape, dtype=output_golden.dtype)
        backward_input_cuda = backward_input.detach().clone().cuda()

        output_golden.backward(backward_input)
        output.backward(backward_input_cuda)
    fw_status = True
    bw_status = True
    if do_acc_check:
        fw_status = torch.allclose(output.cpu(), output_golden)
        if backward:
            g_g = m_cuda.weight.grad.cpu()
            g_c = m.weight.grad
            bw_status = torch.allclose(g_g, g_c, atol=1e-4)
    return (fw_status and bw_status)

def test_embedding(case_id, batch_size, only_run, only_fwd):
    test_cases = {
        0:{"input_shape":(batch_size, 512)}
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        shape = case["input_shape"]

        input_token = torch.zeros(shape, dtype=torch.long, device="cpu")
        input_word = torch.tensor(list(x for x in range(512)) * batch_size, dtype=torch.long,
                                  device="cpu").reshape(shape)
        input_position = torch.tensor(list(x for x in range(512)) * batch_size, dtype=torch.long,
                                      device="cpu").reshape(shape)

        m_token = nn.Embedding(2, 768)
        m_word = nn.Embedding(30522, 768, 0)
        m_position = nn.Embedding(512, 768)
        
        if not run_test(m_token, input_token, do_acc_check=(not only_run), backward=(not only_fwd)):
            print("!!!!!! Error raise with embedding case:", case)
            ret = False
        if not run_test(m_word, input_word, do_acc_check=(not only_run), backward=(not only_fwd)):
            print("!!!!!! Error raise with embedding case:", case)
            ret = False
        if not run_test(m_position, input_position, do_acc_check=(not only_run), backward=(not only_fwd)):
            print("!!!!!! Error raise with embedding case:", case)
            ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret

def test_expand(case_id, only_run):
    test_cases = {
        0:{"input_shape":(512)}
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        shape = case["input_shape"]
        input =  torch.rand(shape)
        if only_run:
            RunSimpleMethodTest("expand", input, 1, -1)
        elif not RunCpuAndGpuTest("expand", input, 1, -1, loop=True):
            print("!!!!!! Error raise with expand case:", case)
            ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret

def test_gelu(case_id, batch_size, only_run, only_fwd):
    test_cases = {
        0:{"input_shape":(batch_size, 512, 768)}
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        input_shape = case["input_shape"]
        input = torch.rand(input_shape)
        m = nn.GELU()
        if only_run:
            RunSimpleGpuTest(m, input, backward=(not only_fwd))
        elif not RunCpuAndGpuTest(m, input, backward=(not only_fwd), loop=True):
            print("!!!!!! Error raise with gelu case:", case)
            ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret

def test_layernorm(case_id, batch_size, only_run, only_fwd):
    test_cases = {
        0:{"input_shape":(batch_size, 512, 768)},
        1:{"input_shape":(batch_size, 512, 768), "MACA_TORCH_PERF_MODE":True},
        2:{"input_shape":(batch_size, 512, 1024)},
        3:{"input_shape":(batch_size, 512, 1024), "MACA_TORCH_PERF_MODE":True},
        4:{"input_shape":(batch_size, 512*16, 768)},
        5:{"input_shape":(batch_size, 512*16, 768), "MACA_TORCH_PERF_MODE":True},
        6:{"input_shape":(batch_size, 512*32, 768)},
        7:{"input_shape":(batch_size, 512*32, 768), "MACA_TORCH_PERF_MODE":True}
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        run_perf_flag = case.get("MACA_TORCH_PERF_MODE", False)
        if run_perf_flag:
            temp_env = "layer_norm"
        else:
            temp_env = ""
        with perfModeEnvGuard(temp_env):
            time_s = time.time()
            input_shape = case["input_shape"]
            input = torch.rand(input_shape)
            m = nn.LayerNorm(normalized_shape=input_shape[-1], elementwise_affine=True)
            if only_run:
                RunSimpleGpuTest(m, input, backward=(not only_fwd))
            elif not RunCpuAndGpuTest(m, input, backward=(not only_fwd), loop=True):
                print("!!!!!! Error raise with layernorm case:", case)
                ret = False
            duration = time.time() - time_s
            print(case, "time: ", duration)
    return ret

def test_layernorm_bww(case_id, batch_size, only_run, only_fwd):
    test_cases = {
        0:{"input_shape":(batch_size, 64, 32)},
        1:{"input_shape":(batch_size, 512, 768)},
        2:{"input_shape":(batch_size, 513, 769)},
        3:{"input_shape":(batch_size, 768, 512)},
        4:{"input_shape":(batch_size, 1024, 512)},
        5:{"input_shape":(batch_size, 16*512, 1024)},
        6:{"input_shape":(batch_size, 32*512, 1024)},
        7:{"input_shape":(batch_size, 64, 32), "MACA_TORCH_PERF_MODE":True},
        8:{"input_shape":(batch_size, 512, 768), "MACA_TORCH_PERF_MODE":True},
        9:{"input_shape":(batch_size, 513, 769), "MACA_TORCH_PERF_MODE":True},
        10:{"input_shape":(batch_size, 768, 512), "MACA_TORCH_PERF_MODE":True},
        11:{"input_shape":(batch_size, 1024, 512), "MACA_TORCH_PERF_MODE":True},
        12:{"input_shape":(batch_size, 16*512, 1024), "MACA_TORCH_PERF_MODE":True},
        13:{"input_shape":(batch_size, 32*512, 1024), "MACA_TORCH_PERF_MODE":True}
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        run_perf_flag = case.get("MACA_TORCH_PERF_MODE", False)
        if run_perf_flag:
            temp_env = "layer_norm"
        else:
            temp_env = ""
        with perfModeEnvGuard(temp_env):
            time_s = time.time()
            input_shape = case["input_shape"]
            input_g = torch.randn(input_shape).requires_grad_(True)
            input = input_g.detach().clone().cuda().requires_grad_(True)
            b_input_g = torch.randn(input_shape)
            b_input = b_input_g.detach().clone().cuda()

            m_g = nn.LayerNorm(normalized_shape=input_shape[-1], eps=1e-12, elementwise_affine=True)
            m = copy.deepcopy(m_g).cuda()

            out_g = m_g(input_g)
            out = m(input)
            x_status = True
            dx_status = True
            w_status = True
            b_status = True
            if not only_run:
                x_status = check_close(out_g, out.cpu())
            if not only_fwd:
                out_g.backward(b_input_g)
                out.backward(b_input)
                if not only_run:
                    dx_status = check_close(input_g.grad, input.grad.cpu())

                    wg = m.weight.grad.cpu()
                    w_status = check_close(m_g.weight.grad, wg)

                    bg = m.bias.grad.cpu()
                    b_status = check_close(m_g.bias.grad, bg)

                    print(f"### x, dx, w, b: {x_status}, {dx_status}, {w_status}, {b_status}")
                    if not (x_status and dx_status and w_status and b_status):
                        print("!!!!!! Error raise with layernorm_bww case:", case)
                        ret = False
            duration = time.time() - time_s
            print(case, "time: ", duration)
    return ret

def test_linear_simple(case_id, batch_size):
    test_cases = {
        0:{"input_shape":(batch_size, 768, 768)}
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        input_shape = case["input_shape"]
        if not MMTest(*input_shape, op=torch.nn.Linear, batch1=batch_size):
            print("!!!!!! Error raise with linear_simple case:", case)
            ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret

def test_linear(case_id, batch_size, only_run, only_fwd):
    test_cases = {
        0:{"input_shape":(batch_size, 512, 768), "linear_shape":(768, 768)},
        1:{"input_shape":(batch_size, 512, 768), "linear_shape":(768, 3072)},
        2:{"input_shape":(batch_size, 512, 3072), "linear_shape":(3072, 768)},
        3:{"input_shape":(batch_size, 768), "linear_shape":(768, 768)}
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        input_shape = case["input_shape"]
        linear_shape = case["linear_shape"]
        input = torch.rand(input_shape)
        m = nn.Linear(*linear_shape)
        if only_run:
            RunSimpleGpuTest(m, input, bakcward=(not only_fwd))
        elif not RunCpuAndGpuTest(m, input, backward=(not only_fwd), loop=True):
            print("!!!!!! Error raise with linear case:", case)
            ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret

def test_matmul_bw(case_id, batch_size, only_run, only_fwd):
    torch.set_printoptions(precision=5)
    test_cases = {
        0:{"input_shape_1":(batch_size, 12, 512, 512), "input_shape_2":(batch_size, 1, 512, 64), "bw_input_shape_1":(1, 12, 512, 64), "bw_input_shape_2":(1, 12, 512, 64)},
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        input11_g = torch.randn(case["input_shape_1"]).requires_grad_(True)
        input12_g = torch.randn(case["input_shape_2"]).requires_grad_(True)

        input11 = input11_g.detach().clone().cuda().requires_grad_(True)
        input12 = input12_g.detach().clone().cuda().requires_grad_(True)

        backward_input_g = torch.ones(case["bw_input_shape_1"])
        backward_input = torch.ones(case["bw_input_shape_2"]).cuda()

        m = torch.matmul

        output_g = m(input11_g, input12_g)
        output = m(input11, input12)
        if not only_run:
            if not check_close(output_g, output.cpu()):
                print("!!!!!! Error raise with matmul_bw case:", case)
                ret = False
        if not only_fwd:
            output_g.backward(backward_input_g)
            output.backward(backward_input)

            b_g = input11_g.grad
            b = input11.grad

            print(f"backward golden:\n {b_g}")
            print(f"backward:\n {b.cpu()}")
            if not only_run:
                status = check_close(b.cpu(), b_g)
                print(f"### backward {status}")
                if not status:
                    print("!!!!!! Error raise with matmul_bw case:", case)
                    ret = False
        print("output:", output)
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret

def test_matmul_simple(case_id, batch_size):
    torch.set_printoptions(precision=5)
    test_cases = {
        0:{"input_shape":(512, 512, 64)},
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        if not MMTest(*case["input_shape"], op=torch.matmul, batch1=2*batch_size, batch2=batch_size):
            print("!!!!!! Error raise with matmul_simple case:", case)
            ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret

def test_matmul(case_id, batch_size, only_run, only_fwd):
    test_cases = {
        0:{"input_shape_1":(batch_size, 12, 512, 64), "input_shape_2":(batch_size, 1, 64, 512)},
        1:{"input_shape_1":(batch_size, 12, 512, 512), "input_shape_2":(batch_size, 1, 512, 64)},
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        input_1 = torch.rand(case["input_shape_1"])
        input_2 = torch.rand(case["input_shape_2"])
        m = torch.matmul
        if only_run:
            RunSimpleGpuTest(m, input_1, input_2, backward=(not only_fwd))
        elif not RunCpuAndGpuTest(m, input_1, input_2, backward=(not only_fwd), loop=True):
            print("!!!!!! Error raise with matmul case:", case)
            ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret

def test_permute(case_id, batch_size, only_run):
    test_cases = {
        0:{"input_shape":(batch_size, 12, 512, 64)},
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        input_shape = case["input_shape"]
        input = torch.rand(input_shape)
        if only_run:
            RunSimpleMethodTest("permute", input, 0, 2, 1, 3)
        elif not RunCpuAndGpuTest("permute", input, 0, 2, 1, 3, loop=True):
            print("!!!!!! Error raise with linear case:", case)
            ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret

def test_softmax(case_id, batch_size, only_run, only_fwd):
    test_cases = {
        0:{"input_shape":(batch_size, 12, 512, 512)},
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        input_shape = case["input_shape"]
        input = torch.rand(input_shape)
        m = nn.Softmax(dim=-1)
        if only_run:
            RunSimpleGpuTest(m, input, backward=(not only_fwd))
        elif not RunCpuAndGpuTest(m, input, backward=(not only_fwd), loop=True):
            print("!!!!!! Error raise with softmax case:", case)
            ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret

def test_tanh(case_id, batch_size, only_run, only_fwd):
    test_cases = {
        0:{"input_shape":(batch_size, 512, 768)},
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        input_shape = case["input_shape"]
        input = torch.rand(input_shape)
        m = nn.Tanh()
        if only_run:
            RunSimpleGpuTest(m, input, backward=(not only_fwd))
        elif not RunCpuAndGpuTest(m, input, backward=(not only_fwd), loop=True):
            print("!!!!!! Error raise with softmax case:", case)
            ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret

def test_transpose_matmul_combine(case_id, batch_size, only_run, only_fwd):
    test_cases = {
        0:{"input_shape":(batch_size, 12, 512, 64)},
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        query_layer_32 = torch.randn(case["input_shape"])
        mixed_key_layer = torch.randn(case["input_shape"])
        key_layer_32 = mixed_key_layer.transpose(-1, -2).contiguous()
        m = torch.matmul
        if only_run:
            RunSimpleGpuTest(m, query_layer_32, key_layer_32, backward=(not only_fwd))
        elif not RunCpuAndGpuTest(m, query_layer_32, key_layer_32, backward=(not only_fwd), loop=True):
            print("!!!!!! Error raise with transpose_matmul_combine case:", case)
            ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret

def test_transpose(case_id, batch_size, only_run):
    test_cases = {
        0:{"input_shape":(batch_size, 12, 512, 64)},
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        input = torch.rand(case["input_shape"])
        if only_run:
            RunSimpleMethodTest("transpose", input, -1, -2)
        elif not RunCpuAndGpuTest("transpose", input, -1, -2, loop=True):
            print("!!!!!! Error raise with transpose case:", case)
            ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret

def test_view(case_id, batch_size, only_run):
    test_cases = {
        0:{"input_shape":(batch_size, 512, 12, 64), "view_dim":(512, 768)},
        1:{"input_shape":(batch_size, 512, 768), "view_dim":(512, 12, 64)},
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        input = torch.rand(case["input_shape"])
        if only_run:
            RunSimpleMethodTest("view", input, batch_size, *case["view_dim"])
        elif not RunCpuAndGpuTest("view", input, batch_size, *case["view_dim"], loop=True):
            print("!!!!!! Error raise with view case:", case)
            ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret

def test_zeros(case_id, batch_size, only_run):
    test_cases = {
        0:{"input_shape":(batch_size, 512)},
    }
    case_id = int(case_id)
    ret = True
    for id, case in test_cases.items():
        if case_id != -1 and case_id != id:
            continue
        time_s = time.time()
        output_golden = torch.zeros(torch.Size(case["input_shape"]), dtype=torch.long)
        output = torch.zeros(torch.Size(case["input_shape"]), dtype=torch.long, device="cuda").cpu()
        if not only_run:
            if not torch.allclose(output, output_golden):
                print("!!!!!! Error raise with zeros case:", case)
                ret = False
        duration = time.time() - time_s
        print(case, "time: ", duration)
    return ret

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--op_type", default="all",
                        help="<all|batchnorm|avgpool|maxpool|relu|linear|add|misc|conv2d>")
    parser.add_argument("--id", default=-1, help="op_type's test case id")
    parser.add_argument("--batch_size", default=1, help="batch size")
    parser.add_argument("--only_run", action="store_true", help="only run and not checkout precision")
    parser.add_argument("--only_fwd", action="store_true", help="only run forward")
    args = parser.parse_args()
    op_type = args.op_type
    case_id = args.id
    only_run = args.only_run
    only_fwd = args.only_fwd
    batch_size = int(args.batch_size)

    ret = True
    if op_type == "all":
        ret = ret and test_add(case_id, batch_size, only_run, only_fwd)
        ret = ret and test_arange(case_id, only_run)
        ret = ret and test_contiguous(case_id, batch_size, only_run)
        ret = ret and test_dividescalar(case_id, batch_size, only_run)
        ret = ret and test_dropout(case_id, batch_size, only_run, only_fwd)
        ret = ret and test_embedding(case_id, batch_size, only_run, only_fwd)
        ret = ret and test_expand(case_id, only_run)
        ret = ret and test_gelu(case_id, batch_size, only_run, only_fwd)
        ret = ret and test_layernorm_bww(case_id, batch_size, only_run, only_fwd)
        ret = ret and test_layernorm(case_id, batch_size, only_run, only_fwd)
        ret = ret and test_linear_simple(case_id, batch_size)
        ret = ret and test_linear(case_id, batch_size, only_run, only_fwd)
        ret = ret and test_matmul_bw(case_id, batch_size, only_run, only_fwd)
        ret = ret and test_matmul_simple(case_id, batch_size)
        ret = ret and test_matmul(case_id, batch_size, only_run, only_fwd)
        ret = ret and test_permute(case_id, batch_size, only_run)
        ret = ret and test_softmax(case_id, batch_size, only_run, only_fwd)
        ret = ret and test_tanh(case_id, batch_size, only_run, only_fwd)
        ret = ret and test_transpose_matmul_combine(case_id, batch_size, only_run, only_fwd)
        ret = ret and test_transpose(case_id, batch_size, only_run)
        ret = ret and test_view(case_id, batch_size, only_run)
        ret = ret and test_zeros(case_id, batch_size, only_run)
    elif op_type == "add":
        ret = ret and test_add(case_id, batch_size, only_run, only_fwd)
    elif op_type == "arange":
        ret = ret and test_arange(case_id, only_run)
    elif op_type == "contiguous":
        ret = ret and test_contiguous(case_id, batch_size, only_run)
    elif op_type == "dividescalar":
        ret = ret and test_dividescalar(case_id, batch_size, only_run)
    elif op_type == "dropout":
        ret = ret and test_dropout(case_id, batch_size, only_run, only_fwd)
    elif op_type == "embedding":
        ret = ret and test_embedding(case_id, batch_size, only_run, only_fwd)
    elif op_type == "expand":
        ret = ret and test_expand(case_id, only_run)
    elif op_type == "gelu":
        ret = ret and test_gelu(case_id, batch_size, only_run, only_fwd)
    elif op_type == "layernorm_bww":
        ret = ret and test_layernorm_bww(case_id, batch_size, only_run, only_fwd)
    elif op_type == "layernorm":
        ret = ret and test_layernorm(case_id, batch_size, only_run, only_fwd)
    elif op_type == "linear_simple":
        ret = ret and test_linear_simple(case_id, batch_size)
    elif op_type == "linear":
        ret = ret and test_linear(case_id, batch_size, only_run, only_fwd)
    elif op_type == "matmul_bw":
        ret = ret and test_matmul_bw(case_id, batch_size, only_run, only_fwd)
    elif op_type == "matmul_simple":
        ret = ret and test_matmul_simple(case_id, batch_size)
    elif op_type == "matmul":
        ret = ret and test_matmul(case_id, batch_size, only_run, only_fwd)
    elif op_type == "permute":
        ret = ret and test_permute(case_id, batch_size, only_run)
    elif op_type == "softmax":
        ret = ret and test_softmax(case_id, batch_size, only_run, only_fwd)
    elif op_type == "tanh":
        ret = ret and test_tanh(case_id, batch_size, only_run, only_fwd)
    elif op_type == "transpose_matmul_combine":
        ret = ret and test_transpose_matmul_combine(case_id, batch_size, only_run, only_fwd)
    elif op_type == "transpose":
        ret = ret and test_transpose(case_id, batch_size, only_run)
    elif op_type == "view":
        ret = ret and test_view(case_id, batch_size, only_run)
    elif op_type == "zeros":
        ret = ret and test_zeros(case_id, batch_size, only_run)
    else:
        print("Error: Not support op_type: ", op_type)

    exit(0 if ret is True else 1)