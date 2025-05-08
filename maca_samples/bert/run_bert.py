#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import os
import argparse
import importlib
import numpy as np
import sys
import time
from itertools import repeat
sys.path.insert(0, os.path.join(os.getcwdb().decode("utf-8"), "../"))
sys.path.insert(0, os.getcwdb().decode("utf-8"))

# don't change this environ,if changed may lead to some accuracy problem
os.environ["PYTORCH_ENABLE_SAME_RAND_A100"] = "1"

# parse the config file
parser = argparse.ArgumentParser()
parser.add_argument("--config_file_path", type=str, default="configs/cfg_batch_1_seqlength_512.py")
parser.add_argument("--mode", type=str, required=True, help="support 'save','verify' and 'fast_verify' only")
parser.add_argument("--ignore_detail_check", action="store_true", help="flag to control whether to check each layer or not")
parser.add_argument("--ignore_final_check",  action="store_true", help="flag to control whether to check final layer or not")
parser.add_argument("--null_hardware", action="store_true", default=False, help="null_hardware for driver")
parser.add_argument("--loop_num", default=1, help="loop num")
args = parser.parse_args()
config_file = os.path.splitext(args.config_file_path)[0].replace("../", "").replace("/", ".")
cfg = importlib.import_module(config_file)

def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace("trace_bert.json")

def run_single_iter(model, inputs, fast_verify_flag, cfg, \
        forward_final_layer_name, backward_final_layer_name):
    encoded_layers, pooled_output = model(*inputs, fast_verify_mode=fast_verify_flag)
    if not cfg.run_backward_flag:
        if forward_final_layer_name:
            golden_data_path = os.path.join(cfg.golden_forward_data_path, forward_final_layer_name + "_forward_output.npy")
            golden_output_data = torch.from_numpy(np.load(golden_data_path))
            if pooled_output is not None:
                result = check_close(pooled_output.cpu(), golden_output_data, 2e-3)
            else:
                result = check_close(encoded_layers[-1].cpu(), golden_output_data, 2e-3)
        else:
            info = "ignore final check and set result True."
            save_log(info)
            result = True
    else:
        encode_layers_labels = torch.zeros(encoded_layers[-1].shape, dtype=torch.long, device=cfg.device)
        if pooled_output is not None:
            pooled_output_labels = torch.zeros(pooled_output.shape, dtype=torch.long, device=cfg.device)
            loss = (encoded_layers[-1] - encode_layers_labels).sum() + \
                (pooled_output - pooled_output_labels).sum()
            loss.backward()
        else:
            loss = (encoded_layers[-1] - encode_layers_labels).sum()
            loss.backward()
        if backward_final_layer_name:
            data_path = os.path.join(cfg.golden_backward_data_path, backward_final_layer_name + "_backward_input_0.npy")
            golden_output_data = torch.from_numpy(np.load(data_path))
            from util import word_embedding_grad_result
            result = check_close(word_embedding_grad_result.cpu(), golden_output_data, 2e-3)
        else:
            info = "ignore final check and set result True."
            save_log(info)
            result = True
    if forward_final_layer_name:
        from util import max_diff_result
        info = f"max diff result is {max_diff_result}"
        save_log(info)
    return result

def run_single_iter_null_hardware(model, inputs, fast_verify_flag):
    encoded_layers, pooled_output = model(*inputs, fast_verify_mode=fast_verify_flag)
    encode_layers_labels = torch.zeros(encoded_layers[-1].shape, dtype=torch.long, device=cfg.device)
    if pooled_output is not None:
        pooled_output_labels = torch.zeros(pooled_output.shape, dtype=torch.long, device=cfg.device)
        loss = (encoded_layers[-1] - encode_layers_labels).sum() + \
            (pooled_output - pooled_output_labels).sum()
        loss.backward()
    else:
        loss = (encoded_layers[-1] - encode_layers_labels).sum()
        loss.backward()
    return True


def run_verify(fast_verify_flag, null_hardware):
    '''
    verify lunch entry
    '''
    if fast_verify_flag:
        assert cfg.run_backward_flag is False, \
            "fast verify would lead to skipping running some forward layers, so backward inference cannot be excuted."
    model = get_model()
    if cfg.train_flag:
        model.train()
    else:
        model.eval()
    inputs = load_input()
    if not null_hardware:
        forward_final_layer_name = add_forward_verify_hook_interface(model, args.ignore_detail_check, args.ignore_final_check)
        backward_final_layer_name = None
        if cfg.run_backward_flag:
            backward_final_layer_name = add_backward_verify_hook_interface(model, args.ignore_detail_check, args.ignore_final_check)

    print("matmul.allow_tf32: ", torch.backends.cuda.matmul.allow_tf32)
    print("cudnn.allow_tf32: ", torch.backends.cudnn.allow_tf32)

    enable_dump_cuda_profile_data = True if "MACA_SAMPLE_BERT_ENABLE_PROFILE_DUMP" in os.environ else False

    if enable_dump_cuda_profile_data is True:
        warmup_cnt = 10
        for i in range(warmup_cnt):
            ts = time.time()
            if not null_hardware:
                result = run_single_iter(model, inputs, fast_verify_flag, cfg,
                                        forward_final_layer_name, backward_final_layer_name)
            else:
                result = run_single_iter_null_hardware(model, inputs, fast_verify_flag)
            te = time.time()
            print("iter time: ", te - ts)

        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                on_trace_ready=trace_handler) as prof:
            if not null_hardware:
                result = run_single_iter(model, inputs, fast_verify_flag, cfg,
                                        forward_final_layer_name, backward_final_layer_name)
            else:
                result = run_single_iter_null_hardware(model, inputs, fast_verify_flag)
    else:
        if not null_hardware:
            result = run_single_iter(model, inputs, fast_verify_flag, cfg,
                                    forward_final_layer_name, backward_final_layer_name)
        else:
            result = run_single_iter_null_hardware(model, inputs, fast_verify_flag)

    return result


def run_save(null_hardware):
    '''
    save lunch entry
    '''
    model = get_model(is_save=True)
    if cfg.train_flag:
        model.train()
    else:
        model.eval()
    inputs = gen_input()
    if not null_hardware:
        add_forward_save_hook_interface(model)
        if cfg.run_backward_flag:
            add_backward_save_hook_interface(model)
        encoded_layers, pooled_output = model(*inputs)
        if cfg.run_backward_flag:
            encode_layers_labels = torch.zeros(encoded_layers[-1].shape, dtype=torch.long, device=cfg.device)
            if pooled_output is not None:
                pooled_output_labels = torch.zeros(pooled_output.shape, dtype=torch.long, device=cfg.device)
                loss = (encoded_layers[-1] - encode_layers_labels).sum() + \
                    (pooled_output - pooled_output_labels).sum()
                loss.backward()
            else:
                loss = (encoded_layers[-1] - encode_layers_labels).sum()
                loss.backward()
    info = f"bert save run finished, and logs are saved in {cfg.save_log_path}."
    save_log(info)


if __name__ == "__main__":
    # import under __main__ to avoid some import conflicts
    from util import gen_input, save_log, get_model, add_forward_verify_hook_interface, \
        add_forward_save_hook_interface, load_input, check_dirs, write_config_file_to_log_file, \
        add_backward_save_hook_interface, add_backward_verify_hook_interface, check_close
    check_dirs()
    info = f"launch command args is {args}"
    save_log(info)
    write_config_file_to_log_file(args.config_file_path)
    torch.manual_seed(cfg.seed)
    if args.mode == "save":
        run_save(args.null_hardware)
    elif args.mode == "verify" or args.mode == "fast_verify":
        if int(args.loop_num) != -1:
            loops = range(int(args.loop_num))
        else:
            loops = repeat(None)
        idx = 0
        for _ in loops:
            print("loop: ", idx)
            result = run_verify(args.mode == "fast_verify", args.null_hardware)
            idx += 1
            if not result:
                exit(1)
    else:
        info = f"mode option {args.mode} is not supported yet, please set mode option as 'save' ,'verify' or 'fast_verify'."
        save_log(info)
