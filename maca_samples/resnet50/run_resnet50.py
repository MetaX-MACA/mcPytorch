import sys
import os
import time
import argparse
from itertools import repeat
import torch
import torch.nn as nn
import resnet50_module
import configs.config_resnet50 as config_resnet50
import configs.config_resnet50_all as config_resnet50_all
import configs.config_resnet50_check_test_data as config_resnet50_check_test_data
import configs.config_resnet50_create_golden_data as config_resnet50_create_golden_data
import configs.config_resnet50_c500_chip as config_resnet50_c500_chip


def test_resnet50(batch_size, mode, type_hardware):
    ret = False
    if mode == "create_golden_data":
        config = config_resnet50_create_golden_data
    elif mode == "check_test_data":
        config = config_resnet50_check_test_data
    elif mode == "ci_mode":
        config = config_resnet50
    elif mode == "all":
        config = config_resnet50_all
    elif mode == "c500_chip":
        config = config_resnet50_c500_chip
    # User need to set the config
    error_eps = config.error_eps
    create_input_tensor = config.create_input_tensor
    create_label_tensor = config.create_label_tensor
    create_golden_data = config.create_golden_data
    create_golden_type = config.create_golden_type
    check_data = config.check_data
    conv_to_cpu = config.conv_to_cpu
    download_model = config.download_model
    # update batch size
    resnet50_module.update_batch_size(batch_size, mode, type_hardware)
    if type_hardware == False:
        # download model parameter
        if download_model:
            resnet50_module.download_resnet50_model()
        # create input tensor
        if create_input_tensor:
            resnet50_module.create_input_tensor(batch_size)
        # create label tensor
        if create_label_tensor:
            resnet50_module.create_label_tensor(batch_size)
        # load input tensor
        input_resnet = resnet50_module.load_input_tensor()
        # load label tensor
        label_resnet = resnet50_module.load_label_tensor()
    else:
        input_resnet = torch.rand(batch_size, 3, 224, 224, dtype=torch.float32)
        label_resnet = torch.empty(batch_size, dtype=torch.long).random_(1000)


    # create resnet50 golden
    if create_golden_data:
        # check create golden file exist
        resnet50_module.check_create_golden_file()
        if create_golden_type == 'all_cpu':
            # resnet50_cpu_golden
            output_resnet = resnet50_module.resnet50_cpu(input_resnet, label_resnet)
        elif create_golden_type == 'all_gpu':
            # resnet50_gpu_golden
            output_resnet = resnet50_module.resnet50_gpu(input_resnet, label_resnet)

    if check_data:
        if type_hardware == False:
            # check golden file exist
            resnet50_module.check_golden_file()
            # check test file exist
            resnet50_module.check_test_file()
        # resnet50_data_test
        output_resnet = resnet50_module.resnet50_data(
            input_resnet, label_resnet, conv_to_cpu, "check_data", error_eps)

    ret = True
    return ret


if __name__ == "__main__":
    time_begin = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, help="batch size")
    parser.add_argument("--mode", default="ci_mode", help="create_golden_data|check_test_data|ci_mode|all|c500_chip")
    parser.add_argument("--null_hardware", action="store_true", default=False, help="null_hardware for driver")
    parser.add_argument("--loop_num", default=1, help="loop num")
    args = parser.parse_args()
    batch_size = int(args.batch_size)
    mode = str(args.mode)
    if int(args.loop_num) != -1:
        loops = range(int(args.loop_num))
    else:
        loops = repeat(None)
    idx = 0
    for _ in loops:
        print("loop: ", idx)
        ret = test_resnet50(batch_size, mode, args.null_hardware)
        idx += 1
    run_resnet50_time = time.time() - time_begin
    print('Resnet50 passed. Using time:', run_resnet50_time, '(s)')
    exit(0 if ret is True else 1)
