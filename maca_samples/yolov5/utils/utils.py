import os
import warnings
import shutil
import copy
from numpy import random

import torch
import torch.nn as nn
from torch import Tensor
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional


params = None
device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda")
error_eps = 1e-3
index_fwd = 0
index_bwd = 0
golden_or_check_data = 'check_data'
golden_data_fwd_path = 'save_data'
golden_data_bwd_path = 'save_data'
result_data_fwd_path = 'save_result'
result_data_bwd_path = 'save_result'
log_path = 'save_result'
model_path = 'save_model'

def update_params(param):
    global params
    params = copy.deepcopy(param)
    global golden_or_check_data
    golden_or_check_data = 'check_data' if param['check_data'] else 'golden_data'
    global golden_data_fwd_path
    golden_data_fwd_path = os.path.join(param['input_path'], 'forward')
    global golden_data_bwd_path
    golden_data_bwd_path = os.path.join(param['input_path'], 'backward')
    global result_data_fwd_path
    result_data_fwd_path = os.path.join(param['result_path'], 'forward')
    global result_data_bwd_path
    result_data_bwd_path = os.path.join(param['result_path'], 'backward')
    global log_path
    log_path = os.path.join(param['result_path'], 'log_msg.log')
    global model_path
    model_path = param['model_dir']
    global error_eps
    error_eps = float(params['error_eps'])
    global index_fwd
    index_fwd = 0
    global index_bwd
    index_bwd = 0


def checkout_error(golden_data, check_data, file_name, device_name):
    if os.path.exists(log_path) is False:
        os.mknod(log_path)
    with open(log_path, 'a') as logflie:
        if golden_data is None and check_data is None:
            print('Check ' + device_name + ' correct: ' + file_name)
        elif golden_data is None and check_data is not None:
            info = 'Error op ' + device_name + ' of ' + file_name + ' is out of eps!'
            print(info)
            logflie.write(info + '\n')
            quit()
        elif golden_data is not None and check_data is None:
            info = 'Error op ' + device_name + ' of ' + file_name + ' is out of eps!'
            print(info)
            logflie.write(info + '\n')
            quit()
        else:
            print('shape:', check_data.shape, ' of ' + device_name + ' of ' + file_name)
            cal = abs(golden_data.to(device_cpu) - check_data.to(device_cpu)).sum() / golden_data.numel()
            if cal > error_eps:
                info = "Error Name op:{}, :{}, error:{:.7f}".format(device_name, file_name, cal)
                print(info)
                logflie.write(info + '\n')
                quit()
            else:
                info = "Check:{}, :{}, correct, eps:{:.7f}".format(device_name, file_name, cal)
                print(info)
                logflie.write(info + '\n')


def hook_forward_fn(module, input, output):
    file_device = golden_data_fwd_path
    os.makedirs(file_device, exist_ok=True)

    global index_fwd
    index_file = index_fwd
    index_fwd = index_fwd + 1
    file_name_input = str(index_file) + '_' + str(module) + '_input' + '.pth'
    file_name_output = str(index_file) + '_' + str(module) + '_output' + '.pth'
    file_path_input = os.path.join(file_device, file_name_input)
    file_path_output = os.path.join(file_device, file_name_output)
    torch.save(input[0], file_path_input)
    torch.save(output[0], file_path_output)


def hook_forward_fn_conv(module, input, output):
    if golden_or_check_data == 'golden_data':
        file_device = golden_data_fwd_path
        os.makedirs(file_device, exist_ok=True)
    if golden_or_check_data == 'check_data':
        file_device = result_data_fwd_path
        os.makedirs(file_device, exist_ok=True)

    global index_fwd
    index_file = index_fwd
    index_fwd = index_fwd + 1
    file_name = str(index_file) + '_' + str(module)
    file_name_input = str(index_file) + '_' + str(module) + '_input.pth'
    file_name_output = str(index_file) + '_' + str(module) + '_output.pth'
    file_path_input = os.path.join(file_device, file_name_input)
    file_path_output = os.path.join(file_device, file_name_output)
    input_write = input[0].to(device_cpu)
    output_write = output[0].to(device_cpu)
    torch.save(input_write, file_path_input)
    torch.save(output_write, file_path_output)

    if golden_or_check_data == 'check_data':
        if os.path.exists(log_path) is False:
            os.mknod(log_path)
        with open(log_path, 'a') as logflie:
            logflie.write(file_name + '\n')

    conv_input = torch.load(os.path.join(golden_data_fwd_path, file_name_input), map_location="cpu")
    conv_output = torch.load(os.path.join(golden_data_fwd_path, file_name_output), map_location="cpu")
    checkout_error(conv_input, input_write, file_name, 'input fwd')
    checkout_error(conv_output, output_write, file_name, 'output fwd')


def hook_backward_fn(module, input, output):
    file_device = golden_data_bwd_path
    os.makedirs(file_device, exist_ok=True)

    global index_bwd
    index_file = index_bwd
    index_bwd = index_bwd + 1
    file_name_input = str(index_file) + '_' + str(module) + '_input' + '.pth'
    file_name_output = str(index_file) + '_' + str(module) + '_output' + '.pth'
    file_path_input = os.path.join(file_device, file_name_input)
    file_path_output = os.path.join(file_device, file_name_output)
    torch.save(input, file_path_input)
    torch.save(output, file_path_output)


def hook_backward_fn_conv(module, input, output):
    if golden_or_check_data == 'golden_data':
        file_device = golden_data_bwd_path
        os.makedirs(file_device, exist_ok=True)
    if golden_or_check_data == 'check_data':
        file_device = result_data_bwd_path
        os.makedirs(file_device, exist_ok=True)

    global index_bwd
    index_file = index_bwd
    index_bwd = index_bwd + 1
    file_name = str(index_file) + '_' + str(module)
    file_name_input = str(index_file) + '_' + str(module) + '_input.pth'
    file_name_output = str(index_file) + '_' + str(module) + '_output.pth'
    file_path_input = os.path.join(file_device, file_name_input)
    file_path_output = os.path.join(file_device, file_name_output)
    input_write = input
    output_write = output
    torch.save(input_write, file_path_input)
    torch.save(output_write, file_path_output)

    if golden_or_check_data == 'check_data':
        if os.path.exists(log_path) is False:
            os.mknod(log_path)
        with open(log_path, 'a') as logflie:
            logflie.write(file_name + '\n')

    conv_input = torch.load(os.path.join(golden_data_bwd_path, file_name_input), map_location="cpu")
    conv_output = torch.load(os.path.join(golden_data_bwd_path, file_name_output), map_location="cpu")
    for idx in range(len(input)):
        checkout_error(conv_input[idx], input_write[idx], file_name, 'input bwd' + str(idx))
        if str(module)[0:11] == 'BatchNorm2d':
            break
    for idx in range(len(output)):
        checkout_error(conv_output[idx], output_write[idx], file_name, 'output bwd' + str(idx))
        if str(module)[0:11] == 'BatchNorm2d':
            break

def create_input_tensor(input_path, batch_size=1):
    tensor_path = os.path.join(input_path, "input_tensor.pth")
    if os.path.exists(tensor_path):
        warnings.warn("Input tensor will be covered", UserWarning)
        os.remove(tensor_path)
    if os.path.exists(input_path) is False:
        os.makedirs(input_path)
    input_tensor = torch.rand(batch_size, 3, params['height'], params['width'], dtype=torch.float32)
    torch.save(input_tensor, tensor_path)

def create_label_tensor(input_path, batch_size=1):
    tensor_path = os.path.join(input_path, "label_tensor.pth")
    if os.path.exists(tensor_path):
        warnings.warn("Label tensor will be covered", UserWarning)
        os.remove(tensor_path)
    if os.path.exists(input_path) is False:
        os.makedirs(input_path)
    labels = []
    for i in range(batch_size):
        label = random.randint(0, params['num_classes'])
        x, y, w, h = random.randint(100, 200), random.randint(100, 200), random.randint(20, 100), random.randint(50, 100)
        labels.append([i, label, x, y, w, h])
    input_tensor = torch.tensor(labels)
    torch.save(input_tensor, tensor_path)

def load_data(input_path, mode="input", device="cuda"):
    if mode == "input":
        input_path = os.path.join(input_path, "input_tensor.pth")
    else:
        input_path = os.path.join(input_path, "label_tensor.pth")
    assert os.path.exists(input_path), "Missing {} data, please check!".format(mode)

    data = torch.load(input_path, map_location="cpu").to(device)
    return data

def check_golden_file():
    assert os.path.exists(model_path), "Error: Missing yolo model file! Please download model file."

    golden_input_tensor = os.path.join(params['input_path'], 'input_tensor.pth')
    assert os.path.exists(golden_input_tensor), "Missing input tensor file! Please generate input tensor."

    golden_label_tensor = os.path.join(params['input_path'], 'label_tensor.pth')
    assert os.path.exists(golden_label_tensor), "Missing label tensor file! Please generate input tensor."

    assert os.path.exists(golden_data_fwd_path), "Missing golden data fwd file! Please generate golden data fwd file."

    if params['is_backward']:
        assert os.path.exists(golden_data_bwd_path), "Missing golden data bwd file! Please generate golden data fwd file."


def check_create_golden_file():
    assert os.path.exists(model_path), "Error: Missing yolo model file! Please download model file."
    
    golden_input_tensor = os.path.join(params['input_path'], 'input_tensor.pth')
    assert os.path.exists(golden_input_tensor), "Missing input tensor file! Please generate input tensor."

    golden_label_tensor = os.path.join(params['input_path'], 'label_tensor.pth')
    assert os.path.exists(golden_label_tensor), "Missing label tensor file! Please generate input tensor."

    if os.path.exists(golden_data_fwd_path):
        warnings.warn("This gloden data fwd file is exists! Result will be covered", UserWarning)
        shutil.rmtree(golden_data_fwd_path)
    else:
        print("INFO: This golden data fwd file will create!")
        os.makedirs(golden_data_fwd_path)
    if params['is_backward']:
        if os.path.exists(golden_data_bwd_path):
            warnings.warn("This gloden data bwd file is exists! Result will be covered", UserWarning)
            shutil.rmtree(golden_data_bwd_path)
        else:
            print("INFO: This golden data bwd file will create!")
            os.makedirs(golden_data_bwd_path)

def check_test_file():
    if os.path.exists(result_data_fwd_path):
        warnings.warn("This result data fwd file is exists! Result will be covered", UserWarning)
        shutil.rmtree(result_data_fwd_path)
    if params['is_backward']:
        if os.path.exists(result_data_bwd_path):
            warnings.warn("This result data bwd file is exists! Result will be covered", UserWarning)
            shutil.rmtree(result_data_bwd_path)
    if os.path.exists(log_path):
        warnings.warn("This log message is exists! Log message will be covered", UserWarning)
        os.remove(log_path)