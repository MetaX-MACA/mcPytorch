# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.
Models and datasets download automatically from the latest YOLOv5 release.

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
"""

import argparse
import os
import sys
import time
from datetime import datetime
from itertools import repeat
from numpy import random

import torch
import torch.nn as nn
import yaml
from pathlib import Path
import copy

FILE = Path(__file__).resolve()
ROOT = str(FILE.parents[0])  # YOLOv5 root directory
if ROOT not in sys.path:
    sys.path.append(ROOT)  # add ROOT to PATH

from models.yolo import Model
from utils.downloads import attempt_download
from utils.loss import ComputeLoss
from utils.general import intersect_dicts
from utils.utils import create_input_tensor, create_label_tensor, load_data, update_params, \
                        check_create_golden_file, check_golden_file, check_test_file


def create_model(model_path, weights, nc, cfg, hyp, params, device, null_hardware=False):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # Model
    if params['pretrained']:
        weights_path = os.path.join(model_path, weights)
        params['model_dir'] = weights_path
        weights = attempt_download(weights_path)  # download if not found locally
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors'), is_check=params['check_data'], null_hardware=null_hardware).to(device)  # create
        exclude = []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), is_check=params['check_data'], null_hardware=null_hardware).to(device)  # create
    
    hyp['label_smoothing'] = 0.01
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    return model

def train(hyp, opt, device):
    batch_size, weights, cfg, mode = opt.batch_size, opt.weights, opt.cfg, opt.mode

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    config_path = os.path.join(ROOT, "utils/config.yaml")
    with open(config_path, errors='ignore') as f:
        config = yaml.safe_load(f)
    if mode == "all":
        params = config['all']
    elif mode == "ci":
        params = config['ci']
    elif mode == "create_golden_data":
        params = config['create_golden_data']
    elif mode == "check_test_data":
        params = config['check_test_data']
    elif mode == "ci":
        params = config['ci']
    elif mode == "null_hardware":
        params = config['null_hardware']
    else:
        raise ValueError("Please set the mode config")
    default_input_dir = os.path.join(config['default_input_dir'], "test_batch{}".format(batch_size))
    default_model_dir = config['default_model_dir']
    default_result_dir = os.path.join(config['default_result_dir'], "test_batch{}".format(batch_size))
    input_path = os.path.join(ROOT, default_input_dir) if params['input_dir'] == '' else params['input_dir']
    if 'test_batch' not in input_path:
        input_path = os.path.join(input_path, "test_batch{}".format(batch_size))
    model_path = os.path.join(ROOT, default_model_dir) if params['model_dir'] == '' else params['model_dir']
    result_path = os.path.join(ROOT, default_result_dir) if params['result_dir'] == '' else params['result_dir']
    if 'test_batch' not in result_path:
        result_path = os.path.join(result_path, "test_batch{}".format(batch_size))
    params['batch_size'] = batch_size
    params['input_path'], params['model_path'], params['result_path'] = input_path, model_path, result_path
    params['model_dir'] = model_path
    params['width'] = int(config['width'])
    params['height'] = int(config['height'])
    params['num_classes'] = int(config['num_classes'])
    update_params(params)

    if mode == "null_hardware":
        input_tensor = torch.rand(batch_size, 3, params['height'], params['width'], dtype=torch.float32)
        labels = []
        for i in range(batch_size):
            label = random.randint(0, params['num_classes'])
            x, y, w, h = random.randint(100, 200), random.randint(100, 200), random.randint(20, 100), random.randint(50, 100)
            labels.append([i, label, x, y, w, h])
        label_tensor = torch.tensor(labels)
    else:
        if params['create_input_tensor']:
            create_input_tensor(input_path, batch_size)
        input_tensor = load_data(input_path, "input", device)

        if params['create_label_tensor']:
            create_label_tensor(input_path, batch_size)
        label_tensor = load_data(input_path, "label", device)

    if params['create_golden_data']:
        params['check_data'] = False
        update_params(params)
        check_create_golden_file()
        # Model
        model = create_model(model_path, weights, 80, cfg, hyp, params, device)
        model = model.double()
        golden_input_tensor = copy.deepcopy(input_tensor).double()
        golden_label_tensor = copy.deepcopy(label_tensor).double()
        output = model(golden_input_tensor)
        if params['is_backward']:
            compute_loss = ComputeLoss(model)
            loss, loss_items = compute_loss(output, golden_label_tensor)
            loss.backward()

    if params['check_data'] or mode == "all":
        params['check_data'] = True
        update_params(params)
        check_golden_file()
        check_test_file()
        # Model
        model = create_model(model_path, weights, 80, cfg, hyp, params, 'cuda')
        # convert input to cuda
        input_tensor = input_tensor.to('cuda')
        label_tensor = label_tensor.to('cuda')
        output = model(input_tensor)
        if params['is_backward']:
            compute_loss = ComputeLoss(model)
            loss, loss_items = compute_loss(output, label_tensor)
            loss.backward()

    if mode == "null_hardware":
        # Model
        model = create_model(model_path, weights, 80, cfg, hyp, params, 'cuda', null_hardware=True)
        # convert input to cuda
        input_tensor = input_tensor.to('cuda')
        label_tensor = label_tensor.to('cuda')
        output = model(input_tensor)
        if params['is_backward']:
            compute_loss = ComputeLoss(model)
            loss, loss_items = compute_loss(output, label_tensor)
            loss.backward()

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5l.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default=ROOT+'/models/yolov5l.yaml', help='model.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT+'/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--batch_size', type=int, default=2, help='total batch size for all GPUs')
    parser.add_argument('--device', default='cpu', help='cuda or cpu')
    parser.add_argument('--mode', type=str, default='all', help='mode in: create_golden_data, check_test_data, all, ci, null_hardware')
    parser.add_argument("--loop_num", default=1, help="loop num")

    return parser.parse_known_args()[0] if known else parser.parse_args()

if __name__ == "__main__":
    start_time = datetime.now()
    opt = parse_opt()
    if int(opt.loop_num) != -1:
        loops = range(int(opt.loop_num))
    else:
        loops = repeat(None)
    idx = 0
    for _ in loops:
        print("loop:", idx)
        try:
            train(opt.hyp, opt, opt.device)
        except Exception as e:
            print(e)
            exit(1)

    end_time = datetime.now()
    print("----------------------{} cost time:{}s----------------------".format(opt.mode, end_time - start_time))
    if opt.mode == "check_test_data" or opt.mode == "all":
        print("YOLOv5 test checkout finish!")
    exit(0)
