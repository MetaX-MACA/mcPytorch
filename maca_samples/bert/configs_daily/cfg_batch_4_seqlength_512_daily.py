#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import torch
from config_util import constuct_log_file_path
# currently only support "bert-base-uncased"
model_type = "bert-base-uncased"
# running bert mode, possible value includes "save","verify" and "fast_verify"
mode_option_list = ["save", "fast_verify", "verify"]
# measuring thresh settings
forward_verify_input_check_thresh = 1e-3
forward_verify_output_check_thresh = 1e-3
backward_verify_input_check_thresh = 1e-3
backward_verify_output_check_thresh = 1e-3
# torch random seed, make sure keep the same when run save,fast_verify and verify mode
seed = 0
# device assign, it could be cuda or cpu
device = torch.device("cuda:0")
# put the model under train or val mode
train_flag = True

# model config parameters
batch_size = 4
seq_length = 512
vocab_size = 30522
hidden_size = 768
num_hidden_layers = 12
num_attention_heads = 12
intermediate_size = 3072
hidden_act = "gelu"
hidden_dropout_prob = 0.1
attention_probs_dropout_prob = 0.1
max_position_embeddings = 512
type_vocab_size = 2
initializer_range = 0.02

# save log flag
save_log_flag = True
# run backward flag
run_backward_flag = False
# add pooling layer flag
add_pooling_layer_flag = True

# golden root path, make sure this directory exist in your computer
golden_root_path = "/netapp/pytorch/golden/bert_daily/"
GOLDEN_PATH = os.getenv("PYTORCH_TEST_GOLDEN_PATH")
if GOLDEN_PATH and os.path.exists(GOLDEN_PATH):
    golden_root_path = os.path.join(GOLDEN_PATH, "bert_daily")

# golden data path, code will generate the directory if it doesn't exist
golden_forward_data_path = os.path.join(golden_root_path,
                                        f"io-64-cuda-gen-batch-{batch_size}-seqlength-{seq_length}/forward")
golden_backward_data_path = os.path.join(golden_root_path,
                                         f"io-64-cuda-gen-batch-{batch_size}-seqlength-{seq_length}/backward")
# pretrained weight path, if you don't need it ,set it as None
pretrained_weight_path = os.path.join(golden_root_path, "weights/pytorch_model.bin")

# construct path to save log
log_dir = "./logs/"
save_log_path = constuct_log_file_path(log_dir, device, os.path.splitext(os.path.basename(__file__))[0])
