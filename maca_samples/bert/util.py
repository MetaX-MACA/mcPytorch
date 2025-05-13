#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import logging
from modeling import BertConfig, BertModel
import os
import numpy as np
import random
import requests
from run_bert import cfg

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin"
}
print("PRETRAINED_MODEL_ARCHIVE_MAP", PRETRAINED_MODEL_ARCHIVE_MAP)


def write_config_file_to_log_file(config_file_path):
    '''
    write the config file's content to the log file
    config_file_path: path of the config file
    '''
    with open(config_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            save_log(line.strip())
    return


def check_dirs():
    '''
    make sure some neccesary dirs exist
    '''
    assert os.path.exists(cfg.golden_root_path)
    if cfg.pretrained_weight_path:
        assert os.path.exists(os.path.dirname(cfg.pretrained_weight_path))
    if not os.path.exists(cfg.golden_forward_data_path):
        os.makedirs(cfg.golden_forward_data_path)
    if not os.path.exists(cfg.golden_backward_data_path):
        os.makedirs(cfg.golden_backward_data_path)
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir, exist_ok=True)
    return


def load_input():
    '''
    load input from npy data file
    '''
    input_ids_path = os.path.join(cfg.golden_forward_data_path, 'input_ids.npy')
    input_ids = torch.from_numpy(np.load(input_ids_path)).to(cfg.device)
    token_type_ids_path = os.path.join(cfg.golden_forward_data_path, 'token_type_ids.npy')
    token_type_ids = torch.from_numpy(np.load(token_type_ids_path)).to(cfg.device)
    position_ids_path = os.path.join(cfg.golden_forward_data_path, 'position_ids.npy')
    position_ids = torch.from_numpy(np.load(position_ids_path)).to(cfg.device)
    info = "load input data finished"
    save_log(info)
    return input_ids, token_type_ids, position_ids


def rename_load_dict(state_dict):
    '''
    rename the keys of the state_dict to match the name of our bert model
    state_dict: original state dict
    '''
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = key.replace("bert.", "")
        if 'gamma' in key:
            new_key = new_key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = new_key.replace('beta', 'bias')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        if old_key.startswith("cls."):
            state_dict.pop(old_key)
            continue
        state_dict[new_key] = state_dict.pop(old_key)
    return state_dict


def get_model(is_save=False):
    '''
    construct BertModel with BertConfig
    is_save: flag to control the model dtype float32 or float64, we need save float64 golden data when is_save is true.
    '''
    config = BertConfig(
        vocab_size_or_config_json_file=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        hidden_act=cfg.hidden_act,
        hidden_dropout_prob=cfg.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.attention_probs_dropout_prob,
        max_position_embeddings=cfg.max_position_embeddings,
        type_vocab_size=cfg.type_vocab_size,
        initializer_range=cfg.initializer_range)

    model = BertModel(config=config, add_pooling_layer_flag=cfg.add_pooling_layer_flag)
    weight_path = cfg.pretrained_weight_path
    if weight_path:
        if not os.path.exists(weight_path):
            info = f"pretrained model not found in {weight_path}, start to download pretrained model"
            save_log(info)
            result = requests.get(PRETRAINED_MODEL_ARCHIVE_MAP[cfg.model_type])
            if result.status_code == 200:
                with open(weight_path, 'wb') as f:
                    f.write(result.content)
            else:
                info = f"error occurs when downloading pretrained model from \
{PRETRAINED_MODEL_ARCHIVE_MAP[cfg.model_type]}, please manually download it and save as {weight_path}"
                save_log(info)
                exit(-1)
        loaded_state_dict = torch.load(weight_path, map_location="cpu")
        renamed_loaded_state_dict = rename_load_dict(loaded_state_dict)
        model.load_state_dict(renamed_loaded_state_dict, strict=True)
    else:
        info = "weight_path is None, use 'init_bert_weights' function to initialize weights \
of the bert model. For more details, please refer to 'init_bert_weights' function in modeling.py"
        save_log(info)
    if is_save:
        model.double()
    model.to(cfg.device)
    info = "load model finished"
    save_log(info)
    return model


def save_log(info):
    '''
    save important infomation in the save_log_path
    '''
    if cfg.save_log_flag:
        logging.basicConfig(filename=cfg.save_log_path,
                            format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
        logging.info(info)

    print(info, flush=True)


max_diff_result = 0.0


def check_close(infer_result_data, golden_data, eps):
    '''
    function to measure the difference between two tensors, replace the old torch.allclose compare function
    '''
    global max_diff_result
    diff = infer_result_data - golden_data
    diff_square = diff * diff
    infer_result_square_double = 2 * infer_result_data * infer_result_data
    sum_diff_square = torch.sum(diff_square)
    sum_infer_result_square_double = torch.sum(infer_result_square_double)
    result = torch.sqrt(sum_diff_square / (sum_infer_result_square_double + 1e-12))
    info = f"diff result:{result}"
    save_log(info)
    if result > max_diff_result:
        max_diff_result = result
    return result < eps


def compare_func(data, golden_data_path, eps):
    '''
    compare two tensors and measure the difference between them.
    data: data gotten from network
    golden_data_path:golden data path saved in save mode
    eps:threshold to judge two tensors' similarity
    '''
    data_cpu = data.to("cpu:0")
    golden = torch.from_numpy(np.load(golden_data_path))
    status = check_close(data_cpu, golden, eps)
    diff = torch.abs(data_cpu - golden)
    max_diff = torch.max(diff).item()
    if status:
        info = f"{golden_data_path} check success, max diff {max_diff}"
        save_log(info)
    else:
        info = f"{golden_data_path} check fail, max diff {max_diff}"
        save_log(info)
    return


def compare_dropout(data, golden_data_path):
    '''
    compare dropout mask 0/1 result.
    data: data gotten from network
    golden_data_path:golden data path saved in save mode
    '''
    result_data = data.cpu().detach().numpy() == 0
    golden_data = np.load(golden_data_path)
    result_golden_data = golden_data == 0
    np.set_printoptions(threshold=np.inf)

    def error_output():
        result = np.not_equal(result_data, result_golden_data).astype(int)
        save_log("result_data.shape:{}".format(result_data.shape))
        save_log("none zero indexes:{}".format(np.nonzero(result)))
        save_log("data_value:{}".format(data[np.nonzero(result)]))
        save_log("golden_data_value:{}".format(golden_data[np.nonzero(result)]))
    if not (result_data == result_golden_data).all():
        error_output()
    return


def forward_verify_input_output_hook(op_name, input_eps=cfg.forward_verify_input_check_thresh,
                                     output_eps=cfg.forward_verify_output_check_thresh):
    '''
    add hook of forward input and output verify check
    op_name:name of operation
    input_eps:threshold for measuring input data
    output_eps:threshold for measuring output data
    '''
    def hook(model, input, output):
        if isinstance(input, tuple):
            for i in range(len(input)):
                if input[i] is None:
                    continue
                input_golden_path = os.path.join(
                    cfg.golden_forward_data_path,
                    op_name + "_forward_input_" + f"{i}" + ".npy")
                assert os.path.exists(input_golden_path), f"{input_golden_path} is not exist"
                compare_func(input[i], input_golden_path, input_eps)
                if "dropout" in op_name:
                    compare_dropout(input[i], input_golden_path)
        else:
            input_golden_path = os.path.join(cfg.golden_forward_data_path, op_name + "_forward_input.npy")
            assert os.path.exists(input_golden_path), f"{input_golden_path} is not exist"
            compare_func(input, input_golden_path, input_eps)
            if "dropout" in op_name:
                compare_dropout(input, input_golden_path)
        if isinstance(output, tuple):
            for i in range(len(output)):
                if output[i] is None:
                    continue
                output_golden_path = os.path.join(
                    cfg.golden_forward_data_path,
                    op_name + "_forward_output_" + f"{i}" + ".npy")
                assert os.path.exists(output_golden_path), f"{output_golden_path} is not exist"
                compare_func(output[i], output_golden_path, output_eps)
                if "dropout" in op_name:
                    compare_dropout(output[i], output_golden_path)
        else:
            output_golden_path = os.path.join(cfg.golden_forward_data_path, op_name + "_forward_output.npy")
            assert os.path.exists(output_golden_path), f"{output_golden_path} is not exist"
            compare_func(output, output_golden_path, output_eps)
            if "dropout" in op_name:
                compare_dropout(output, output_golden_path)

    return hook


word_embedding_grad_result = None


def backward_verify_input_output_hook(op_name, input_eps=cfg.backward_verify_input_check_thresh,
                                      output_eps=cfg.backward_verify_output_check_thresh):
    '''
    add hook of backward input and output verify check
    op_name:name of operation
    input_eps:threshold for measuring input data
    output_eps:threshold for measuring output data
    '''
    def hook(model, input, output):
        if isinstance(input, tuple):
            for i in range(len(input)):
                if input[i] is None:
                    continue
                input_golden_path = os.path.join(
                    cfg.golden_backward_data_path,
                    op_name + "_backward_input_" + f"{i}" + ".npy")
                if op_name == "embeddings_word_embeddings":
                    global word_embedding_grad_result
                    word_embedding_grad_result = input[i]
                assert os.path.exists(input_golden_path), f"{input_golden_path} is not exist"
                compare_func(input[i], input_golden_path, input_eps)
        else:
            input_golden_path = os.path.join(cfg.golden_backward_data_path, op_name + "_backward_input.npy")
            assert os.path.exists(input_golden_path), f"{input_golden_path} is not exist"
            compare_func(input, input_golden_path, input_eps)
        if isinstance(output, tuple):
            for i in range(len(output)):
                if output[i] is None:
                    continue
                output_golden_path = os.path.join(
                    cfg.golden_backward_data_path,
                    op_name + "_backward_output_" + f"{i}" + ".npy")
                assert os.path.exists(output_golden_path), f"{output_golden_path} is not exist"
                compare_func(output[i], output_golden_path, output_eps)
        else:
            output_golden_path = os.path.join(cfg.golden_backward_data_path, op_name + "_backward_output.npy")
            assert os.path.exists(output_golden_path), f"{output_golden_path} is not exist"
            compare_func(output, output_golden_path, output_eps)

    return hook


def add_forward_verify_hook_interface(model, ignore_detail_check, ignore_final_check):
    '''
    interface of adding forward hook, which can compare input and output result of the model with the forward golden data
    model:the fully bert model
    '''
    forward_final_layer_name = None
    if not ignore_detail_check:
        # compare embedding
        model.embeddings.word_embeddings.register_forward_hook(
            forward_verify_input_output_hook("embeddings_word_embeddings"))
        model.embeddings.token_type_embeddings.register_forward_hook(
            forward_verify_input_output_hook("embeddings_token_type_embeddings"))
        model.embeddings.position_embeddings.register_forward_hook(
            forward_verify_input_output_hook("embeddings_position_embeddings"))
        model.embeddings.LayerNorm.register_forward_hook(
            forward_verify_input_output_hook("embeddings_LayerNorm"))
        model.embeddings.dropout.register_forward_hook(
            forward_verify_input_output_hook("embeddings_dropout"))
        for i in range(cfg.num_hidden_layers):
            # compare attention
            model.encoder.layer[i].attention.self.query.register_forward_hook(
                forward_verify_input_output_hook(f"encoder_layer[{i}]_attention_self_query"))
            model.encoder.layer[i].attention.self.key.register_forward_hook(
                forward_verify_input_output_hook(f"encoder_layer[{i}]_attention_self_key"))
            model.encoder.layer[i].attention.self.value.register_forward_hook(
                forward_verify_input_output_hook(f"encoder_layer[{i}]_attention_self_value"))
            model.encoder.layer[i].attention.self.dropout.register_forward_hook(
                forward_verify_input_output_hook(f"encoder_layer[{i}]_attention_self_dropout"))
            model.encoder.layer[i].attention.output.dense.register_forward_hook(
                forward_verify_input_output_hook(f"encoder_layer[{i}]_attention_output_dense"))
            model.encoder.layer[i].attention.output.dropout.register_forward_hook(
                forward_verify_input_output_hook(f"encoder_layer[{i}]_attention_output_dropout"))
            model.encoder.layer[i].attention.output.LayerNorm.register_forward_hook(
                forward_verify_input_output_hook(f"encoder_layer[{i}]_attention_output_LayerNorm"))

            # compare encoder intermediate
            model.encoder.layer[i].intermediate.dense.register_forward_hook(
                forward_verify_input_output_hook(f"encoder_layer[{i}]_intermediate_dense"))
            # compare encoder output
            model.encoder.layer[i].output.dense.register_forward_hook(
                forward_verify_input_output_hook(f"encoder_layer[{i}]_output_dense"))
            model.encoder.layer[i].output.LayerNorm.register_forward_hook(
                forward_verify_input_output_hook(f"encoder_layer[{i}]_output_LayerNorm"))
            model.encoder.layer[i].output.dropout.register_forward_hook(
                forward_verify_input_output_hook(f"encoder_layer[{i}]_output_dropout"))
        # compare pooler output
        if cfg.add_pooling_layer_flag:
            model.pooler.dense.register_forward_hook(
                forward_verify_input_output_hook("pooler_dense"))
            model.pooler.activation.register_forward_hook(
                forward_verify_input_output_hook("pooler_activation"))
            forward_final_layer_name = "pooler_activation"
        else:
            forward_final_layer_name = f"encoder_layer[{cfg.num_hidden_layers - 1}]_output_LayerNorm"
        if ignore_final_check:
            forward_final_layer_name = None
    elif not ignore_final_check:
        if cfg.add_pooling_layer_flag:
            model.pooler.activation.register_forward_hook(
                forward_verify_input_output_hook("pooler_activation"))
            forward_final_layer_name = "pooler_activation"
        else:
            model.encoder.layer[cfg.num_hidden_layers - 1].output.LayerNorm.register_forward_hook(
                forward_verify_input_output_hook(f"encoder_layer[{cfg.num_hidden_layers - 1}]_output_LayerNorm"))
            forward_final_layer_name = f"encoder_layer[{cfg.num_hidden_layers - 1}]_output_LayerNorm"
    else:
        info = "don't do any forward layer check"
        save_log(info)
    return forward_final_layer_name


    info = "add forward hook finished"
    save_log(info)


def add_backward_verify_hook_interface(model, ignore_detail_check, ignore_final_check):
    '''
    interface of adding backward hook, which can compare input and output result of the model with the backward golden data
    model:the fully bert model
    '''
    backward_final_layer_name = None
    if not ignore_detail_check:
        # compare embedding
        model.embeddings.word_embeddings.register_backward_hook(
            backward_verify_input_output_hook("embeddings_word_embeddings"))
        model.embeddings.token_type_embeddings.register_backward_hook(
            backward_verify_input_output_hook("embeddings_token_type_embeddings"))
        model.embeddings.position_embeddings.register_backward_hook(
            backward_verify_input_output_hook("embeddings_position_embeddings"))
        model.embeddings.LayerNorm.register_backward_hook(
            backward_verify_input_output_hook("embeddings_LayerNorm"))
        model.embeddings.dropout.register_backward_hook(
            backward_verify_input_output_hook("embeddings_dropout"))

        for i in range(cfg.num_hidden_layers):
            # compare attention
            model.encoder.layer[i].attention.self.query.register_backward_hook(
                backward_verify_input_output_hook(f"encoder_layer[{i}]_attention_self_query"))
            model.encoder.layer[i].attention.self.key.register_backward_hook(
                backward_verify_input_output_hook(f"encoder_layer[{i}]_attention_self_key"))
            model.encoder.layer[i].attention.self.value.register_backward_hook(
                backward_verify_input_output_hook(f"encoder_layer[{i}]_attention_self_value"))
            model.encoder.layer[i].attention.self.dropout.register_backward_hook(
                backward_verify_input_output_hook(f"encoder_layer[{i}]_attention_self_dropout"))

            model.encoder.layer[i].attention.output.dense.register_backward_hook(
                backward_verify_input_output_hook(f"encoder_layer[{i}]_attention_output_dense"))
            model.encoder.layer[i].attention.output.dropout.register_backward_hook(
                backward_verify_input_output_hook(f"encoder_layer[{i}]_attention_output_dropout"))
            model.encoder.layer[i].attention.output.LayerNorm.register_backward_hook(
                backward_verify_input_output_hook(f"encoder_layer[{i}]_attention_output_LayerNorm"))

            # compare encoder intermediate
            model.encoder.layer[i].intermediate.dense.register_backward_hook(
                backward_verify_input_output_hook(f"encoder_layer[{i}]_intermediate_dense"))

            # compare encoder output
            model.encoder.layer[i].output.dense.register_backward_hook(
                backward_verify_input_output_hook(f"encoder_layer[{i}]_output_dense"))
            model.encoder.layer[i].output.LayerNorm.register_backward_hook(
                backward_verify_input_output_hook(f"encoder_layer[{i}]_output_LayerNorm"))
            model.encoder.layer[i].output.dropout.register_backward_hook(
                backward_verify_input_output_hook(f"encoder_layer[{i}]_output_dropout"))

        # compare pooler output
        if cfg.add_pooling_layer_flag:
            model.pooler.dense.register_backward_hook(
                backward_verify_input_output_hook("pooler_dense"))
            model.pooler.activation.register_backward_hook(
                backward_verify_input_output_hook("pooler_activation"))
        if ignore_final_check:
            backward_final_layer_name = None
        else:
            backward_final_layer_name = "embeddings_word_embeddings"
    elif not ignore_final_check:
        model.embeddings.word_embeddings.register_backward_hook(
            backward_verify_input_output_hook("embeddings_word_embeddings"))
        backward_final_layer_name = "embeddings_word_embeddings"
    else:
        info = "don't do any backward layer check"
        save_log(info)
    info = "add backward hook finished"
    save_log(info)
    return backward_final_layer_name


def ids_tensor(shape, vocab_size):
    """
    creates a random int32 tensor of the shape within the vocab size.
    shape: shape of the tensor to created
    vocab_size: length of the vocab
    """
    rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))
    return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


def gen_input():
    '''
    when cfg.mode_option is save, generate input data by random generating method, \
    include inputs_ids, token_type_ids and position_ids .
    '''
    input_ids = ids_tensor([cfg.batch_size, cfg.seq_length], cfg.vocab_size).to(cfg.device)
    token_type_ids = ids_tensor([cfg.batch_size, cfg.seq_length], cfg.type_vocab_size).to(cfg.device)
    position_ids = ids_tensor([cfg.batch_size, cfg.seq_length], 2).to(cfg.device)

    input_ids_path = os.path.join(cfg.golden_forward_data_path, "input_ids.npy")
    np.save(input_ids_path, input_ids.cpu().numpy(),
            allow_pickle=False, fix_imports=False)
    token_type_ids_path = os.path.join(cfg.golden_forward_data_path, "token_type_ids.npy")
    np.save(token_type_ids_path, token_type_ids.cpu().numpy(),
            allow_pickle=False, fix_imports=False)
    position_ids_path = os.path.join(cfg.golden_forward_data_path, "position_ids.npy")
    np.save(position_ids_path, position_ids.cpu().numpy(),
            allow_pickle=False, fix_imports=False)
    return input_ids, token_type_ids, position_ids


def forward_save_input_output_hook(op_name):
    '''
    add hook to save forward input and output
    op_name:name of operation
    '''
    def hook(model, input, output):
        input_save_name_base = os.path.join(cfg.golden_forward_data_path, op_name + '_forward_input')
        if isinstance(input, tuple):
            for i in range(len(input)):
                if input[i] is None:
                    continue
                np.save(input_save_name_base + f"_{i}", input[i].clone().detach().cpu().numpy(),
                        allow_pickle=False, fix_imports=False)
        else:
            np.save(input_save_name_base, input.clone().detach().cpu().numpy(),
                    allow_pickle=False, fix_imports=False)

        output_save_name_base = os.path.join(cfg.golden_forward_data_path, op_name + '_forward_output')
        if isinstance(output, tuple):
            for i in range(len(output)):
                if output[i] is None:
                    continue
                np.save(output_save_name_base + f"_{i}", output[i].clone().detach().cpu().numpy(),
                        allow_pickle=False, fix_imports=False)
        else:
            np.save(output_save_name_base, output.clone().detach().cpu().numpy(),
                    allow_pickle=False, fix_imports=False)
    return hook


def backward_save_input_output_hook(op_name):
    '''
    add hook to save forward input and output
    op_name:name of operation
    '''
    def hook(model, input, output):
        input_save_name_base = os.path.join(cfg.golden_backward_data_path, op_name + '_backward_input')
        if isinstance(input, tuple):
            for i in range(len(input)):
                if input[i] is None:
                    continue
                np.save(input_save_name_base + f"_{i}", input[i].clone().detach().cpu().numpy(),
                        allow_pickle=False, fix_imports=False)
        else:
            np.save(input_save_name_base, input.clone().detach().cpu().numpy(),
                    allow_pickle=False, fix_imports=False)

        output_save_name_base = os.path.join(cfg.golden_backward_data_path, op_name + '_backward_output')
        if isinstance(output, tuple):
            for i in range(len(output)):
                if output[i] is None:
                    continue
                np.save(output_save_name_base + f"_{i}", output[i].clone().detach().cpu().numpy(),
                        allow_pickle=False, fix_imports=False)
        else:
            np.save(output_save_name_base, output.clone().detach().cpu().numpy(),
                    allow_pickle=False, fix_imports=False)
    return hook


def add_forward_save_hook_interface(model):
    '''
    interface of adding forward hook, which can save input and output as the golden data
    model:the fully bert model
    '''
    model.embeddings.word_embeddings.register_forward_hook(
        forward_save_input_output_hook("embeddings_word_embeddings"))
    model.embeddings.token_type_embeddings.register_forward_hook(
        forward_save_input_output_hook("embeddings_token_type_embeddings"))
    model.embeddings.position_embeddings.register_forward_hook(
        forward_save_input_output_hook("embeddings_position_embeddings"))

    model.embeddings.LayerNorm.register_forward_hook(
        forward_save_input_output_hook("embeddings_LayerNorm"))
    model.embeddings.dropout.register_forward_hook(
        forward_save_input_output_hook("embeddings_dropout"))

    for i in range(cfg.num_hidden_layers):
        # attention.self
        model.encoder.layer[i].attention.self.key.register_forward_hook(
            forward_save_input_output_hook(f"encoder_layer[{i}]_attention_self_key"))
        model.encoder.layer[i].attention.self.value.register_forward_hook(
            forward_save_input_output_hook(f"encoder_layer[{i}]_attention_self_value"))
        model.encoder.layer[i].attention.self.query.register_forward_hook(
            forward_save_input_output_hook(f"encoder_layer[{i}]_attention_self_query"))
        model.encoder.layer[i].attention.self.dropout.register_forward_hook(
            forward_save_input_output_hook(f"encoder_layer[{i}]_attention_self_dropout"))

        # attention.output
        model.encoder.layer[i].attention.output.dense.register_forward_hook(
            forward_save_input_output_hook(f"encoder_layer[{i}]_attention_output_dense"))
        model.encoder.layer[i].attention.output.dropout.register_forward_hook(
            forward_save_input_output_hook(f"encoder_layer[{i}]_attention_output_dropout"))
        model.encoder.layer[i].attention.output.LayerNorm.register_forward_hook(
            forward_save_input_output_hook(f"encoder_layer[{i}]_attention_output_LayerNorm"))

        # intermediate
        model.encoder.layer[i].intermediate.dense.register_forward_hook(
            forward_save_input_output_hook(f"encoder_layer[{i}]_intermediate_dense"))

        # output
        model.encoder.layer[i].output.dense.register_forward_hook(
            forward_save_input_output_hook(f"encoder_layer[{i}]_output_dense"))
        model.encoder.layer[i].output.LayerNorm.register_forward_hook(
            forward_save_input_output_hook(f"encoder_layer[{i}]_output_LayerNorm"))
        model.encoder.layer[i].output.dropout.register_forward_hook(
            forward_save_input_output_hook(f"encoder_layer[{i}]_output_dropout"))
    if cfg.add_pooling_layer_flag:
        model.pooler.activation.register_forward_hook(forward_save_input_output_hook("pooler_activation"))
        model.pooler.dense.register_forward_hook(forward_save_input_output_hook("pooler_dense"))


def add_backward_save_hook_interface(model):
    '''
    interface of adding backward hook, which can save input and output of backward as the golden data
    model:the fully bert model
    '''
    model.embeddings.word_embeddings.register_backward_hook(
        backward_save_input_output_hook("embeddings_word_embeddings"))
    model.embeddings.token_type_embeddings.register_backward_hook(
        backward_save_input_output_hook("embeddings_token_type_embeddings"))
    model.embeddings.position_embeddings.register_backward_hook(
        backward_save_input_output_hook("embeddings_position_embeddings"))

    model.embeddings.LayerNorm.register_backward_hook(
        backward_save_input_output_hook("embeddings_LayerNorm"))
    model.embeddings.dropout.register_backward_hook(
        backward_save_input_output_hook("embeddings_dropout"))

    for i in range(cfg.num_hidden_layers):
        # attention.self
        model.encoder.layer[i].attention.self.key.register_backward_hook(
            backward_save_input_output_hook(f"encoder_layer[{i}]_attention_self_key"))
        model.encoder.layer[i].attention.self.value.register_backward_hook(
            backward_save_input_output_hook(f"encoder_layer[{i}]_attention_self_value"))
        model.encoder.layer[i].attention.self.query.register_backward_hook(
            backward_save_input_output_hook(f"encoder_layer[{i}]_attention_self_query"))
        model.encoder.layer[i].attention.self.dropout.register_backward_hook(
            backward_save_input_output_hook(f"encoder_layer[{i}]_attention_self_dropout"))

        # attention.output
        model.encoder.layer[i].attention.output.dense.register_backward_hook(
            backward_save_input_output_hook(f"encoder_layer[{i}]_attention_output_dense"))
        model.encoder.layer[i].attention.output.dropout.register_backward_hook(
            backward_save_input_output_hook(f"encoder_layer[{i}]_attention_output_dropout"))
        model.encoder.layer[i].attention.output.LayerNorm.register_backward_hook(
            backward_save_input_output_hook(f"encoder_layer[{i}]_attention_output_LayerNorm"))

        # intermediate
        model.encoder.layer[i].intermediate.dense.register_backward_hook(
            backward_save_input_output_hook(f"encoder_layer[{i}]_intermediate_dense"))

        # output
        model.encoder.layer[i].output.dense.register_backward_hook(
            backward_save_input_output_hook(f"encoder_layer[{i}]_output_dense"))
        model.encoder.layer[i].output.LayerNorm.register_backward_hook(
            backward_save_input_output_hook(f"encoder_layer[{i}]_output_LayerNorm"))
        model.encoder.layer[i].output.dropout.register_backward_hook(
            backward_save_input_output_hook(f"encoder_layer[{i}]_output_dropout"))
    if cfg.add_pooling_layer_flag:
        model.pooler.activation.register_backward_hook(backward_save_input_output_hook("pooler_activation"))
        model.pooler.dense.register_backward_hook(backward_save_input_output_hook("pooler_dense"))
