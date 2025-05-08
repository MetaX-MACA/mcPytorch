import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.optim as optim
import transformers
import numpy as np
import logging
import argparse
import os

device = "cuda"
seed = 0
torch.manual_seed(seed)
SAVE_LOG = False

golden_root_path = "/netapp/pytorch/golden/amp/bert/"
GOLDEN_PATH = os.getenv("PYTORCH_TEST_GOLDEN_PATH")
if GOLDEN_PATH and os.path.exists(GOLDEN_PATH):
    golden_root_path = os.path.join(GOLDEN_PATH, "amp", "bert")
golden_debug_root_path = os.path.join(golden_root_path, "debug")

def log(info, save_log=False):
    '''
    save important infomation in the save_log_path
    '''
    if SAVE_LOG:
        logging.basicConfig(filename="bert_amp_maca.log",
                            format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
        logging.info(info)

    print(info, flush=True)


def save_forward(name):
    def hook(model, input, output):
        output_name = name + "_output"
        save_output_name = args.debug_golden_path + output_name
        if isinstance(output, tuple):
            for idx, op in enumerate(output):
                if torch.is_tensor(op):
                    save_name = save_output_name + f"_{idx}"
                    np.save(save_name, op.clone().detach().cpu(
                        ).numpy(), allow_pickle=False, fix_imports=False)
        else:
            np.save(save_output_name, output.clone().detach().cpu(
                ).numpy(), allow_pickle=False, fix_imports=False)

        input_name = name + "_input"
        save_input_name = args.debug_golden_path + input_name
        if isinstance(input, tuple):
            for idx, ip in enumerate(input):
                if torch.is_tensor(ip):
                    save_name = save_input_name + f"_{idx}"
                    np.save(save_name, ip.clone().detach().cpu(
                        ).numpy(), allow_pickle=False, fix_imports=False)
        else:
            np.save(save_input_name, input.clone().detach().cpu(
                ).numpy(), allow_pickle=False, fix_imports=False)

    return hook


def add_save_forward_hook(model):
    model.bert_model.embeddings.word_embeddings.register_forward_hook(
        save_forward("embeddings_word_embeddings"))
    model.bert_model.embeddings.token_type_embeddings.register_forward_hook(
        save_forward("embeddings_token_type_embeddings"))
    model.bert_model.embeddings.position_embeddings.register_forward_hook(
        save_forward("embeddings_position_embeddings"))
    model.bert_model.embeddings.LayerNorm.register_forward_hook(
        save_forward("embeddings_LayerNorm"))
    model.bert_model.embeddings.dropout.register_forward_hook(
        save_forward("embeddings_dropout"))

    for i in range(1):
        model.bert_model.encoder.layer[i].attention.self.key.register_forward_hook(
            save_forward(f"encoder_layer[{i}]_attention_self_key"))
        model.bert_model.encoder.layer[i].attention.self.value.register_forward_hook(
            save_forward(f"encoder_layer[{i}]_attention_self_value"))
        model.bert_model.encoder.layer[i].attention.self.query.register_forward_hook(
            save_forward(f"encoder_layer[{i}]_attention_self_query"))
        model.bert_model.encoder.layer[i].attention.self.dropout.register_forward_hook(
            save_forward(f"encoder_layer[{i}]_attention_self_dropout"))

        model.bert_model.encoder.layer[i].attention.output.dense.register_forward_hook(
            save_forward(f"encoder_layer[{i}]_attention_output_dense"))
        model.bert_model.encoder.layer[i].attention.output.LayerNorm.register_forward_hook(
            save_forward(f"encoder_layer[{i}]_attention_output_LayerNorm"))
        model.bert_model.encoder.layer[i].attention.output.dropout.register_forward_hook(
            save_forward(f"encoder_layer[{i}]_attention_output_dropout"))

        model.bert_model.encoder.layer[i].attention.register_forward_hook(
            save_forward(f"encoder_layer[{i}]_attention"))

        model.bert_model.encoder.layer[i].intermediate.dense.register_forward_hook(
            save_forward(f"encoder_layer[{i}]_intermediate_dense"))
        model.bert_model.encoder.layer[i].intermediate.register_forward_hook(
            save_forward(f"encoder_layer[{i}]_intermediate"))

        model.bert_model.encoder.layer[i].output.dense.register_forward_hook(
            save_forward(f"encoder_layer[{i}]_output_dense"))
        model.bert_model.encoder.layer[i].output.LayerNorm.register_forward_hook(
            save_forward(f"encoder_layer[{i}]_output_LayerNorm"))
        model.bert_model.encoder.layer[i].output.dropout.register_forward_hook(
            save_forward(f"encoder_layer[{i}]_output_dropout"))

    model.bert_model.pooler.activation.register_forward_hook(
        save_forward(f"pooler_output"))
    model.bert_model.pooler.dense.register_forward_hook(
        save_forward(f"pooler_dense"))

    info = "add forward hook finished"
    log(info)


def save_backward(name):
    def hook(model, grad_input, grad_output):
        output_name = name + "_gradout"
        input_name = name + "_gradin"
        save_output_name = args.debug_golden_path + output_name
        save_input_name = args.debug_golden_path + input_name

        if isinstance(grad_output, tuple):
            for idx, op in enumerate(grad_output):
                if torch.is_tensor(op):
                    save_name = save_output_name + f"_{idx}"
                    np.save(save_name, op.clone().detach().cpu(
                        ).numpy(), allow_pickle=False, fix_imports=False)
        else:
            np.save(save_output_name, grad_output[0].clone().detach().cpu(
                ).numpy(), allow_pickle=False, fix_imports=False)

        if isinstance(grad_input, tuple):
            for idx, ip in enumerate(grad_input):
                if torch.is_tensor(ip):
                    save_name = save_input_name +  f"_{idx}"
                    np.save(save_name, ip.clone().detach().cpu(
                        ).numpy(), allow_pickle=False, fix_imports=False)
        else:
            np.save(save_input_name, grad_input[0].clone().detach().cpu(
                ).numpy(), allow_pickle=False, fix_imports=False)

    return hook


def add_save_backward_hook(model):
    model.bert_model.embeddings.word_embeddings.register_backward_hook(
        save_backward("embeddings_word_embeddings"))
    model.bert_model.embeddings.token_type_embeddings.register_backward_hook(
        save_backward("embeddings_token_type_embeddings"))
    model.bert_model.embeddings.position_embeddings.register_backward_hook(
        save_backward("embeddings_position_embeddings"))
    model.bert_model.embeddings.LayerNorm.register_backward_hook(
        save_backward("embeddings_LayerNorm"))
    model.bert_model.embeddings.dropout.register_backward_hook(
        save_backward("embeddings_dropout"))

    for i in range(1):
        model.bert_model.encoder.layer[i].attention.self.key.register_backward_hook(
            save_backward(f"encoder_layer[{i}]_attention_self_key"))
        model.bert_model.encoder.layer[i].attention.self.value.register_backward_hook(
            save_backward(f"encoder_layer[{i}]_attention_self_value"))
        model.bert_model.encoder.layer[i].attention.self.query.register_backward_hook(
            save_backward(f"encoder_layer[{i}]_attention_self_query"))
        model.bert_model.encoder.layer[i].attention.self.dropout.register_backward_hook(
            save_backward(f"encoder_layer[{i}]_attention_self_dropout"))

        model.bert_model.encoder.layer[i].attention.output.dense.register_backward_hook(
            save_backward(f"encoder_layer[{i}]_attention_output_dense"))
        model.bert_model.encoder.layer[i].attention.output.LayerNorm.register_backward_hook(
            save_backward(f"encoder_layer[{i}]_attention_output_LayerNorm"))
        model.bert_model.encoder.layer[i].attention.output.dropout.register_backward_hook(
            save_backward(f"encoder_layer[{i}]_attention_output_dropout"))

        model.bert_model.encoder.layer[i].intermediate.dense.register_backward_hook(
            save_backward(f"encoder_layer[{i}]_intermediate_dense"))

        model.bert_model.encoder.layer[i].output.dense.register_backward_hook(
            save_backward(f"encoder_layer[{i}]_output_dense"))
        model.bert_model.encoder.layer[i].output.LayerNorm.register_backward_hook(
            save_backward(f"encoder_layer[{i}]_output_LayerNorm"))
        model.bert_model.encoder.layer[i].output.dropout.register_backward_hook(
            save_backward(f"encoder_layer[{i}]_output_dropout"))

    model.bert_model.pooler.activation.register_backward_hook(
        save_backward(f"pooler_output"))
    model.bert_model.pooler.dense.register_backward_hook(
        save_backward(f"pooler_dense"))

    info = "add backward hook finished"
    log(info)


def check_close(infer_result_data, golden_data, eps=1e-4):
    diff = infer_result_data - golden_data
    diff_square = diff * diff
    infer_result_square_double = 2 * infer_result_data * infer_result_data
    sum_diff_square = torch.sum(diff_square)
    sum_infer_result_square_double = torch.sum(infer_result_square_double)
    result = torch.sqrt(sum_diff_square / (sum_infer_result_square_double + 1e-12))
    log(f"check close error: {result}")

    return result < eps


def compare(golden_name, eps=1e-3, mode="forward"):
    def compare_impl(golden_data_path, data, eps):
        data = data.clone().detach().cpu().float()
        golden = torch.from_numpy(np.load(golden_data_path))
        status = check_close(data, golden, eps)
        data_name = golden_data_path.split("/")[-1].split(".")[0]

        if status:
            info = f"##### {data_name} check success"
            log(info)
        else:
            diff = torch.abs(data - golden)
            max_diff = torch.max(diff).item()
            info = f"***** {data_name} check fail, max diff {max_diff}"
            log(info)

    def comp(golden_path, data, eps):
        if os.path.exists(golden_path):
            compare_impl(golden_path, data, eps)

    def hook(model, input, output):
        root_path = args.debug_golden_path
        if mode == "forward":   # forward
            input_golden_name = golden_name + "_input"
            output_golden_name = golden_name + "_output"
        else:   # backward
            input_golden_name = golden_name + "_gradin"
            output_golden_name = golden_name + "_gradout"

        if isinstance(input, tuple):
            for idx, ip in enumerate(input):
                in_golden_path = os.path.join(root_path, input_golden_name + f"_{idx}.npy")
                comp(in_golden_path, ip, eps)
        else:
            in_golden_path = os.path.join(root_path, input_golden_name + ".npy")
            comp(in_golden_path, input, eps)

        if isinstance(output, tuple):
            for idx, op in enumerate(output):
                out_golden_path = os.path.join(root_path, output_golden_name + f"_{idx}.npy")
                comp(out_golden_path, op, eps)
        else:
            out_golden_path = os.path.join(root_path, output_golden_name + ".npy")
            comp(out_golden_path, output, eps)

    return hook


def add_check_forward_hook(model):
    # compare embedding
    model.bert_model.embeddings.word_embeddings.register_forward_hook(
        compare("embeddings_word_embeddings",mode="forward"))
    model.bert_model.embeddings.position_embeddings.register_forward_hook(
        compare("embeddings_position_embeddings",mode="forward"))
    model.bert_model.embeddings.token_type_embeddings.register_forward_hook(
        compare("embeddings_token_type_embeddings",mode="forward"))
    model.bert_model.embeddings.LayerNorm.register_forward_hook(
        compare("embeddings_LayerNorm",mode="forward"))
    model.bert_model.embeddings.dropout.register_forward_hook(
        compare("embeddings_dropout",mode="forward"))

    # compare attention
    model.bert_model.encoder.layer[0].attention.self.query.register_forward_hook(
        compare("encoder_layer[0]_attention_self_query",mode="forward"))
    model.bert_model.encoder.layer[0].attention.self.key.register_forward_hook(
        compare("encoder_layer[0]_attention_self_key",mode="forward"))
    model.bert_model.encoder.layer[0].attention.self.value.register_forward_hook(
        compare("encoder_layer[0]_attention_self_value",mode="forward"))
    model.bert_model.encoder.layer[0].attention.self.dropout.register_forward_hook(
        compare("encoder_layer[0]_attention_self_dropout",mode="forward"))

    model.bert_model.encoder.layer[0].attention.output.dense.register_forward_hook(
        compare(f"encoder_layer[0]_attention_output_dense",mode="forward"))
    model.bert_model.encoder.layer[0].attention.output.LayerNorm.register_forward_hook(
        compare(f"encoder_layer[0]_attention_output_LayerNorm",mode="forward"))
    model.bert_model.encoder.layer[0].attention.output.dropout.register_forward_hook(
        compare(f"encoder_layer[0]_attention_output_dropout",mode="forward"))

    model.bert_model.encoder.layer[0].attention.register_forward_hook(
        compare("encoder_layer[0]_attention",mode="forward"))

    # compare encoder intermediate
    model.bert_model.encoder.layer[0].intermediate.dense.register_forward_hook(
        compare("encoder_layer[0]_intermediate_dense",mode="forward"))
    model.bert_model.encoder.layer[0].intermediate.register_forward_hook(
        compare(f"encoder_layer[0]_intermediate"))

    # compare encoder output
    model.bert_model.encoder.layer[0].output.dense.register_forward_hook(
        compare("encoder_layer[0]_output_dense",mode="forward"))
    model.bert_model.encoder.layer[0].output.LayerNorm.register_forward_hook(
        compare("encoder_layer[0]_output_LayerNorm",mode="forward"))
    model.bert_model.encoder.layer[0].output.dropout.register_forward_hook(
        compare("encoder_layer[0]_output_dropout",mode="forward"))

    # compare pooler output
    model.bert_model.pooler.dense.register_forward_hook(
        compare("pooler_dense",mode="forward"))
    model.bert_model.pooler.activation.register_forward_hook(
        compare("pooler_output",mode="forward"))

    info = "add forward hook finished"
    log(info)


def add_check_backward_hook(model):
    # compare embedding
    model.bert_model.embeddings.word_embeddings.register_backward_hook(
        compare("embeddings_word_embeddings",eps=bwd_eps,mode="backward"))
    model.bert_model.embeddings.position_embeddings.register_backward_hook(
        compare("embeddings_position_embeddings",eps=bwd_eps,mode="backward"))
    model.bert_model.embeddings.token_type_embeddings.register_backward_hook(
        compare("embeddings_token_type_embeddings",eps=bwd_eps,mode="backward"))
    model.bert_model.embeddings.LayerNorm.register_backward_hook(
        compare("embeddings_LayerNorm",eps=bwd_eps,mode="backward"))
    model.bert_model.embeddings.dropout.register_backward_hook(
        compare("embeddings_dropout",eps=bwd_eps,mode="backward"))

    # compare attention
    model.bert_model.encoder.layer[0].attention.self.query.register_backward_hook(
        compare("encoder_layer[0]_attention_self_query",eps=bwd_eps,mode="backward"))
    model.bert_model.encoder.layer[0].attention.self.key.register_backward_hook(
        compare("encoder_layer[0]_attention_self_key",eps=bwd_eps,mode="backward"))
    model.bert_model.encoder.layer[0].attention.self.value.register_backward_hook(
        compare("encoder_layer[0]_attention_self_value",eps=bwd_eps,mode="backward"))
    model.bert_model.encoder.layer[0].attention.self.dropout.register_backward_hook(
        compare("encoder_layer[0]_attention_self_dropout",eps=bwd_eps,mode="backward"))
    model.bert_model.encoder.layer[0].attention.output.dense.register_backward_hook(
        compare(f"encoder_layer[0]_attention_output_dense",eps=bwd_eps,mode="backward"))
    model.bert_model.encoder.layer[0].attention.output.LayerNorm.register_backward_hook(
        compare(f"encoder_layer[0]_attention_output_LayerNorm",eps=bwd_eps,mode="backward"))
    model.bert_model.encoder.layer[0].attention.output.dropout.register_backward_hook(
        compare(f"encoder_layer[0]_attention_output_dropout",eps=bwd_eps,mode="backward"))

    # compare encoder intermediate
    model.bert_model.encoder.layer[0].intermediate.dense.register_backward_hook(
        compare("encoder_layer[0]_intermediate_dense",eps=bwd_eps,mode="backward"))

    # compare encoder output
    model.bert_model.encoder.layer[0].output.dense.register_backward_hook(
        compare("encoder_layer[0]_output_dense",eps=bwd_eps,mode="backward"))
    model.bert_model.encoder.layer[0].output.LayerNorm.register_backward_hook(
        compare("encoder_layer[0]_output_LayerNorm",eps=bwd_eps,mode="backward"))
    model.bert_model.encoder.layer[0].output.dropout.register_backward_hook(
        compare("encoder_layer[0]_output_dropout",eps=bwd_eps,mode="backward"))

    # compare pooler output
    model.bert_model.pooler.dense.register_backward_hook(
        compare("pooler_dense",eps=bwd_eps,mode="backward"))
    model.bert_model.pooler.activation.register_backward_hook(
        compare("pooler_output",eps=bwd_eps,mode="backward"))

    info = "add backward hook finished"
    log(info)


class BERT(nn.Module):
    def __init__(self, class_num):
        super(BERT, self).__init__()

        def disable_dropout(op):
            if isinstance(op, nn.Dropout):
                op.p = 0

        if not args.null_hardware:
            if not args.offline:
                self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased") # load pretrained model from website
            else:
                self.bert_model = transformers.BertModel.from_pretrained(args.weight_path)   # load pretrained model from local
        else:
            self.bert_model = transformers.BertModel(transformers.BertConfig())   # init model by config

        if not args.enable_dropout:
            self.bert_model.apply(disable_dropout)

        self.out = nn.Linear(768, class_num)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)

        out = self.out(o2)
        out = nn.Softmax(1)(out)
        return out


def load_data(data_path):
    input = np.load(data_path, allow_pickle=True)
    return input


def finetune_amp(model, dataset, epochs, batch_size, data_num, lr, save_golden):
    assert data_num >= batch_size, "invalid data_num"
    dataset = dataset[:data_num]  # only choose first data_num data for train
    iteration = data_num // batch_size

    model = model.to(device)
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = amp.GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2, enabled=True)

    if args.data_path != "":
        golden_path = args.data_path
    else:
        golden_path = "./data/"
    if args.enable_dropout:
        golden_path += "golden_with_dropout.npy"
    else:
        golden_path += "golden_without_dropout.npy"

    golden_data = None
    golden_loss = []
    golden_scale = []

    if not args.null_hardware and not save_golden:
        assert os.path.exists(golden_path), f"invalid golden path: {golden_path}"
        golden_data = torch.from_numpy(np.load(golden_path)).float()

    for epoch in range(epochs):
        num_correct = 0
        mean_loss = 0
        start_index = 0
        for it in range(iteration):
            log(f"epoch: {epoch + 1}/{epochs} iteration: {it + 1}/{iteration} finetune start...")

            end_index = start_index + batch_size
            data = dataset[start_index:end_index]
            start_index += batch_size

            # prepare data
            input_ids = [item[0] for item in data]
            token_type_ids = [item[1] for item in data]
            attention_mask = [item[2] for item in data]
            label = [item[3] for item in data]

            input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).to(device)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(device)
            label = torch.tensor(label, dtype=torch.long).to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16):
                output = model(ids=input_ids, mask=attention_mask, token_type_ids=token_type_ids)
                loss = loss_fn(output, label)
            if not args.null_hardware and args.debug:
                save_output_name = args.debug_golden_path + "output.npy"
                save_loss_name = args.debug_golden_path + "loss.npy"
                if args.save_golden:
                    np.save(save_output_name, output.clone().detach().cpu(
                        ).numpy(), allow_pickle=False, fix_imports=False)
                    np.save(save_loss_name, loss.clone().detach().cpu(
                        ).numpy(), allow_pickle=False, fix_imports=False)
                else:
                    golden_output = torch.from_numpy(np.load(save_output_name))
                    status_output = check_close(output.clone().detach().cpu().float(), golden_output, 1e-4)
                    if status_output:
                        log(f"##### model output check success")
                    else:
                        diff = torch.abs(output.cpu() - golden_output)
                        max_diff = torch.max(diff).item()
                        log(f"***** model output check fail, max diff {max_diff}")
                    golden_loss = torch.from_numpy(np.load(save_loss_name))
                    status_loss = check_close(loss.clone().detach().cpu().float(), golden_loss, 1e-4)
                    if status_loss:
                        log(f"##### loss check success")
                    else:
                        diff = torch.abs(loss - golden_loss)
                        max_diff = torch.max(diff).item()
                        log(f"***** loss check fail, max diff {max_diff}")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if not args.null_hardware:
                log(f"epoch: {epoch + 1}, growth_factor: {scaler.get_growth_factor()}, {scaler.get_backoff_factor()}, {scaler.get_growth_interval()}, {scaler.get_scale()}")

            pred = torch.argmax(output, dim=1)
            num_correct += sum(1 for a, b in zip(pred, label) if a == b)
            mean_loss += loss

            # only save/check the first iteration
            if not args.null_hardware and args.debug:
                if save_golden:
                    return True
                else:
                    return status_loss

        accuracy = num_correct / data_num
        mean_loss = mean_loss / data_num
        if not args.null_hardware:
            log(f"epoch: {epoch + 1}/{epochs} accuracy: {accuracy} loss: {mean_loss}")

        if not args.null_hardware:
            if not save_golden:
                status1 = torch.allclose(mean_loss, golden_data[0][epoch], 1e-1)
                status2 = torch.allclose(torch.tensor(scaler.get_scale(), dtype=torch.float), golden_data[1][epoch])
                if not status1:
                    log(f"epoch: {epoch + 1}/{epochs} loss check fail")
                    return False
                elif not status2:
                    log(f"epoch: {epoch + 1}/{epochs} scale check fail")
                    return False
                else:
                    log(f"epoch: {epoch + 1}/{epochs} loss and scale check success")
            else:
                golden_loss.append(mean_loss.clone().detach().cpu().numpy())
                golden_scale.append(scaler.get_scale())

    if not args.null_hardware and save_golden:
        golden_data = np.array([golden_loss, golden_scale])
        np.save(golden_path, golden_data)

    return True


def run_finetune(epochs, batch_size, data_num, lr, save_golden):
    model = BERT(2)

    if args.data_path != "":
        dataset_path = args.data_path
    else:
        dataset_path = "./data/"
    dataset_path += "sentiment-classification-2-6920.npy"
    dataset = load_data(dataset_path).tolist()

    if not args.null_hardware and args.debug:
        if args.save_golden:
            add_save_forward_hook(model)
            add_save_backward_hook(model)
        else:
            add_check_forward_hook(model)
            add_check_backward_hook(model)

        return finetune_amp(model, dataset, 1, batch_size, data_num, lr, save_golden) # debug only run 1 epoch
    return finetune_amp(model, dataset, epochs, batch_size, data_num, lr, save_golden)


# parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--save_golden", action="store_true", help="save_golden_data use to save golden data, or it would run valid mode to compare output with golden data")
parser.add_argument("--save_log", action="store_true", help="save_log use to save log in text file")
parser.add_argument("--offline", action="store_true", help="run test under offline mode means use local pretained model weight")
parser.add_argument("--debug", action="store_true", help="run test under debug mode means save/check intermediate result")
parser.add_argument("--epochs", type=int, default=3, help="train epochs")
parser.add_argument("--batch_size", type=int, default=1, help="train batch size")
parser.add_argument("--data_num", type=int, choices=range(1, 6921), default=2, help="the number of input data sliced from dataset")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--weight_path", type=str, default=golden_root_path, help="the path of pretrained model weight, it's only needed under offline mode")
parser.add_argument("--data_path", type=str, default="", help="the path of golden data and dataset, default would search under work directory")
parser.add_argument("--debug_golden_path", type=str, default=golden_debug_root_path, help="the path of golden data for debug mode, it's only needed under debug mode")
parser.add_argument("--enable_dropout", action="store_true", help="enable dropout with p=0.1")
parser.add_argument("--null_hardware", action="store_true", help="run under null hardware mode")

args = parser.parse_args()

if args.save_log:
    SAVE_LOG = True
if args.enable_dropout:
    args.debug_golden_path = args.debug_golden_path + "with_dropout/"
else:
    args.debug_golden_path = args.debug_golden_path + "without_dropout/"

bwd_eps = 5e-3

if __name__ == "__main__":
    status = run_finetune(epochs=args.epochs, batch_size=args.batch_size, data_num=args.data_num, lr=args.lr, save_golden=args.save_golden)

    if status:
        exit(0)
    else:
        exit(1)
