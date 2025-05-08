import torch
import torch.nn as nn
import transformers
import numpy as np
import logging
import torch.optim as optim
import argparse
import os, sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import *

torch.manual_seed(0)
device = "cuda"

model_path = r"/netapp/pytorch/golden/amp/bert/"
torch.set_float32_matmul_precision('high')
SAVE_LOG = False

def log(info, save_log=False):
    '''
    save important infomation in the save_log_path
    '''
    if SAVE_LOG:
        logging.basicConfig(filename="bert_compile.log",
                            format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
        logging.info(info)

    print(info, flush=True)


class BERT(nn.Module):
    def __init__(self, class_num):
        super(BERT, self).__init__()

        def disable_dropout(op):
            if isinstance(op, nn.Dropout):
                op.p = 0
        self.bert_model = transformers.BertModel.from_pretrained(model_path)   # load pretrained model from local
        self.bert_model.apply(disable_dropout)  # disable dropout
        self.out = nn.Linear(768, class_num)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)

        out = self.out(o2)
        out = nn.Softmax(1)(out)
        return out


def run(epochs, batch_size, data_num, lr, save_golden):
    model = BERT(2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    model_com = torch.compile(model, mode="max-autotune", backend="inductor")

    if args.data_path != "":
        data_path = args.data_path
    else:
        data_path = "./data/"
    dataset_path =data_path +  "sentiment-classification-2-6920.npy"
    dataset = load_data(dataset_path).tolist()

    assert data_num >= batch_size, "invalid data_num"
    dataset = dataset[:data_num]  # only choose first data_num data for train
    iteration = data_num // batch_size

    golden_output = []
    golden_loss = []
    save_output_name = data_path + "output.npy"
    save_loss_name = data_path + "loss.npy"

    if not save_golden:
        golden_output = torch.from_numpy(np.load(save_output_name))
        golden_loss = torch.from_numpy(np.load(save_loss_name))
        assert (golden_output.shape[0] == epochs) and (golden_output.shape[1] == iteration), "golden_output size not match"
        assert (golden_loss.shape[0] == epochs), "golden_loss size not match"

    for epoch in range(epochs):
        start_index = 0
        mean_loss = 0
        output = []
        for it in range(iteration):
            log(f"epoch: {epoch + 1}/{epochs}, iteration: {it + 1}/{iteration} finetune start...")

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
            ret = timed(lambda: model_com(ids=input_ids, mask=attention_mask, token_type_ids=token_type_ids))
            log(f"     forward time(ms): {ret[1]}")

            if save_golden:
                output.append(ret[0].clone().detach().cpu().numpy())
            else:
                status = check_close(ret[0].clone().detach().cpu(), golden_output[epoch][it], 1e-3)
                if status:
                    log(f"epoch: {epoch + 1}/{epochs}, iteration: {it + 1}/{iteration} output check pass")
                else:
                    log(f"epoch: {epoch + 1}/{epochs}, iteration: {it + 1}/{iteration} output check fail")
                    return False

            loss = loss_fn(ret[0], label)
            mean_loss += loss
            _, time = timed(lambda: loss.backward())
            log(f"     backward time(ms): {time}")
            optimizer.step()
            log(f"epoch: {epoch + 1}/{epochs}, iteration: {it + 1}/{iteration} finetune finished")

                
        mean_loss = mean_loss / data_num

        if not save_golden:
            status1 = check_close(mean_loss, golden_loss[epoch], 1e-1)
            if not status:
                log(f"epoch: {epoch + 1}/{epochs} loss check fail")
                return False
            else:
                log(f"epoch: {epoch + 1}/{epochs} loss check pass")
        else:
            golden_loss.append(mean_loss.clone().detach().cpu().numpy())
            golden_output.append(output)

        log(f"epoch: {epoch + 1}/{epochs} finetune finished")

    if save_golden:
        golden_loss = np.array(golden_loss)
        golden_output = np.array(golden_output)
        np.save(save_loss_name, golden_loss)
        np.save(save_output_name, golden_output)
        log(f"save golden finished")
    return True


parser = argparse.ArgumentParser()
parser.add_argument("--save_golden", action="store_true", help="save_golden_data use to save golden data, or it would run valid mode to compare output with golden data")
parser.add_argument("--save_log", action="store_true", help="save_log use to save log in text file")
parser.add_argument("--debug", action="store_true", help="run test under debug mode means save/check intermediate result")
parser.add_argument("--epochs", type=int, default=2, help="train epochs")
parser.add_argument("--batch_size", type=int, default=1, help="train batch size")
parser.add_argument("--data_num", type=int, choices=range(1, 6921), default=1, help="the number of input data sliced from dataset")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--data_path", type=str, default="", help="the path of golden data and dataset, default would search under work directory")
parser.add_argument("--enable_dropout", action="store_true", help="enable dropout with p=0.1")

args = parser.parse_args()

if args.save_log:
    SAVE_LOG = True


if __name__ == "__main__":
    status = run(epochs=args.epochs, batch_size=args.batch_size, data_num=args.data_num, lr=args.lr, save_golden=args.save_golden)

    if status:
        log("##### success")
        exit(0)
    else:
        log(f"##### fail")
        exit(1)