import csv
import os
import socket
import argparse
import torch
import torch.multiprocessing as mp
from torch.utils.data import dataloader, distributed
from torchvision import datasets, transforms, models
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import config_dataset_path


# Set config
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=256, help="batch_size")
parser.add_argument("--total_epoch", default=20, help="total_epoch")
parser.add_argument("--learn_rate", default=1e-4, help="learn_rate")
parser.add_argument("--amp_mode", action="store_true", default=False, help="amp_mode")
parser.add_argument("--ddp_mode", action="store_true", default=False, help="ddp_mode")
parser.add_argument("--world_size", default=-1, help="world_size")
parser.add_argument("--dataset_mode", default="tiny", help="normal|small|tiny")
batch_sz = int(parser.parse_args().batch_size)
total_epoch = int(parser.parse_args().total_epoch)
learn_rate = float(parser.parse_args().learn_rate)
amp_mode = bool(parser.parse_args().amp_mode)
ddp_mode = bool(parser.parse_args().ddp_mode)
if ddp_mode:
    world_size = int(parser.parse_args().world_size)
    assert world_size!=-1, "Please set world size with DDP mode!"
dataset_mode = str(parser.parse_args().dataset_mode)
if dataset_mode == "normal":
    imagenet_path = config_dataset_path.imagenet_normal_path
elif dataset_mode == "small":
    imagenet_path = config_dataset_path.imagenet_small_path
else:  # dataset_mode == "tiny"
    imagenet_path = config_dataset_path.imagenet_tiny_path


# DDP init
if ddp_mode:
    def selectPort(ip_addr, port):
        for _ in range(10):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((ip_addr, port))
                s.shutdown(2)
                print("port %d has been used." % port)
                port += 100
            except:
                print("port %d is unused." % port)
                return str(port)
        raise(Exception("Couldn't find an available port."))
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = selectPort("127.0.0.1", 22782)


# Set datasets
imagenet_path_train = imagenet_path + "/train/"
imagenet_path_test = imagenet_path + "/val/"
train_imagenet_data = datasets.ImageFolder(root=imagenet_path_train,
    transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
)
test_imagenet_data = datasets.ImageFolder(root=imagenet_path_test,
    transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
)


# Set result file
if (os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/resnet50_result/")) is False:
    os.mkdir(os.path.dirname(os.path.abspath(__file__)) + "/resnet50_result/")
result_csv_path = os.path.dirname(os.path.abspath(__file__)) + "/resnet50_result/resnet50_"
if amp_mode:
    result_csv_path += "amp_"
else:
    result_csv_path += "fp32_"
if ddp_mode:
    result_csv_path += "ddp_"
result_csv_path += dataset_mode + ".csv"
if (os.path.exists(result_csv_path)):
    os.remove(result_csv_path)
result_title = ["epoch", "loss", "acc"]
with open(result_csv_path, "w") as f:
    writer = csv.writer(f)
    writer.writerow(result_title)


def resnet_train_eval(train_data, test_data, model, criteon, optimizer, device):
    for epoch in range(total_epoch):
        if ddp_mode:
            train_data.sampler.set_epoch(epoch)
        model.train()
        for batch_idx, (x, label) in enumerate(train_data):
            print("[train iter]:", batch_idx)
            x, label = x.to(device), label.to(device)
            if amp_mode:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
            else:
                logits = model(x)
            loss = criteon(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            eval_iter = 0
            for x, label in test_data:
                print("[eval iter]:", eval_iter)
                x, label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                eval_iter += 1
            acc = total_correct / total_num
        
        print("[epoch]:", epoch, "[loss]:", loss.item(), "[acc]:", acc)
        with open(result_csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss.item(), acc])


def resnet50_ddp_program(local_rank, world_size):
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    train_sampler = distributed.DistributedSampler(train_imagenet_data)
    train_data = dataloader.DataLoader(train_imagenet_data, batch_size=batch_sz * world_size, sampler=train_sampler)
    test_data = dataloader.DataLoader(test_imagenet_data, shuffle=True, batch_size=batch_sz)
    device = torch.device("cuda:%d" % local_rank)
    model = models.resnet50(pretrained=False)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    criteon = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    resnet_train_eval(train_data, test_data, model, criteon, optimizer, device)
            

def resnet50_program():
    train_data = dataloader.DataLoader(train_imagenet_data, shuffle=True, batch_size=batch_sz)
    test_data = dataloader.DataLoader(test_imagenet_data, shuffle=True, batch_size=batch_sz)
    model = models.resnet50(pretrained=False).cuda()
    criteon = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    resnet_train_eval(train_data, test_data, model, criteon, optimizer, "cuda")


if __name__ == '__main__':

    print("##### Test config #####")
    print("[batch_size]:", batch_sz)
    print("[total_epoch]:", total_epoch)
    print("[amp_mode]:", amp_mode)
    print("[ddp_mode]:", ddp_mode)
    if ddp_mode:
        print("[world_size]:", world_size)
    print("[dataset_mode]:", dataset_mode)
    print("[dataset_path]:", imagenet_path)
    print("##### Test config #####")

    # Strat train and val
    if ddp_mode:
        mp.spawn(resnet50_ddp_program, args=(world_size,), nprocs=world_size, join=True)
    else:
        resnet50_program()
