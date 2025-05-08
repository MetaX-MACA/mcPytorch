import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import copy
import itertools
from itertools import repeat
import random
import sys
import socket
import argparse
import resnet50_module

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

def step_model(model, input_resnet, optimizer, rank):
    '''
    calculate backward and update parameter value
    '''
    model.train()
    output_resnet_data = model(input_resnet)
    label_resnet = torch.zeros(output_resnet_data.shape, dtype=torch.float32).to(rank)
    label_resnet[:, 0] = 1
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output_resnet_data, label_resnet)
    loss.backward()
    optimizer.step()


def ddp_program(rank, world_size, local_batch_size, null_hardware):
    '''
    program to verify ddp runs correctly
    '''
    print("null_hardware:", null_hardware)
    torch.manual_seed(0)
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device("cuda:%d" % rank)
    model = resnet50_module.resnet50()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    # construct DDP model
    ddp_model = DDP(copy.deepcopy(model).to(device), device_ids=[device])
    optimizer = optim.SGD(model.parameters(), lr=1)
    optimizer_ddp = optim.SGD(ddp_model.parameters(), lr=1)

    for iteration in range(1):
        print("iteration:", iteration)
        global_batch_size = world_size * local_batch_size
        input_resnet = torch.randn(global_batch_size, 3, 256, 256, dtype=torch.float32).to(rank)
        step_model(model, input_resnet, optimizer, rank)
        step_model(
            ddp_model,
            input_resnet[rank * local_batch_size: (rank + 1) * local_batch_size],
            optimizer_ddp,
            rank
        )
        if not null_hardware:
            print("do accuracy check.")
            for i, j in zip(model.parameters(), ddp_model.parameters()):
                result_grad = torch.allclose(i.grad, j.grad, rtol=1.3e-06, atol=5e-5)
                result_weight = torch.allclose(i, j, rtol=1.3e-06, atol=5e-5)
                print("result_grad:", result_grad)
                print("result_weight:", result_weight)
                if result_grad is False or result_weight is False:
                    print("rank:{}, i.grad:{},\nj.grad:{}".format(rank, i.grad.cpu().numpy(), j.grad.cpu().numpy()))
                    print("rank:{}, i:{},\nj:{}".format(rank, i.detach().cpu().numpy(), j.detach().cpu().numpy()))
                    raise Exception("result_grad and result weight should both be True")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--null_hardware", action="store_true", default=False, help="null_hardware for driver")
    parser.add_argument("--loop_num", type=int, default=1, help="loop num")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    args = parser.parse_args()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = selectPort("127.0.0.1", 22782)
    world_sizse = [2]
    batch_sizes = [args.batch_size]
    if int(args.loop_num) != -1:
        loops = range(int(args.loop_num))
    else:
        loops = repeat(None)
    idx = 0
    for _ in loops:
        print("loop: ", idx)
        for world_size, batch_size in itertools.product(world_sizse, batch_sizes):
            try:
                mp.spawn(ddp_program, args=(world_size, batch_size, args.null_hardware,), nprocs=world_size, join=True)
            except Exception as e:
                print("error msg:", e)
                print("ddp program runs failed.")
                exit(1)
        idx += 1

    print("ddp program runs successfully.")
    exit(0)
