import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import copy
import itertools


class Net(nn.Module):
    '''
    Network to verify ddp function
    '''

    def __init__(self):
        super(Net, self).__init__()
        torch.manual_seed(0)
        self.fc1 = nn.Linear(2, 4, bias=False)
        self.fc2 = nn.Linear(4, 6, bias=False)
        self.fc3 = nn.Linear(6, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


def step_model(model, input, target, optimizer=None):
    '''
    calculate backward and update parameter value
    '''
    model.train()
    output = model(input)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()


def ddp_program(local_batch_size):
    '''
    program to verify ddp runs correctly
    '''
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    model = Net()
    device = torch.device("cuda:%d" % local_rank)
    # construct DDP model
    ddp_model = DDP(copy.deepcopy(model).to(device), device_ids=[device])
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=1)
    optimizer_ddp = optim.SGD(ddp_model.parameters(), lr=1)

    for iteration in range(2):
        print("iteration:", iteration)
        global_batch_size = world_size * local_batch_size
        input_data = torch.randn(global_batch_size, 2).to(local_rank)
        target_label = torch.randn(global_batch_size, 4).to(local_rank)
        step_model(model, input_data, target_label, optimizer)
        step_model(
            ddp_model,
            input_data[rank * local_batch_size: (rank + 1) * local_batch_size],
            target_label[rank * local_batch_size: (rank + 1) * local_batch_size],
            optimizer_ddp
        )
        for i, j in zip(model.parameters(), ddp_model.parameters()):
            print("local_rank:{}, i.grad:{},\nj.grad:{}".format(local_rank, i.grad, j.grad))
            print("local_rank:{}, i:{},\nj:{}".format(local_rank, i, j))
            result_grad = torch.allclose(i.grad, j.grad, rtol=1.3e-06, atol=5e-5)
            result_weight = torch.allclose(i, j, rtol=1.3e-06, atol=5e-5)
            if result_grad is False or result_weight is False:
                raise Exception("result_grad and result weight should both be True")


if __name__ == "__main__":
    #single node launch command:python -m torch.distributed.launch --nproc_per_node=4 --master_port=55546 test_ddp_with_mccl_mlp_torch_launch.py
    #multi node launch command1:python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="172.24.0.49" --master_port=55546 test_ddp_with_mccl_mlp_torch_launch.py
    #multi node launch command2:python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="172.24.0.49" --master_port=55546 test_ddp_with_mccl_mlp_torch_launch.py
    try:
        ddp_program(1)
    except:
        print("ddp program runs failed.")
        exit(1)
    print("ddp program runs successfully.")
    exit(0)
