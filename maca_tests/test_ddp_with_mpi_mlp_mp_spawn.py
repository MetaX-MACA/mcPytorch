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
import numpy as np
from utils import selectPort
np.set_printoptions(threshold=np.inf)

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


def ddp_program(rank, world_size, local_batch_size, backend):
    '''
    program to verify ddp runs correctly
    '''
    # create default process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    # create local model
    model = Net()
    device = torch.device("cuda:%d" % rank)
    # construct DDP model
    ddp_model = DDP(copy.deepcopy(model).to(device), device_ids=[device])
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=1)
    optimizer_ddp = optim.SGD(ddp_model.parameters(), lr=1)

    for iteration in range(1):
        print("iteration:", iteration)
        global_batch_size = world_size * local_batch_size
        input_data = torch.randn(global_batch_size, 2).to(rank)
        target_label = torch.randn(global_batch_size, 4).to(rank)
        step_model(model, input_data, target_label, optimizer)
        step_model(
            ddp_model,
            input_data[rank * local_batch_size: (rank + 1) * local_batch_size],
            target_label[rank * local_batch_size: (rank + 1) * local_batch_size],
            optimizer_ddp
        )
        for i, j in zip(model.parameters(), ddp_model.parameters()):
            result_grad = torch.allclose(i.grad, j.grad, rtol=1.3e-06, atol=5e-5)
            result_weight = torch.allclose(i, j, rtol=1.3e-06, atol=5e-5)
            print("result_grad:", result_grad)
            print("result_weight:", result_weight)
            if result_grad is False or result_weight is False:
                print("rank:{}, i.grad:{},\nj.grad:{}".format(rank, i.grad.cpu().numpy(), j.grad.cpu().numpy()))
                print("rank:{}, i:{},\nj:{}".format(rank, i.detach().cpu().numpy(), j.detach().cpu().numpy()))
                raise Exception("result_grad and result weight should both be True")

#command:/opt/maca/ompi/bin/mpirun -np 2 python test_ddp_with_mpi_mlp_mp_spawn.py
if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = selectPort("127.0.0.1", 29110)
    batch_size = 2
    backend = "mpi"
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])

    try:
        ddp_program(world_rank, world_size, batch_size, backend)
    except Exception as e:
        print("error msg:", e)
        print("ddp program runs failed.")
        exit(1)
    print("ddp program runs successfully.")
    exit(0)

