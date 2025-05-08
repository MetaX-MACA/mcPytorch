import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import selectPort

def run(rank_id, world_size, backend):
    dist.init_process_group(backend, rank=rank_id, world_size=world_size)
    if backend == "nccl":
        tensor = torch.zeros(1).cuda(device=rank_id)
    elif backend == "gloo":
        tensor = torch.zeros(1)

    if rank_id == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
        print('after send, Rank ', rank_id, ' has data ', tensor[0])
        if(tensor[0] != 1):
            raise Exception(f"tensor[0] in rank {rank_id} and backend {backend} should equal to 1")
        dist.recv(tensor=tensor, src=1)
        print('after recv, Rank ', rank_id, ' has data ', tensor[0])
        if(tensor[0] != 2):
            raise Exception(f"tensor[0] in rank {rank_id} and backend {backend} should equal to 2")
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
        print('after recv, Rank ', rank_id, ' has data ', tensor[0])
        if(tensor[0] != 1):
            raise Exception(f"tensor[0] in rank {rank_id} and backend {backend} should equal to 1")
        tensor += 1
        dist.send(tensor=tensor, dst=0)
        print('after send, Rank ', rank_id, ' has data ', tensor[0])
        if(tensor[0] != 2):
            raise Exception(f"tensor[0] in rank {rank_id} and backend {backend} should equal to 2")


if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = '127.0.0.1'
    os.environ["MASTER_PORT"] = selectPort("127.0.0.1", 29112)
    world_size = 2
    backend_list = ["nccl", "gloo"]
    try:
        for backend in backend_list:
            mp.spawn(run, args=(world_size, backend, ), nprocs=world_size, join=True)
    except Exception as e:
        print("error msg:", e)
        print("test distributed send/recv program runs failed.")
        exit(1)