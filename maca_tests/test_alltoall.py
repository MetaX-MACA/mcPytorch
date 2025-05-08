import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from utils import selectPort

def alltoall_program(rank, world_size):
    '''
    program to verify alltoall_program runs correctly
    '''
    torch.manual_seed(0)
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    #check dist.all_to_all_single with even split
    input = torch.arange(2).cuda(rank) + rank * 2
    output = torch.empty([2], dtype=torch.int64).cuda(rank)
    dist.all_to_all_single(output, input)
    print(f"output:{rank}:", output)
    if rank == 0:
        assert(torch.allclose(output.cpu(), torch.tensor([0, 2])))
    if rank == 1:
        assert(torch.allclose(output.cpu(), torch.tensor([1, 3])))

    #check dist.all_to_all_single with uneven split
    if rank == 0:
        input = torch.tensor([0, 1, 2, 3, 4, 5]).cuda(rank)
        input_splits = [3, 3]
        output_splits = [3, 4]
        output = torch.empty([7], dtype=torch.int64).cuda(rank)
    else:
        input = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18]).cuda(rank)
        input_splits = [4, 5]
        output_splits = [3, 5]
        output = torch.empty([8], dtype=torch.int64).cuda(rank)

    dist.all_to_all_single(output, input, output_splits, input_splits)
    print(f"output:{rank}:", output)
    if rank == 0:
        assert(torch.allclose(output.cpu(), torch.tensor([0, 1, 2, 10, 11, 12, 13])))
    if rank == 1:
        assert(torch.allclose(output.cpu(), torch.tensor([3, 4, 5, 14, 15, 16, 17, 18])))

    #check dist.all_to_all with same tensor size
    input = torch.arange(2).cuda(rank) + rank * 2
    input = list(input.chunk(2))
    output = list(torch.empty([2], dtype=torch.int64).cuda(rank).chunk(2))
    dist.all_to_all(output, input)
    print(f"output:{rank}:", output)
    if rank == 0:
        assert(torch.allclose(output[0].cpu(), torch.tensor([0])))
        assert(torch.allclose(output[1].cpu(), torch.tensor([2])))
    if rank == 1:
        assert(torch.allclose(output[0].cpu(), torch.tensor([1])))
        assert(torch.allclose(output[1].cpu(), torch.tensor([3])))

    #check dist.all_to_all with differentss tensor size
    input = torch.arange(2).cuda(rank) + rank * 2
    input = list(input.chunk(2))
    if rank == 0:
        input = [torch.tensor([0, 1]).cuda(rank), torch.tensor([2, 3, 4, 5]).cuda(rank)]
        output = [torch.empty([2], dtype=torch.int64).cuda(rank), torch.empty([3], dtype=torch.int64).cuda(rank)]
    else:
        input = [torch.tensor([10, 11, 12]).cuda(rank), torch.tensor([13, 14, 15, 16, 17]).cuda(rank)]
        output = [torch.empty([4], dtype=torch.int64).cuda(rank), torch.empty([5], dtype=torch.int64).cuda(rank)]

    dist.all_to_all(output, input)
    print(f"output:{rank}:", output)
    if rank == 0:
        assert(torch.allclose(output[0].cpu(), torch.tensor([0, 1])))
        assert(torch.allclose(output[1].cpu(), torch.tensor([10, 11, 12])))
    if rank == 1:
        assert(torch.allclose(output[0].cpu(), torch.tensor([2, 3, 4, 5])))
        assert(torch.allclose(output[1].cpu(), torch.tensor([13, 14, 15, 16, 17])))



if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = selectPort("127.0.0.1", 27552)
    world_sizse = [2]
    for world_size in world_sizse:
        try:
            mp.spawn(alltoall_program, args=(world_size,), nprocs=world_size, join=True)
        except Exception as e:
            print("error msg:", e)
            print("alltoall_program runs failed.")
            exit(1)
    print("alltoall_program runs successfully.")
    exit(0)