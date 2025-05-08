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
import random
import sys
sys.path.insert(0, "../maca_samples/")
from bert.modeling import BertConfig, BertModel

def ids_tensor(shape, vocab_size):
    """
    creates a random int32 tensor of the shape within the vocab size.
    shape: shape of the tensor to created
    vocab_size: length of the vocab
    """
    random.seed(0)

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(random.randint(0, vocab_size - 1))
    return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()

def step_model(model, input_ids, token_type_ids, position_ids, optimizer, rank):
    '''
    calculate backward and update parameter value
    '''
    model.train()
    encoded_layers, pooled_output = model(input_ids, token_type_ids, position_ids)
    encode_layers_labels = torch.zeros(encoded_layers[-1].shape, dtype=torch.float32).to(rank)
    pooled_output_labels = torch.zeros(pooled_output.shape, dtype=torch.float32).to(rank)
    loss = F.mse_loss(encoded_layers[-1], encode_layers_labels) + F.mse_loss(pooled_output, pooled_output_labels)
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
    # create local model and set hidden_dropout_prob and attention_probs_dropout_prob
    # to 0.0 to make sure same behavior between model and ddp model.
    config = BertConfig(
        vocab_size_or_config_json_file=30522,
        hidden_size=768,
        num_hidden_layers=1,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02)
    device = torch.device("cuda:%d" % local_rank)
    model = BertModel(config=config)
    model.to(device)
    # construct DDP model
    ddp_model = DDP(copy.deepcopy(model).to(device), device_ids=[device])
    optimizer = optim.SGD(model.parameters(), lr=1)
    optimizer_ddp = optim.SGD(ddp_model.parameters(), lr=1)

    for iteration in range(1):
        print("iteration:", iteration)
        global_batch_size = world_size * local_batch_size
        input_ids = ids_tensor([global_batch_size, 128], 30522).to(local_rank)
        token_type_ids = ids_tensor([global_batch_size, 128], 2).to(local_rank)
        position_ids = ids_tensor([global_batch_size, 128], 2).to(local_rank)
        step_model(model, input_ids, token_type_ids, position_ids, optimizer, local_rank)
        step_model(
            ddp_model,
            input_ids[rank * local_batch_size: (rank + 1) * local_batch_size],
            token_type_ids[rank * local_batch_size: (rank + 1) * local_batch_size],
            position_ids[rank * local_batch_size: (rank + 1) * local_batch_size],
            optimizer_ddp,
            local_rank
        )
        for i, j in zip(model.parameters(), ddp_model.parameters()):
            print("local_rank:{}, i.grad:{},\nj.grad:{}".format(local_rank, i.grad, j.grad))
            print("local_rank:{}, i:{},\nj:{}".format(local_rank, i, j))
            result_grad = torch.allclose(i.grad, j.grad, rtol=1.3e-06, atol=5e-5)
            result_weight = torch.allclose(i, j, rtol=1.3e-06, atol=5e-5)
            if result_grad is False or result_weight is False:
                raise Exception("result_grad and result weight should both be True")


if __name__ == "__main__":
    #single node launch command:python -m torch.distributed.launch --nproc_per_node=4 --master_port=55566 test_ddp_with_mccl_bert_torch_launch.py
    #multi node launch command1:python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="172.24.0.49" --master_port=55566 test_ddp_with_mccl_bert_torch_launch.py
    #multi node launch command2:python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="172.24.0.49" --master_port=55566 test_ddp_with_mccl_bert_torch_launch.py
    try:
        ddp_program(1)
    except:
        print("ddp program runs failed.")
        exit(1)
    print("ddp program runs successfully.")
    exit(0)