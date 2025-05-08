import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as fsdp
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
import os
import copy
import itertools
import random
import numpy as np
import sys
from functools import partial
from utils import selectPort
sys.path.insert(0, "../maca_samples/")
from bert.modeling import BertConfig, BertModel
np.set_printoptions(threshold=np.inf)
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


def fsdp_program(rank, world_size, local_batch_size):
    '''
    program to verify fsdp runs correctly
    '''
    torch.manual_seed(0)
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    # create local model and set hidden_dropout_prob and attention_probs_dropout_prob
    # to 0.0 to make sure same behavior between model and fsdp model.
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

    model = BertModel(config=config)
    model.to(rank)
    # construct FSDP model
    fsdp_model = fsdp(
        cpu_offload=CPUOffload(offload_params=True),
        module=copy.deepcopy(model),
        device_id=rank,
        auto_wrap_policy=partial(
            size_based_auto_wrap_policy,
            min_num_params=200)
        )
    optimizer_fsdp = optim.SGD(fsdp_model.parameters(), lr=1)
    max_memory_allocated = 0
    for iteration in range(1000):
        input_ids = ids_tensor([local_batch_size, 128], 30522).to(rank)
        token_type_ids = ids_tensor([local_batch_size, 128], 2).to(rank)
        position_ids = ids_tensor([local_batch_size, 128], 2).to(rank)
        step_model(
            fsdp_model,
            input_ids,
            token_type_ids,
            position_ids,
            optimizer_fsdp,
            rank
        )
        if iteration <= 10:
            max_memory_allocated = max(max_memory_allocated, torch.cuda.max_memory_reserved())
        else:
            memory = torch.cuda.max_memory_reserved()
            if memory > max_memory_allocated:
                print("memory:", memory)
                print("max_memory_allocated:", max_memory_allocated)
                exit(-1)
    dist.destroy_process_group()

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = selectPort("127.0.0.1", 27552)
    world_sizse = [2]
    batch_sizes = [16]
    for world_size, batch_size in itertools.product(world_sizse, batch_sizes):
        try:
            mp.spawn(fsdp_program, args=(world_size, batch_size,), nprocs=world_size, join=True)
        except Exception as e:
            print("error msg:", e)
            print("fsdp program runs failed.")
            exit(1)
    print("fsdp program runs successfully.")
    exit(0)
