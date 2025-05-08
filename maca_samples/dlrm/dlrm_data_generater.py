import torch
import numpy as np
from numpy import random as ra
from torch.utils.data import Dataset
import configs.config_dlrm as config_dlrm


# uniform ditribution (input data)
class RandomDataset(Dataset):

    def __init__(
            self,
            m_den,
            ln_emb,
            mini_batch_size,
            data_size=1,
            num_batches=1000,
            num_indices_per_lookup=100,
            num_indices_per_lookup_fixed=False,
            num_targets=1,
            round_targets=False,
            data_generation="random",
            trace_file="",
            enable_padding=False,
            reset_seed_on_access=True,
            rand_data_dist="uniform",
            rand_data_min=0,
            rand_data_max=1,
            rand_data_mu=-1,
            rand_data_sigma=1,
            rand_seed=config_dlrm.rand_seed
    ):
        # compute batch size
        nbatches = int(np.ceil((data_size * 1.0) / mini_batch_size))
        if num_batches != 0:
            nbatches = num_batches
            data_size = nbatches * mini_batch_size
            # print("Total number of batches %d" % nbatches)

        # save args (recompute data_size if needed)
        self.m_den = m_den
        self.ln_emb = ln_emb
        self.data_size = data_size
        self.num_batches = nbatches
        self.mini_batch_size = mini_batch_size
        self.num_indices_per_lookup = num_indices_per_lookup
        self.num_indices_per_lookup_fixed = num_indices_per_lookup_fixed
        self.num_targets = num_targets
        self.round_targets = round_targets
        self.data_generation = data_generation
        self.trace_file = trace_file
        self.enable_padding = enable_padding
        self.reset_seed_on_access = reset_seed_on_access
        self.rand_seed = rand_seed
        self.rand_data_dist = rand_data_dist
        self.rand_data_min = rand_data_min
        self.rand_data_max = rand_data_max
        self.rand_data_mu = rand_data_mu
        self.rand_data_sigma = rand_data_sigma

    def reset_numpy_seed(self, numpy_rand_seed):
        np.random.seed(numpy_rand_seed)
        # torch.manual_seed(numpy_rand_seed)

    def __getitem__(self, index):

        if isinstance(index, slice):
            return [
                self[idx] for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]

        # WARNING: reset seed on access to first element
        # (e.g. if same random samples needed across epochs)
        if self.reset_seed_on_access and index == 0:
            self.reset_numpy_seed(self.rand_seed)

        # number of data points in a batch
        n = min(self.mini_batch_size, self.data_size - (index * self.mini_batch_size))

        # generate a batch of dense and sparse features
        (X, lS_o, lS_i) = generate_dist_input_batch(
            self.m_den,
            self.ln_emb,
            n,
            self.num_indices_per_lookup,
            self.num_indices_per_lookup_fixed,
            rand_data_dist=self.rand_data_dist,
            rand_data_min=self.rand_data_min,
            rand_data_max=self.rand_data_max,
            rand_data_mu=self.rand_data_mu,
            rand_data_sigma=self.rand_data_sigma,
        )

        # generate a batch of target (probability of a click)
        T = generate_random_output_batch(n, self.num_targets, self.round_targets)
        return (X, lS_o, lS_i, T)

    def __len__(self):
        # WARNING: note that we produce bacthes of outputs in __getitem__
        # therefore we should use num_batches rather than data_size below
        return self.num_batches


def make_random_data_and_loader(batch_size, ln_emb, m_den):

    train_data = RandomDataset(
        m_den,
        ln_emb,
        batch_size,
    )  # WARNING: generates a batch of lookups at once

    collate_wrapper_random = collate_wrapper_random_offset

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_wrapper_random,
        pin_memory=False,
        drop_last=False,  # True
    )
    return train_loader


# random data from uniform or gaussian ditribution (input data)
def generate_dist_input_batch(
    m_den,
    ln_emb,
    n,
    num_indices_per_lookup,
    num_indices_per_lookup_fixed,
    rand_data_dist,
    rand_data_min,
    rand_data_max,
    rand_data_mu,
    rand_data_sigma,
):
    # dense feature
    Xt = torch.tensor(ra.rand(n, m_den).astype(np.float32))

    # sparse feature (sparse indices)
    lS_emb_offsets = []
    lS_emb_indices = []
    # for each embedding generate a list of n lookups,
    # where each lookup is composed of multiple sparse indices
    for size in ln_emb:
        lS_batch_offsets = []
        lS_batch_indices = []
        offset = 0
        for _ in range(n):
            # num of sparse indices to be used per embedding (between
            if num_indices_per_lookup_fixed:
                sparse_group_size = np.int64(num_indices_per_lookup)
            else:
                # random between [1,num_indices_per_lookup])
                r = ra.random(1)
                sparse_group_size = np.int64(
                    np.round(max([1.0], r * min(size, num_indices_per_lookup)))
                )
            # sparse indices to be used per embedding
            if rand_data_dist == "gaussian":
                if rand_data_mu == -1:
                    rand_data_mu = (rand_data_max + rand_data_min) / 2.0
                r = ra.normal(rand_data_mu, rand_data_sigma, sparse_group_size)
                sparse_group = np.clip(r, rand_data_min, rand_data_max)
                sparse_group = np.unique(sparse_group).astype(np.int64)
            elif rand_data_dist == "uniform":
                r = ra.random(sparse_group_size)
                sparse_group = np.unique(np.round(r * (size - 1)).astype(np.int64))
            else:
                raise(rand_data_dist, "distribution is not supported. \
                     please select uniform or gaussian")

            # reset sparse_group_size in case some index duplicates were removed
            sparse_group_size = np.int64(sparse_group.size)
            # store lengths and indices
            lS_batch_offsets += [offset]
            lS_batch_indices += sparse_group.tolist()
            # update offset for next iteration
            offset += sparse_group_size
        lS_emb_offsets.append(torch.tensor(lS_batch_offsets))
        lS_emb_indices.append(torch.tensor(lS_batch_indices))

    return (Xt, lS_emb_offsets, lS_emb_indices)


def generate_random_output_batch(n, num_targets, round_targets=False):
    # target (probability of a click)
    if round_targets:
        P = np.round(ra.rand(n, num_targets).astype(np.float32)).astype(np.float32)
    else:
        P = ra.rand(n, num_targets).astype(np.float32)

    return torch.tensor(P)


def collate_wrapper_random_offset(list_of_tuples):
    # where each tuple is (X, lS_o, lS_i, T)
    (X, lS_o, lS_i, T) = list_of_tuples[0]
    return (X,
            torch.stack(lS_o),
            lS_i,
            T)


def collate_wrapper_random_length(list_of_tuples):
    # where each tuple is (X, lS_o, lS_i, T)
    (X, lS_o, lS_i, T) = list_of_tuples[0]
    return (X,
            offset_to_length_converter(torch.stack(lS_o), lS_i),
            lS_i,
            T)


# Conversion from offset to length
def offset_to_length_converter(lS_o, lS_i):
    def diff(tensor):
        return tensor[1:] - tensor[:-1]

    return torch.stack(
        [
            diff(torch.cat((S_o, torch.tensor(lS_i[ind].shape))).int())
            for ind, S_o in enumerate(lS_o)
        ]
    )
