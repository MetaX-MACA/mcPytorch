import torch
import numpy as np
import itertools
from torch.testing._internal.common_utils import random_hermitian_pd_matrix
device = "cuda"
dtype = torch.float32
def cholesky(A):
    expected_L = np.linalg.cholesky(A.cpu().numpy())
    expected_info = torch.zeros(A.shape[:-2], dtype=torch.int32, device=device)
    actual_L  = 0
    actual_L, actual_info = torch.linalg.cholesky_ex(A)
    # print("expected_L:", expected_L)
    # print("actual_L:", actual_L)
    if not torch.allclose(torch.from_numpy(expected_L), actual_L.cpu()):
        print("-----------------Error in test sparse!")
        return False
    if A.numel() > 0 :
        expected_norm = np.linalg.norm(expected_L, ord=1, axis=(-2, -1))
        actual_norm = torch.linalg.norm(actual_L, ord=1, axis=(-2, -1))
        # print("expected_norm:", expected_norm)
        # print("actual_norm:", actual_norm)
        if not torch.allclose(torch.tensor(expected_norm), actual_norm.cpu()):
            print("-----------------Error in test sparse!")
            return False
    return True

def run_test(n, batch):
    print("n, batch: ", n, batch)
    A = random_hermitian_pd_matrix(n, *batch, dtype=dtype, device=device)
    return cholesky(A)


if __name__ == "__main__":
    ns = (3, )
    batches = ((), )
    for n, batch in itertools.product(ns, batches):
        if not run_test(n, batch):
            exit(1)
    exit(0)