import os
import shutil
import torch
import torch.cuda

def pt_test():
    a = torch.randn(10,10, dtype = torch.float).cuda()
    b = torch.randn(1000,10000, dtype = torch.float).cuda()
    del a
    torch.cuda.empty_cache()

if __name__ == "__main__":
    os.environ["TORCH_GEN_MEM_TEST"]="1"
    for file_name in os.listdir():
        if file_name.startswith("memory_test_"):
            shutil.rmtree(file_name)
    pt_test()



