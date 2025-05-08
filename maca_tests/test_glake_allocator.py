import torch
import os

a = torch.randn(100000000, device="cuda")
b = torch.randn(100000000, device="cuda")
c = torch.randn(100000000, device="cuda")
d = torch.randn(100000000, device="cuda")
del a
del b
del c
del d
e = torch.randn(400000000, device="cuda")
print("allocated: ", torch.cuda.memory_allocated())
print("cached : ", torch.cuda.memory_reserved())
ratio = torch.cuda.memory_allocated() / torch.cuda.memory_reserved()
if "vmmDefragment" in os.environ and os.environ["vmmDefragment"] == '1':
    if ratio <= 0.6:
        print("disable glake error!")
        exit(1)
else:
    if ratio > 0.6:
        print("disable glake error!")
        exit(1)
exit(0)
