import torch
import os

# the env var to set default memory format as NCHW
env_var = os.getenv("PYTORCH_DEFAULT_NCHW")
env_var_ndhwc = os.getenv("PYTORCH_DEFAULT_NDHWC")
env_var_nlc = os.getenv("PYTORCH_DEFAULT_NLC")

input = torch.randn(2,3,4,5).cuda()
if env_var and not input.is_contiguous():
    print("Error Raise: 4 dim tensor nchw to nhwc!")
    exit(1)
if not env_var and not input.is_contiguous(memory_format = torch.channels_last):
    print("Error Raise: 4 dim tensor nchw to nhwc!")
    exit(1)

input_1 = torch.randn(2,3,4,5,6).cuda()
if not env_var_ndhwc and not input_1.is_contiguous():
    print("Error Raise: 5 dim tensor nchw to nhwc!")
    exit(1)
if env_var_ndhwc and not input_1.is_contiguous(memory_format = torch.channels_last_3d):
    print("Error Raise: 5 dim tensor nchw to nhwc!")
    exit(1)

input_2 = torch.randn(2,1,4,5).cuda()
if not input_2.is_contiguous(memory_format = torch.channels_last) and not input_2.is_contiguous()\
        and not input_2.stride() == (20, 20, 5, 1):
    print("Error Raise: tensor with channels == 1 nchw to nhwc!")
    exit(1)
exit(0)

input_3 = torch.randn(2,3,4).cuda()
if env_var and (input_3.is_contiguous() or input_3.stride() != (12, 1, 3)):
    print("Error Raise: 3 dim tensor ncl to nlc!")
    exit(1)
if not env_var and not input_3.is_contiguous():
    print("Error Raise: 3 dim tensor ncl to nlc!")
    exit(1)
