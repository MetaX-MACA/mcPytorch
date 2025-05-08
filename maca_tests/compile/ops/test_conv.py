import model
import torch
import torch.nn as nn
import logging
import torch._dynamo
import torch._inductor
import torch._inductor.config
import os, sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import check_close

torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.verbose = True
torch._inductor.config.debug = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.triton.exclude_aten = True


conv_confs = [
    {
        "BATCH": BATCH,
        "IN_H": IN_H,
        "IN_W": IN_W,
        "IN_C": IN_C,
        "KERNEL_N": KERNEL_N,
        "KERNEL_H": KERNEL_H,
        "KERNEL_W": KERNEL_W,
        "stride": stride,
        "padding": padding,
    }
    for i, (
        IN_H,
        IN_W,
        IN_C,
        KERNEL_H,
        KERNEL_W,
        KERNEL_N,
        stride,
        padding,
    ) in enumerate(model.resnet50_layers)
    for BATCH in [32]
]

def run_conv_accuracy(conv_config):
    conv = nn.Conv2d(conv_config["IN_C"], conv_config["KERNEL_N"], 
                     (conv_config["KERNEL_H"], conv_config["KERNEL_W"]),
                       stride=conv_config["stride"], padding=conv_config["padding"]).cuda()
    def foo(input):
        x = conv(input)
        return x
    new_foo = torch.compile(foo, backend="inductor", mode="max-autotune")
    input = torch.randn(conv_config["BATCH"], conv_config["IN_C"],
                         conv_config["IN_H"], conv_config["IN_W"]).cuda()
    inductor_result = new_foo(input)
    foo_result = foo(input)
    return check_close(inductor_result, foo_result)

if __name__ == "__main__":
    for conv_conf in conv_confs:
        if run_conv_accuracy(conv_conf):
            print("{} passed".format(conv_conf))
        else:
            print("{} failed".format(conv_conf))
            break





