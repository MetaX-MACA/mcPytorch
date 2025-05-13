import torch
import torch.nn as nn
import copy


# only support input type: (tensor, ) or tensor
def get_tensor(input):
    output = []
    if isinstance(input, tuple):
            for i in input:
                if torch.is_tensor(i):
                    output.append(i)
    else:
        if torch.is_tensor(input):
            output = [input]
    return output


def is_abnormal(x):
    if x.isnan().any() == False and x.isinf().any() == False:
        return False
    else:
        return True
        

def check(mode="forward"):
    def check_hook(model, input, output):
        inputs = get_tensor(input)
        outputs = get_tensor(output)

        for i in inputs:
            if not is_abnormal(i):
                info = f"check pass model: {model} {i.device} {mode} input: {i.shape}, max: {torch.max(i)}, min: {torch.min(i)}, sum: {torch.sum(i.double())}"
            else:
                info = f"check error model: {model} {i.device} {mode} input: {i.shape}, max: {torch.max(i)}, min: {torch.min(i)}, sum: {torch.sum(i.double())}"
            print(info, flush=True)

        for o in outputs:
            print(f"model: {model}, {mode} output: {o.shape}")
            if not is_abnormal(o):
                info = f"check pass model: {model} {o.device} {mode} output: {o.shape}, max: {torch.max(o)}, min: {torch.min(o)}, sum: {torch.sum(o.double())}"
            else:
                info = f"check error model: {model} {o.device} {mode} output: {o.shape}, max: {torch.max(o)}, min: {torch.min(o)}, sum: {torch.sum(o.double())}"
            print(info, flush=True)

        for p in model.parameters():
            if torch.is_tensor(p):
                if not is_abnormal(p):
                    info = f"check pass model: {model} {p.device} {mode} weight: {p.shape}, max: {torch.max(p)}, min: {torch.min(p)}, sum: {torch.sum(p.double())}"
                else:
                    info = f"check error model: {model} {p.device} {mode} weight: {p.shape}, max: {torch.max(p)}, min: {torch.min(p)},  sum: {torch.sum(p.double())}"
                print(info, flush=True)

    return check_hook


def check_forward_abnormal(op):
        op.register_forward_hook(check(mode="forward"))


def check_backward_abnormal(op):
    # if isinstance(op, nn.ReLU) or isinstance(op, nn.Conv2d) or isinstance(op, nn.SyncBatchNorm) or isinstance(op, nn.AdaptiveAvgPool2d) or isinstance(op, nn.Dropout2d):
        op.register_backward_hook(check(mode="backward"))


def abnormal_val_check(model):
    model.apply(check_forward_abnormal)
    model.apply(check_backward_abnormal)