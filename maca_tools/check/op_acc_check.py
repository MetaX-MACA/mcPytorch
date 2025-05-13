import torch
import torch.nn as nn
import copy
from collections import OrderedDict

record_data = {}

def cal_rel_err(infer, golden):
    diff = infer - golden
    diff_square = diff * diff
    infer_result_square_double = 2 * infer * infer
    sum_diff_square = torch.sum(diff_square)
    sum_infer_result_square_double = torch.sum(infer_result_square_double)
    result = torch.sqrt(sum_diff_square / (sum_infer_result_square_double + 1e-12))
    return result


def peek(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return first


def get_tensor(input):
    output = []
    if isinstance(input, tuple):
            for i in input:
                if torch.is_tensor(i):
                    output = output.aapend(i)
    else:
        output = [input]


def check_forward_impl(eps=1e-3, check_bwd=True):
    def check_forward_hook(model, input, output):
        inputs = []
        if isinstance(input, tuple):
            for i in input:
                if torch.is_tensor(i):
                    if i.dtype == torch.float16 or i.dtype == torch.bfloat16 or i.dtype == torch.float:
                        inputs.append(i.clone().detach().float().cpu().requires_grad_(True))
                    else:
                        inputs.append(i.clone().detach().cpu())
                else:
                    print(".... forward skiped as input type not supported")
                    return
        elif torch.is_tensor(input):
            if input.dtype == torch.float16 or input.dtype == torch.bfloat16 or input.dtype == torch.float:
                inputs = [input.clone().detach().float().cpu().requires_grad_(True)]
            else:
                inputs = [input.clone().detach().cpu()]
        else:
            print("..... forward skiped as input type not supported")
            return 
        
        if torch.is_tensor(output):
            if output.dtype == torch.float16 or output.dtype == torch.bfloat16 or output.dtype == torch.float:
                output = output.clone().detach().float().cpu()
            else:
                output = output.clone().detach().cpu()
        else:
            print("..... forward skiped as output type not supported")
            return

        # check model
        if peek(model.parameters()) is not None:
            model_cpu = copy.deepcopy(model).to(torch.float32).cpu()
        else:
            model_cpu = model
        # remove forward hook
        dict_items = list(model._forward_hooks.items())
        model_cpu._forward_hooks = OrderedDict([(i, fn) for i, fn in dict_items if fn.__name__ != "check_forward_hook"])
        if model.training:
            model_cpu.train()
        else:
            model_cpu.eval()
        if hasattr(model_cpu, "inplace"):
            model_cpu.inplace = False
        golden = model_cpu(*inputs)
        
        if check_bwd:
            record_data[model] = {}
            record_data[model]["inputs"] = inputs
            record_data[model]["output"] = golden

        rel_err = cal_rel_err(output.float(), golden.float())
        mark = "error" if rel_err > eps else "     "
        print(f"{mark} forward rel_err: {rel_err.item()}, input shape: {inputs[0].shape}, op: {model}, input max : {torch.max(inputs[0]).item()}, input min {torch.min(inputs[0]).item()}, output max: {torch.max(output).item()}, output min: {torch.min(output).item()}")

    return check_forward_hook


def check_backward_impl(eps=1e-3):
    r_grad = None
    def get_cpu_grad(model, input, output):
        global r_grad
        r_grad = input

    def check_backward_hook(model, input, output):
        # find model in tensor_dict
        if model not in record_data.keys():
            print(f"..... skipped as model not exist in record_data: {model}")
            return
        r_inputs =  record_data[model]["inputs"]
        # r_inputs = [Variable(i.data, requires_grad=True) for i in r_inputs]
        if peek(model.parameters()) is not None:
            r_model = copy.deepcopy(model).to(torch.float32).cpu()
        else:
            r_model = model
        r_model.train()

        #remove hook
        dict_items = list(r_model._forward_hooks.items())
        r_model._forward_hooks = OrderedDict([(i, fn) for i, fn in dict_items if fn.__name__ != "check_forward_hook"])
        dict_items = list(r_model._backward_hooks.items())
        r_model._backward_hooks = OrderedDict([(i, fn) for i, fn in dict_items if fn.__name__ != "check_backward_hook"])

        # cal backward
        r_model.register_backward_hook(get_cpu_grad)
        grad_input = output[0].clone().detach().float().cpu()
        with torch.enable_grad():
            r_output = r_model(*r_inputs)
            r_output.backward(grad_input)
        global r_grad

        assert len(input) == len(r_grad), f"len of current grads({len(input)}) not match with record grads({len(r_grad)})"
        for r_g, g in zip(r_grad, input):
            # assert r_g.shape == g.shape, f"shape of current grads({g.shape}) not match with record grads({r_g.shape})"
            if r_g == None and g == None:
                continue
            if r_g == None or g == None:
                shp = r_g.shape if r_g != None else g.shape
                print(f"..... backward skip as tensor is None in {model}: {shp} r_g is tensor: {torch.is_tensor(r_g)}, g is tensor: {torch.is_tensor(g)}")
                continue
            if r_g.shape == g.shape:
                rel_err = cal_rel_err(g.float().cpu(), r_g.float())
                mark = "error" if rel_err > eps else "     "
                print(f"{mark} backward rel_err: {rel_err.item()}, grad shape: {r_g.shape}, op: {model} grad max : {torch.max(g).item()}, min {torch.min(g).item()}")
            else:
                print(f"..... backward skip as shape not match in {model}: record shape{r_g.shape} != current shape{g.shape}")

    return check_backward_hook


def disable_inplace_(op):
    if hasattr(op, "inplace"):
        op.inplace = False

def check_forward(op):
    if isinstance(op, nn.ReLU) or isinstance(op, nn.Conv2d) or isinstance(op, nn.BatchNorm2d) or isinstance(op, nn.MaxPool2d) or isinstance(op, nn.Linear):
    # if isinstance(op, nn.Conv2d) or isinstance(op, nn.SyncBatchNorm):
    # if isinstance(op, nn.ReLU): 
    # if isinstance(op, RoIAlign) or isinstance(op, nn.MaxPool2d) or isinstance(op, nn.Linear):
    # if isinstance(op, Embedding1D) or isinstance(op, RotaryEmbedding) or isinstance(op, ColumnParallelLinearTorch) or isinstance(op, RowParallelLinearTorch) or isinstance(op, ScaleColumnParallelLinear) or isinstance(op, RMSNormTorch) or isinstance(op, nn.Dropout2d) or isinstance(op, RotaryEmbedding):
    # if isinstance(op, ColumnParallelLinearTorch) or isinstance(op, RowParallelLinearTorch) or isinstance(op, ScaleColumnParallelLinear) or isinstance(op, RMSNormTorch) or isinstance(op, nn.Dropout2d):
        op.register_forward_hook(check_forward_impl())


def check_backward(op):
    if isinstance(op, nn.ReLU) or isinstance(op, nn.Conv2d) or isinstance(op, nn.SyncBatchNorm):
    # if isinstance(op, nn.Conv2d) or isinstance(op, nn.SyncBatchNorm) or isinstance(op, RoIAlign):
    # if isinstance(op, Embedding1D) or isinstance(op, RotaryEmbedding) or isinstance(op, ColumnParallelLinearTorch) or isinstance(op, RowParallelLinearTorch) or isinstance(op, ScaleColumnParallelLinear) or isinstance(op, RMSNormTorch) or isinstance(op, nn.Dropout2d) or isinstance(op, RotaryEmbedding):
    # if isinstance(op, nn.ReLU):
        op.register_backward_hook(check_backward_impl())


def check_forward_acc(model, disable_inplace=False):
    if disable_inplace:
        model.apply(disable_inplace_)
    model.apply(check_forward)


def check_forward_and_backward_acc(model, disable_inplace=False):
    if disable_inplace:
        model.apply(disable_inplace_)
    model.apply(check_forward)
    model.apply(check_backward)