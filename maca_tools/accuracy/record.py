import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _rebuild_tensor

import functools
import warnings
import os
import sys
import importlib
import shutil
import copyreg
import dill
import traceback
import pickle
from contextlib import contextmanager
import inspect
import copy

from .ops import *
# from .utils import add_tensor_info, TensorSummary, get_min_val, get_max_val, get_log_name, peek, cal_rel_err
from .utils import *


"""
level = 2
The default record level, all the inputs and outputs including the whole tensors of ops
are allowed to be recorded.

level = 1
All the inputs and outputs including tensor summary info of ops are allowed to be recorded.

level = 0
Do not allowed to record the inputs and outputs of ops.

"""

LOG_NAME = get_log_name()
eps = 1e-3
record_cpu_data = {}

_ctx = None


def get_context():
    return _ctx

def clear_context():
    global _ctx
    if _ctx:
        recover_hijack()
    _ctx = None
    return

class Context(object):
    def __init__(
            self,
            enabled,
            output_dir,
            record_level,
            record_stack,
            record_input,
            record_output,
            op_range,
            op_list,
            skip):
        self._enabled = enabled
        self._record_input = record_input
        self._record_output = record_output
        self._op_range = op_range
        self._op_list = [api.lower() for api in op_list]
        self._pid = os.getpid()
        self._is_in_range = False if op_range else True
        self._first_in_range = False
        self._is_in_list = False if op_list else True
        self._skip = skip + 1
        self._cnt = 0
        self._record_files = []
        self._files_seqs_path = os.path.join(output_dir, "files_seqs.pt")
        self.output_dir = output_dir
        self.fwd_cnt = 0
        self.wrapped_depth = 0
        self.record_level = record_level
        self.record_stack = record_stack

    def set_enabled(self, enabled):
        self._enabled = enabled

    @property
    def enabled(self):
        is_main_process = self._pid == os.getpid()
        if is_main_process:
            return self._enabled
        elif self._enabled:
            warnings.warn("recording in subprocess is not supported, "
                "because this behavior will cause the sequence numbers to be messed up.")
        return False

    def check_record_input(self, op_name, wrapped_depth = None):
        if self.enabled and self._record_input:
            return self.check_record_helper(op_name, wrapped_depth)
        return False

    def check_record_output(self, op_name, wrapped_depth = None):
        if self.enabled and self._record_output:
            return self.check_record_helper(op_name, wrapped_depth)
        return False

    def check_record_helper(self, op_name, wrapped_depth):
        depth = wrapped_depth if wrapped_depth else _ctx.wrapped_depth
        res = (depth == 1 or (self._op_list and self._is_in_list)) \
            and self._is_in_range and self._is_in_list
        if res and self._skip > 1:
            self._cnt += 1
            return res and (self._cnt % self._skip == 0)
        return res

    def check_in_list(self, op_name):
        if not (self.enabled and self._op_list):
            return
        for api in self._op_list:
            if api in op_name.lower():
                self._is_in_list = True
                return
        self._is_in_list = False
        return

    def check_in_range(self, op_name):
        if not (self.enabled and self._op_range):
            return
        if self._op_range[0] == self._op_range[1]:
            if self._op_range[0] is None:
                self._is_in_range = True
            return
        if not self._is_in_range and (op_name == self._op_range[0] \
            or (not self._first_in_range and self._op_range[0] is None)):
            self._is_in_range = True
            self._first_in_range = True
        elif self._is_in_range and (op_name == self._op_range[1] and self._op_range[1]):
            self._is_in_range = False
        return

    def check_in_user_scope(self, op_name):
        self.check_in_range(op_name)
        self.check_in_list(op_name)

    def record_record_file(self, record_path):
        self._record_files.append(record_path.split('/')[-1])
        torch.save(self._record_files, self._files_seqs_path,
            pickle_protocol=pickle.HIGHEST_PROTOCOL)


def rebuild_tensor_wrapper(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    tensor = _rebuild_tensor(storage, storage_offset, size, stride)
    tensor.requires_grad = requires_grad
    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    tensor._backward_hooks = backward_hooks
    add_tensor_info(tensor)
    return tensor

def build_summary_wapper(dtype, size, stride, min_val, max_val):
    summary = TensorSummary(dtype, size, stride, min_val, max_val)
    add_tensor_info(summary)
    return summary

def reduce_to_summary(self, proto):
    cpu_self = torch._C._TensorBase.cpu(self)
    min_val = get_min_val(cpu_self)
    max_val = get_max_val(cpu_self)
    return (build_summary_wapper, (self.dtype, tuple(self.size()), self.stride(), min_val, max_val))

@contextmanager
def record_guard(record_path):
    _ctx.record_record_file(record_path)
    if _ctx.record_level <= 1:
        f0_bak = torch.Tensor._reduce_ex_internal
        f1_bak = torch.nn.Parameter.__reduce_ex__
        torch.Tensor._reduce_ex_internal = reduce_to_summary
        torch.nn.Parameter.__reduce_ex__ = reduce_to_summary
    elif _ctx.record_level == 2:
        f_bak = torch._utils._rebuild_tensor_v2
        torch._utils._rebuild_tensor_v2 = rebuild_tensor_wrapper
    yield
    if _ctx.record_level <= 1:
        torch.Tensor._reduce_ex_internal = f0_bak
        torch.nn.Parameter.__reduce_ex__ = f1_bak
    elif _ctx.record_level == 2:
        torch._utils._rebuild_tensor_v2 = f_bak

def record_save(record_cont, op_name, is_input):
    if record_cont:
        file_suffix = ".input.pt" if is_input else ".output.pt"
        record_path = os.path.join(_ctx.output_dir, op_name + file_suffix)
        with record_guard(record_path):
            torch.save(record_cont, record_path, pickle_module=dill,
                pickle_protocol=pickle.HIGHEST_PROTOCOL)

backward_input_list = []

def bwd_hook_cpu(op_name, wrapped_depth, grad_in, grad_out):
    global backward_input_list
    backward_input_list = grad_in
    return

def bwd_hook(op_name, wrapped_depth, grad_in, grad_out):
    op_name_ = op_name
    op_name = op_name + ".bwd"
    stack = ''
    if _ctx.record_stack:
        stack = traceback.format_stack()
    _ctx.check_in_user_scope(op_name)
    if _ctx.check_record_input(op_name, wrapped_depth):
        record_cont = {}
        if _ctx.record_level >= 0:
            record_cont.update({"grad_out": grad_out})
        if stack:
            record_cont["stack"] = stack
        if _ctx.record_level > 0:
            record_save(record_cont, op_name, True)
    if _ctx.check_record_output(op_name, wrapped_depth):
        record_cont = {}
        if _ctx.record_level >= 0:
            record_cont.update({"grad_in": grad_in})
        if stack:
            record_cont["stack"] = stack
        if _ctx.record_level > 0:
            record_save(record_cont, op_name, False)
    if _ctx.record_level == 0:
        recover_hijack()
        if op_name_ not in record_cpu_data.keys():
            log(f"..... backward {op_name_} skipped as model not exist in record_data", LOG_NAME)
            return
        out_cpu, ins_cpu, args, kwargs = record_cpu_data[op_name_]
        g_ins = []
        for g_out in grad_out:
            if torch.is_tensor(g_out):
                g_out_cpu = g_out.float().cpu()
                out_cpu.backward(g_out_cpu)
                global backward_input_list
                g_ins = backward_input_list

        for g_in, grad in zip(g_ins, grad_in):
            if g_in == None or grad == None:
                continue
            if g_in.shape == grad.shape:
                rel_err = cal_rel_err(g_in.float().cpu(), grad.cpu().float())
                mark = "error" if rel_err > eps else "     "
                dev_id = get_tensor_device(grad)
                log(f"{mark} backward rel_err: {rel_err.item()}, device.id: {dev_id}, op: {op_name_} dx shape: {grad.shape}, max : {torch.max(grad).item()}, min {torch.min(grad).item()}, sum {torch.sum(grad).item()}", LOG_NAME)
            else:
                log(f"..... backward {op_name_} skipped as shape not match: cpu shape {g_in.shape} != cuda shape {grad.shape}", LOG_NAME)
        ops_hijack()
    return

# torch.return_types currently can not be saved by torch.save,
# used to judge whether obj is a torch.return_types,
# may be able to be removed when upgrade pytorch.
def is_structseq(obj):
    cls = type(obj)
    if (
        cls.__base__ is tuple
        and isinstance(getattr(cls, 'n_sequence_fields', None), int)
        and isinstance(getattr(cls, 'n_fields', None), int)
        and isinstance(getattr(cls, 'n_unnamed_fields', None), int)
    ):
        try:
            class subcls(cls):
                pass
        except (
            TypeError,       # CPython
            AssertionError,  # PyPy
        ):
            return True

    return False

class HookTemplate(nn.Module):
    def __init__(self, op, op_name):
        super(HookTemplate, self).__init__()
        self._op = op
        self._op_name = op_name + "." + str(_ctx.fwd_cnt)
        self._wrapped_depth = _ctx.wrapped_depth

    # Compared with PyTorch's native __call__ function, we have deleted some unused hooks,
    # supported the scenario where the return value of the operator API is None,
    # and solved the problem of getting grad_fn will report an error in the case shown below:
    # a = torch.randn(3).requires_grad_()
    # with torch.no_grad():
    #   a.view(-1).abs_().grad_fn
    def __call__(self, *input, **kwargs):
        result = self.forward(*input, **kwargs)

        # Modifying inputs or outputs inplace is not supported by register_full_backward_hook,
        # so we can only use the logic of register_backward_hook, despite some bwd input or
        # output may be lost.
        if torch.is_grad_enabled():
            var = result
            while not isinstance(var, torch.Tensor):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
                elif isinstance(var, (list, tuple)):
                    if var:
                        var = var[0]
                    else:
                        return result
                else:
                    return result
            grad_fn = var.grad_fn
            if grad_fn is not None:
                wrapper = functools.partial(bwd_hook, self._op_name, self._wrapped_depth)
                functools.update_wrapper(wrapper, bwd_hook)
                grad_fn.register_hook(wrapper)

        return result

    def forward(self, *args, **kwargs):
        op_name = self._op_name + ".fwd"
        stack = ''
        if _ctx.record_stack:
            stack = traceback.format_stack()
        _ctx.check_in_user_scope(op_name)
        if _ctx.check_record_input(op_name):
            record_cont = {}
            if _ctx.record_level > 0:
                if len(args) > 0 and isinstance(args[0], nn.Module):
                    record_cont.update({"module": args[0], "args": args[1:], "kwargs": kwargs})
                else:
                    record_cont.update({"args": args, "kwargs": kwargs})
            if stack:
                record_cont["stack"] = stack
            if _ctx.record_level > 0:
                record_save(record_cont, op_name, True)
        res = self._op(*args, **kwargs)
        if _ctx.record_level == 0:
            recover_hijack()
            res_cpu, in_tsr_cpu, args_cpu, kwargs_cpu = self.run_on_cpu(*args, **kwargs)
            if res_cpu is not None:
                record_cpu_data[self._op_name] = [res_cpu, in_tsr_cpu, args_cpu, kwargs_cpu]
                self.compare_tensor(op_name, "output", res, res_cpu, eps)
            else:
                log(f"     {op_name} output not check as it is not tensor", LOG_NAME)
            ops_hijack()
        if _ctx.check_record_output(op_name):
            record_cont = {}
            if _ctx.record_level > 0:
                record_res = res
                if is_structseq(res):
                    record_res = tuple(res)
                record_cont["res"] = record_res
            if stack:
                record_cont["stack"] = stack
            if _ctx.record_level > 0:
                record_save(record_cont, op_name, False)
        return res
    
    def run_on_cpu(self, *args, **kwargs):
        args_cpu = []
        input_tensors = []
        for arg in args:
            if hasattr(arg, "parameters"):
                if peek(arg.parameters()) is not None:
                    op = copy.deepcopy(arg).to(torch.float32).cpu()
                    if hasattr(arg, "training"):
                        args_cpu.append(op.train() if arg.training else op.eval())
                    else:
                        args_cpu.append(op)
                else:
                    args_cpu.append(arg)
            elif torch.is_tensor(arg):
                if arg.dtype == torch.float16 or arg.dtype == torch.bfloat16 or arg.dtype == torch.float:
                    t = arg.clone().detach().float().cpu().requires_grad_(True)
                else:
                    t = arg.clone().detach().cpu()
                args_cpu.append(t)
                input_tensors.append(t)
            else:
                args_cpu.append(arg)
        try:
            ret_cpu = self._op(*args_cpu, **kwargs)
            grad_fn = ret_cpu.grad_fn
            if grad_fn is not None:
                wrapper = functools.partial(bwd_hook_cpu, self._op_name, self._wrapped_depth)
                functools.update_wrapper(wrapper, bwd_hook_cpu)
                grad_fn.register_hook(wrapper)
        except:
            return None
        return ret_cpu, input_tensors, args_cpu, kwargs


    def compare_tensor(self, op_name, tensor_name, t, t_cpu, eps):
        if torch.is_tensor(t):
            rel_err = cal_rel_err(t.cpu().float(), t_cpu.float())
            mark = "error" if rel_err > eps else "     "
            dev_id = get_tensor_device(t)
            log(f"{mark} forward rel_err: {rel_err.item()}, device.id: {dev_id}, op: {op_name}, {tensor_name} shape: {t.shape},  max : {torch.max(t).item()}, min {torch.min(t).item()}, sum: {torch.sum(t.double()).item()}", LOG_NAME)
        elif isinstance(t, (list, tuple)):
            log(f"..... skip compare with cpu as [{op_name}] output is list/tuple", LOG_NAME)
        else:
            # log(f"..... skip compare with cpu as [{op_name}] output is not a tensor", LOG_NAME)
            pass


def ops_hijack():
    def wrap_op(op_name, f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            _ctx.wrapped_depth += 1
            res = HookTemplate(f, op_name)(*args, **kwargs)
            if _ctx.wrapped_depth == 1:
                _ctx.fwd_cnt += 1
            _ctx.wrapped_depth -= 1
            return res

        return wrapper

    for attr_name in torch_ops:
        if not hasattr(torch, attr_name):
            continue
        setattr(torch, attr_name, wrap_op("torch." + attr_name, getattr(torch, attr_name)))
        if hasattr(torch._VF, attr_name):
            setattr(torch._VF, attr_name, wrap_op("torch._VF." + attr_name,
                getattr(torch._VF, attr_name)))

    for attr_name in tensor_ops:
        if not hasattr(torch.Tensor, attr_name):
            continue
        setattr(torch.Tensor, attr_name, wrap_op("torch.Tensor." + attr_name,
            getattr(torch.Tensor, attr_name)))

    for attr_name in nn_functional_ops:
        if not hasattr(torch.nn.functional, attr_name):
            continue
        setattr(torch.nn.functional, attr_name, wrap_op("torch.nn.functional." + attr_name,
            getattr(torch.nn.functional, attr_name)))

    for attr_name in torch_fft_ops:
        if not hasattr(torch.fft, attr_name):
            continue
        setattr(torch.fft, attr_name, wrap_op("torch.fft." + attr_name,
            getattr(torch.fft, attr_name)))
        alias_name = "fft_" + attr_name
        setattr(torch._C._fft, alias_name, wrap_op("torch._C._fft." + alias_name,
            getattr(torch._C._fft, alias_name)))

    for attr_name in torch_linalg_ops:
        if not hasattr(torch.linalg, attr_name):
            continue
        setattr(torch.linalg, attr_name, wrap_op("torch.linalg." + attr_name,
            getattr(torch.linalg, attr_name)))
        alias_name = "linalg_" + attr_name
        setattr(torch._C._linalg, alias_name, wrap_op("torch._C._linalg." + alias_name,
            getattr(torch._C._linalg, alias_name)))

    for attr_name in torch_special_ops:
        if not hasattr(torch.special, attr_name):
            continue
        setattr(torch.special, attr_name, wrap_op("torch.special." + attr_name,
            getattr(torch.special, attr_name)))
        alias_name = "special_" + attr_name
        setattr(torch._C._special, alias_name, wrap_op("torch._C._special." + alias_name,
            getattr(torch._C._special, alias_name)))

    for attr_name in nn_module_ops:
        if not hasattr(torch.nn, attr_name):
            continue
        module_op = getattr(torch.nn, attr_name)
        setattr(module_op, "forward", wrap_op("torch.nn." + attr_name,
            getattr(module_op, "forward")))

def recover_hijack():
    for attr_name in torch_ops:
        if not hasattr(torch, attr_name):
            continue
        setattr(torch, attr_name, getattr(torch, attr_name).__wrapped__)
        if hasattr(torch._VF, attr_name):
            setattr(torch._VF, attr_name, getattr(torch._VF, attr_name).__wrapped__)

    for attr_name in tensor_ops:
        if not hasattr(torch.Tensor, attr_name):
            continue
        setattr(torch.Tensor, attr_name, getattr(torch.Tensor, attr_name).__wrapped__)

    for attr_name in nn_functional_ops:
        if not hasattr(torch.nn.functional, attr_name):
            continue
        setattr(torch.nn.functional, attr_name, getattr(
            torch.nn.functional, attr_name).__wrapped__)

    for attr_name in torch_fft_ops:
        if not hasattr(torch.fft, attr_name):
            continue
        setattr(torch.fft, attr_name, getattr(torch.fft, attr_name).__wrapped__)
        alias_name = "fft_" + attr_name
        setattr(torch._C._fft, alias_name, getattr(torch._C._fft, alias_name).__wrapped__)

    for attr_name in torch_linalg_ops:
        if not hasattr(torch.linalg, attr_name):
            continue
        setattr(torch.linalg, attr_name, getattr(torch.linalg, attr_name).__wrapped__)
        alias_name = "linalg_" + attr_name
        setattr(torch._C._linalg, alias_name, getattr(torch._C._linalg, alias_name).__wrapped__)

    for attr_name in torch_special_ops:
        if not hasattr(torch.special, attr_name):
            continue
        setattr(torch.special, attr_name, getattr(torch.special, attr_name).__wrapped__)
        alias_name = "special_" + attr_name
        setattr(torch._C._special, alias_name, getattr(torch._C._special, alias_name).__wrapped__)

    for attr_name in nn_module_ops:
        if not hasattr(torch.nn, attr_name):
            continue
        module_op = getattr(torch.nn, attr_name)
        setattr(module_op, "forward", getattr(module_op, "forward").__wrapped__)

def pickle_memory_format(memory_format):
    return str(memory_format).split('.')[1]

def fake_pickle_torch_generator(generator):
    return torch._C.Generator, ()

def create_output_dir(output_dir, ranks = None):
    if os.path.exists(output_dir):
        warnings.warn("Output directory of recordtool has already exists and will be overwritten.")
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    if ranks:
        for rank in ranks:
            sub_dir = os.path.join(output_dir, "rank" + str(rank))
            os.mkdir(sub_dir)

def start_record(
        enabled = True,
        output_dir = "./record_dir",
        record_level = 2,
        record_stack = True,
        record_input = True,
        record_output = True,
        op_range = [],
        op_list = [],
        skip = 0,
        process_group = None,
        ranks = []):
    assert all(isinstance(a, bool) for a in [enabled, record_stack, record_input, record_output]), \
        "The types of enabled, record_stack, record_input and record_output must all be bool."

    if not enabled:
        return
    global _ctx
    if _ctx:
        warnings.warn("start_record has already been called and will not be overwitten.")
        return

    assert isinstance(record_level, int) and record_level >= 0 and record_level <= 2, \
        "The type of record_level param must be int and the value must be in the range [0, 2]."
    assert isinstance(op_range, list) or isinstance(op_range, tuple), \
        "The type of op_range param must be list or tuple."
    assert len(op_range) == 0 or len(op_range) == 2, \
        "op_range must be empty or only contains two elements."
    assert isinstance(op_list, list), "The type of op_list must be list."
    assert isinstance(skip, int) and skip >= 0, \
        "The type of skip must be int and the value cannot be less than 0."
    assert isinstance(ranks, list), "The type of ranks must be list."

    dist_check = True
    real_path = os.path.realpath(output_dir)
    will_record = record_level > 0 or record_stack
    if dist.is_initialized():
        cur_rank = dist.get_rank(process_group)
        dist_check = cur_rank != -1 and (len(ranks) == 0 or cur_rank in ranks)
        if len(ranks) == 0:
            ranks = list(range(dist.get_world_size(process_group)))
        if will_record and cur_rank == ranks[0]:
            create_output_dir(real_path, ranks=ranks)
        dist.barrier(group=process_group)
        real_path = os.path.join(real_path, "rank" + str(cur_rank))
    elif will_record:
        create_output_dir(real_path)

    _ctx = Context(enabled, real_path, record_level, record_stack,
        record_input, record_output, op_range, op_list, skip)

    if not dist_check:
        return

    ops_hijack()

    # PyTorch currently do not support serialize these types,
    # we need register the serialization method by ourselves.
    # https://github.com/pytorch/pytorch/issues/56525
    # https://github.com/pytorch/pytorch/issues/43672
    # https://github.com/pytorch/pytorch/issues/71398
    copyreg.pickle(torch.memory_format, pickle_memory_format)
    copyreg.pickle(torch.Generator, fake_pickle_torch_generator)

@contextmanager
def record(
        enabled = True,
        output_dir = "./record_dir",
        record_level = 2,
        record_stack = True,
        record_input = True,
        record_output = True,
        op_range = [],
        op_list = [],
        skip = 0,
        process_group = None,
        ranks = []):
    double_init = False
    if _ctx:
        double_init = True
        prev = _ctx.enabled
    start_record(enabled, output_dir, record_level, record_stack, record_input,
        record_output, op_range, op_list, skip, process_group, ranks)
    yield
    if double_init:
        record_switch(prev)
    else:
        record_switch(False)

def record_switch(enabled):
    assert isinstance(enabled, bool), "The type of enabled must be bool."
    if _ctx:
        _ctx.set_enabled(enabled)
