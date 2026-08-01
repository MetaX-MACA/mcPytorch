import functools
import inspect
import logging
import math

import torch
from torch import nn

from ..._dynamo.utils import counters
from ..pattern_matcher import (
    filter_nodes,
    fwd_only,
    gen_register_replacement,
    joint_fwd_bwd,
    Match,
)

log = logging.getLogger(__name__)
aten = torch.ops.aten
dtype = torch.bfloat16

# USE_MACA: replace fused kernel
_custom_fused_op = aten.custom_fused_rsm


def _fuse_custom_pattern(q, variance, weight_0):
    var = torch.ops.aten.div.Tensor(variance, 8)
    q = torch.ops.aten.split_with_sizes.default(q, [768, 128, 128], -1) # slice [16384, 1024] -> [16384, 768]
    q = q[0]
    orig_dtype = q.dtype
    q = q.to(torch.float32)
    q = q * torch.rsqrt(var + 1e-6)
    out = (q * weight_0).to(orig_dtype)
    return out, var, q


def _fuse_custom_replacement(q, variance, weight_0):
    counters["inductor"]["fused_rsm_custom"] += 1
    out, variance, q = _custom_fused_op(q, variance, weight_0)
    return out, variance, q


def is_forward_graph(match: Match) -> bool:
    gm = match.graph
    for node in gm.nodes:
        if node.op == "call_function":
            if "backward" in str(node.target).lower():
                return False
    return True


def check_custom_rsm_inputs(match: Match) -> bool:

    """
        check input name, shape, dtype
        q: [16384, 768], float32
        variance: [16384, 1], float32
        weight_0: [768, ], bfloat16
    """
    nodes_map = match.kwargs

    # check input names
    q_node = nodes_map.get("q")
    var_node = nodes_map.get("variance")
    w_node = nodes_map.get("weight_0")

    if not all([q_node, var_node, w_node]):
        return False

    # check input shape & dtype
    q = nodes_map["q"]
    if hasattr(q, 'meta') and 'val' in q.meta:
        v = q.meta['val']
        if not (isinstance(v, torch.Tensor) and
                v.dtype == torch.bfloat16 and
                list(v.shape) == [16384, 1024]):
            return False

    var = nodes_map["variance"]
    if hasattr(var, 'meta') and 'val' in var.meta:
        v = var.meta['val']
        if not (isinstance(v, torch.Tensor) and
                v.dtype == torch.float32 and
                list(v.shape) == [16384, 1]):
            return False

    w = nodes_map["weight_0"]
    if hasattr(w, 'meta') and 'val' in w.meta:
        v = w.meta['val']
        if not (isinstance(v, torch.Tensor) and
                v.dtype == torch.bfloat16 and
                list(v.shape) == [768]):
            return False

    return True


# check points: 1) check graph is forward; 2) check input shape && dtype
def _fuse_custom_params_check(match):
    # check device
    if not torch.cuda.is_available():
        return False

   # check in forward graph mode
    if not is_forward_graph(match):
        return False

    # check parameter match
    if not check_custom_rsm_inputs(match):
        return False

    return True

def _get_fuse_custom_patterns():
    from .joint_graph import patterns

    device = "cuda"

    g_inp1 = functools.partial(
        torch.empty, (16384, 1024), device=device, requires_grad=True
    )

    g_inp2 = functools.partial(
        torch.empty, (16384, 1), device=device, requires_grad=True
    )

    g_inp3 = functools.partial(
        torch.empty, (768), device=device, requires_grad=True
    )

    g1 = functools.partial(g_inp1, dtype=torch.bfloat16)
    g2 = functools.partial(g_inp2, dtype=torch.float)
    g3 = functools.partial(g_inp3, dtype=torch.bfloat16)

    candidates = [
        (
            _fuse_custom_pattern,
            _fuse_custom_replacement,
            [g1(), g2(), g3() ],
            {},
            _fuse_custom_params_check,
        ),
    ]

    for pattern, replacement, args, workaround, extra_check in candidates:
        assert isinstance(workaround, dict)
        name = pattern.__name__

        #training_name = name + "_training"
        #yield (
        #    training_name,
        #    {
        #        "search_fn": pattern,
        #        "replace_fn": replacement,
        #        "example_inputs": args,
        #        "trace_fn": joint_fwd_bwd,
        #        "pass_dicts": patterns,
        #        # "extra_check": extra_check,
        #        "scalar_workaround": workaround,
        #    },
        #)

        inference_name = name + "_inference"
        yield (
            inference_name,
            {
                "search_fn": pattern,
                "replace_fn": replacement,
                "example_inputs": args,
                "trace_fn": fwd_only,
                "pass_dicts": patterns,
                "extra_check": extra_check,
                "scalar_workaround": workaround,
                # with dropout turned into clone, we end up with a number of
                # semantically identical graphs
                "skip_duplicates": True,
            },
        )


@functools.cache
def _fuse_custom_init():
    for key, register_replacement_kwargs in _get_fuse_custom_patterns():
        gen_register_replacement(key, **register_replacement_kwargs)
