# mypy: ignore-errors

# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

import torch
import torch._inductor
import operator

aten = torch.ops.aten
prims = torch.ops.prims

from torch._inductor.pattern_matcher import (
   Arg,
   CallFunction,
   CallFunctionVarArgs,
   CallMethod,
   CallMethodVarArgs,
   CallModule,
   CallModuleVarArgs,
   ExclusiveKeywordArg,
   Ignored,
   KeywordArg,
   ListOf,
   MultiOutputPattern,
   PatternExpr,
   RepeatedExpr,
   _TargetArgsExpr,
   _TargetExpr,
   _TargetExprVarArgs,
)
split_with_sizes_default = CallFunction(aten.split_with_sizes.default, KeywordArg('q'), Ignored(), Ignored())
operator_getitem = CallFunction(operator.getitem, split_with_sizes_default, 0)
convert_element_type_default = CallFunction(prims.convert_element_type.default, operator_getitem, Ignored())
div_Tensor = CallFunction(aten.div.Tensor, KeywordArg('variance'), Ignored(), _users=2)
add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, Ignored())
rsqrt_default = CallFunction(aten.rsqrt.default, add_Tensor)
mul_Tensor = CallFunction(aten.mul.Tensor, convert_element_type_default, rsqrt_default, _users=2)
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, KeywordArg('weight_0'))
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, mul_Tensor_1, Ignored())
_fuse_custom_pattern_inference = MultiOutputPattern([convert_element_type_default_1,
  div_Tensor,
  mul_Tensor
])