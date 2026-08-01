import functools
from torch._utils_internal import USE_MACA
import torch._inductor.config as cfg


def reduction(original_func):
    """
    wrapper triton_heuristics._reduction_configs to
    add maca reduction configs
    """
    from torch._inductor.runtime.triton_heuristics import (
        triton_config_reduction,
        get_total_reduction_numel,
    )
    from torch._inductor.runtime.hints import ReductionHint

    @functools.wraps(original_func)
    def wrapper(
        *,
        size_hints,
        inductor_meta=None
    ):
        reduction_hint = inductor_meta.get("reduction_hint", None)
        if (USE_MACA 
            and not cfg.maca.disable_maca_triton_heuristics
            and reduction_hint == ReductionHint.INNER
            and "y" not in size_hints   # not handle 3D
            and not (inductor_meta.get("max_autotune")
                or inductor_meta.get("max_autotune_pointwise"))
        ):
            register_intensive = True
            rnumel = get_total_reduction_numel(size_hints)
            MAX_R0_BLOCK = 2048
            contiguous_config = triton_config_reduction(
                size_hints,
                1,
                min(rnumel, MAX_R0_BLOCK),
                register_intensive=register_intensive,
            )
            maca_config_1 = triton_config_reduction(
                size_hints,
                1,
                (rnumel if 256 <= rnumel < MAX_R0_BLOCK else MAX_R0_BLOCK),
                num_warps=4, register_intensive=register_intensive
            )
            maca_config_2 = triton_config_reduction(size_hints, 1,
                                                    (rnumel if 256 <= rnumel < MAX_R0_BLOCK else MAX_R0_BLOCK),
                                                    num_warps=8, register_intensive=register_intensive)
            configs = [contiguous_config, maca_config_1, maca_config_2]
        else:
            configs = original_func(size_hints=size_hints, inductor_meta=inductor_meta)
        return configs
    return wrapper


maca_reduction = reduction
