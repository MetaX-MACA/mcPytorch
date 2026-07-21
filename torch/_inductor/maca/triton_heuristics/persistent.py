import functools
from torch._utils_internal import USE_MACA
import torch._inductor.config as cfg


def persistent(original_func):
    """
    wrapper triton_heuristics._persistent_reduction_configs to
    add maca persistent configs
    """
    from torch._inductor.runtime.triton_heuristics import (
        get_total_reduction_numel,
        triton_config_reduction,
    )
    from torch._inductor.utils import prefix_is_reduction
    from torch._inductor.runtime.hints import ReductionHint

    @functools.wraps(original_func)
    def wrapper(
        size_hints,
        reduction_hint=False,
        inductor_meta=None,
        triton_meta=None,
    ):
        if (
            USE_MACA
            and not cfg.maca.disable_maca_triton_heuristics
            and "y" not in size_hints
            and reduction_hint == ReductionHint.INNER
        ):
            xnumel = size_hints["x"]
            rnumel = get_total_reduction_numel(size_hints)
            MAX_PERSISTENT_BLOCK_NUMEL_MACA = 8192
            configs = []
            for xblock in (1, 8, 32, 128):
                nel = rnumel * xblock
                if xblock == 1 or (nel <= MAX_PERSISTENT_BLOCK_NUMEL_MACA and xblock <= xnumel):
                    configs.append(
                        triton_config_reduction(
                            size_hints,
                            xblock,
                            rnumel,
                            register_intensive=True
                        )
                    )
                    if not inductor_meta.get("coordinate_descent_tuning", False):
                        num_warps_4 = max(1, min(nel // (4 * 64), 16)) # each thread handle 4 element
                        num_warps_8 = max(1, min(nel // (8 * 64), 16)) # each thread handle 8 element
                        configs.append(
                            triton_config_reduction(
                                size_hints, xblock,
                                rnumel,
                                num_warps=num_warps_4,
                                register_intensive=True
                            )
                        )
                        if num_warps_4 != num_warps_8:
                            configs.append(
                                triton_config_reduction(
                                    size_hints,
                                    xblock,
                                    rnumel,
                                    num_warps=num_warps_8,
                                    register_intensive=True
                                )
                            )
        
            for c in configs:
                # we don't need Rn_BLOCK for persistent reduction
                for prefix in size_hints:
                    key = f"{prefix.upper()}BLOCK"
                    if prefix_is_reduction(prefix) and key in c.kwargs:
                        c.kwargs.pop(key)

            if not (inductor_meta.get("autotune_pointwise", True)
                    or inductor_meta.get("max_autotune")
                    or inductor_meta.get("max_autotune_pointwise")):
                configs = configs[:1]
        else:
            configs = original_func(size_hints, reduction_hint, inductor_meta, triton_meta) 
        return configs
    return wrapper


maca_persistent = persistent


