import functools
import operator
from torch._utils_internal import USE_MACA
import torch._inductor.config as cfg


def pointwise(original_func):
    """
    wrapper triton_heuristics.pointwise to
    add maca pointwise configs
    """
    from torch._inductor.runtime.triton_heuristics import (
        triton_config,
        autotune_hints_to_configs,
        cached_autotune
    )
    from torch._inductor.runtime.hints import HeuristicType

    @functools.wraps(original_func)
    def wrapper(
        size_hints,
        triton_meta,
        tile_hint=None,
        filename=None,
        min_elem_per_thread=0,
        inductor_meta=None,
    ):
        inductor_meta = {} if inductor_meta is None else inductor_meta
        assert not inductor_meta.get("no_x_dim")
        if (USE_MACA
            and not cfg.maca.disable_maca_triton_heuristics
            and len(size_hints) == 1
            and (inductor_meta.get("autotune_pointwise", True)
                or inductor_meta.get("max_autotune")
                    or inductor_meta.get("max_autotune_pointwise"))
            ):
            numel = functools.reduce(operator.mul, size_hints.values())
            bs = max(256, min(numel // 128, 2048))
            hinted_configs = autotune_hints_to_configs(
                inductor_meta.get("autotune_hints", set()),
                size_hints,
                bs,
                triton_meta["device"],
            )
            triton_config_with_settings = functools.partial(
                triton_config, min_elem_per_thread=min_elem_per_thread
            )
            configs = [
                triton_config_with_settings(size_hints, bs, num_elements_per_warp=256),
                triton_config_with_settings(
                    size_hints, bs // 2, num_elements_per_warp=64
                ),
                # add below config for some situation use bigger block size
                # num_elements_per_warp is a suggested value to calculate num_warps,
                # however the num_warps would be adjusted base on other limits
                triton_config_with_settings(
                    size_hints, bs * 2, num_elements_per_warp=64
                ),
                *hinted_configs,
            ]
            if not configs:
                raise NotImplementedError(f"size_hints: {size_hints}")
            return cached_autotune(
                size_hints,
                configs,
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
        else:
            autotune_configs = original_func(size_hints, triton_meta, tile_hint, filename,
                        min_elem_per_thread, inductor_meta)
            return autotune_configs
    return wrapper


maca_pointwise = pointwise
