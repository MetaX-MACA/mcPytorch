# USE_MACA: This file is added by mcPytorch
#
# For those configs already exist in torch/_inductor/config.py and needed to be modified by MACA_TORCH_COMPILE_CONF，
# we should add them outside class maca, e.g. source_fusion_memory_threshold below.
#
# Configs for maca specifically should be added inside class maca.

# If fusing two nodes only save less then score_fusion_memory_threshold memory,
# we should not bother fusing the nodes.
#
# This is especially helpful to resolve https://github.com/pytorch/pytorch/issues/133242
# Previously we fuse two nodes because of common read of a scalar tensor.
# If we skip it, the loop ordering after fusion mechanism kicks in and can
# brings more savings.
#
# For the cases loop ordering after fusion does not help, we don't lose much.
# this option use to control nodes which have common reads to fuse if their common reads >= threshold
# the overhead is low
score_fusion_memory_threshold = 10

# nodes try to fuse even they have no common reads if aggressive_fusion is True
# this option is useful for small kernels which have no common reads
# the overhead is moderate
aggressive_fusion = False

# we can use option to control kernel fusion, but it's not always improve performance
# we can enbale benchmark_fusion to benchmark kernels to decide use fused kernel if its performance is better
# the overhead is heavy
benchmark_fusion = False


class maca:
    # maca config for torch.compile should be added inside this class with default value
    # e.g. maca_config_0 = 'value_0'
    enable_maca_triton_mm = 0

    # disable maca_tirotn_heuristics configs for maca hardware
    disable_maca_triton_heuristics = 0

    #maca config for enable_custom_fused_rsm_kernel in compile mode
    custom_fused_rsm_enable = 0

    # 1) Adjust the admission threshold for reduce persistent in compile from 1024 to 4096 (typically
    #    used for the non-tunable mode of torch compile standalone):
    #    export MACA_TORCH_COMPILE_CONF=maca.reduce_persistent_threshold:1
    # 2) If you need to adjust the range of the admission threshold, you can use the pesistent_threshold_scale
    #    variable to configure the scale value:
    #    export MACA_TORCH_COMPILE_CONF=maca.reduce_persistent_threshold:1,maca.pesistent_threshold_scale:16
    # 3) The reduce_persistent_threshold environment variable switch is disabled by default.
    reduce_persistent_threshold = 0

    # maca reduction persistent mode scale value
    pesistent_threshold_scale = 4


# env format: MACA_TORCH_COMPILE_CONF=key_0:value_0,key_1:value_1,...
# we can use all env to set all configs config.py, value support int and string e.g
# export MACA_TORCH_COMPILE_CONF=kernel_name_max_ops:20,triton.cudagraphs:1,triton.descriptive_names:torch
# parse MACA_TORCH_COMPILE_CONF into dict
# after parsing: configs={'config_0':'value_0', 'config_1':'value1', ...}
def _parse_maca_compile_conf(env: str) -> dict:
    configs = {}
    sub_envs = env.split(',')
    for sub_env in sub_envs:
        sub_env = sub_env.strip()
        key_value = sub_env.split(':', 1)
        assert len(key_value) == 2, "'MACA_TORCH_COMPILE_CONF' format invalid! E.g.: export MACA_TORCH_COMPILE_CONF=config_0:value_0,config_1:value_1"
        key, value = key_value
        configs[key.strip()] = value.strip()
    return configs


def _updateConfigs(cfgs):
    import os
    import logging
    log = logging.getLogger(__name__)
    maca_env = os.getenv('MACA_TORCH_COMPILE_CONF')
    if not maca_env:
        return

    configs = _parse_maca_compile_conf(maca_env)
    for key, value in configs.items():
        # the value is int or string
        try:
            typed_value = int(value)
        except ValueError:
            typed_value = value
        if "." in key:  # namespace.key
            nk = key.split('.')
            assert len(nk) == 2, f"'MACA_TORCH_COMPILE_CONF' format invalid: {nk} should be formated as namespace.key"
            namespace, key = nk
            if namespace in cfgs:
                setattr(cfgs[namespace], key, typed_value)
                log.info(f"setting {namespace}.{key} = {typed_value} for maca platform")
            else:
                log.warning(f"{namespace}.{key} is not exist in config")
        else:
            cfgs[key] = typed_value
            log.info(f"setting {key} = {typed_value} for maca platform")


__all__ = [k for k in globals() if not k.startswith("_")]
