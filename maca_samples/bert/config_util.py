import time
import torch
import os


def constuct_log_file_path(log_root_dir, running_device, config_file_name):
    '''
    construct save log path based on platform and local time
    '''
    now_time = time.time()
    tuple_time = time.localtime(now_time)
    str_time = time.strftime("_%y_%m_%d_%H_%M_%S", tuple_time)
    platform_prefix = ""
    try:
        maca_version = torch.version.maca
    except BaseException:
        maca_version = None
    if "cuda" in str(running_device) and maca_version is not None:
        platform_prefix = "_maca"
    elif "cuda" in str(running_device) and maca_version is None:
        platform_prefix = "_cuda"
    else:
        platform_prefix = "_cpu"
    save_log_path = os.path.join(log_root_dir, config_file_name + str_time + platform_prefix + ".txt")
    print("save_log_path:", save_log_path)
    return save_log_path
