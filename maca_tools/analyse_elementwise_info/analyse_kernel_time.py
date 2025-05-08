import argparse
import os
import json
import collections
from functools import reduce
import warnings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_file",
        type=str,
        nargs="?",
        default="./log.txt",
    )
    parser.add_argument(
        "--json_name",
        type=str,
        nargs="?",
        default="./trace_stablediffusion.json",
    )
    args = parser.parse_args()
    return args


def analyse_kernel(json_name, kernel_name):
    with open(json_name, "r") as f:
        data = json.load(f)

    events = []
    for event in data['traceEvents']:
        if kernel_name in event['name']:
            if "broadcast" not in event['name'] and "transpose" not in event['name']:
                events.append(event)
    kernel_lst = []
    block_grid_lst = []
    for event in events:
        block_grid = {}
        block_grid['grid'] = event['args']['grid']
        block_grid['block'] = event['args']['block']
        if block_grid not in block_grid_lst:
            kernel_lst.append([event])
            block_grid_lst.append(block_grid)
        else:
            idx = block_grid_lst.index(block_grid)
            kernel_lst[idx].append(event)
    final_info = []
    for events in kernel_lst:
        name = events[0]['name']
        name_split = name.split(",")
        vt = int(name_split[1])
        max_dur = 0
        total_dur = 0 
        for i, event in enumerate(events):
            if event['dur'] > max_dur:
                max_dur = event['dur']
            total_dur += event['dur']
        final_info.append({"max_dur": max_dur, "total_dur": total_dur, 'idx': i, 'vt': vt})
    max_block_grid = 0 
    max_total_cur = 0
    max_info = {}
    total_time = 0
    for i, infos in enumerate(final_info):
        total_time += infos["total_dur"]
        if infos["total_dur"] > max_total_cur:
            max_total_cur = infos["total_dur"]
            max_info = infos
            max_block_grid = i

    res_list = []
    for i, block_grid in enumerate(block_grid_lst):
        res_dict = {}
        res_dict["block_grid"] = block_grid
        res_dict["max_dur"] = final_info[i]["max_dur"]
        res_dict["total_dur"] = final_info[i]["total_dur"]
        res_dict["dur_ratio"] = float(final_info[i]["total_dur"]) / float(total_time)
        grids = reduce((lambda x, y: int(x) * int(y)), block_grid['grid'])
        blocks = reduce((lambda x, y: int(x) * int(y)), block_grid['block'])
        res_dict["threads"] = grids * blocks * final_info[i]["vt"]
        res_list.append(res_dict)

    return res_list, total_time

def main(args):
    useful_lines = []
    with open(args.log_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if "p_e_" in line:
                useful_lines.append(line)
    useful_lines_set = set(useful_lines)
    dict_info = {}
    for line in useful_lines_set:
        if "broadcast" not in line and "transpose" not in line:
            try:
                split_items = line.split(',')
                ndim = int(split_items[1])
                narity = int(split_items[2])
                shape = [int(split_items[3+i]) for i in range(ndim)]
                # for kernel like p_e_launch_legacy_kernel_N2at6native13BinaryFunctorIN3c104HalfES3_S3_NS0_15binary_internal10MulFunctorIfEEEE,4,2,
                # should check by your self
                kernel_name = f"at::native::elementwise_kernel_{ndim}_{narity}"
                if kernel_name not in dict_info:
                    dict_info[kernel_name] = [shape]
                else:
                    dict_info[kernel_name].append(shape)
            except:
                print(line)
                pass
    for name, shapes in dict_info.items():
        print("---------------------kernel name:", name)
        print("---------------------all {} shapes:{}".format(name, shapes))
        splits = name.split("_")
        ndim = int(splits[-2])
        narity = int(splits[-1])
        rel_shapes = {}
        res_list, total_time = analyse_kernel(args.json_name, name)
        res_list = sorted(res_list, key=lambda x : x["threads"])
        for shape in shapes:
            num = reduce((lambda x, y: int(x) * int(y)), shape)
            for i, info in enumerate(res_list):
                if info['threads'] >= num:
                    if i not in rel_shapes:
                        rel_shapes[i] = [shape]
                    else:
                        rel_shapes[i].append(shape)
                    break
        for i, info in enumerate(res_list):
            if i in rel_shapes:
                print("{}, total dur:{}, ratio:{:2%}, related shape:{}".format(info["block_grid"], info["total_dur"], info["dur_ratio"], rel_shapes[i]))
            else:
                print("{}, total dur:{}, ratio:{:2%}, related shape:{}".format(info["block_grid"], info["total_dur"], info["dur_ratio"], []))
        print("================================================================================")

if __name__ == "__main__":
    # tool for analyse the time ratio of different elementwise kernel
    # args:
    #      --log_file: the log file by set 'PYTORCH_ENABLE_ELEMENTWISE_KERNEL_INFO' environment
    #      --json_name: the json file obtained from profile
    # the log as followed will get
    # 1)
    # ---------------------kernel name: at::native::elementwise_kernel_3_2
    # ---------------------all at::native::elementwise_kernel_3_2 shapes:[[512, 512, 13], [512, 512, 16], [768, 512, 16]]
    # {'grid': [16384, 1, 1], 'block': [128, 1, 1]}, total dur:972, ratio:40.215143%, related shape:[[512, 512, 13], [512, 512, 16]]
    # {'grid': [24576, 1, 1], 'block': [128, 1, 1]}, total dur:1445, ratio:59.784857%, related shape:[[768, 512, 16]]
    # all at::native::elementwise_kernel_3_2 shapes get from log_file
    # related shape means the shape related the grid and block, we can prioritize based on ratio
    # you may get the log as followed:
    # ---------------------kernel name: at::native::elementwise_kernel_4_3
    # ---------------------all at::native::elementwise_kernel_4_3 shapes:[[257, 12, 256, 13], [257, 12, 256, 16]]
    # no grid and block print, that is because kernel like 
    # 'p_e_launch_legacy_kernel_ZN2at6native12_GLOBAL__N_119masked_scale_kernelIbffEEvRNS_6TensorERKS3_S6_T1_EUlfbE_,\
    # 4,2,513,12,512,16,4,2052,24624,12607488,4,1050624,2052,12607488,1,513,6156,3151872 ' encounter, you should
    # check the time manually

    # warnings.warn(f"kernel like p_e_cp_launch_legacy_kernel_ZZZN2at6native23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE0_clEvENKUlvE9_clEvEUlbE_,3,2, check by yourself")
    args = parse_args()
    main(args)