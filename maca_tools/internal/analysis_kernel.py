import argparse, os
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_name",
        type=str,
        nargs="?",
        default="./trace_stablediffusion.json",
    )
    parser.add_argument(
        "--kernel_name",
        type=str,
        nargs="?",
        default="at::native::elementwise_kernel_2_1<128, 4, c10::Half, c10::Half, at::native::direct_copy_kernel_cuda",
    )
    opt = parser.parse_args()
    return opt

def main(opt):
    with open(opt.json_name, "r") as f:
        data = json.load(f)
    
    events = []
    for event in data['traceEvents']:
        if opt.kernel_name in event['name']:
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
        max_dur = 0
        total_dur = 0 
        for i, event in enumerate(events):
            if event['dur'] > max_dur:
                max_dur = event['dur']
            total_dur += event['dur']
        final_info.append({"max_dur": max_dur, "total_dur": total_dur, 'idx': i})
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

    for i, block_grid in enumerate(block_grid_lst):
        print(block_grid, " max_dur: ", final_info[i]["max_dur"], " total_dur: ", final_info[i]["total_dur"])

    print("kernel with max total time: ", kernel_lst[max_block_grid][max_info['idx']])
    print("kernel with max total time / total kernel time: ", max_info['total_dur']/total_time)


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
