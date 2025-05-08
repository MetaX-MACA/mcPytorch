import numpy as np
import torch
import torch.nn as nn
import argparse

from utils import *
from kernel_map import *

def resolution_log(inp_path, out_path):
    line_info = {}
    with open(inp_path,"r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if is_legal_elementwise_info(line):
                if line not in line_info:
                    line_info[line] = {}
                    line_info[line]["NumOfCalls"] =  1
                else:
                    line_info[line]["NumOfCalls"] += 1
    middle_list = ["isGetTime dim arity shapeStride dtype isOpt func substride NumOfCalls eleInfo"]
    for (line,info) in line_info.items():
        middle_list.append(get_middle_log(line, info))

    save_tmp_log(out_path, middle_list, sort = True, sortIdx = 8, reverse = True)



def get_device_time(middle_log, device_log):
    with open(middle_log,"r") as f:
        lines = f.readlines()[1:]

    device_list = ["isGetTime kernel dim arity shapeStride dtype time totalTime isOpt func substride NumOfCalls eleInfo"]
    for line in lines:
        print(line)
        line_list = line.strip().split()
        kernel = "__"
        Time = "__"
        totalTime = "__"
        if line_list[0] == "1":   #isGetTime
            kernel, Time = get_kernel_time(line_list)
            totalTime = round(float(Time[:-2]) * int(line_list[8]), 4)
            if Time[-2:] == "us" and totalTime >= 1000000:
                totalTime = str(round(totalTime/1000000.0, 4)) + "s"
            elif Time[-2:] == "us" and totalTime >= 1000:
                totalTime = str(round(totalTime/1000.0, 4)) + "ms"
            elif Time[-2:] == "ms" and totalTime >= 1000:
                totalTime = str(round(totalTime/1000.0, 4)) + "s"
            else:
                totalTime = str(totalTime) + Time[-2:]
        line_list.insert(1, kernel)
        line_list.insert(6, Time)
        line_list.insert(7, totalTime)
        device_list.append(" ".join(line_list))

    # save_tmp_log(device_log, device_list, sort = True, sortIdx = 7, reverse = True)
    save_tmp_log(device_log, device_list)


def compare_time(maca_time_log, cuda_time_log, compare_time_log, filter_no_time):
    with open(maca_time_log,"r") as f:
        macalines = f.readlines()[1:]
    with open(cuda_time_log,"r") as f:
        cudalines = f.readlines()[1:]
    assert(len(macalines)==len(cudalines))

    device_list = ["isGetTime kernel dim arity shapeStride dtype macaTime cudaTime macaTotalTime cudaTotalTime (cuda/maca) isOpt func substride NumOfCalls eleInfo"]
    for i in range(len(macalines)):
        maca_line_list = macalines[i].split()
        cuda_line_list = cudalines[i].split()
        
        shapeStride = "(" + maca_line_list[4] + ")"
        maca_line_list[4] = shapeStride

        maca_time = "__"
        cuda_time = "__"
        maca_total_time = "__"
        cuda_total_time = "__"
        radio = "__"
        if maca_line_list[0] == "1":   #isGetTime
            maca_time = maca_line_list[6]
            cuda_time = cuda_line_list[6]
            mt = transform_time(maca_time)
            ct = transform_time(cuda_time)
            radio = str(round((ct+0.0000001)/(mt+0.0000001), 4))

            maca_total_time = str(transform_time(maca_line_list[6]) * int(maca_line_list[-2])) + "us"
            cuda_total_time = str(transform_time(cuda_line_list[6]) * int(maca_line_list[-2])) + "us"

        maca_line_list[7] = maca_total_time
        maca_line_list.insert(7, cuda_time)
        maca_line_list.insert(9, cuda_total_time)
        maca_line_list.insert(10, radio)
        
        if not (maca_line_list[0] == "0" and filter_no_time):
            device_list.append(" ".join(maca_line_list))
    
    # save_tmp_log(compare_time_log, device_list, sort = True, sortIdx = 8, reverse = True)
    save_tmp_log(compare_time_log, device_list)

    #get totalTime ration
    maca_time_all=0.0
    cuda_time_all=0.0
    for i in range(1,len(device_list)):
        macaTotalTime = device_list[i].split()[8]
        cudaTotalTime = device_list[i].split()[9]
        if macaTotalTime!="__":
            maca_time_all += transform_time(macaTotalTime)
            cuda_time_all += transform_time(cudaTotalTime)
    radio = cuda_time_all / maca_time_all
    with open(compare_time_log,"a") as f:
        f.write("cudaTotalTime/macaTotalTime:" + str(radio))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--get_middle_log", action="store_true", help="resolution tmp log from origin log")
    parser.add_argument("--origin_log", help="resolution_log input file")
    parser.add_argument("--middle_log", default="middle.log", help="resolution_log output file")

    parser.add_argument("--get_time_log", action="store_true", help="resolution tmp log from origin log")
    parser.add_argument("--device_log", default="device.log", help="device time log")

    parser.add_argument("--get_compare_time_log", action="store_true", help="resolution tmp log from origin log")
    parser.add_argument("--maca_time_log", default="maca_time.log", help="macatime log")
    parser.add_argument("--cuda_time_log", default="cuda_time.log", help="cudatime log")
    parser.add_argument("--compare_time_log", default="compare_time.log", help="comparetime log")
    parser.add_argument("--filter_no_time", action="store_true", help="resolution tmp log from origin log")

    args = parser.parse_args()
    if args.get_middle_log:
        resolution_log(args.origin_log, args.middle_log)
    elif args.get_time_log:
        get_device_time(args.middle_log, args.device_log)
    elif args.get_compare_time_log:
        compare_time(args.maca_time_log, args.cuda_time_log, args.compare_time_log, args.filter_no_time)
 

