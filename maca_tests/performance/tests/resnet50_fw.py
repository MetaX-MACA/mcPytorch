import os
import subprocess
import argparse
from datetime import datetime
import time
import torch
import torch.nn.functional as F
import re
import copy

from utils.pt_reporter import PyTorchReporter
from utils.utils import launch_prof

parser = argparse.ArgumentParser()
parser.add_argument('-platform', type=str, metavar='platform', help='chip platform')
parser.add_argument('-branch', type=str, metavar='branch', help='git branch')
parser.add_argument('-path', type=str, metavar='path', help='output json path')
args = parser.parse_args()


def get_time_ips(info):
    val = float(info.strip().split("\n")[-4].split("train.total_ips : ")[-1].split("images/s train.lr")[0])
    return val

class Resnet50Test(PyTorchReporter):
    def __init__(self, platform, branch):
        super().__init__()
        self.function = "resnet50_test_"
        self.platform = platform
        self.branch = branch

    @launch_prof(args.path, warmup_num=1, active_num=1, max_repeat_num=1)
    def resnet50_test(self, shape, dtype, is_optim=True, func_name="", feature=""):
        cur_dir = os.path.abspath(os.curdir)
        command  = "bash " + cur_dir + "/../../../maca_samples/resnet50_ngc/scripts_resnet50/fw_" + func_name + "_" + str(shape) + ".sh"
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            print(result.stdout)
            t = get_time_ips(result.stdout)
        except Exception as e:
            print("Error: ", str(e))
        return t

    def run_common(self, is_optim=True):
        self.feature = "common"
        self.is_optim = is_optim
        bs_list = [256, 512]
        format_list = ["nchw", "nhwc"]
        dtypes = [torch.float16,]
        for bs in bs_list:
            for dtype in dtypes:
                for form in format_list:
                    self.resnet50_test(shape=bs, dtype=dtype, is_optim=is_optim, func_name="resnet50_AMP_perf_1C_"+form, feature=self.feature)


if __name__ == '__main__':
    f = Resnet50Test(args.platform, args.branch)
    f.run_common()
