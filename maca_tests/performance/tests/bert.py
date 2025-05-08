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


def get_throughput(info):
    val = float(info.strip().split("\n")[-1].split("training_sequences_per_second : ")[-1].split("sequences/s final_loss")[0])
    return val

class BertTest(PyTorchReporter):
    def __init__(self, platform, branch):
        super().__init__()
        self.function = "bert_test_"
        self.platform = platform
        self.branch = branch

    @launch_prof(args.path, warmup_num=1, active_num=1, max_repeat_num=1)
    def bert_test(self, shape, dtype, is_optim=True, func_name="", feature=""):
        cur_dir = os.path.abspath(os.curdir)
        command  = "bash " + cur_dir + "/../../../maca_samples/LanguageModeling/BERT/spd_" + func_name + ".sh"
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            print(result.stdout)
            t = get_throughput(result.stdout)
        except Exception as e:
            print("Error: ", str(e))
            t = 0.0
        return t

    def run_common(self, is_optim=True):
        self.feature = "common"
        self.is_optim = is_optim
        model_dict = {"base": [32, 128], "large": [32, 64]}
        dtypes = [torch.float16,]
        for dtype in dtypes:
            for model in model_dict:
                for bs in model_dict[model]:
                    self.bert_test(shape=bs, dtype=dtype, is_optim=is_optim, func_name=f"bert_amp_bs_{bs}_{model}", feature=self.feature)


if __name__ == '__main__':
    f = BertTest(args.platform, args.branch)
    f.run_common()
