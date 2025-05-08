import argparse
from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

from utils.pt_reporter import PyTorchReporter


class NlllossPerfTest(PyTorchReporter):
    def __init__(self, platform, branch, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.function = "nllloss_test" + "_" + reduction
        self.platform = platform
        self.branch = branch

    def launch(self, shape, dtype, is_optim, niter=1):
        date_start = datetime.now()
        self.testgroup = str(dtype)[6:]
        input = torch.randn(shape, dtype = dtype, device="cuda")
        target = torch.randperm(shape[0], device="cuda")
        loss_fn = nn.NLLLoss(reduction=self.reduction)
        # warm up
        output = loss_fn(input, target)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(niter):
            output = loss_fn(input, target)
        torch.cuda.synchronize()
        end = time.perf_counter()
        mean_time = (end - start) / niter
        metrics = {"second": mean_time}
        self.performance = {"metrics":metrics}


        date_end = datetime.now()
        self.teststart = date_start.strftime("%Y-%m-%d %H:%M:%S")
        self.duration = "%H:%M:%S"
        self.duration = re.sub("%H", str(date_end.hour - date_start.hour), self.duration)
        self.duration = re.sub("%M", str(date_end.minute - date_start.minute), self.duration)
        self.duration = re.sub("%S", str(date_end.second - date_start.second), self.duration)
        self.is_optim = is_optim
        self.testcase = str(shape)

        self.dumpJson(args.path, self.function + "_" + self.testgroup + "_" +
                              self.testcase + ".json")

    def run(self):
        self.feature = "common,from_model"
        for dtype in [torch.float16, torch.float32, torch.bfloat16]:
            for shape in [(4096, 8000)]:
                self.launch(shape, dtype, True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-platform', type=str, metavar='platform', help='chip platform')
    parser.add_argument('-branch', type=str, metavar='branch', help='git branch')
    parser.add_argument('-path', type=str, metavar='path', help='output json path')
    args = parser.parse_args()
    f = NlllossPerfTest(args.platform, args.branch)
    f.run()


