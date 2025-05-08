import argparse
from datetime import datetime
import time
import torch
import torch.nn.functional as F
import re

from utils.pt_reporter import PyTorchReporter
torch.manual_seed(107)


class ThresholdPerfTest(PyTorchReporter):
    def __init__(self, platform, branch):
        super().__init__()
        self.function = "threshold_test"
        self.platform = platform
        self.branch = branch

    def run(self, niter=1):
        m = torch.nn.Threshold(0.1, 20, True)
        for dtype in [torch.float16]:
            for (shape, stride) in [((2, 16, 419430), (0, 4194, 1))]:
                date_start = datetime.now()
                self.testgroup = str(dtype)[6:]
                input = torch.randn(shape, dtype = dtype, device = "cuda")
                input = input.as_strided(shape, stride)
                # warm up
                output = m(input)


                torch.cuda.synchronize()
                start = time.perf_counter()
                for i in range(niter):
                    output = m(input)
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
                self.testcase = str(shape)

                self.dumpJson(args.path, self.function + "_" + self.testgroup + "_" +
                              self.testcase + ".json")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-platform', type=str, metavar='platform', help='chip platform')
    parser.add_argument('-branch', type=str, metavar='branch', help='git branch')
    parser.add_argument('-path', type=str, metavar='path', help='output json path')
    args = parser.parse_args()
    f = ThresholdPerfTest(args.platform, args.branch)
    f.run()


