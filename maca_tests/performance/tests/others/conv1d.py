import argparse
from datetime import datetime
import time
import torch
import torch.nn.functional as F
import re
import torch.nn as nn

from utils.pt_reporter import PyTorchReporter
torch.manual_seed(107)


class Conv1dPerfTest(PyTorchReporter):
    def __init__(self, platform, branch):
        super().__init__()
        self.function = "conv1d_test"
        self.platform = platform
        self.branch = branch

    def run(self, niter=1):
        for dtype in [torch.float32, torch.float16]:
            for (shape, out_c, k, s, p, d, g) in [((256, 1, 3600), 4, 5, 1, 2, 1, 1), ((256, 4, 3600), 16, 5, 1, 2, 1, 1),\
                                                  ((256, 16, 3600), 512, 19, 5, 9, 1, 1), ((256, 512, 720), 1024, 1, 1, 0, 1, 1),\
                                                  ((256, 512, 720), 512, 15, 1, 7, 1, 512), ((256, 512, 720), 512, 1, 1, 0, 1, 1)]:
                date_start = datetime.now()
                self.testgroup = str(dtype)[6:]
                input = torch.randn(shape, dtype = dtype).cuda()
                m = nn.Conv1d(shape[1], out_c, k, stride = s, padding = p, dilation = d, groups = g, dtype = dtype).to("cuda")
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
    f = Conv1dPerfTest(args.platform, args.branch)
    f.run()


