import os
import json
import torch

class PyTorchReporter:
    def __init__(self):
        self.status = "Passed"
        self.software_version = torch.__version__
        maca_version = os.getenv("MACA_VERSION")
        if maca_version:
            self.software_version += "_"
            self.software_version += maca_version
        self.job_name = "pytorch_performance"
        self.function = None # op name
        self.teststart = None
        self.testcase = None # op shape
        self.duration = None
        self.testgroup = None # op dtype
        self.platform = None
        self.branch = None
        self.performance = None
        self.is_optim = True
        self.feature = "common"

    def dumpJson(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, name), "w") as file:
            json.dump(self.__dict__, file, indent=11)

        print(name, " has dumped!")


