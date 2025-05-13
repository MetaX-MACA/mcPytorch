import unittest
import ctypes
import contextlib

import torch
from torch.testing._internal.common_utils import run_tests, skipCUDANonDefaultStreamIf

TEST_CUDA = torch.cuda.is_available()
TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2

class TestMacaCUDA(unittest.TestCase):

    def setUp(self):
        super(TestMacaCUDA, self).setUp()

    def test_mem_cpy_device2device(self):
        x = torch.ones(100, dtype=torch.uint8).cuda(0)
        y = torch.zeros(100, dtype=torch.uint8).cuda(0)

        x.copy_(y, non_blocking=True)
        # self.assertEqual(x, y)   # x == y not support, use print
        print(x)

    def test_device_primary_context(self):
        for device in range(torch.cuda.device_count()):
            # Ensure context has not been created beforehand
            self.assertFalse(torch._C._cuda_hasPrimaryContext(device), TestCudaPrimaryCtx.CTX_ALREADY_CREATED_ERR_MSG)

    def test_show_config(self):
        print(torch._C._show_config())

    def test_mem_info(self):
        dc = torch.cuda.device_count()
        curt = torch.cuda.cudart()
        for i in range(dc):
            print(curt.cudaMemGetInfo(0))


if __name__ == "__main__":
    testunit = unittest.TestSuite()
    testunit.addTest(TestMacaCUDA("test_mem_info"))
    unittest.TextTestRunner(verbosity=2).run(testunit)
    # unittest.main()

