import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../test/".format(cur_dir))
import test_nn
test_nn.TestNNDeviceTypeCUDA.primary_device = "cuda"

# functions in TestNNDeviceType
# TODO(liuyuxin): torch.randn not supported by maca.
nn_devicetype_functions = ["test_max_pool2d_cuda"]

if __name__ == '__main__':
    obj1 = test_nn.TestNNDeviceTypeCUDA()
    for name in nn_devicetype_functions:
        func = getattr(obj1, name, None)
        import pdb
        pdb.set_trace()
        if func is not None:
            func()
            print(name + " passed!")
        else:
            print(name + " is not existed in TestNNDeviceType!")
