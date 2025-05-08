import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../test/".format(cur_dir))
import test_cuda

# functions in TestCuda
test_cuda_functions = ["test_memory_stats"]

if __name__ == '__main__':
    obj1 = test_cuda.TestCuda()
    for name in test_cuda_functions:
        func = getattr(obj1, name, None)
        if func is not None:
            func()
            print(name + " passed!")
        else:
            print(name + " is not existed in TestCuda!")
