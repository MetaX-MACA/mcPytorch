# Instruction for running cpp_extension_cuda_demo

在 cpp_extension_cuda_demo 目录下执行：

bash run_cpp_extension_cuda_demo.sh

使用 setuptools build 使用到的文件包括：

    cpp_extension_cuda_demo
    ├── cuda_extension.cpp
    ├── cuda_extension_kernel.cu
    ├── headers.h
    ├── mypackage
    │   └── __init__.py
    └── setup.py

cuda_device.cpp: 测试获取device信息需要的cpp文件
cuda_extension.cu: jit compile extension需要的.cu文件
cudnn_extension.cpp: cudnn extension需要的cpp文件
test_cpp_extension_cuda.py: 测试python文件
run_cpp_extension_cuda_demo.sh: build及测试脚本