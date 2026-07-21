# Instruction for running cpp_extension_cuda_demo

本示例要求具备在 maca 平台上利用 cu-bridge 进行编译的环境，因此除去 maca 平台运行环境之外，运行本示例之前还需检查以下环境：

- 检查${your_maca_install_dir}/tools/cu-bridge/ 目录是否存在

以上环境如果没有问题，在 cpp_extension_cuda_demo 目录下执行：

```
bash run_cpp_extension_cuda_demo.sh
```

使用 setuptools build 使用到的文件包括：

    cpp_extension_cuda_demo
    ├── cuda_extension.cpp
    ├── cuda_extension_kernel.cu
    ├── headers.h
    ├── mypackage
    │   └── __init__.py
    └── setup.py

- cuda_device.cpp: 测试获取device信息需要的cpp文件
- cuda_extension.cu: jit compile extension需要的.cu文件
- cudnn_extension.cpp: cudnn extension需要的cpp文件
- test_cpp_extension_cuda.py: 测试python文件
- run_cpp_extension_cuda_demo.sh: build及测试脚本