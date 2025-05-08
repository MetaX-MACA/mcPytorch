## 概述

PyTorch 是一款流行的开源机器学习库，被广泛用于深度学习、自然语言处理、计算机视觉等任务，并已成为许多研究人员和开发人员在人工智能和机器学习领域的首选工具之一。

本工程在 PyTorch 2.4 的基础上增加了对沐曦 (MetaX，https://www.metax-tech.com/) GPU 的支持。MetaX GPU 提供了和 cuda 生态高度兼容的软件栈，包括驱动、编译器以及各类算子库，用户可以使用和 cuda 类似的支持方式完成对 PyTorch 2.4 的 MetaX GPU backend 支持。

## 安装

安装前需要首先完成 MetaX 软件栈的环境准备，包括驱动，编译器和算子库。

### 发布包

```shell
python -m pip install torch-2.4.0*.whl
```

### 源码编译
源码编译之前需要先同步 submodule。
```shell
git submodule update --init --recursive
```

本工程借助 cuBridge 项目（https://gitee.com/p4ul/cu-bridge）以最小成本完成 MetaX Pytorch 的构建。开始构建前用户需要参考 cuBridge 文档准备好 cuBridge 使用环境 (包括 cuda-toolkit、cudnn 和 nccl， 建议版本 cuda11.6 + cudnn8.5 + nccl 2.12)。

编译之前请检查以下环境是否已经安装好。
- 检查 cuda 环境, 通过检查以下文件是否存在，确保已经安装 cuda、cudnn 以及 nccl。
```
ls -l /usr/local/cuda/lib64/libcublas.so
ls -l /usr/local/cuda/lib64/libcudnn.so
ls -l /usr/local/cuda/lib64/libnccl.so
```

- 检查 maca 环境（默认安装在 /opt/maca 目录下）。
```
ls -l /opt/maca/tools/cu-bridge/tools/cmake_maca
ls -l /opt/maca/tools/cu-bridge/tools/make_maca
```

环境准备好后就可以开始编译。在项目根目录执行：
```shell
bash maca_tools/build_and_run_impl.sh  \
    --maca_path /opt/maca/             \  # 指定安装的 MetaX 软件栈
    --py_setup_cmd bdist_wheel            # 生成安装包，也可以使用 --py_set_cmd install 在构建成功后直接安装 PyTorch 到当前 Python 环境
```
成功运行结束后在 dist 目录生成安装包。

更多选项可以参考：
```shell
bash maca_tools/build_and_run_impl.sh --help
```

推荐使用 Python3.8/3.10 + Ubuntu20.04 环境构建。

### 镜像及容器
TODO

### 安装验证

运行前需要导入必须的环境变量：
```shell
    export MACA_PATH=/opt/maca
    export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib
    export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
```
然后执行：
```shell
    python -c "import torch; print(torch.ones(2).cuda())"
```
会得到如下打印输出：
```shell
    tensor([1., 1.], device='cuda:0')
```
即表明安装成功。
