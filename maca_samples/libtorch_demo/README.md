# Instruction for running libtorch_demo

本示例要求具备在 maca 平台上利用 cu-bridge 进行编译的环境，因此除去 maca 平台运行环境之外，运行本示例之前还需检查以下环境：

- 检查是否安装了 CUDA 环境 (包括 cuda-toolkit、cudnn 和 nccl， 建议版本 cuda11.6 + cudnn8.5 + nccl 2.12)
- 确认 /usr/local/cuda 文件夹是否存在，如果没有，需要软链接到 cuda 的安装目录

以上环境如果没有问题，在 cpp_extension_cuda_demo 目录下执行：

```
bash run_cpp_extension_cuda_demo.sh
```
