import torch
import torch.utils.cpp_extension
from torch.utils.cpp_extension import CUDA_HOME
import torch_test_cpp_extension_cuda.cuda as cuda_extension

def test_cuda_extension():

    x = torch.zeros(100, device="cuda", dtype=torch.float32)
    y = torch.zeros(100, device="cuda", dtype=torch.float32)

    z = cuda_extension.sigmoid_add(x, y).cpu()

    # 2 * sigmoid(0) = 2 * 0.5 = 1
    # print("Diff: ", z - torch.ones_like(z))

    return torch.equal(z, torch.ones_like(z))

# JIT compiling extensions
def test_jit_cuda_extension():
    # NOTE: The name of the extension must equal the name of the module.
    module = torch.utils.cpp_extension.load(
        name="test_jit_cpp_extension_cuda",
        sources=[
            "cuda_extension.cpp",
            "cuda_extension.cu",
        ],
        extra_cuda_cflags=["-O2"],
        verbose=True,
        keep_intermediates=False,
    )

    x = torch.zeros(100, device="cuda", dtype=torch.float32)
    y = torch.zeros(100, device="cuda", dtype=torch.float32)

    z = module.sigmoid_add(x, y).cpu()

    return torch.equal(z, torch.ones_like(z))

# JIT inline compiling extensions
def test_inline_jit_compile_extension_cuda():
    cuda_source = """
    __global__ void cos_add_kernel(
        const float* __restrict__ x,
        const float* __restrict__ y,
        float* __restrict__ output,
        const int size) {
      const auto index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index < size) {
        output[index] = __cosf(x[index]) + __cosf(y[index]);
      }
    }

    torch::Tensor cos_add(torch::Tensor x, torch::Tensor y) {
      auto output = torch::zeros_like(x);
      const int threads = 1024;
      const int blocks = (output.numel() + threads - 1) / threads;
      cos_add_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), output.data_ptr<float>(), output.numel());
      return output;
    }
    """

    # Here, the C++ source need only declare the function signature.
    cpp_source = "torch::Tensor cos_add(torch::Tensor x, torch::Tensor y);"

    module = torch.utils.cpp_extension.load_inline(
        name="inline_jit_extension_cuda",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["cos_add"],
        verbose=True,
    )

    x = torch.randn(4, 4, device="cuda", dtype=torch.float32)
    y = torch.randn(4, 4, device="cuda", dtype=torch.float32)

    z = module.cos_add(x, y)

    # use allclose for cos add
    return torch.allclose(z, x.cos() + y.cos())

# cudnn test
def test_jit_cudnn_extension():
    # implementation of cuDNN ReLU
    module = torch.utils.cpp_extension.load(
        name="torch_test_cudnn_extension",
        sources=["cudnn_extension.cpp"],
        verbose=True,
        with_cuda=True,
    )

    x = torch.randn(100, device="cuda", dtype=torch.float32)
    y = torch.zeros(100, device="cuda", dtype=torch.float32)
    module.cudnn_relu(x, y)  # y=relu(x)
    return torch.equal(torch.nn.functional.relu(x), y)

def test_cuda_device():
    module = torch.utils.cpp_extension.load(
        name="torch_test_cuda_device",
        sources=[
            "cuda_device.cpp",
        ],
        verbose=True,
        with_cuda=True,
        keep_intermediates=False,
    )
    module.check_cuda_device()

def main():
    TEST_CUDA = torch.cuda.is_available() and CUDA_HOME is not None
    TEST_CUDNN = torch.cuda.is_available() and torch.backends.cudnn.is_available()

    if not TEST_CUDA:
      print("CUDA NOT FOUND! SKIP TESTS!")
    else:
      test_cuda_device()

      if not test_cuda_extension():
        return 1
      elif not test_jit_cuda_extension():
        return 2
      elif not test_inline_jit_compile_extension_cuda():
        return 3

    if not TEST_CUDNN:
       print("CUDNN NOT FOUND! SKIP TESTS!")
    else:
      if not test_jit_cudnn_extension():
        return 4
    
    return 0

ret = main()
exit(ret)
