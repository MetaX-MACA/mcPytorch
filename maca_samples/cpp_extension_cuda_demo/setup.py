import sys
import torch.cuda
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME

print("xxx: ", CUDA_HOME)

if sys.platform == 'win32':
    vc_version = os.getenv('VCToolsVersion', '')
    if vc_version.startswith('14.16.'):
        CXX_FLAGS = ['/sdl']
    else:
        CXX_FLAGS = ['/sdl', '/permissive-']
else:
    CXX_FLAGS = ['-g']

USE_NINJA = os.getenv('USE_NINJA') == '1'
ext_modules = []

if torch.cuda.is_available() and (CUDA_HOME is not None):
    extension = CUDAExtension(
        'torch_test_cpp_extension_cuda.cuda', [
            'cuda_extension.cpp',
            'cuda_extension_kernel.cu',
        ],
        extra_compile_args={'cxx': CXX_FLAGS,
                            'nvcc': ['-O2']})
    ext_modules.append(extension)

# packages name can't be same as setup name
setup(
    name='torch_test_cpp_extension_cuda',
    packages=['mypackage'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=USE_NINJA)})