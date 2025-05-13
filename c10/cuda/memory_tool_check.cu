#include <c10/cuda/memory_tool_check.h>
#include <c10/macros/Macros.h>

__global__ void memory_tool_check(char* x, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size / sizeof(char)) {
    if (x[idx] != 0x01) {
      printf("PyTorch memory tools. Error: memory violation index:%ld\n", idx);
      CUDA_KERNEL_ASSERT(0);
    }
  }
}

__global__ void memory_tool_set(char* x, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size / sizeof(char)) {
    x[idx] = 0x01;
  }
}

void launch_memory_tool_check(void* x, size_t size) {
  memory_tool_check<<<(size/(sizeof(char)*1024) + 1), 1024>>>(static_cast<char*>(x), size);
  cudaError_t error_code = cudaGetLastError();
  if (error_code != cudaSuccess) {
    void *buffer[256];
    int nptrs = backtrace(buffer, 256);
    char **strings = backtrace_symbols(buffer, nptrs);
    if (strings != NULL) {
      printf("Stack trace:\n");
      for (int i = 0; i < nptrs; i++) {
          printf("%s\n", strings[i]);
      }
    }
    free(strings);
  }
}

void launch_memory_tool_set(void* x, size_t size) {
  memory_tool_set<<<(size/(sizeof(char)*1024) + 1), 1024>>>(static_cast<char*>(x), size);
}