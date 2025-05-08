#include "headers.h"

void check_cuda_device() {
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  // printf("GetDeviceProperties = %d.%d\n", prop->major, prop->minor);
  TORCH_CHECK(prop->major==8, "The major version of A100 is 8!, but now is ", prop->major);
  TORCH_CHECK(prop->minor==0, "The minor version of A100 is 0!, but now is ", prop->minor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("check_cuda_device", &check_cuda_device, "check the cuda device properties.");
}
