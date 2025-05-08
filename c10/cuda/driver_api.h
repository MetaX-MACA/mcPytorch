#pragma once
#include <cuda.h>
#define NVML_NO_UNVERSIONED_FUNC_DEFS
#include <nvml.h>

#ifdef USE_MACA
  #include <mxsml/MxSml.h>
#endif

#define C10_CUDA_DRIVER_CHECK(EXPR)                                        \
  do {                                                                     \
    CUresult __err = EXPR;                                                 \
    if (__err != CUDA_SUCCESS) {                                           \
      const char* err_str;                                                 \
      CUresult get_error_str_err C10_UNUSED =                              \
          c10::cuda::DriverAPI::get()->cuGetErrorString_(__err, &err_str); \
      if (get_error_str_err != CUDA_SUCCESS) {                             \
        AT_ERROR("CUDA driver error: unknown error");                      \
      } else {                                                             \
        AT_ERROR("CUDA driver error: ", err_str);                          \
      }                                                                    \
    }                                                                      \
  } while (0)

#ifdef USE_MACA
  #define C10_LIBCUDA_DRIVER_API(_)              \
    _(cuMemAddressReserve, wcuMemAddressReserve) \
    _(cuMemRelease, wcuMemRelease)               \
    _(cuMemMap, wcuMemMap)                       \
    _(cuMemAddressFree, wcuMemAddressFree)       \
    _(cuMemSetAccess, wcuMemSetAccess)           \
    _(cuMemUnmap, wcuMemUnmap)                   \
    _(cuMemCreate, wcuMemCreate)                 \
    _(cuGetErrorString, wcuGetErrorString)

  #define C10_NVML_DRIVER_API(_)                                                      \
    _(nvmlInit_v2, mxSmlInit)                                                         \
    _(nvmlDeviceGetHandleByPciBusId_v2, mxSmlExDeviceGetHandleByUUID)                 \
    _(nvmlDeviceGetNvLinkRemoteDeviceType, mxSmlExDeviceGetMetaXLinkRemoteDeviceType) \
    _(nvmlDeviceGetNvLinkRemotePciInfo_v2, mxSmlExDeviceGetMetaXLinkRemotePciInfo_v2) \
    _(nvmlDeviceGetComputeRunningProcesses, mxSmlExDeviceGetComputeRunningProcesses)
#else
  #define C10_LIBCUDA_DRIVER_API(_) \
    _(cuMemAddressReserve)          \
    _(cuMemRelease)                 \
    _(cuMemMap)                     \
    _(cuMemAddressFree)             \
    _(cuMemSetAccess)               \
    _(cuMemUnmap)                   \
    _(cuMemCreate)                  \
    _(cuGetErrorString)

  #define C10_NVML_DRIVER_API(_)           \
    _(nvmlInit_v2)                         \
    _(nvmlDeviceGetHandleByPciBusId_v2)    \
    _(nvmlDeviceGetNvLinkRemoteDeviceType) \
    _(nvmlDeviceGetNvLinkRemotePciInfo_v2) \
    _(nvmlDeviceGetComputeRunningProcesses)
#endif


namespace c10::cuda {

#ifdef USE_MACA
  struct DriverAPI {
  #define CREATE_MEMBER(name,name1) decltype(&name1) name##_;
    C10_LIBCUDA_DRIVER_API(CREATE_MEMBER)
    C10_NVML_DRIVER_API(CREATE_MEMBER)
  #undef CREATE_MEMBER
    static DriverAPI* get();
    static void* get_nvml_handle();
  };
#else
  struct DriverAPI {
  #define CREATE_MEMBER(name) decltype(&name) name##_;
    C10_LIBCUDA_DRIVER_API(CREATE_MEMBER)
    C10_NVML_DRIVER_API(CREATE_MEMBER)
  #undef CREATE_MEMBER
    static DriverAPI* get();
    static void* get_nvml_handle();
  };
#endif

} // namespace c10::cuda
