find_package(MKL QUIET)

if(NOT TARGET caffe2::mkl)
  add_library(caffe2::mkl INTERFACE IMPORTED)
endif()

if(NOT DEFINED ENV{PYTORCH_ENABLE_MKL_DYNAMIC_LIB} AND DEFINED pytorch_compile_flag)
  foreach(ITEM ${MKL_LIBRARIES})
     message(${ITEM})
     STRING(FIND ${ITEM} "libmkl_intel_lp64.so" po0)
     STRING(FIND ${ITEM} "libmkl_gnu_thread.so" po1)
     STRING(FIND ${ITEM} "libmkl_core.so" po2)
     if(${po0} GREATER -1)
       set(MKL_FOUND FALSE)
     endif()
     if(${po1} GREATER -1)
       set(MKL_FOUND FALSE)
     endif()
     if(${po2} GREATER -1)
       set(MKL_FOUND FALSE)
     endif()
  endforeach()
  if(NOT ${MKL_FOUND})
   set(MKL_LIBRARIES "")
   set(MKL_THREAD_LIB "")
   set(MKL_INCLUDE_DIR "")
   set(MKL_ROOT "")
 endif()
endif()

set_property(
  TARGET caffe2::mkl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${MKL_INCLUDE_DIR})
set_property(
  TARGET caffe2::mkl PROPERTY INTERFACE_LINK_LIBRARIES
  ${MKL_LIBRARIES} ${MKL_THREAD_LIB})
# TODO: This is a hack, it will not pick up architecture dependent
# MKL libraries correctly; see https://github.com/pytorch/pytorch/issues/73008
set_property(
  TARGET caffe2::mkl PROPERTY INTERFACE_LINK_DIRECTORIES
  ${MKL_ROOT}/lib)
