if(USE_CUDA AND NOT USE_MACA)
  add_library(c10d_cuda_test CUDATest.cu)
  if(USE_MACA)
    target_compile_definitions(c10d_cuda_test PRIVATE USE_MACA)
    target_compile_definitions(c10d_cuda_test PRIVATE NV_ARCH_A100)
  endif()
  target_include_directories(c10d_cuda_test PRIVATE $<BUILD_INTERFACE:${TORCH_SRC_DIR}/csrc/distributed>)
  target_link_libraries(c10d_cuda_test torch_cuda)
  add_dependencies(c10d_cuda_test torch_cuda)
endif()

function(c10d_add_test test_src)
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} "${test_src}")
  if(USE_MACA)
    target_compile_definitions(${test_name} PRIVATE USE_MACA)
    target_compile_definitions(${test_name} PRIVATE NV_ARCH_A100)
  endif()
  target_include_directories(${test_name} PRIVATE $<BUILD_INTERFACE:${TORCH_SRC_DIR}/csrc/distributed>)
  target_link_libraries(${test_name} ${ARGN})
  if(NOT WIN32)
    target_link_libraries(${test_name} pthread)
  endif()
  add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
endfunction()

if(NOT USE_MACA)
  c10d_add_test(FileStoreTest.cpp torch_cpu gtest_main)
  c10d_add_test(TCPStoreTest.cpp torch_cpu gtest_main)
  if(INSTALL_TEST)
    install(TARGETS FileStoreTest DESTINATION bin)
    install(TARGETS TCPStoreTest DESTINATION bin)
  endif()
  if(NOT WIN32)
    c10d_add_test(HashStoreTest.cpp torch_cpu gtest_main)
    if(INSTALL_TEST)
      install(TARGETS HashStoreTest DESTINATION bin)
    endif()
  endif()

  if(USE_CUDA)
    if(USE_GLOO AND USE_C10D_GLOO)
      c10d_add_test(ProcessGroupGlooTest.cpp torch_cpu c10d_cuda_test gtest_main)
      if(INSTALL_TEST)
        install(TARGETS ProcessGroupGlooTest DESTINATION bin)
      endif()
      c10d_add_test(ProcessGroupGlooAsyncTest.cpp torch_cpu c10d_cuda_test gtest_main)
    endif()
    if(USE_NCCL AND USE_C10D_NCCL)
      # NCCL is a private dependency of libtorch, but the tests include some
      # private headers of libtorch, which in turn include NCCL. As a hacky
      # alternative to making NCCL a public dependency of libtorch, we make it
      # a private dependency of the tests as well.
      c10d_add_test(
        ProcessGroupNCCLTest.cpp
        torch_cpu c10d_cuda_test gtest_main __caffe2_nccl)
      c10d_add_test(
        ProcessGroupNCCLErrorsTest.cpp
        torch_cpu c10d_cuda_test gtest_main __caffe2_nccl)
      if(INSTALL_TEST)
        install(TARGETS ProcessGroupNCCLTest DESTINATION bin)
        install(TARGETS ProcessGroupNCCLErrorsTest DESTINATION bin)
        install(TARGETS c10d_cuda_test DESTINATION lib)
      endif()
    endif()
    if(USE_UCC AND USE_C10D_UCC)
      # UCC is a private dependency of libtorch, but the tests include some
      # private headers of libtorch, which in turn include UCC. As a hacky
      # alternative to making UCC a public dependency of libtorch, we make it
      # a private dependency of the tests as well.
      c10d_add_test(
        ProcessGroupUCCTest.cpp
        torch_cpu c10d_cuda_test gtest_main __caffe2_ucc)
      if(INSTALL_TEST)
        install(TARGETS ProcessGroupUCCTest DESTINATION bin)
        install(TARGETS c10d_cuda_test DESTINATION lib)
      endif()
    endif()
  else()
    if(USE_GLOO AND USE_C10D_GLOO)
      c10d_add_test(ProcessGroupGlooTest.cpp torch_cpu gtest_main)
    endif()
  endif()
endif()

if(USE_MPI AND USE_C10D_MPI)
  add_definitions(-DMPIEXEC=${MPIEXEC})
  # MPI is a private dependency of libtorch, but the tests include some
  # private headers of libtorch, which in turn include MPI. As a hacky
  # alternative to making MPI a public dependency of libtorch, we make it
  # a private dependency of the tests as well.
  c10d_add_test(ProcessGroupMPITest.cpp torch_cpu ${MPI_CXX_LIBRARIES})
  if(INSTALL_TEST)
    install(TARGETS ProcessGroupMPITest DESTINATION bin)
  endif()
endif()

if(LINUX AND USE_GLOO AND USE_C10D_GLOO AND NOT USE_MACA)
  add_executable(example_allreduce example/allreduce.cpp)
  if(USE_MACA)
    target_compile_options(example_allreduce PRIVATE "-DUSE_MACA")
    target_compile_options(example_allreduce PRIVATE "-DNV_ARCH_A100")
  endif()
  target_include_directories(example_allreduce PRIVATE $<BUILD_INTERFACE:${TORCH_SRC_DIR}/csrc/distributed>)
  target_link_libraries(example_allreduce pthread torch_cpu)
  if(USE_CUDA)
    target_link_libraries(example_allreduce torch_cuda)
  endif()
endif()